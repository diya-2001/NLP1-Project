"""The three architectures (A1-A3) behind one interface: ``fit(X, y)`` / ``predict(X)``.

A1 loglin : multinomial logistic (softmax) regression, scikit-learn, sparse input.
A2 mlp    : the tutorial's one-hidden-layer ``part1.smlp.SparseMLP`` (reused unchanged).
A3 deep   : ours, two hidden layers with a sparse first layer.

Neural models are trained with our own loop (``train_torch``) because the tutorial's
``train_neural_network_model`` (a) builds a dense vector per document (≈8 GB for our pool) and
(b) chooses the best epoch on the validation fold *and then reports that same fold*, which is
optimistic. We instead carve a small **inner hold-out** out of the training fold for early
stopping, so the validation fold is touched exactly once, at the end.
"""
from __future__ import annotations

import copy
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from part1.smlp import SparseMLP   # tutorial code, reused as-is

ARCHITECTURES = ("loglin", "mlp", "deep")

# Default hyperparameters (one place, so the notebook can print and justify them).
DEFAULTS = {
    "loglin": dict(C=1.0, max_iter=500),
    "mlp":    dict(hidden=128, dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=128,
                   max_epochs=20, patience=3, inner_frac=0.1),
    "deep":   dict(hidden=(256, 128), dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=128,
                   max_epochs=20, patience=3, inner_frac=0.1),
}


# --------------------------------------------------------------------------- A3 network
class DeepSparseMLP(nn.Module):
    """Two hidden layers. Layer 1 multiplies the sparse input directly (like SparseMLP).

    sparse h(x) --[V x H1]--> ReLU, dropout --[H1 x H2]--> ReLU, dropout --[H2 x K]--> log-softmax
    """

    def __init__(self, input_dim: int, hidden=(256, 128), num_classes: int = 28, dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden
        self.weight = nn.Parameter(torch.empty(input_dim, h1))
        self.bias = nn.Parameter(torch.zeros(h1))
        nn.init.xavier_uniform_(self.weight)
        self.rest = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, num_classes), nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        return self.rest(torch.sparse.mm(x, self.weight) + self.bias)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_count(arch: str, V: int, K: int = 28, **hp) -> int:
    """Number of trainable parameters, computed from the shapes (no model needed)."""
    hp = {**DEFAULTS.get(arch, {}), **hp}
    if arch == "loglin":
        return V * K + K
    if arch == "mlp":
        H = hp["hidden"]
        return V * H + H + H * K + K
    if arch == "deep":
        h1, h2 = hp["hidden"]
        return V * h1 + h1 + h1 * h2 + h2 + h2 * K + K
    raise ValueError(arch)


# --------------------------------------------------------------------------- batching
def csr_to_torch(X: sp.csr_matrix) -> torch.Tensor:
    """Same result as the tutorial's ``smlp.scipy_to_torch_sparse``, but without a slow Python list."""
    X = X.tocoo()
    idx = torch.from_numpy(np.vstack([X.row, X.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(X.data.astype(np.float32)), size=X.shape).coalesce()


def _batches(X: sp.csr_matrix, y: np.ndarray, batch_size: int, rng=None):
    """Yield (torch sparse batch, label tensor). Slicing CSR rows is fast, no per-row Python loop."""
    order = rng.permutation(X.shape[0]) if rng is not None else np.arange(X.shape[0])
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        yield csr_to_torch(X[idx]), torch.as_tensor(y[idx], dtype=torch.long)


@torch.no_grad()
def _predict_logprobs(model, X, batch_size=1024):
    model.eval()
    dummy = np.zeros(X.shape[0], dtype=np.int64)
    return torch.cat([model(xb) for xb, _ in _batches(X, dummy, batch_size)]).numpy()


def train_torch(model, X, y, *, lr, weight_decay, batch_size, max_epochs, patience, inner_frac,
                seed=0, verbose=False, **_):
    """AdamW + NLLLoss with early stopping on an inner hold-out carved from (X, y).

    Returns the model with the weights of its best epoch (highest inner hold-out macro-F1) and a
    per-epoch history list.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx_fit, idx_es = train_test_split(np.arange(len(y)), test_size=inner_frac,
                                       stratify=y, random_state=seed)
    X_fit, y_fit, X_es, y_es = X[idx_fit], y[idx_fit], X[idx_es], y[idx_es]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()
    best_score, best_state, bad_epochs, history = -np.inf, None, 0, []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for xb, yb in _batches(X_fit, y_fit, batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)

        lp = _predict_logprobs(model, X_es)
        es_loss = float(nn.functional.nll_loss(torch.from_numpy(lp), torch.from_numpy(y_es)))
        es_f1 = f1_score(y_es, lp.argmax(1), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total / n, "holdout_loss": es_loss,
                        "holdout_macro_f1": es_f1, "seconds": time.time() - t0})
        if verbose:
            h = history[-1]
            print(f"epoch {epoch:2d} | train loss {h['train_loss']:.3f} | hold-out loss {es_loss:.3f}"
                  f" | hold-out macro-F1 {es_f1:.3f} | {h['seconds']:.0f}s")

        # early stopping on hold-out macro-F1, our evaluation metric (higher is better)
        if es_f1 > best_score + 1e-4:
            best_score, best_state, bad_epochs = es_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- unified wrapper
class Classifier:
    """``Classifier(arch, **hyperparams).fit(X, y).predict(X)`` for all three architectures."""

    def __init__(self, arch: str, num_classes: int = 28, seed: int = 0, n_threads: int | None = None,
                 verbose: bool = False, **hp):
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch!r}")
        self.arch, self.K, self.seed, self.verbose = arch, num_classes, seed, verbose
        self.hp = {**DEFAULTS[arch], **hp}
        self.n_threads = n_threads
        self.history = []
        self.converged = True

    def fit(self, X: sp.csr_matrix, y: np.ndarray):
        if self.arch == "loglin":
            # lbfgs: a quasi-Newton solver; handles sparse input and the multinomial (softmax) loss.
            # C = inverse L2-regularisation strength (sklearn default 1.0).
            self.model = LogisticRegression(C=self.hp["C"], max_iter=self.hp["max_iter"], solver="lbfgs")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                self.model.fit(X, y)
            self.converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
            self.history = [{"n_iter": int(self.model.n_iter_[0])}]
            return self

        if self.n_threads:
            torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.seed)
        V = X.shape[1]
        if self.arch == "mlp":
            net = SparseMLP(V, self.hp["hidden"], self.K, dropout=self.hp["dropout"])
        else:
            net = DeepSparseMLP(V, tuple(self.hp["hidden"]), self.K, dropout=self.hp["dropout"])
        self.model, self.history = train_torch(net, X, y, seed=self.seed, verbose=self.verbose, **self.hp)
        return self

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        if self.arch == "loglin":
            return self.model.predict_proba(X)
        return np.exp(_predict_logprobs(self.model, X))

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.predict_proba(X).argmax(1)
