"""The three architectures (A1-A3) behind one interface: ``fit(X, y)`` / ``predict(X)``.

A1 loglin : multinomial logistic (softmax) regression, scikit-learn, sparse input.
A2 mlp    : the tutorial's one-hidden-layer ``part1.smlp.SparseMLP`` (reused unchanged).
A3 deep   : ours, the MLP plus one residual, LayerNorm-ed hidden layer (sparse first layer,
            input token dropout).

The neural models (A2, A3) run on ``DEVICE``: the GPU (CUDA) if there is one, otherwise the CPU.
The notebook may overwrite it (``models.DEVICE = ...``). A1 always runs on the CPU (sklearn).

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

# Where the neural models train and predict. Read at call time, so setting models.DEVICE works.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters (one place, so the notebook can print and justify them).
DEFAULTS = {
    "loglin": dict(C=1.0, max_iter=500),
    "mlp":    dict(hidden=128, dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=128,
                   max_epochs=20, patience=3, inner_frac=0.1),
    "deep":   dict(hidden=(128, 128), dropout=0.3, input_dropout=0.2, rezero=True, lr=1e-3,
                   weight_decay=1e-4, batch_size=128, max_epochs=20, patience=3, inner_frac=0.1),
}


# --------------------------------------------------------------------------- A3 network
class DeepSparseMLP(nn.Module):
    """The MLP plus one residual hidden layer. Layer 1 multiplies the sparse input directly (like SparseMLP).

    sparse h(x) --token dropout--> [V x H] --> LayerNorm, ReLU, dropout = z1
    z2 = z1 + dropout(ReLU(LayerNorm([H x H] z1)))          (residual shortcut; with rezero: z1 + alpha * ...)
    z2 --[H x K]--> log-softmax

    Why each piece (our first version, a plain 256 -> 128 stack, scored below the MLP, worst on TF-IDF):
    - LayerNorm: TF-IDF rows have length 1 (counts ~9.7), so the bio-dependent signal reaching layer 2
      was tiny next to the biases; normalising makes the model behave the same for every representation.
    - Residual shortcut: if the extra layer learns nothing useful, the model reduces to the MLP.
    - Input token dropout: hides a random share of each bio's tokens while training, which regularises the
      first layer (the 10,000 x H word lists, >90% of all weights) directly.
    - ReZero (optional, Bachlechner et al. 2020): the extra layer is multiplied by a learnable alpha that
      starts at 0, so training starts from exactly an MLP and only uses the extra layer as far as it helps.
    """

    def __init__(self, input_dim: int, hidden=(128, 128), num_classes: int = 28, dropout: float = 0.3,
                 input_dropout: float = 0.2, rezero: bool = False):
        super().__init__()
        h1, h2 = hidden
        if h1 != h2:
            raise ValueError("the residual shortcut needs equal hidden sizes, e.g. (128, 128)")
        self.weight = nn.Parameter(torch.empty(input_dim, h1))
        self.bias = nn.Parameter(torch.zeros(h1))
        nn.init.xavier_uniform_(self.weight)
        self.norm1 = nn.LayerNorm(h1)
        self.block = nn.Sequential(nn.Linear(h1, h2), nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(h2, num_classes), nn.LogSoftmax(dim=-1))
        self.input_dropout = input_dropout
        self.alpha = nn.Parameter(torch.zeros(1)) if rezero else None

    def _drop_tokens(self, x):
        """Training only: remove each non-zero input entry (a token of a bio) with prob. input_dropout."""
        x = x.coalesce()
        keep = torch.rand_like(x.values()) >= self.input_dropout
        return torch.sparse_coo_tensor(x.indices()[:, keep], x.values()[keep], x.shape)

    def forward(self, x):
        if self.training and self.input_dropout > 0:
            x = self._drop_tokens(x)
        z1 = self.drop(torch.relu(self.norm1(torch.sparse.mm(x, self.weight) + self.bias)))
        z2 = z1 + (self.alpha * self.block(z1) if self.alpha is not None else self.block(z1))
        return self.out(z2)


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
        # layer 1 + LayerNorm (2*h1) + residual layer + LayerNorm (2*h2) + output layer
        return V * h1 + h1 + 2 * h1 + h1 * h2 + h2 + 2 * h2 + h2 * K + K + int(hp["rezero"])  # + alpha
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
        yield csr_to_torch(X[idx]).to(DEVICE), torch.as_tensor(y[idx], dtype=torch.long).to(DEVICE)


@torch.no_grad()
def _predict_logprobs(model, X, batch_size=1024):
    model.eval()
    dummy = np.zeros(X.shape[0], dtype=np.int64)
    return torch.cat([model(xb).cpu() for xb, _ in _batches(X, dummy, batch_size)]).numpy()


def train_torch(model, X, y, *, lr, weight_decay, batch_size, max_epochs, patience, inner_frac,
                seed=0, verbose=False, **_):
    """AdamW + NLLLoss with early stopping on an inner hold-out carved from (X, y).

    Returns the model with the weights of its best epoch (highest inner hold-out macro-F1) and a
    per-epoch history list.
    """
    torch.manual_seed(seed)                 # also seeds every GPU
    rng = np.random.default_rng(seed)
    model = model.to(DEVICE)
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
            net = DeepSparseMLP(V, tuple(self.hp["hidden"]), self.K, dropout=self.hp["dropout"],
                                input_dropout=self.hp["input_dropout"], rezero=self.hp["rezero"])
        self.model, self.history = train_torch(net, X, y, seed=self.seed, verbose=self.verbose, **self.hp)
        return self

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        if self.arch == "loglin":
            return self.model.predict_proba(X)
        return np.exp(_predict_logprobs(self.model, X))

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.predict_proba(X).argmax(1)
