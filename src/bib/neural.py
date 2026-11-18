"""Part 2 neural models over pretrained word vectors: embed-and-pool and recurrent (LSTM/GRU) encoders.

How a bio reaches a model
    tokens (bib.embeddings.tokenize, cached)  --back-off-->  row ids into a vector TABLE
    TABLE row 0 = <pad>: zeros, used to make the bios of a batch equally long (ignored by every model);
    TABLE row 1 = <unk>: the mean of all word vectors, as in the tutorial's ``PretrainedEmbeddings``;
    TABLE rows 2.. = the words of our bios that have a pretrained vector.

Two kinds of input
    * DENSE features: with FROZEN embeddings an embed-and-pool model always computes the same pooled vector
      for a bio, so we compute it once (``pool_features``) and train only the classifier on it. This gives
      the same model as the tutorial's ``EmbedAndPoolEncoder(freeze_embeddings=True)``, but an epoch takes
      seconds instead of minutes.
    * ID SEQUENCES: with trainable embeddings, and for every LSTM/GRU, the model looks the vectors up itself.

Training (``fit``) follows Part 1's recipe: AdamW, batch 128, at most 20 epochs, early stopping on the
macro-F1 of an inner 10% hold-out carved out of the training fold (patience 3). The validation fold is used
once, at the end. (The tutorial's ``experiment.train`` picks the best epoch on the validation fold itself,
which is optimistic, so we don't reuse it.)
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import embeddings as emb

PAD, UNK = 0, 1

# Where models train; set per system by ``device_for`` (the notebook or a script may override it).
DEVICE = torch.device("cpu")


def device_for(spec: dict) -> torch.device:
    """CUDA if there is one (Snellius). On a Mac: the GPU (MPS) for LSTM/GRU, which our timing test found
    2-3.5x faster there; the CPU for embed-and-pool, whose small models run faster on the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if spec.get("kind") == "rnn" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================================================================== vocabulary, ids, features
def build_table(words: list[str], vectors: np.ndarray):
    """Put <pad> (zeros) and <unk> (mean vector) in front of the word vectors. Returns ({word: row}, table)."""
    unk = vectors.mean(0)
    table = np.vstack([np.zeros_like(unk), unk, vectors]).astype(np.float32)
    return {w: i + 2 for i, w in enumerate(words)}, table


def to_ids(token_lists, index: dict, use_backoff: bool = True) -> list[np.ndarray]:
    """One int32 array of table rows per bio; words without a vector become UNK."""
    known = index.keys()
    out = []
    for toks in token_lists:
        if use_backoff:
            toks = emb.backoff(toks, known)
        out.append(np.fromiter((index.get(t, UNK) for t in toks), dtype=np.int32, count=len(toks)))
    return out


def save_ids(path, ids):
    np.savez(path, flat=np.concatenate(ids), lengths=np.array([len(a) for a in ids], dtype=np.int32))


def load_ids(path) -> list[np.ndarray]:
    d = np.load(path)
    return np.split(d["flat"], np.cumsum(d["lengths"])[:-1])


def load_table_and_ids(emb_name: str, results_dir: Path, token_lists=None):
    """Vector table + per-bio ids for one embedding model (ids cached in ``ids_{emb_name}.npz``)."""
    results_dir = Path(results_dir)
    d = np.load(results_dir / f"emb_{emb_name}.npz", allow_pickle=True)
    index, table = build_table(list(d["words"]), d["vectors"])
    path = results_dir / f"ids_{emb_name}.npz"
    if path.exists():
        ids = load_ids(path)
    else:
        if token_lists is None:
            raise FileNotFoundError(f"{path} missing; pass token_lists to build it")
        ids = to_ids(token_lists, index)
        save_ids(path, ids)
    return table, ids


def reduce_table(table: np.ndarray, dims: int, seed: int = 0) -> np.ndarray:
    """Shrink the word vectors to ``dims`` numbers with PCA (keeps the directions with the most variance).

    Used to test embedding SIZE with the corpus held fixed. PCA centres the vectors, so the <unk> row
    (the mean) becomes zeros; <pad> stays zeros.
    """
    from sklearn.decomposition import PCA
    red = PCA(dims, random_state=seed).fit_transform(table[2:]).astype(np.float32)
    return np.vstack([np.zeros((1, dims), np.float32), red.mean(0, keepdims=True), red])


def pad(seqs) -> torch.Tensor:
    """List of id arrays -> (batch, longest) LongTensor, shorter bios filled with PAD."""
    L = max(1, max(len(s) for s in seqs))
    x = np.zeros((len(seqs), L), dtype=np.int64)
    for r, s in enumerate(seqs):
        x[r, :len(s)] = s
    return torch.from_numpy(x)


def pool(e: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool word vectors e (batch, length, dims) over the real (non-pad) positions.

    mean: average of the word vectors (what the tutorial's EmbedAndPoolEncoder does);
    max: per dimension, the largest value in the bio (picks up a single strong word);
    meanmax: both, side by side (twice as many numbers).
    """
    m = mask.unsqueeze(-1)
    mean = (e * m).sum(1) / m.sum(1).clamp_min(1)
    if pooling == "mean":
        return mean
    mx = e.masked_fill(~m, float("-inf")).max(1).values
    mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)    # a bio with no tokens -> zeros
    if pooling == "max":
        return mx
    if pooling == "meanmax":
        return torch.cat([mean, mx], -1)
    raise ValueError(pooling)


@torch.no_grad()
def pool_features(ids, table: np.ndarray, pooling: str = "mean", batch_size: int = 1024) -> np.ndarray:
    """The pooled vector of every bio, computed once (for FROZEN embeddings). Shape (n_bios, d or 2d)."""
    E = torch.from_numpy(table)
    order = np.argsort([len(s) for s in ids], kind="stable")     # similar lengths together -> little padding
    out = np.zeros((len(ids), table.shape[1] * (2 if pooling == "meanmax" else 1)), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        rows = order[i:i + batch_size]
        x = pad([ids[r] for r in rows])
        out[rows] = pool(E[x], x != PAD, pooling).numpy()
    return out


# =========================================================================== models
def make_head(in_dim: int, hidden: tuple, num_classes: int, dropout: float) -> nn.Sequential:
    """The classifier on top of a bio vector.

    hidden = ()        -> log-linear (one linear layer + log-softmax);
    hidden = (h,)      -> one tanh hidden layer;
    hidden = (h1, h2)  -> two tanh hidden layers, the shape of the tutorial's ``SoftmaxRegressor``.
    Dropout before every linear layer, as in the tutorial.
    """
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Dropout(dropout), nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Dropout(dropout), nn.Linear(d, num_classes), nn.LogSoftmax(dim=-1)]
    return nn.Sequential(*layers)


class Standardize(nn.Module):
    """Fixed (not learned) per-dimension (x - mean) / std, with mean and std from the training fold.

    Pooled fastText vectors have tiny, uneven scales; standardising them is what sklearn's StandardScaler
    did in the §3 quick test, and it lets one learning rate suit every pooling and embedding size.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class DenseModel(nn.Module):
    """Frozen embed-and-pool, trained on precomputed pooled vectors: standardise -> head."""

    def __init__(self, mean, std, hidden=(), num_classes=28, dropout=0.1):
        super().__init__()
        self.scale = Standardize(mean, std)
        self.head = make_head(len(mean), tuple(hidden), num_classes, dropout)

    def forward(self, x):
        return self.head(self.scale(x))


def _embedding(table: np.ndarray, mode: str) -> nn.Embedding:
    """mode: 'frozen' (pretrained, fixed), 'finetune' (pretrained, trained further), 'random' (trained from scratch).

    Trainable tables use sparse gradients on CPU/CUDA: a batch touches a few thousand of the 200k rows, so only
    those rows are updated (SparseAdam), which keeps fine-tuning fast. On a Mac GPU (MPS) ordinary dense
    gradients were faster in our test (61 vs 144 ms per LSTM step), so there we use those.
    """
    sparse = DEVICE.type != "mps"
    if mode == "frozen":
        return nn.Embedding.from_pretrained(torch.from_numpy(table), freeze=True, padding_idx=PAD)
    if mode == "finetune":
        return nn.Embedding.from_pretrained(torch.from_numpy(table.copy()), freeze=False, padding_idx=PAD,
                                            sparse=sparse)
    if mode == "random":
        w = torch.randn(table.shape) * float(table[2:].std())     # same scale as the pretrained vectors
        w[PAD] = 0
        return nn.Embedding(*table.shape, padding_idx=PAD, sparse=sparse, _weight=w)
    raise ValueError(mode)


class PoolModel(nn.Module):
    """Embed-and-pool with the embeddings inside the model (needed when they are trained).

    With mode='frozen' this computes exactly what DenseModel does on ``pool_features`` output.
    """

    def __init__(self, table, mean, std, pooling="mean", embeddings="finetune", hidden=(), num_classes=28,
                 dropout=0.1):
        super().__init__()
        self.embedding = _embedding(table, embeddings)
        self.pooling = pooling
        self.scale = Standardize(mean, std)
        self.head = make_head(len(mean), tuple(hidden), num_classes, dropout)

    def forward(self, x):
        return self.head(self.scale(pool(self.embedding(x), x != PAD, self.pooling)))


class RNNModel(nn.Module):
    """Embedding -> word dropout -> LSTM or GRU -> readout -> head.

    Generalises the tutorial's ``LSTMEncoder``: same packed sequences (padding is skipped) and the same word
    dropout (each real word's vector is zeroed with probability ``word_dropout`` while training), plus
    a GRU option, a bidirectional option (a second RNN reads the bio backwards) and a choice of readout:
    'last' = the final state(s), as in the tutorial; 'mean' = the average of the states after every word,
    so the start of a long bio is not forgotten.
    """

    def __init__(self, table, cell="lstm", hidden_size=128, bidirectional=False, embeddings="frozen",
                 word_dropout=0.0, hidden=(), num_classes=28, dropout=0.1, readout="last"):
        super().__init__()
        self.embedding = _embedding(table, embeddings)
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(table.shape[1], hidden_size, batch_first=True, bidirectional=bidirectional)
        self.bidirectional, self.word_dropout, self.readout = bidirectional, word_dropout, readout
        self.head = make_head(hidden_size * (2 if bidirectional else 1), tuple(hidden), num_classes, dropout)

    def forward(self, x):
        mask = x != PAD
        e = self.embedding(x)
        if self.training and self.word_dropout > 0:
            drop = (torch.rand(x.shape, device=x.device) < self.word_dropout) & mask
            e = e.masked_fill(drop.unsqueeze(-1), 0.0)
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.readout == "mean":        # average of the states after every real word
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            m = mask.unsqueeze(-1)
            return self.head((out * m).sum(1) / m.sum(1).clamp_min(1))
        if isinstance(h, tuple):          # the LSTM returns (hidden state, cell state)
            h = h[0]
        h = torch.cat([h[-2], h[-1]], -1) if self.bidirectional else h[-1]
        return self.head(h)


def n_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# =========================================================================== batching and training
class Inputs:
    """Either dense features (n x d array) or id sequences (list of arrays); serves batches by row."""

    def __init__(self, X=None, ids=None):
        self.X, self.ids = X, ids
        self.lengths = None if ids is None else np.array([len(s) for s in ids])

    def batch(self, rows) -> torch.Tensor:
        if self.X is not None:
            return torch.from_numpy(self.X[rows])
        return pad([self.ids[r] for r in rows])

    def order(self, rows, batch_size, rng=None):
        """Batches of rows. Training (rng given): random; sequences are first grouped by similar length
        within random chunks of 50 batches ("bucketing"), which cuts padding and LSTM time a lot."""
        rows = np.asarray(rows)
        if rng is None:
            if self.lengths is not None:
                rows = rows[np.argsort(self.lengths[rows], kind="stable")]
            return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        rows = rows[rng.permutation(len(rows))]
        if self.lengths is None:
            return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        batches, chunk = [], 50 * batch_size
        for i in range(0, len(rows), chunk):
            c = rows[i:i + chunk]
            c = c[np.argsort(self.lengths[c], kind="stable")]
            batches += [c[j:j + batch_size] for j in range(0, len(c), batch_size)]
        return [batches[k] for k in rng.permutation(len(batches))]


@torch.no_grad()
def predict_logprobs(model, inputs: Inputs, rows, batch_size: int = 512) -> np.ndarray:
    model.eval()
    rows = np.asarray(rows)
    out = np.zeros((len(rows), model.head[-2].out_features), dtype=np.float32)
    pos = {r: i for i, r in enumerate(rows)}
    for b in inputs.order(rows, batch_size):
        lp = model(inputs.batch(b).to(DEVICE)).cpu().numpy()
        out[[pos[r] for r in b]] = lp
    return out


def fit(model, inputs: Inputs, y: np.ndarray, rows, *, lr=1e-3, weight_decay=1e-4, batch_size=128,
        max_epochs=20, patience=3, inner_frac=0.1, clip=5.0, seed=0, verbose=False, max_minutes=None, **_):
    """Train on ``rows`` (a training fold) with early stopping on an inner hold-out; return (model, history).

    y: labels for ALL bios (indexed by row). ``max_minutes`` stops early (for timing tests only).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    fit_rows, es_rows = train_test_split(rows, test_size=inner_frac, stratify=y[rows], random_state=seed)
    model = model.to(DEVICE)

    sparse = [p for m in model.modules() if isinstance(m, nn.Embedding) and m.sparse
              for p in m.parameters() if p.requires_grad]
    dense = [p for p in model.parameters() if p.requires_grad and all(p is not q for q in sparse)]
    opts = [torch.optim.AdamW(dense, lr=lr, weight_decay=weight_decay)]
    if sparse:
        opts.append(torch.optim.SparseAdam(sparse, lr=lr))
    loss_fn = nn.NLLLoss()
    best, best_state, bad, history, t_start = -np.inf, None, 0, [], time.time()

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in inputs.order(fit_rows, batch_size, rng):
            xb = inputs.batch(b).to(DEVICE)
            yb = torch.as_tensor(y[b], dtype=torch.long, device=DEVICE)
            for o in opts:
                o.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(dense, clip)     # keeps RNN updates from exploding
            for o in opts:
                o.step()
            total += loss.item() * len(b)
            n += len(b)
            if max_minutes and time.time() - t_start > 60 * max_minutes:
                break

        lp = predict_logprobs(model, inputs, es_rows)
        es_loss = float(nn.functional.nll_loss(torch.from_numpy(lp), torch.as_tensor(y[es_rows])))
        es_f1 = f1_score(y[es_rows], lp.argmax(1), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "holdout_loss": es_loss,
                        "holdout_macro_f1": es_f1, "seconds": time.time() - t0})
        if verbose:
            h = history[-1]
            print(f"  epoch {epoch:2d} | train loss {h['train_loss']:.3f} | hold-out loss {es_loss:.3f} | "
                  f"hold-out macro-F1 {es_f1:.3f} | {h['seconds']:.0f}s")
        if es_f1 > best + 1e-4:
            best, best_state, bad = es_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if max_minutes and time.time() - t_start > 60 * max_minutes:
            break

    model.load_state_dict(best_state)
    return model, history
