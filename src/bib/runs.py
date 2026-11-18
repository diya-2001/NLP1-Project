"""Part 2 systems: a recipe per system, its name, and cross-validated runs with caching.

A *recipe* (``spec``) is a dict of every setting; ``system(**changes)`` starts from the defaults below and
changes only what an experiment varies, so each comparison differs in exactly one choice. The name is built
from the changes, e.g. ``pool-max`` or ``rnn-gru-bi``, and results are cached in
``results/part2/{name}.pkl`` with the same structure as Part 1 (``scores`` per seed/fold, out-of-fold
predictions ``oof``, training ``history``, the settings ``hp``), so Part 1's comparison code works on them.

The notebook (§4-§6) and ``scripts/run_part2.py`` (for a GPU or Snellius) both run systems through ``run``.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from . import evaluation, neural

# Every setting, with its default. Pool defaults = the §3 quick test (fastText Common Crawl, mean, frozen,
# log-linear), trained with Part 1's neural recipe.
POOL = dict(kind="pool", emb="ft_crawl", dims=None, pooling="mean", embeddings="frozen", hidden=(),
            dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=20, patience=3,
            inner_frac=0.1, train_frac=1.0)
# RNN defaults = the tutorial's LSTMEncoder (LSTM, 128 numbers of state, one direction, last state, no word dropout)
# with frozen fastText, and the same classifier as the best embed-and-pool system (one hidden layer of 256).
RNN = {**POOL, "kind": "rnn", "pooling": None, "hidden": (256,), "cell": "lstm", "hidden_size": 128,
       "bidirectional": False, "word_dropout": 0.0, "readout": "last"}

# how a setting is written in a system's name
_SHORT = {"emb": lambda v: v.replace("_", ""), "dims": lambda v: f"{v}d", "pooling": str, "embeddings": str,
          "hidden": lambda v: "loglin" if not v else "mlp" + "x".join(map(str, v)), "cell": str,
          "hidden_size": lambda v: f"h{v}", "bidirectional": lambda v: "bi" if v else "uni",
          "word_dropout": lambda v: f"wd{v:g}", "train_frac": lambda v: f"data{int(round(100 * v))}",
          "lr": lambda v: f"lr{v:g}", "dropout": lambda v: f"do{v:g}", "weight_decay": lambda v: f"decay{v:g}",
          "readout": lambda v: f"read{v}"}
# the main choices always appear in the name; any other setting only when it differs from the default
_ALWAYS = {"pool": ("pooling", "hidden"), "rnn": ("cell", "hidden_size", "bidirectional", "embeddings")}

# None = automatic (bib.neural.device_for); scripts/run_part2.py can force one, e.g. "cuda"
DEVICE = None


def system(kind: str = "pool", **changes) -> tuple[str, dict]:
    """(name, spec) for a system that differs from the defaults in ``changes``.

    Examples: system() -> 'pool-mean-loglin'; system(pooling='max', hidden=(256,)) -> 'pool-max-mlp256';
    system('rnn', cell='gru', bidirectional=True) -> 'rnn-gru-h128-bi-frozen'.
    """
    base = POOL if kind == "pool" else RNN
    unknown = set(changes) - set(base)
    if unknown:
        raise ValueError(f"unknown settings {unknown}")
    spec = {**base, **changes}
    spec["hidden"] = tuple(spec["hidden"])
    fmt = lambda k: _SHORT[k](spec[k]) if k in _SHORT else f"{k}{spec[k]}"
    parts = [kind] + [fmt(k) for k in _ALWAYS[kind]]
    parts += [fmt(k) for k in base if k not in _ALWAYS[kind] and spec[k] != base[k]]
    return "-".join(parts), spec


def group(name: str, best: dict | None = None) -> list[tuple[str, dict]]:
    """One planned experiment group, as a list of (system name, spec).

    pool_stage1    pooling x classifier, frozen fastText (9 systems; CPU, minutes).
    pool_stage2    stage-1 winner (``best``, e.g. {"pooling": "mean", "hidden": (256,)}) with trained embeddings
                   (fine-tuned, random start) and other sizes (PCA 50/100 of fastText; GloVe 50).
    pool_datasize  the best pool system with frozen / fine-tuned / random-start embeddings on 10% and 30% of each
                   training fold: when does pretraining matter most? (CPU, minutes)
    rnn_stage1     the tutorial-like LSTM and 7 one-change variants (GPU): GRU, bidirectional, bigger state, mean
                   readout, fine-tuned or random-start embeddings, word dropout.
    rnn_stage2     the stage-1 winners combined (``best``, e.g. {"cell": "gru", "readout": "mean"}), with every
                   embedding mode and a data-size curve (10%, 30%) for its embeddings vs random start (GPU).
    """
    best = dict(best or {})
    if name == "pool_stage1":
        return [system(pooling=p, hidden=h) for p in ("mean", "max", "meanmax") for h in ((), (256,), (256, 256))]
    if name == "pool_stage2":
        return ([system(**best, embeddings=e) for e in ("finetune", "random")]
                + [system(**best, dims=d) for d in (50, 100)] + [system(**best, emb="glove50")])
    if name == "pool_datasize":
        return [system(pooling="mean", hidden=(256,), embeddings=e, train_frac=f)
                for e in ("frozen", "finetune", "random") for f in (0.1, 0.3)]
    if name == "rnn_stage1":
        return [system("rnn")] + [system("rnn", **c) for c in (
            {"cell": "gru"}, {"bidirectional": True}, {"hidden_size": 256}, {"readout": "mean"},
            {"embeddings": "finetune"}, {"embeddings": "random"}, {"word_dropout": 0.25})]
    if name == "rnn_stage2":
        out = [system("rnn", **best)]
        out += [system("rnn", **{**best, "embeddings": e}) for e in ("frozen", "finetune", "random")]
        out += [system("rnn", **{**best, "embeddings": e, "train_frac": f})
                for e in (best.get("embeddings", "frozen"), "random") for f in (0.1, 0.3)]
        seen = set()
        return [(n, s) for n, s in out if not (n in seen or seen.add(n))]
    raise ValueError(f"unknown group {name!r}; one of {GROUPS}")


GROUPS = ("pool_stage1", "pool_stage2", "pool_datasize", "rnn_stage1", "rnn_stage2")


class Data:
    """What every run needs, loaded once: labels Y, gender G, folds; tables/ids/pooled features on demand."""

    def __init__(self, Y, G, folds, results_dir, tokens=None):
        self.Y, self.G, self.folds = np.asarray(Y), np.asarray(G), folds
        self.results_dir, self.tokens = Path(results_dir), tokens
        self._tables, self._feats = {}, {}

    @classmethod
    def from_disk(cls, root: Path):
        """Load without the HF dataset (for Snellius): results/part2/labels.npz + results/part1/splits.json."""
        root = Path(root)
        lab = np.load(root / "results" / "part2" / "labels.npz")
        folds = load_folds(root / "results" / "part1" / "splits.json", len(lab["Y"]))
        return cls(lab["Y"], lab["G"], folds, root / "results" / "part2")

    def save_labels(self):
        np.savez(self.results_dir / "labels.npz", Y=self.Y, G=self.G)

    def table_ids(self, emb_name: str, dims: int | None = None):
        key = (emb_name, dims)
        if key not in self._tables:
            table, ids = neural.load_table_and_ids(emb_name, self.results_dir, self.tokens)
            if dims is not None and dims < table.shape[1]:
                table = neural.reduce_table(table, dims)
            self._tables[key] = (table, ids)
        return self._tables[key]

    def features(self, emb_name: str, dims: int | None, pooling: str) -> np.ndarray:
        key = (emb_name, dims, pooling)
        if key not in self._feats:
            table, ids = self.table_ids(emb_name, dims)
            self._feats[key] = neural.pool_features(ids, table, pooling)
        return self._feats[key]


def load_folds(splits_path: Path, n: int) -> dict:
    """{seed: [(train_idx, val_idx), ...]} from Part 1's splits file (same as ``bib.data.load_pool``)."""
    splits = json.loads(Path(splits_path).read_text())
    folds = {}
    for seed, val_lists in splits["val_folds"].items():
        folds[int(seed)] = []
        for va in val_lists:
            va = np.asarray(va)
            mask = np.ones(n, bool)
            mask[va] = False
            folds[int(seed)].append((np.flatnonzero(mask), va))
    return folds


def _subsample(rows, y, frac_or_n, seed):
    """A stratified random subset of rows: a fraction (<= 1) or a count (> 1)."""
    n = int(round(frac_or_n * len(rows))) if frac_or_n <= 1 else int(frac_or_n)
    if n >= len(rows):
        return rows
    keep, _ = train_test_split(rows, train_size=n, stratify=y[rows], random_state=seed)
    return np.sort(keep)


def build(spec: dict, data: Data, train_rows, seed: int):
    """The model and its inputs for one training fold."""
    torch.manual_seed(seed)
    if spec["kind"] == "pool" and spec["embeddings"] == "frozen":
        X = data.features(spec["emb"], spec["dims"], spec["pooling"])
        Xtr = X[train_rows]
        model = neural.DenseModel(Xtr.mean(0), Xtr.std(0), spec["hidden"], dropout=spec["dropout"])
        return model, neural.Inputs(X=X)
    table, ids = data.table_ids(spec["emb"], spec["dims"])
    if spec["kind"] == "pool":
        d = table.shape[1] * (2 if spec["pooling"] == "meanmax" else 1)
        model = neural.PoolModel(table, np.zeros(d), np.ones(d), spec["pooling"], spec["embeddings"],
                                 spec["hidden"], dropout=spec["dropout"])
        # standardise with the pooled vectors of the model's own starting table (20k training bios)
        sample = np.random.default_rng(seed).choice(train_rows, min(20_000, len(train_rows)), replace=False)
        P = neural.pool_features([ids[r] for r in sample],
                                 model.embedding.weight.detach().numpy(), spec["pooling"])
        model.scale = neural.Standardize(P.mean(0), P.std(0))
        return model, neural.Inputs(ids=ids)
    model = neural.RNNModel(table, spec["cell"], spec["hidden_size"], spec["bidirectional"], spec["embeddings"],
                            spec["word_dropout"], spec["hidden"], dropout=spec["dropout"], readout=spec["readout"])
    return model, neural.Inputs(ids=ids)


def train_eval_fold(spec, data: Data, tr, va, seed=0, verbose=False, max_minutes=None):
    t0 = time.time()
    Y = data.Y
    if spec["train_frac"] < 1:
        tr = _subsample(tr, Y, spec["train_frac"], seed)
    neural.DEVICE = torch.device(DEVICE) if DEVICE else neural.device_for(spec)
    model, inputs = build(spec, data, tr, seed)
    model, history = neural.fit(model, inputs, Y, tr, seed=seed, verbose=verbose, max_minutes=max_minutes,
                                **spec)
    pred = neural.predict_logprobs(model, inputs, va).argmax(1)
    out = {**evaluation.scores(Y[va], pred), "seconds": time.time() - t0, "epochs": len(history),
           "best_epoch": int(np.argmax([h["holdout_macro_f1"] for h in history])) + 1,
           "n_train": len(tr), "params_trained": neural.n_params(model)}
    return out, pred, history


def run(name: str, spec: dict, data: Data, rerun=False, verbose=False, seeds=None, limit=None,
        max_minutes=None):
    """Cross-validate one system over every (seed, fold) and cache it as ``{results_dir}/{name}.pkl``.

    seeds: run only these seeds (default all). limit=(n_train, n_val): quick test on stratified subsets of
    seed 0 / fold 0 only, never cached. Re-running loads the cache unless the folds or settings changed.
    """
    folds = data.folds if seeds is None else {s: data.folds[s] for s in seeds}
    if limit is not None:
        tr, va = data.folds[0][0]
        tr, va = _subsample(tr, data.Y, limit[0], 0), _subsample(va, data.Y, limit[1], 0)
        out, _, history = train_eval_fold(spec, data, tr, va, 0, verbose, max_minutes)
        return {"name": name, "scores": pd.DataFrame([{"system": name, "seed": 0, "fold": 0, **out}]),
                "history": {(0, 0): history}, "hp": spec}

    path = data.results_dir / f"{name}.pkl"
    signature = {s: len(f) for s, f in folds.items()}
    if path.exists() and not rerun:
        res = pickle.load(open(path, "rb"))
        if res.get("hp") == spec and res.get("folds") == signature:
            return res
        print(f"[{name}] cache does not match the current settings or folds -> retraining")

    rows, oof, history = [], {}, {}
    for seed, fold_list in folds.items():
        oof[seed] = np.full(len(data.Y), -1)
        for f, (tr, va) in enumerate(fold_list):
            out, pred, hist = train_eval_fold(spec, data, tr, va, seed, verbose, max_minutes)
            oof[seed][va] = pred
            history[(seed, f)] = hist
            rows.append({"system": name, "seed": seed, "fold": f, **out})
            print(f"[{name}] seed {seed} fold {f}: macro-F1 {out['macro_f1']:.4f} | acc {out['accuracy']:.4f}"
                  f" | {out['epochs']} epochs | {out['seconds']:.0f}s", flush=True)
    res = {"name": name, "scores": pd.DataFrame(rows), "oof": oof, "history": history, "hp": spec,
           "folds": signature}
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


def load(name: str, results_dir: Path):
    return pickle.load(open(Path(results_dir) / f"{name}.pkl", "rb"))
