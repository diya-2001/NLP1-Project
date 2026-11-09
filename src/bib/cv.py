"""Cross-validation runner with on-disk caching.

``run_system(rep, arch, C, Y, folds, ...)`` trains one system on every (seed, fold), and returns
and caches to ``results/week1/{arch}_{rep}.pkl``:
  scores : DataFrame, one row per (seed, fold): macro_f1, accuracy, balanced_accuracy, seconds, ...
  oof    : {seed: array of out-of-fold predictions}  (every pool bio predicted exactly once per seed)
  history: training curves per (seed, fold)
Re-running a cell loads the cache instead of retraining, unless ``rerun=True``.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import encoders, evaluation, models


def system_name(rep: str, arch: str) -> str:
    return f"{arch}_{rep}"


def train_eval_fold(rep, arch, C, Y, train_idx, val_idx, seed=0, n_threads=None, verbose=False, hp=None):
    """Fit representation + model on one training fold, predict its validation fold."""
    t0 = time.time()
    X_tr, X_va, _ = encoders.fold_features(rep, C, train_idx, val_idx)
    clf = models.Classifier(arch, num_classes=int(Y.max()) + 1, seed=seed, n_threads=n_threads,
                            verbose=verbose, **(hp or {})).fit(X_tr, Y[train_idx])
    pred = clf.predict(X_va)
    out = {**evaluation.scores(Y[val_idx], pred), "seconds": time.time() - t0,
           "converged": clf.converged, "epochs": len(clf.history) if arch != "loglin" else None}
    return out, pred, clf


def run_system(rep, arch, C, Y, folds: dict, results_dir: Path, rerun=False, n_threads=None,
               verbose=False, hp=None, tag: str = ""):
    """All seeds x folds for one system, with caching. ``folds`` = {seed: [(train_idx, val_idx), ...]}."""
    name = system_name(rep, arch) + tag
    path = Path(results_dir) / f"{name}.pkl"
    settings = {**models.DEFAULTS[arch], **(hp or {})}
    if path.exists() and not rerun:
        with open(path, "rb") as f:
            res = pickle.load(f)
        same_folds = set(res["oof"]) == set(folds) and len(res["oof"][next(iter(folds))]) == len(Y)
        same_settings = res.get("hp") == settings      # a changed model or setting must not reuse old results
        if same_folds and same_settings:
            return res
        print(f"[{name}] cache does not match the current {'folds' if not same_folds else 'model settings'}"
              " -> retraining")

    rows, oof, history = [], {}, {}
    for seed, fold_list in folds.items():
        oof[seed] = np.full(len(Y), -1)
        for f, (tr, va) in enumerate(fold_list):
            out, pred, clf = train_eval_fold(rep, arch, C, Y, tr, va, seed=seed,
                                             n_threads=n_threads, verbose=verbose, hp=hp)
            oof[seed][va] = pred
            history[(seed, f)] = clf.history
            rows.append({"system": name, "rep": rep, "arch": arch, "seed": seed, "fold": f, **out})
            print(f"[{name}] seed {seed} fold {f}: macro-F1 {out['macro_f1']:.4f} | "
                  f"acc {out['accuracy']:.4f} | {out['seconds']:.0f}s")
    res = {"name": name, "rep": rep, "arch": arch, "scores": pd.DataFrame(rows), "oof": oof,
           "history": history, "hp": {**models.DEFAULTS[arch], **(hp or {})}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def load_all(results_dir: Path, reps=encoders.REPRESENTATIONS, archs=models.ARCHITECTURES) -> dict:
    """Load every cached system that exists: {name: result dict}."""
    out = {}
    for arch in archs:
        for rep in reps:
            p = Path(results_dir) / f"{system_name(rep, arch)}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    out[system_name(rep, arch)] = pickle.load(f)
    return out
