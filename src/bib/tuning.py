"""Pick ONE key hyperparameter per system before cross-validation.

Why: representations live on different scales (a TF-IDF row has length 1, a count row has length
about 9), so a single regularisation strength C or learning rate would favour some representations
over others. Tuning each system's most important knob makes the comparison fair.

How (no leakage): only the *training* part of seed-0 / fold-0 is used.
(An earlier 5-fold run of this search picked C = 1 for every representation, with C = 10 and 100 worse
and often not converging, so the grid is now centred on 1.) We split it 90/10 again, train on
the 90%, and score macro-F1 on the 10%. The CV validation folds are never looked at.

  loglin : C  (inverse L2 strength)  in GRIDS["loglin"]
  mlp/deep: learning rate            in GRIDS["mlp"] / GRIDS["deep"]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import encoders, evaluation, models

GRIDS = {"loglin": {"C": [0.3, 1.0, 3.0]},
         "mlp":    {"lr": [1e-3, 3e-3]},
         "deep":   {"lr": [1e-3, 3e-3]}}


def tune(C, Y, train_idx, reps=encoders.REPRESENTATIONS, archs=models.ARCHITECTURES,
         seed=0, n_threads=None) -> pd.DataFrame:
    """Hold-out macro-F1 for every (arch, rep, value) in GRIDS."""
    fit_idx, ho_idx = train_test_split(train_idx, test_size=0.1, stratify=Y[train_idx], random_state=seed)
    rows = []
    for rep in reps:
        X_fit, X_ho, _ = encoders.fold_features(rep, C, fit_idx, ho_idx)
        for arch in archs:
            (param, values), = GRIDS[arch].items()
            for v in values:
                t0 = time.time()
                clf = models.Classifier(arch, num_classes=int(Y.max()) + 1, seed=seed,
                                        n_threads=n_threads, **{param: v}).fit(X_fit, Y[fit_idx])
                s = evaluation.scores(Y[ho_idx], clf.predict(X_ho))
                rows.append({"arch": arch, "rep": rep, "param": param, "value": v,
                             "holdout_macro_f1": s["macro_f1"], "seconds": time.time() - t0,
                             "converged": clf.converged})
                print(f"{arch:6s} {rep:9s} {param}={v:<6g} hold-out macro-F1 {s['macro_f1']:.4f}"
                      f"  ({rows[-1]['seconds']:.0f}s)")
    return pd.DataFrame(rows)


def best_params(table: pd.DataFrame) -> dict:
    """{system_name: {param: best value}} from a ``tune`` table."""
    best = table.loc[table.groupby(["arch", "rep"])["holdout_macro_f1"].idxmax()]
    return {f"{r.arch}_{r.rep}": {r.param: float(r.value)} for r in best.itertuples()}


def save(params: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(params, indent=2))


def load(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
