"""Rebuild the exact CV pool and folds from ``results/week1/splits.json`` (used by scripts/run_grid.py).

The notebook builds the pool in §2 and saves the original ``train`` row ids plus each seed's
validation folds. Anything that needs the same data later (a script, a teammate) calls
``load_pool`` instead of re-deriving it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


def load_pool(splits_path: Path | str):
    """Returns TEXTS (list[str]), Y, G (int arrays) and FOLDS {seed: [(train_idx, val_idx), ...]}."""
    splits = json.loads(Path(splits_path).read_text())
    train = load_dataset("LabHC/bias_in_bios", split="train").to_pandas()
    pool = train.iloc[splits["pool_train_rows"]]
    texts = pool["hard_text"].tolist()
    Y, G = pool["profession"].to_numpy(), pool["gender"].to_numpy()
    n = len(Y)
    folds = {}
    for seed, val_lists in splits["val_folds"].items():
        folds[int(seed)] = []
        for va in val_lists:
            va = np.asarray(va)
            mask = np.ones(n, bool)
            mask[va] = False
            folds[int(seed)].append((np.flatnonzero(mask), va))
    return texts, Y, G, folds
