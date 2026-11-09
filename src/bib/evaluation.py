"""Metrics: overall performance (per fold) and gender fairness (on pooled out-of-fold predictions).

Performance (one number per fold -> used by the Bayesian comparison):
  accuracy           fraction of bios classified correctly
  macro_f1           mean over the 28 occupations of F1 (each occupation weighs 1/28)  <- primary
  balanced_accuracy  mean over the 28 occupations of recall

Fairness (per occupation y, following De-Arteaga et al. 2019):
  TPR_g,y  = recall of occupation y among bios of gender g
  gap_y    = TPR_female,y - TPR_male,y     (> 0: women of that occupation are recognised more often)
  RMS gap  = sqrt(mean_y gap_y^2)          (one-number summary, Romanov et al. 2019)
Only occupations where *both* gender groups have >= ``min_size`` bios are included.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

METRICS = ("macro_f1", "accuracy", "balanced_accuracy")


def scores(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def per_class_f1(y_true, y_pred, names) -> pd.Series:
    f1 = f1_score(y_true, y_pred, average=None, labels=range(len(names)), zero_division=0)
    return pd.Series(f1, index=names, name="F1")


def tpr_by_gender(y_true, y_pred, gender, names, min_size: int = 10) -> pd.DataFrame:
    """One row per occupation: group sizes, recall per gender, gap, % female, and whether it is reported."""
    y_true, y_pred, gender = map(np.asarray, (y_true, y_pred, gender))
    rows = []
    for k, name in enumerate(names):
        is_k = y_true == k
        f, m = is_k & (gender == 1), is_k & (gender == 0)
        n_f, n_m = int(f.sum()), int(m.sum())
        tpr_f = (y_pred[f] == k).mean() if n_f else np.nan
        tpr_m = (y_pred[m] == k).mean() if n_m else np.nan
        rows.append({"occupation": name, "n_female": n_f, "n_male": n_m,
                     "pct_female": 100 * n_f / max(n_f + n_m, 1),
                     "tpr_female": tpr_f, "tpr_male": tpr_m, "gap": tpr_f - tpr_m,
                     "reported": min(n_f, n_m) >= min_size})
    return pd.DataFrame(rows).set_index("occupation")


def rms_gap(tpr_table: pd.DataFrame) -> float:
    g = tpr_table.loc[tpr_table["reported"], "gap"]
    return float(np.sqrt(np.mean(g ** 2)))


def normalized_confusion(y_true, y_pred, K: int) -> np.ndarray:
    """Row-normalised confusion matrix: row = true occupation, entries sum to 1 per row."""
    cm = confusion_matrix(y_true, y_pred, labels=range(K)).astype(float)
    return cm / cm.sum(1, keepdims=True).clip(min=1)


def top_confusions(y_true, y_pred, names, n: int = 10) -> pd.DataFrame:
    """The largest off-diagonal entries of the row-normalised confusion matrix."""
    cm = normalized_confusion(y_true, y_pred, len(names))
    counts = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:n]
    rows = [{"true": names[i], "predicted as": names[j], "% of true class": 100 * cm[i, j],
             "bios": int(counts[i, j])} for i, j in zip(*np.unravel_index(flat, cm.shape))]
    return pd.DataFrame(rows)


def gendered_errors(y_true, y_pred, gender, names, occupations) -> pd.DataFrame:
    """For each occupation: which gender is recognised less often, and where its missed bios end up.

    Example row: nurse | disadvantaged: male | TPR 0.62 | most common wrong prediction: physician (21%).
    """
    y_true, y_pred, gender = map(np.asarray, (y_true, y_pred, gender))
    rows = []
    for occ in occupations:
        k = names.index(occ)
        tpr = {g: (y_pred[(y_true == k) & (gender == g)] == k).mean() for g in (0, 1)}
        worse = min(tpr, key=tpr.get)
        missed = y_pred[(y_true == k) & (gender == worse) & (y_pred != k)]
        top = pd.Series(missed).value_counts(normalize=True).head(1)
        rows.append({"occupation": occ, "recognised less often": {0: "men", 1: "women"}[worse],
                     "TPR women": tpr[1], "TPR men": tpr[0],
                     "their misses mostly go to": names[int(top.index[0])] if len(top) else "—",
                     "share of misses": float(top.iloc[0]) if len(top) else np.nan})
    return pd.DataFrame(rows).set_index("occupation")
