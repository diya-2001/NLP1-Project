"""The three representation spaces (R1-R3), all as sparse CSR matrices.

Every representation starts from the same **count matrix** C (one row per bio, one column per
BPE token, C[i, v] = how often token v occurs in bio i). It is built once from the cached
token ids, so no representation re-tokenizes text.

R1 indicator : 1 if the token occurs, else 0                  (tutorial "token indicators")
R2 bow       : the raw count                                  (tutorial "bag-of-words")
R3 tfidf     : (1 + log count) * idf, each row scaled to length 1   (ours)

R1 and R2 need no fitting. R3 has to *learn* the idf weights (how rare each token is), and it
must learn them **from the training fold only**; otherwise the validation fold leaks into
training. That's why every representation has the same ``fit`` / ``transform`` interface.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

REPRESENTATIONS = ("indicator", "bow", "tfidf")


def count_matrix(ids: list[np.ndarray], vocab_size: int) -> sp.csr_matrix:
    """Sparse document-term count matrix from per-document token-id arrays."""
    lengths = np.fromiter((len(a) for a in ids), dtype=np.int64, count=len(ids))
    rows = np.repeat(np.arange(len(ids)), lengths)
    cols = np.concatenate(ids) if len(ids) else np.zeros(0, dtype=np.int64)
    data = np.ones(len(cols), dtype=np.float32)
    C = sp.csr_matrix((data, (rows, cols)), shape=(len(ids), vocab_size), dtype=np.float32)
    C.sum_duplicates()   # repeated (row, col) pairs are added up -> counts
    return C


class Indicator:
    """R1: token presence (binary)."""
    name = "indicator"

    def fit(self, C):
        return self

    def transform(self, C):
        X = C.copy()
        X.data[:] = 1.0
        return X


class BagOfWords:
    """R2: raw token counts."""
    name = "bow"

    def fit(self, C):
        return self

    def transform(self, C):
        return C.copy()


class TfIdf:
    """R3: sublinear tf * idf, L2-normalised rows. ``fit`` learns idf on the training rows only."""
    name = "tfidf"

    def __init__(self):
        # idf(v) = ln((1 + n) / (1 + df(v))) + 1, where df(v) = number of training bios containing v
        self._t = TfidfTransformer(sublinear_tf=True, smooth_idf=True, norm="l2")

    def fit(self, C):
        self._t.fit(C)
        return self

    def transform(self, C):
        return self._t.transform(C).astype(np.float32).tocsr()

    @property
    def idf(self) -> np.ndarray:
        return self._t.idf_


def make(rep: str):
    """Fresh, unfitted representation object by name."""
    return {"indicator": Indicator, "bow": BagOfWords, "tfidf": TfIdf}[rep]()


def fold_features(rep: str, C: sp.csr_matrix, train_idx, val_idx):
    """Fit the representation on the training rows, then transform both parts."""
    enc = make(rep).fit(C[train_idx])
    return enc.transform(C[train_idx]), enc.transform(C[val_idx]), enc
