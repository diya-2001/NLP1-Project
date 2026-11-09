"""Our BPE tokenizer (a variant of the tutorial's ``experiment.train_tokenizer``).

Two deliberate differences from the tutorial:
1. normaliser NFKC + Lowercase  ("He"/"he" and "Professor"/"professor" become one unit);
2. pre-tokeniser Punctuation + Metaspace  (punctuation is split off, so "engineer." -> "engineer", ".").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

UNK = "[UNK]"


def train_bpe(texts, vocab_size: int = 10_000, min_frequency: int = 5) -> Tokenizer:
    """Fit a BPE vocabulary on raw texts (no labels involved)."""
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), pre_tokenizers.Metaspace()])
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                         special_tokens=[UNK], show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_or_train(path: Path | str, texts, vocab_size: int = 10_000, min_frequency: int = 5) -> Tokenizer:
    """Load the tokenizer from ``path`` if it exists, otherwise train it and save it there."""
    path = Path(path)
    if path.exists():
        return Tokenizer.from_file(str(path))
    tok = train_bpe(texts, vocab_size, min_frequency)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def encode_all(tok: Tokenizer, texts, batch_size: int = 10_000) -> list[np.ndarray]:
    """Tokenize every text once; returns one int64 array of token ids per text."""
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(np.asarray(e.ids, dtype=np.int64) for e in tok.encode_batch(list(texts[i:i + batch_size])))
    return out
