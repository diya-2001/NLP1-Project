"""Pretrained word embeddings for Part 2: a word tokenizer, a memory-light loader, coverage measures.

Why our own loader instead of ``part2.clslib.PretrainedEmbeddings``: that class keeps *every* row of the
file (fastText Common Crawl = 2M words x 300 numbers ≈ 2.4 GB in memory). We only need the rows for words
that actually occur in our bios, so ``load_vectors`` streams the zip file and keeps just those.

Case: GloVe is lowercased; the fastText files are cased and sorted by frequency. We lowercase our tokens,
so for a cased file we keep the FIRST (= most frequent) spelling of each lowercased word, the same rule
``part2.clslib.PretrainedEmbeddings`` uses.
"""
from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from pathlib import Path

import nltk
import numpy as np

# the candidates compared in Part 2, step 1 (files live in data/embeddings/, not in git)
MODELS = {
    "glove50": dict(file="glove.2024.wikigiga.50d.zip", dims=50, skip_first_line=False, cased=False,
                    label="GloVe 2024, 50-d", trained_on="Wikipedia + Gigaword newswire"),
    "ft_wiki": dict(file="wiki-news-300d-1M.vec.zip", dims=300, skip_first_line=True, cased=True,
                    label="fastText wiki-news, 300-d", trained_on="Wikipedia 2017 + UMBC web + news"),
    "ft_crawl": dict(file="crawl-300d-2M.vec.zip", dims=300, skip_first_line=True, cased=True,
                     label="fastText Common Crawl, 300-d", trained_on="Common Crawl (600B tokens)"),
}

# typographic characters that the embedding vocabularies spell with plain ASCII
_ASCII_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " "})


def tokenize(text: str, normalise_quotes: bool = True) -> list[str]:
    """Lowercase + (optionally) straighten curly quotes/dashes + NLTK word tokenization.

    This mirrors the tutorial's ``BasicTokenizer(lowercase=True, remove_punct=False,
    remove_stopwords=False)``; the quote normalisation is our addition, tested in Part 2 step 1.
    """
    text = text.lower()
    if normalise_quotes:
        text = text.translate(_ASCII_MAP)
    return nltk.tokenize.word_tokenize(text)


def vocab_counts(token_lists) -> Counter:
    """How often each token type occurs across all tokenized texts."""
    counts = Counter()
    for toks in token_lists:
        counts.update(toks)
    return counts


def load_vectors(path: Path | str, dims: int, keep: set[str] | None = None, skip_first_line: bool = False):
    """Stream a .zip/.txt/.vec embedding file; return (words, matrix float32, n_lines_in_file).

    keep: only words (lowercased) in this set are kept; None keeps everything.
    """
    path = Path(path)
    words, rows, seen, n_lines = [], [], set(), 0
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        inner = [n for n in zf.namelist() if not n.endswith("/")][0]
        handle = io.TextIOWrapper(zf.open(inner), encoding="utf-8", errors="replace")
    else:
        handle = open(path, encoding="utf-8", errors="replace")
    with handle:
        for i, line in enumerate(handle):
            if skip_first_line and i == 0:
                continue
            n_lines += 1
            word = line.split(" ", 1)[0].lower()
            if word in seen or (keep is not None and word not in keep):
                continue
            parts = line.rstrip().rsplit(" ", dims)          # robust to words that contain spaces
            if len(parts) != dims + 1:
                continue
            seen.add(word)
            words.append(word)
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    matrix = np.stack(rows) if rows else np.zeros((0, dims), dtype=np.float32)
    return words, matrix, n_lines


def coverage(counts: Counter, known: set[str]) -> dict:
    """Share of distinct words (types) and of running words (tokens) that have a vector."""
    n_types = len(counts)
    n_tokens = sum(counts.values())
    found_types = sum(1 for w in counts if w in known)
    found_tokens = sum(c for w, c in counts.items() if w in known)
    return {"type coverage (%)": 100 * found_types / max(n_types, 1),
            "token coverage (%)": 100 * found_tokens / max(n_tokens, 1),
            "distinct words": n_types, "distinct words covered": found_types}


def top_missing(counts: Counter, known: set[str], n: int = 25) -> list[tuple[str, int]]:
    """The most frequent tokens that have no vector."""
    return [(w, c) for w, c in counts.most_common() if w not in known][:n]


def backoff(tokens: list[str], known: set[str]) -> list[str]:
    """Make a token list fit an embedding vocabulary, without changing the words themselves.

    - a token with a vector is kept;
    - an unknown token with '-' or '/' inside (in-network, cross/blue) is replaced by its known parts
      (in, network): GloVe was trained on text where such compounds were split;
    - NLTK's quote tokens `` and '' become a plain " when the model has it (else stay unknown);
    - n't becomes not when the model lacks n't (the fastText files do);
    - anything else stays as it is (it will map to the <unk> vector).
    """
    out = []
    for t in tokens:
        if t in known:
            out.append(t)
        elif ("-" in t or "/" in t) and len(t) > 1:
            parts = [p for p in re.split(r"[-/]", t) if p]
            out.extend(parts if any(p in known for p in parts) else [t])
        elif t in ("``", "''") and '"' in known:
            out.append('"')
        elif t == "n't" and "not" in known:
            out.append("not")
        else:
            out.append(t)
    return out


def words_to_keep(counts: Counter) -> set[str]:
    """Every token in our bios, plus the parts of hyphen/slash compounds and the back-off targets."""
    keep = set(counts)
    for w in counts:
        if "-" in w or "/" in w:
            keep.update(p for p in re.split(r"[-/]", w) if p)
    return keep | {'"', "not"}


def mean_pool(token_lists, words: list[str], vectors: np.ndarray, use_backoff: bool = True) -> np.ndarray:
    """Average the vectors of each text's known tokens (unknown tokens are skipped). Shape (n_texts, dims).

    Built as one sparse (texts x vocabulary) count matrix times the vector table, so 257k bios take seconds.
    """
    import scipy.sparse as sp
    index = {w: i for i, w in enumerate(words)}
    known = set(index)
    rows, cols = [], []
    for r, toks in enumerate(token_lists):
        toks = backoff(toks, known) if use_backoff else toks
        ids = [index[t] for t in toks if t in index]
        rows.extend([r] * len(ids))
        cols.extend(ids)
    counts = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(token_lists), len(words)))
    n_known = np.asarray(counts.sum(axis=1)).ravel()
    pooled = counts @ vectors
    return pooled / np.maximum(n_known, 1)[:, None]


_WORDLIKE = re.compile(r"^[a-z]+(?:['-][a-z]+)*$")


def is_wordlike(token: str) -> bool:
    """True for alphabetic words (with inner ' or -), False for numbers, punctuation, codes, e-mails."""
    return bool(_WORDLIKE.match(token))
