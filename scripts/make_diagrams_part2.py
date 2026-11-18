"""Draw the Part 2 architecture diagrams embedded in the notebook's markdown (§4 embed-and-pool, §5 LSTM/GRU).

    python scripts/make_diagrams_part2.py

Writes PNGs to figures/part2/diagrams/. The coloured squares are the first 12 of the 300 numbers of the
REAL fastText Common Crawl vectors of each word (results/part2/emb_ft_crawl.npz, made by notebook §2).
"""
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from bib import style
from make_diagrams import arrow, blank, box   # Part 1's drawing helpers (same folder)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "part2" / "diagrams"
WORDS = ["she", "is", "a", "registered", "nurse", "…"]
N_SHOW = 12


def vectors():
    d = np.load(ROOT / "results" / "part2" / "emb_ft_crawl.npz", allow_pickle=True)
    index = {w: i for i, w in enumerate(d["words"])}
    V = np.stack([d["vectors"][index[w]][:N_SHOW] for w in WORDS[:-1]])
    return V, float(np.abs(V).max())


def squares(ax, x, y, values, vmax, size=0.28):
    """One word vector as a row of coloured squares (blue = positive, grey = negative)."""
    for k, v in enumerate(values):
        t = abs(v) / vmax
        col = style.BLUES(0.15 + 0.85 * t) if v >= 0 else (0.55 + 0.4 * (1 - t),) * 3
        ax.add_patch(Rectangle((x + k * size, y), size * 0.92, size * 0.92, color=col, lw=0))


def embed_and_pool():
    V, vmax = vectors()
    fig, ax = blank((13, 4.8), (0, 13), (0, 5))
    ax.text(0.1, 4.6, "1  look up each word", fontsize=10.5, weight="bold", color=style.INK)
    for i, w in enumerate(WORDS):
        y = 3.9 - i * 0.55
        ax.text(1.15, y + 0.12, w, ha="right", va="center", fontsize=10, color=style.INK)
        if i < len(V):
            squares(ax, 1.3, y, V[i], vmax)
            ax.text(1.3 + N_SHOW * 0.28 + 0.1, y + 0.12, "…", va="center", fontsize=10, color=style.MUTED)
    ax.text(0.1, 0.25, "fastText table: 202,182 words × 300 numbers\nfrozen (fixed) or fine-tuned (trained further)",
            fontsize=8.8, color=style.INK_2)
    arrow(ax, 5.0, 2.6, 5.7, 2.6)
    box(ax, 5.75, 2.05, 1.6, 1.1, "2  pool\nmean or max\nper column", fc=style.LIGHT_FILL, size=9.5)
    arrow(ax, 7.4, 2.6, 7.9, 2.6)
    ax.text(7.95, 3.15, "one bio vector h(x)", fontsize=9.5, weight="bold", color=style.INK)
    squares(ax, 7.95, 2.45, V.mean(0), vmax)
    ax.text(7.95, 2.05, "300 numbers (here: the mean)", fontsize=8.5, color=style.INK_2)
    arrow(ax, 11.4, 2.6, 11.55, 2.6)
    box(ax, 11.6, 1.75, 1.35, 1.7, "3  classifier\nlog-linear\nor MLP\n→ 28 probs", fc=style.ACCENT, color="white",
        size=9.5)
    ax.text(5.75, 0.25, "Word order is lost: 'she is a registered nurse' and 'a nurse registered is she' give the same h(x).",
            fontsize=9, color=style.INK_2)
    style.save(fig, OUT / "embed_and_pool.png")


def rnn():
    V, vmax = vectors()
    fig, ax = blank((13, 4.6), (0, 13), (0, 4.6))
    n = len(WORDS) - 1
    xs = [0.4 + i * 2.15 for i in range(n)]
    for i, (x, w) in enumerate(zip(xs, WORDS)):
        ax.text(x + 0.85, 0.25, w, ha="center", fontsize=10.5, color=style.INK)
        squares(ax, x + 0.85 - N_SHOW * 0.28 / 4, 0.75, V[i][:6], vmax)
        arrow(ax, x + 0.85, 1.1, x + 0.85, 1.75)
        box(ax, x + 0.2, 1.8, 1.3, 0.95, "LSTM\nor GRU", fc=style.ACCENT, color="white", size=9.5)
        if i < n - 1:
            arrow(ax, x + 1.52, 2.27, xs[i + 1] + 0.18, 2.27)
            ax.text(x + 1.84, 2.4, "state", ha="center", fontsize=8, color=style.MUTED)
    xe = xs[-1] + 1.52
    arrow(ax, xe, 2.27, xe + 0.45, 2.27)
    box(ax, xe + 0.5, 1.75, 1.0, 1.05, "classifier\n→ 28\nprobs", fc=style.INK_2, color="white", size=9)
    ax.text(0.4, 3.85, "The same cell reads the bio word by word and updates a state (128 numbers): a summary of what it has "
            "read so far.", fontsize=9.5, color=style.INK)
    ax.text(0.4, 3.4, "Word order now matters. Bidirectional: a second cell reads right to left, and both final states are "
            "joined. Word dropout: words are randomly blanked while training.", fontsize=9, color=style.INK_2)
    style.save(fig, OUT / "rnn.png")


if __name__ == "__main__":
    style.apply()
    embed_and_pool()
    rnn()
    print("saved to", OUT)
