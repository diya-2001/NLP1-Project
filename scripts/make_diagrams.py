"""Draw the explanatory diagrams embedded in the notebook's markdown cells.

    python scripts/make_diagrams.py

Writes PNGs to figures/week1/diagrams/. The representations diagram uses our trained BPE tokenizer
and real document frequencies from deduplicated `train`, so run the notebook up to §3 first.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from bib import style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "week1" / "diagrams"
TOKENIZER = ROOT / "results" / "week1" / "bpe_lower_punct_v10000_mf5.json"
style.apply()


def box(ax, x, y, w, h, text, fc=style.LIGHT_FILL, ec="none", color=style.INK, size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=style.MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.4))


def blank(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------------- 1. pipeline
def pipeline():
    fig, ax = blank((12, 3.0), (0, 12), (0, 3))
    steps = [
        ("bio text", '"She completed her\nnursing degree at…"'),
        ("BPE tokens", "▁she ▁completed\n▁her ▁nursing …"),
        ("vector h(x)", "10,000 numbers,\nmostly 0  (§4)"),
        ("model", "log-linear / MLP /\ndeeper MLP  (§5)"),
        ("probabilities", "p(nurse)=0.81\np(physician)=0.09 …"),
        ("prediction", "argmax\n→ nurse"),
    ]
    w, gap = 1.7, 0.32
    for i, (title, body) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        fc = style.ACCENT if title == "model" else style.LIGHT_FILL
        col = "white" if title == "model" else style.INK
        box(ax, x, 0.55, w, 1.5, body, fc=fc, color=col, size=9)
        ax.text(x + w / 2, 2.3, title, ha="center", fontsize=10.5, weight="bold", color=style.INK)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.03, 1.3, x + w + gap - 0.03, 1.3)
    ax.text(0.1, 0.12, "§3 tokenizer  →  §4 representation  →  §5 architecture  →  decision rule (most probable class)",
            fontsize=9, color=style.INK_2)
    style.save(fig, OUT / "pipeline.png")


# ----------------------------------------------------------------------------- 2. CV folds
def cv_folds():
    fig, ax = blank((11, 4.6), (-1.6, 10.6), (-1.5, 6.4))
    for s, seed_y in enumerate([5.6, 2.6]):
        ax.text(-1.5, seed_y + 0.35, f"seed {s}", fontsize=10.5, weight="bold", color=style.INK)
        for f in range(3):
            y = seed_y - 0.4 - f * 0.8
            ax.text(-0.15, y + 0.3, f"fold {f}", ha="right", va="center", fontsize=9.5, color=style.INK_2)
            # a different shuffle per seed: show it by shifting which third is validation
            for b in range(3):
                val = b == (f + s) % 3
                box(ax, b * 3.3, y, 3.2, 0.6, "validation (86k)" if val else "train (171k)",
                    fc=style.FEMALE if val else style.LIGHT_FILL, color="white" if val else style.INK_2, size=9)
    ax.text(0, -0.75, "Each seed shuffles the bios differently, then cuts them into 3 parts. Each bio is validated once per seed "
            "→ 3 folds × 2 seeds = 6 scores per system.", fontsize=9.5, color=style.INK)
    ax.text(0, -1.3, "Inside the training part, neural nets hold out 10% for early stopping, and tuning (§5) also uses only "
            "training data. Validation is used once, at the end.", fontsize=9.5, color=style.INK_2)
    style.save(fig, OUT / "cv_folds.png")


# ----------------------------------------------------------------------------- 3. representations
def representations():
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from bib import encoders
    from bib import tokenizer as btok

    tok = Tokenizer.from_file(str(TOKENIZER))
    train = load_dataset("LabHC/bias_in_bios", split="train").to_pandas()
    texts = train["hard_text"].drop_duplicates().tolist()
    C = encoders.count_matrix(btok.encode_all(tok, texts), tok.get_vocab_size())
    tfidf = encoders.TfIdf().fit(C)
    df_share = np.asarray((C > 0).mean(0)).ravel()

    bio = "She is a nurse. She trained as a nurse in Utrecht and teaches nursing students."
    enc = tok.encode(bio)
    row = encoders.count_matrix([np.asarray(enc.ids)], tok.get_vocab_size())
    ids, first = [], {}
    for t, i in zip(enc.tokens, enc.ids):          # unique tokens in order of appearance
        if i not in first:
            first[i] = t
            ids.append(i)
    labels = [first[i].replace("▁", "") or "▁" for i in ids]
    r1 = encoders.Indicator().transform(row)[0, ids].toarray().ravel()
    r2 = row[0, ids].toarray().ravel()
    r3 = tfidf.transform(row)[0, ids].toarray().ravel()

    fig, axes = plt.subplots(4, 1, figsize=(12, 5.6), gridspec_kw={"height_ratios": [0.7, 1, 1, 1]})
    ax0 = axes[0]
    ax0.axis("off")
    ax0.text(0, 0.65, f'bio: "{bio}"', fontsize=10.5, color=style.INK, transform=ax0.transAxes)
    ax0.text(0, 0.1, "BPE tokens: " + " ".join(t for t in enc.tokens), fontsize=9, color=style.INK_2,
             transform=ax0.transAxes, family="monospace")
    panels = [("R1 indicator: is the token there? (0/1)", r1, "{:.0f}"),
              ("R2 bag-of-words: how many times?", r2, "{:.0f}"),
              ("R3 TF-IDF: repeats damped, common tokens down-weighted, row scaled to length 1", r3, "{:.2f}")]
    for ax, (title, vals, fmt) in zip(axes[1:], panels):
        ax.imshow(vals[None, :] / max(vals.max(), 1e-9), cmap=style.BLUES, vmin=0, vmax=1.15, aspect="auto")
        for j, v in enumerate(vals):
            ax.text(j, 0, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if v / max(vals.max(), 1e-9) > 0.6 else style.INK)
        ax.set_yticks([])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels if ax is axes[-1] else [""] * len(labels), fontsize=9.5, color=style.INK)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10.5, loc="left", pad=4, weight="bold")
    axes[-1].set_xlabel("each column = one of the 10,000 vocabulary dimensions (the other ~9,980 are 0 for this bio)",
                        color=style.INK_2)
    fig.tight_layout()
    style.save(fig, OUT / "representations.png")

    # print the document frequencies used in the notebook text
    for i, lab in zip(ids, labels):
        print(f"{lab:>10s}: in {100 * df_share[i]:5.1f}% of bios, idf {tfidf.idf[i]:.2f}")


# ----------------------------------------------------------------------------- 4. architectures
def architectures():
    from bib.models import param_count
    V, K = 10_000, 28
    specs = [
        ("A1 · log-linear", style.ARCH_COLORS["loglin"], [("input h(x)", V)], "loglin",
         "one weight per (token, occupation):\nscore = Σ weights of the bio's tokens"),
        ("A2 · MLP (tutorial)", style.ARCH_COLORS["mlp"], [("input h(x)", V), ("hidden + ReLU", 128)], "mlp",
         "hidden layer = 128 learned 'feature\ndetectors' that combine tokens"),
        ("A3 · deeper MLP (ours)", style.ARCH_COLORS["deep"],
         [("input h(x)", V), ("hidden + ReLU", 256), ("hidden + ReLU", 128)], "deep",
         "a second hidden layer combines\nthe detectors once more"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for ax, (title, col, layers, arch, note) in zip(axes, specs):
        ax.set(xlim=(0, 10), ylim=(0, 10))
        ax.axis("off")
        ax.set_title(title, color=style.INK, fontsize=12.5)
        stack = layers + [("softmax → 28 probs", K)]
        n = len(stack)
        ys = np.linspace(8.4, 1.6, n)
        for k, ((name, size), y) in enumerate(zip(stack, ys)):
            width = 1.6 + 6.0 * np.log10(size) / np.log10(V)
            is_out = k == n - 1
            is_in = k == 0
            fc = style.LIGHT_FILL if is_in else (style.INK_2 if is_out else col)
            box(ax, 5 - width / 2, y - 0.45, width, 0.9, f"{name}\n{size:,}", fc=fc,
                color=style.INK if is_in else "white", size=9.5)
            if k < n - 1:
                arrow(ax, 5, y - 0.5, 5, ys[k + 1] + 0.5)
                if k == 0:
                    ax.text(5.25, (y + ys[1]) / 2, "sparse × dense", fontsize=8, color=style.MUTED, va="center")
        ax.text(5, 0.55, f"{param_count(arch, V, K):,} parameters", ha="center", fontsize=11, weight="bold",
                color=style.INK)
        ax.text(5, -0.6, note, ha="center", fontsize=9, color=style.INK_2, va="top")
    fig.tight_layout()
    style.save(fig, OUT / "architectures.png")


# ----------------------------------------------------------------------------- 5. TPR gap
def tpr_gap():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    rng = np.random.default_rng(3)
    for ax, (who, color, n_hit) in zip(axes, [("female nurses", style.FEMALE, 90), ("male nurses", style.MALE, 70)]):
        hits = np.zeros(100, bool)
        hits[rng.choice(100, n_hit, replace=False)] = True
        xs, ys = np.meshgrid(range(10), range(10))
        ax.scatter(xs.ravel()[hits], ys.ravel()[hits], s=90, color=color)
        ax.scatter(xs.ravel()[~hits], ys.ravel()[~hits], s=90, facecolor="none", edgecolor=color, lw=1.4)
        ax.set(xlim=(-0.8, 9.8), ylim=(-0.8, 9.8), aspect="equal")
        ax.axis("off")
        ax.set_title(f"100 {who}: {n_hit} predicted 'nurse'\nTPR = {n_hit / 100:.2f}", fontsize=11.5)
    fig.suptitle("TPR gap (nurse) = TPR women − TPR men = 0.90 − 0.70 = +0.20",
                 x=0.02, ha="left", fontsize=13, weight="bold")
    fig.text(0.02, 0.01, "filled = recognised as nurse, hollow = predicted something else.  "
             "Positive gap → the model favours women in this occupation (made-up numbers).",
             fontsize=9, color=style.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    style.save(fig, OUT / "tpr_gap.png")


if __name__ == "__main__":
    pipeline()
    cv_folds()
    architectures()
    tpr_gap()
    if TOKENIZER.exists():
        representations()
    else:
        print("tokenizer not found: run the notebook up to §3, then rerun this script")
