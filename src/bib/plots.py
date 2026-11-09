"""All chart-drawing code, so notebook cells only *compute* things and then call one plot function.

Every function takes the data plus a ``title`` (the chart's takeaway, written in the notebook),
draws one chart in the project style (`bib.style`), and returns the figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from . import style

WORD_START = "▁"   # the tokenizer marks "a new word starts here" with this character
REP_LABEL = {"indicator": "R1 indicator", "bow": "R2 bag-of-words", "tfidf": "R3 TF-IDF"}
ARCH_LABEL = {"loglin": "A1 log-linear", "mlp": "A2 MLP", "deep": "A3 deeper MLP"}


def _label_points(ax, xs, ys, names, nudge=None):
    """Write each point's name next to it; ``nudge`` = {name: (dx, dy, ha)} moves crowded labels."""
    nudge = nudge or {}
    for x, y, n in zip(xs, ys, names):
        dx, dy, ha = nudge.get(n, (5, -3, "left"))
        ax.annotate(n.replace("_", " "), (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=style.INK_2, ha=ha)


# =============================================================================== §1 data
def class_shares(counts: pd.Series, title: str, sub: str):
    share = 100 * counts / counts.sum()
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.barh(share.index, share.values, height=0.72,
            color=[style.ACCENT if i == 0 else "#b7d3f6" for i in range(len(share))])
    ax.invert_yaxis()
    for i, (name, s) in enumerate(share.items()):          # label only the biggest and smallest
        if i < 3 or i >= len(share) - 3:
            ax.text(s + 0.4, i, f"{s:.1f}%  ({counts[name]:,})", va="center", fontsize=8.5, color=style.INK_2)
    ax.set(xlabel="% of training bios", xlim=(0, share.max() * 1.25))
    ax.grid(axis="y", visible=False)
    style.title(ax, title, sub)
    return fig


def baseline_bars(table: pd.DataFrame, title: str, sub: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(table.shape[1])
    for j, (name, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + (j - 0.5) * 0.36, row.values, width=0.34, label=name,
                      color=[style.INK_2, style.BASELINE][j])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9, color=style.INK_2)
    ax.set_xticks(x, table.columns)
    ax.set(ylim=(0, 0.36), ylabel="score (higher = better)")
    ax.legend(loc="upper right")
    style.title(ax, title, sub)
    return fig


def gender_scatter(gender_occ: pd.DataFrame, title: str, sub: str):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(50, color=style.BASELINE, lw=1.2, ls="--")
    ax.text(1.02e5, 51.2, "50 / 50", color=style.MUTED, fontsize=9, ha="right")
    ax.scatter(gender_occ["bios"], gender_occ["% female"], s=70, c=gender_occ["% female"],
               cmap=style.GENDER_DIVERGING, vmin=0, vmax=100, edgecolor=style.MUTED, lw=0.6, zorder=3)
    _label_points(ax, gender_occ["bios"], gender_occ["% female"], gender_occ.index,
                  {"yoga_teacher": (6, -12, "left"), "software_engineer": (6, -12, "left"), "teacher": (6, -12, "left")})
    ax.set_xscale("log")
    ax.set(xlabel="number of training bios (log scale)", ylabel="% of bios that are women", ylim=(0, 100),
           xlim=(700, 1.3e5))
    style.title(ax, title, sub)
    return fig


def length_hist(n_words: pd.Series, gender: pd.Series, title: str, sub: str):
    med = n_words.median()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.arange(0, 185, 5)
    for g, col in [("female", style.FEMALE), ("male", style.MALE)]:
        w = n_words[gender == g]
        ax.hist(w, bins=bins, histtype="step", lw=2, color=col, label=g, weights=np.full(len(w), 100 / len(w)))
    ax.axvline(med, color=style.INK, lw=1, ls="--")
    ax.text(med + 2, ax.get_ylim()[1] * 0.92, f"median {med:.0f} words", fontsize=9.5)
    ax.set(xlabel="words per bio", ylabel="% of that gender's bios")
    ax.legend()
    style.title(ax, title, sub)
    return fig


def pronoun_bars(tab: pd.DataFrame, title: str, sub: str):
    """``tab``: rows = true gender (female, male), columns = rule's guess, values = % of the row."""
    fig, ax = plt.subplots(figsize=(9, 2.8))
    left = np.zeros(2)
    for guess, col in [("female", style.FEMALE), ("male", style.MALE), ("no clear pronoun", style.BASELINE)]:
        vals = tab.loc[["female", "male"], guess].to_numpy()
        ax.barh(["women", "men"], vals, left=left, color=col, height=0.55, edgecolor=style.SURFACE, lw=2,
                label=f"rule says: {guess}")
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9.5)
        left += vals
    ax.set(xlim=(0, 100), xlabel="% of bios with that gender label")
    ax.grid(False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    style.title(ax, title, sub)
    return fig


def hbars(values: pd.Series, title: str, sub: str = None, xlabel: str = "", fmt: str = "%.1f%%", color=None):
    """Simple horizontal bar chart, largest on top."""
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(values) + 1.2))
    bars = ax.barh(values.index, values.values, color=color or style.ACCENT, height=0.6)
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9, color=style.INK_2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=xlabel, xlim=(0, values.max() * 1.2))
    style.title(ax, title, sub)
    return fig


def keyword_scatter(kp: pd.DataFrame, title: str, sub: str):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axvline(50, color=style.BASELINE, lw=1, ls="--")
    ax.axhline(50, color=style.BASELINE, lw=1, ls="--")
    ax.scatter(kp["coverage (%)"], kp["reliability (%)"], s=60, color=style.ACCENT, edgecolor="white", lw=1.5, zorder=3)
    _label_points(ax, kp["coverage (%)"], kp["reliability (%)"], kp.index,
                  {"photographer": (5, -12, "left"), "dj": (-5, -4, "right"), "composer": (-5, -12, "right"),
                   "psychologist": (0, -13, "center"), "rapper": (5, 5, "left"), "architect": (5, -10, "left")})
    ax.text(98, 3, "often present,\nrarely trustworthy", ha="right", fontsize=9, color=style.MUTED)
    ax.text(98, 97, "easy: present and trustworthy", ha="right", va="top", fontsize=9, color=style.MUTED)
    ax.text(2, 97, "trustworthy but rarely present", va="top", fontsize=9, color=style.MUTED)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="coverage: % of the occupation's bios that contain its keyword",
           ylabel="reliability: % of keyword bios that are that occupation")
    style.title(ax, title, sub)
    return fig


# =============================================================================== §3 tokenizer
def _draw_tokens(ax, tokens, y0=0.0, max_width=100.0, row_h=1.0) -> float:
    """Tokens as coloured chips (blue = starts a word, amber = continues a word, grey = punctuation)."""
    x, y = 0.0, y0
    for t in tokens:
        label = t.replace(WORD_START, "")
        w = 0.9 * max(len(label), 1) + 1.2
        if x + w > max_width:
            x, y = 0.0, y - row_h
        if not any(ch.isalnum() for ch in label):
            fc = style.LIGHT_FILL
        elif t.startswith(WORD_START):
            fc = "#cde2fb"
        else:
            fc = "#fbe3b0"
        ax.add_patch(FancyBboxPatch((x, y - 0.38), w - 0.35, 0.76, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=fc, ec="none"))
        ax.text(x + (w - 0.35) / 2, y, label, ha="center", va="center", fontsize=9, family="monospace")
        x += w
    return y


def token_comparison(rows: dict, title: str, sub: str):
    """``rows`` = {caption: list of tokens}; draws each tokenization under its caption."""
    fig, ax = plt.subplots(figsize=(12, 1.8 * len(rows) + 0.4))
    ax.axis("off")
    y = 0.0
    for caption, tokens in rows.items():
        ax.text(0, y + 0.9, caption, fontsize=10, weight="bold")
        y = _draw_tokens(ax, tokens, y0=y) - 2.2
    ax.set(xlim=(0, 100), ylim=(y + 1.2, 1.4))
    style.title(ax, title, sub)
    return fig


def hist_with_median(values, title: str, xlabel: str, bins=60):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(values, bins=bins, color=style.ACCENT)
    med = np.median(values)
    ax.axvline(med, color=style.INK, ls="--", lw=1)
    ax.text(med + 3, ax.get_ylim()[1] * 0.9, f"median {med:.0f}", fontsize=9.5)
    ax.set(xlabel=xlabel, ylabel="number of bios")
    style.title(ax, title)
    return fig


# =============================================================================== §4 representations
def count_distribution(dist: pd.Series, title: str, sub: str):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(dist.index, dist.values, color=[style.ACCENT] + ["#86b6ef"] * (len(dist) - 1), width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9, color=style.INK_2)
    ax.set(xlabel="count of a token in a bio (non-zero entries only)", ylabel="% of non-zero entries")
    ax.grid(axis="x", visible=False)
    style.title(ax, title, sub)
    return fig


def one_bio_three_reps(values: dict, tokens: list, n_top: int, title: str):
    """``values`` = {rep: array of that bio's values for ``tokens``}. The first ``n_top`` tokens are the
    highest TF-IDF ones, the rest the lowest (a dashed line separates them)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for ax, (rep, v) in zip(axes, values.items()):
        ax.barh(range(len(tokens)), v, color=style.REP_COLORS[rep], height=0.7)
        ax.set_title(REP_LABEL[rep], fontsize=11.5)
        ax.grid(axis="y", visible=False)
        ax.axhline(n_top - 0.5, color=style.BASELINE, lw=1, ls="--")
    axes[0].set_yticks(range(len(tokens)), tokens)
    axes[0].invert_yaxis()
    axes[2].text(axes[2].get_xlim()[1], n_top + 0.3, "lowest TF-IDF weights", ha="right", fontsize=8.5, color=style.MUTED)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


def idf_bars(idf: pd.Series, pronouns: list, job_tokens: list, title: str, sub: str):
    pron_col = {"▁she": style.FEMALE, "▁her": style.FEMALE, "▁he": style.MALE, "▁his": style.MALE, "▁him": style.MALE}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([t.replace(WORD_START, "") or "␣ (space)" for t in idf.index], idf.values, height=0.7,
            color=[pron_col.get(t, style.ACCENT if t in job_tokens else style.BASELINE) for t in idf.index])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="idf (1 = token is in every bio; higher = rarer = more weight)", xlim=(0.9, idf.max() * 1.08))
    style.title(ax, title, sub)
    return fig


# =============================================================================== §5 architectures
def training_curves(histories: dict):
    """``histories`` = {arch: list of per-epoch dicts from models.train_torch}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arch, hist in histories.items():
        h = pd.DataFrame(hist)
        col = style.ARCH_COLORS[arch]
        axes[0].plot(h["epoch"], h["train_loss"], color=col, lw=2, label=f"{arch}: training data")
        axes[0].plot(h["epoch"], h["holdout_loss"], color=col, lw=2, ls="--", label=f"{arch}: hold-out")
        axes[1].plot(h["epoch"], h["holdout_macro_f1"], color=col, lw=2, marker="o", label=arch)
        best = h["holdout_macro_f1"].idxmax()
        axes[1].scatter(h.loc[best, "epoch"], h.loc[best, "holdout_macro_f1"], s=160, facecolor="none",
                        edgecolor=col, lw=2)
    axes[0].set(xlabel="epoch (one pass over the training data)", ylabel="loss (lower = better)")
    axes[1].set(xlabel="epoch", ylabel="hold-out macro-F1")
    axes[0].legend()
    axes[1].legend()
    style.title(axes[0], "Loss on training data keeps falling; on hold-out it stalls", "the gap = overfitting (memorising)")
    style.title(axes[1], "Early stopping keeps the best epoch (circled)", "measured on the inner 10% hold-out")
    fig.tight_layout()
    return fig


def tuning_lines(table: pd.DataFrame, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, arch in zip(axes, ARCH_LABEL):
        t = table[table["arch"] == arch]
        for rep in REP_LABEL:
            r = t[t["rep"] == rep].sort_values("value")
            ax.plot(r["value"], r["holdout_macro_f1"], marker="o", lw=2, color=style.REP_COLORS[rep])
            ax.annotate(rep, (r["value"].iloc[-1], r["holdout_macro_f1"].iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", fontsize=8.5, color=style.INK_2, va="center")
        vals = sorted(t["value"].unique())
        ax.set_xscale("log")
        ax.set_xticks(vals, [f"{v:g}" for v in vals])
        ax.minorticks_off()
        ax.set(xlabel=t["param"].iloc[0])
        ax.set_title(ARCH_LABEL[arch], fontsize=11.5)
    axes[0].set_ylabel("hold-out macro-F1")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


# =============================================================================== §8 results
def fold_scores(scores: pd.DataFrame, order, title: str, sub: str, metric="macro_f1"):
    """One column per system: every fold score as a dot, the mean as a black bar."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    rng = np.random.default_rng(0)
    for i, name in enumerate(order):
        s = scores[scores["system"] == name]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(s)), s[metric], s=34, color=style.ARCH_COLORS[s["arch"].iloc[0]],
                   alpha=0.9, edgecolor="white", lw=0.8, zorder=3)
        ax.hlines(s[metric].mean(), i - 0.28, i + 0.28, color=style.INK, lw=2, zorder=4)
    ax.set_xticks(range(len(order)), [n.replace("_", "\n") for n in order])
    ax.grid(axis="x", visible=False)
    for arch, col in style.ARCH_COLORS.items():
        ax.scatter([], [], color=col, label=ARCH_LABEL[arch])
    ax.legend(loc="lower left", ncol=3)
    ax.set(ylabel=f"{metric.replace('_', '-')} (validation fold)")
    style.title(ax, title, sub)
    return fig


def grid_heatmap(table: pd.DataFrame, title: str, sub: str, fmt="{:.3f}"):
    """3x3 heatmap (rows = representations, columns = architectures) with the values written in."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    vals = table.to_numpy(dtype=float)
    ax.imshow(vals, cmap=style.BLUES, vmin=vals.min() - (vals.max() - vals.min()) * 0.6, vmax=vals.max())
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            dark = (vals[i, j] - vals.min()) / max(vals.max() - vals.min(), 1e-9) > 0.55
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=12,
                    weight="bold" if vals[i, j] == vals.max() else "normal", color="white" if dark else style.INK)
    ax.set_xticks(range(vals.shape[1]), [ARCH_LABEL[c] for c in table.columns])
    ax.set_yticks(range(vals.shape[0]), [REP_LABEL[r] for r in table.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    style.title(ax, title, sub)
    return fig


def f1_vs_keyword(f1: pd.Series, reliability: pd.Series, color, title_left: str, title_right: str, sub_right: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1, 1.3]})
    s = f1.sort_values()
    axes[0].barh([n.replace("_", " ") for n in s.index], s.values, color=color, height=0.7)
    axes[0].grid(axis="y", visible=False)
    axes[0].set(xlabel="F1", xlim=(0, 1))
    style.title(axes[0], title_left)
    axes[1].scatter(reliability, f1[reliability.index], s=60, color=style.ACCENT, edgecolor="white", lw=1.5, zorder=3)
    _label_points(axes[1], reliability, f1[reliability.index], reliability.index)
    axes[1].set(xlabel="keyword reliability from §1.9 (%)", ylabel="F1", xlim=(0, 100), ylim=(0, 1))
    style.title(axes[1], title_right, sub_right)
    fig.tight_layout()
    return fig


def confusion_heatmap(cm: np.ndarray, names, title: str, sub: str):
    """Row-normalised confusion matrix; the diagonal (correct) is grey so the errors stand out."""
    fig, ax = plt.subplots(figsize=(10, 9))
    off = cm.copy()
    np.fill_diagonal(off, np.nan)
    ax.imshow(off, cmap=style.BLUES, vmin=0, vmax=np.nanmax(off))
    for i in range(len(names)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color=style.LIGHT_FILL))
        ax.text(i, i, f"{100 * cm[i, i]:.0f}", ha="center", va="center", fontsize=6.5, color=style.INK_2)
    ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names], rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=8)
    ax.set(xlabel="predicted occupation", ylabel="true occupation")
    ax.grid(False)
    style.title(ax, title, sub)
    return fig


# =============================================================================== §9 Bayesian
def comparison_bars(table: pd.DataFrame, rope: float, title: str, sub: str):
    """One row per comparison: stacked P(A better) / P(practically equal) / P(B better)."""
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(table) + 1.6))
    labels = [f"{a}  vs  {b}" for a, b in zip(table["A"], table["B"])]
    left = np.zeros(len(table))
    for col, colour, name in [("P(A better)", style.INK_2, "A better"), ("P(equal)", style.BASELINE, "practically equal"),
                              ("P(B better)", style.ACCENT, "B better")]:
        v = table[col].to_numpy()
        ax.barh(labels, v, left=left, color=colour, height=0.65, edgecolor=style.SURFACE, lw=1.5, label=name)
        for i, x in enumerate(v):
            if x > 0.08:
                ax.text(left[i] + x / 2, i, f"{x:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if colour != style.BASELINE else style.INK)
        left += v
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel=f"posterior probability (ROPE = ±{rope})")
    ax.grid(False)
    style.title(ax, title, sub)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), ncol=3, loc="lower center", frameon=False)
    return fig


# =============================================================================== §10 fairness
def gap_vs_female(tpr: pd.DataFrame, title: str, sub: str):
    t = tpr[tpr["reported"]]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(0, color=style.BASELINE, lw=1.2)
    ax.axvline(50, color=style.BASELINE, lw=1, ls="--")
    ax.scatter(t["pct_female"], t["gap"], s=70, c=t["pct_female"], cmap=style.GENDER_DIVERGING, vmin=0, vmax=100,
               edgecolor=style.MUTED, lw=0.6, zorder=3)
    slope, icept = np.polyfit(t["pct_female"], t["gap"], 1)
    xs = np.array([0, 100])
    ax.plot(xs, icept + slope * xs, color=style.INK_2, lw=1, ls=":")
    _label_points(ax, t["pct_female"], t["gap"], t.index)
    ax.set(xlim=(0, 100), xlabel="% of the occupation's bios that are women",
           ylabel="TPR gap = recall(women) − recall(men)")
    ax.text(2, ax.get_ylim()[1] * 0.92, "women recognised more often ↑", fontsize=9, color=style.FEMALE)
    ax.text(2, ax.get_ylim()[0] * 0.92, "men recognised more often ↓", fontsize=9, color=style.MALE)
    style.title(ax, title, sub)
    return fig


def system_bars(values: pd.Series, title: str, sub: str, xlabel: str, fmt="%.3f"):
    """One bar per system, coloured by architecture (system names look like 'mlp_tfidf')."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols = [style.ARCH_COLORS[n.split("_")[0]] for n in values.index]
    bars = ax.barh(values.index, values.values, color=cols, height=0.65)
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9, color=style.INK_2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=xlabel, xlim=(0, values.max() * 1.2))
    style.title(ax, title, sub)
    return fig


def before_after(table: pd.DataFrame, title: str, sub: str):
    """``table``: rows = metrics, columns = ['original', 'scrubbed'] -> paired bars per metric."""
    fig, axes = plt.subplots(1, len(table), figsize=(3.6 * len(table), 3.8))
    for ax, (metric, row) in zip(np.atleast_1d(axes), table.iterrows()):
        bars = ax.bar(row.index, row.values, color=[style.ACCENT, style.BASELINE], width=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9.5, color=style.INK_2)
        ax.set_title(metric, fontsize=11)
        ax.set(ylim=(0, row.max() * 1.25))
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, weight="bold")
    fig.text(0.01, 0.9, sub, fontsize=9.5, color=style.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def pronoun_weight_scatter(df: pd.DataFrame, title: str, sub: str):
    """``df``: index = occupation, columns 'pct_female' and 'she_minus_he' (log-linear weight difference)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color=style.BASELINE, lw=1.2)
    ax.scatter(df["pct_female"], df["she_minus_he"], s=70, c=df["pct_female"], cmap=style.GENDER_DIVERGING,
               vmin=0, vmax=100, edgecolor=style.MUTED, lw=0.6, zorder=3)
    _label_points(ax, df["pct_female"], df["she_minus_he"], df.index)
    ax.set(xlim=(0, 100), xlabel="% of the occupation's bios that are women",
           ylabel="weight('she') − weight('he')")
    style.title(ax, title, sub)
    return fig
