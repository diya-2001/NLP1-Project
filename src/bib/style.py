"""One visual style for every figure in the project.

Colours were checked with a colour-blindness validator:
women/men (orange/blue) and the three architectures (aqua/violet/amber) stay
distinguishable for common types of colour blindness.

Usage::

    from bib import style
    style.apply()
    ax.bar(..., color=style.FEMALE)
    style.save(fig, FIGURES / "1_5_gender.png")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- people ---------------------------------------------------------------
FEMALE = "#eb6834"   # orange
MALE = "#2a78d6"     # blue
GENDER_COLORS = {"female": FEMALE, "male": MALE, 1: FEMALE, 0: MALE}

# --- systems --------------------------------------------------------------
ARCH_COLORS = {"loglin": "#1baf7a", "mlp": "#6250d6", "deep": "#eda100"}   # aqua, violet, amber
ACCENT = "#2a78d6"   # default single-series colour (blue)

# --- ink & chrome ---------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"    # secondary text
MUTED = "#898781"    # axis labels, de-emphasised marks
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
LIGHT_FILL = "#e9e8e2"

# sequential (one hue, light -> dark) for heatmaps / confusion matrices
BLUES = LinearSegmentedColormap.from_list(
    "bib_blues", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# diverging: men-leaning (blue) <- grey -> women-leaning (orange)
GENDER_DIVERGING = LinearSegmentedColormap.from_list(
    "bib_gender", [MALE, "#f0efec", FEMALE])


def apply() -> None:
    """Set matplotlib defaults: light surface, hairline grid, no top/right spines."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "figure.dpi": 110,
    })


def title(ax, main: str, sub: str | None = None) -> None:
    """Bold takeaway title, with an optional grey one-line explanation underneath."""
    ax.set_title(main, pad=26 if sub else 12)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


def save(fig, path: Path | str, dpi: int = 200) -> None:
    """Save a figure for slides/blog (tight bounding box, consistent dpi)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

# representations: three steps of one blue (light -> dark), always shown with direct labels
REP_COLORS = {"indicator": "#86b6ef", "bow": "#2a78d6", "tfidf": "#0d366b"}
