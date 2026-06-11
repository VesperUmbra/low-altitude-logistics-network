from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
FIGURES_DIR = ROOT.parent / "SI_output"


PALETTE = {
    "navy": "#1f4e79",
    "navy_dark": "#15324e",
    "blue": "#2f6690",
    "blue_light": "#a9cce3",
    "teal": "#1b6f7a",
    "red": "#c23b22",
    "orange": "#e38b06",
    "gold": "#d19a00",
    "purple": "#7a3e9d",
    "grey": "#9aa1a9",
    "grey_dark": "#5d6670",
    "grey_light": "#d8dde3",
    "sand": "#f7f4ee",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.8,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.4,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.color": PALETTE["grey_light"],
            "grid.linewidth": 0.65,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def finish_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    fig.savefig(FIGURES_DIR / f"{stem}.png")
    fig.savefig(FIGURES_DIR / f"{stem}.svg")
