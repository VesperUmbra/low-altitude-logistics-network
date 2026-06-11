from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
OUTPUT_DIR = ROOT.parent / "output_revised"


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
    return None


def finish_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_map_range(name: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "map_canvas_lonlat_ranges.txt"
    rows: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("panel\t"):
                rows[line.split("\t", 1)[0]] = line
    rows[name] = (
        f"{name}\t"
        f"lon_min={xlim[0]:.8f}\t"
        f"lon_max={xlim[1]:.8f}\t"
        f"lat_min={ylim[0]:.8f}\t"
        f"lat_max={ylim[1]:.8f}"
    )
    content = "panel\tlon_min\tlon_max\tlat_min\tlat_max\n" + "\n".join(rows[k] for k in sorted(rows)) + "\n"
    path.write_text(content, encoding="utf-8")


def write_map_canvas_range(fig: plt.Figure, ax: plt.Axes, name: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.06)
    bbox_display = bbox.transformed(fig.dpi_scale_trans)
    inv = ax.transData.inverted()
    lon_left, lat_bottom = inv.transform((bbox_display.x0, bbox_display.y0))
    lon_right, lat_top = inv.transform((bbox_display.x1, bbox_display.y1))
    write_map_range(
        name,
        (min(lon_left, lon_right), max(lon_left, lon_right)),
        (min(lat_bottom, lat_top), max(lat_bottom, lat_top)),
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg")


def save_panel(fig: plt.Figure, axes: plt.Axes | list[plt.Axes], stem: str, transparent: bool = False) -> None:
    if not isinstance(axes, list):
        axes = [axes]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
    bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.06)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches=bbox, transparent=transparent)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches=bbox, transparent=transparent)
