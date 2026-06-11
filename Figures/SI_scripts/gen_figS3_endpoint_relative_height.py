from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_figure


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
FOR_REVIEW = WORKSPACE_ROOT / "for_review"
REVIEW_DATA_DIR = FOR_REVIEW / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

INPUT_CSV = REVIEW_DATA_DIR / "endpoint_relative_height_classification.csv"
INPUT_JSON = SOURCE_JSON_DIR / "endpoint_relative_height_summary.json"

FIGURE_STEM = "S3_endpoint_relative_height_distribution"

CLASS_ORDER = ["ground_like", "building_compatible", "elevated_unresolved"]
CLASS_LABELS = {
    "ground_like": "Ground-like",
    "building_compatible": "Building-compatible",
    "elevated_unresolved": "Elevated / unresolved",
}
CLASS_COLORS = {
    "ground_like": PALETTE["navy"],
    "building_compatible": PALETTE["gold"],
    "elevated_unresolved": PALETTE["grey"],
}


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_stacked_bar(ax: plt.Axes, summary: dict) -> None:
    groups = [
        ("Endpoint\ncells\n(n=217)", summary["overall"], "endpoint_cells"),
        ("Endpoint\ntouches\n(n=92,818)", summary["overall"], "endpoint_touches"),
        ("Station-adjacent\ntouches\n(n=34,710)", summary["station_adjacent_subset"], "endpoint_touches"),
    ]

    x = np.arange(len(groups))
    bottoms = np.zeros(len(groups))

    for cls in CLASS_ORDER:
        heights = []
        for _, group_summary, metric in groups:
            if metric == "endpoint_cells":
                heights.append(group_summary[cls]["cell_share"])
            else:
                heights.append(group_summary[cls]["touch_share"])

        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=CLASS_COLORS[cls],
            edgecolor="white",
            linewidth=0.8,
            label=CLASS_LABELS[cls],
        )

        for xi, base, height in zip(x, bottoms, heights, strict=True):
            if height >= 0.085:
                ax.text(
                    xi,
                    base + height / 2,
                    percent(height),
                    ha="center",
                    va="center",
                    color="white" if cls != "building_compatible" else "black",
                    fontsize=8.2,
                    fontweight="bold",
                )
        bottoms += np.array(heights)

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Share")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

def main() -> None:
    apply_style()
    df = pd.read_csv(INPUT_CSV)
    summary = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax0, ax1 = axes

    bins = np.arange(-20, 171, 10)
    ax0.hist(
        df["rel_ground_m"],
        bins=bins,
        color=PALETTE["blue_light"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax0.axvspan(
        bins[0],
        summary["analysis_spec"]["ground_like_threshold_m"],
        color=PALETTE["navy"],
        alpha=0.12,
        label="Ground-like band",
    )
    ax0.axvline(summary["overall"]["median_rel_ground_m"], color=PALETTE["red"], linewidth=1.6, linestyle="--")
    ax0.text(
        summary["overall"]["median_rel_ground_m"] + 2,
        ax0.get_ylim()[1] * 0.90,
        f"Median {summary['overall']['median_rel_ground_m']:.1f} m",
        color=PALETTE["red"],
        fontsize=8.4,
    )
    ax0.set_xlabel("Median endpoint altitude above local terrain (m)")
    ax0.set_ylabel("Recurrent endpoint cells")
    ax0.set_xlim(-20, 170)
    ax0.grid(axis="y", linestyle="--", alpha=0.5)
    ax0.legend(frameon=False, loc="upper right")

    build_stacked_bar(ax1, summary)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, columnspacing=1.1, handlelength=1.8)

    add_panel_label(ax0, "a")
    add_panel_label(ax1, "b")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94], w_pad=1.3)
    save_figure(fig, FIGURE_STEM)


if __name__ == "__main__":
    main()
