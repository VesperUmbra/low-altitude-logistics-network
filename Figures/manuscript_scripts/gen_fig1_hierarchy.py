# 产出Fig 1 panel a,b,c
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, finish_map_axis, save_panel, write_map_canvas_range


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

CELL_CSV = REVIEW_DATA_DIR / "figure_cell_metrics.csv"
RANKED_CSV = REVIEW_DATA_DIR / "ranked_cell_traffic_table.csv"
SUMMARY_JSON = SOURCE_JSON_DIR / "summary_results.json"


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1.0) / n)


def main() -> None:
    apply_style()

    cell = pd.read_csv(CELL_CSV).query("total_points > 0").sort_values("rank")
    ranked = pd.read_csv(RANKED_CSV)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    active_share = summary["support_ratio_boundary"]["original_ratio_percent"]
    gini_coeff = gini(ranked["total_points"].to_numpy())

    fig = plt.figure(figsize=(7.35, 5.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.22, 1.0])

    ax_map = fig.add_subplot(gs[0, :])
    ax_lorenz = fig.add_subplot(gs[1, 0])
    ax_curve = fig.add_subplot(gs[1, 1])

    ax_map.set_facecolor(PALETTE["sand"])
    ax_map.scatter(
        cell["lon"],
        cell["lat"],
        marker="o",
        s=1.0,
        color="#4A9FD0",
        linewidths=0.0,
        alpha=0.95,
    )

    backbone = cell[cell["is_backbone_cell"] == True]
    backbone_full_share = len(backbone) / len(cell) * active_share
    ax_map.scatter(
        backbone["lon"],
        backbone["lat"],
        marker="o",
        s=1.0,
        color="#D0164D",
        linewidths=0.0,
        alpha=0.98,
    )

    lon_pad = (cell["lon"].max() - cell["lon"].min()) * 0.02
    lat_pad = (cell["lat"].max() - cell["lat"].min()) * 0.02
    fig1a_xlim = (cell["lon"].min() - lon_pad, cell["lon"].max() + lon_pad)
    fig1a_ylim = (cell["lat"].min() - lat_pad, cell["lat"].max() + lat_pad)
    ax_map.set_xlim(*fig1a_xlim)
    ax_map.set_ylim(*fig1a_ylim)
    finish_map_axis(ax_map)
    ax_map.set_facecolor("none")
    add_panel_label(ax_map, "a")

    ranked_low = ranked.sort_values("total_points", ascending=True).reset_index(drop=True)
    x = np.arange(1, len(ranked_low) + 1) / len(ranked_low)
    y = ranked_low["traffic_share"].cumsum().to_numpy()
    ax_lorenz.plot(x * 100, y * 100, color=PALETTE["navy"], lw=2.1)
    ax_lorenz.plot([0, 100], [0, 100], color=PALETTE["grey"], lw=1.1, ls="--")

    for pct in [90, 99]:
        idx = max(0, int(round(len(ranked_low) * pct / 100.0)) - 1)
        yy = y[idx] * 100
        ax_lorenz.scatter(pct, yy, s=28, color=PALETTE["navy_dark"], zorder=3)

    ax_lorenz.set_xlabel("Bottom-ranked active cells (%)")
    ax_lorenz.set_ylabel("Cumulative traffic share (%)")
    ax_lorenz.set_xlim(0, 100)
    ax_lorenz.set_ylim(0, 100)
    ax_lorenz.grid(axis="y")
    add_panel_label(ax_lorenz, "b")

    ranked_high = ranked.sort_values("total_points", ascending=False).reset_index(drop=True)
    x_zoom = (np.arange(1, len(ranked_high) + 1) / len(ranked_high)) * 100
    y_zoom = ranked_high["traffic_share"].cumsum().to_numpy() * 100
    mask = x_zoom <= 20
    ax_curve.fill_between(x_zoom[mask], 0, y_zoom[mask], color=PALETTE["blue_light"], alpha=0.14, zorder=1)
    ax_curve.plot(x_zoom[mask], y_zoom[mask], color=PALETTE["teal"], lw=2.3, zorder=3)
    ax_curve.axvline(10, color=PALETTE["navy_dark"], lw=1.2, ls="--", zorder=2)

    for pct in [1, 5, 10, 20]:
        idx = max(0, int(round(len(ranked_high) * pct / 100.0)) - 1)
        yy = y_zoom[idx]
        xx = pct
        ax_curve.scatter(xx, yy, s=28, color=PALETTE["teal"], edgecolor="white", linewidth=0.7, zorder=3)
    ax_curve.set_xlabel("Active-cell budget (%)")
    ax_curve.set_ylabel("Traffic captured (%)")
    ax_curve.set_xlim(0, 20)
    ax_curve.set_ylim(0, 92)
    ax_curve.xaxis.set_major_locator(mtick.MultipleLocator(5))
    ax_curve.grid(axis="y")
    add_panel_label(ax_curve, "c")

    write_map_canvas_range(fig, ax_map, "fig1a")
    save_panel(fig, ax_map, "manuscript_fig1a_active_footprint_backbone", transparent=True)
    save_panel(fig, ax_lorenz, "manuscript_fig1b_traffic_concentration")
    save_panel(fig, ax_curve, "manuscript_fig1c_traffic_captured_by_footprint")
    plt.close(fig)
    print("Saved manuscript Figure 1 panels")


if __name__ == "__main__":
    main()
