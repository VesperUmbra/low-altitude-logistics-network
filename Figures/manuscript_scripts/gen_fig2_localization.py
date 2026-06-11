# 产出Fig 2 panel a,b,c,d
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from paper_plot_style import OUTPUT_DIR, PALETTE, add_panel_label, apply_style, finish_map_axis, save_panel, write_map_canvas_range


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

CELL_CSV = REVIEW_DATA_DIR / "figure_cell_metrics.csv"
ANCHOR_CSV = REVIEW_DATA_DIR / "figure_anchor_summary.csv"
STATION_CSV = REVIEW_DATA_DIR / "real_station_anchor_sites_mapped.csv"
STATION_ANCHOR_JSON = SOURCE_JSON_DIR / "real_station_anchor_results.json"


def main() -> None:
    apply_style()

    cell = pd.read_csv(CELL_CSV)
    anchors = pd.read_csv(ANCHOR_CSV)
    station = pd.read_csv(STATION_CSV)
    station_anchor = json.loads(STATION_ANCHOR_JSON.read_text(encoding="utf-8"))
    station = station[station["anchor_set"] == "operational_sites"].copy()

    fig = plt.figure(figsize=(7.3, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.22, 1.0])

    ax_map = fig.add_subplot(gs[0, :])
    ax_curve = fig.add_subplot(gs[1, 0])
    ax_anchor = fig.add_subplot(gs[1, 1])

    active = cell[cell["total_points"] > 0].copy()
    backbone = cell[cell["is_backbone_cell"] == True]
    hotspots = cell[cell["is_hotspot_cell"] == True].copy()
    endpoints = cell[cell["is_endpoint_cell"] == True]
    ax_map.set_facecolor(PALETTE["sand"])

    ax_map.scatter(
        active["lon"],
        active["lat"],
        marker="s",
        s=7.0,
        color="#006FB0",
        linewidths=0,
        alpha=1.0,
    )
    ax_map.scatter(
        backbone["lon"],
        backbone["lat"],
        marker="s",
        s=8.2,
        color="#499ECF",
        linewidths=0,
        alpha=1.0,
    )

    hotspot_sizes = np.interp(hotspots["exceedance_count"], (hotspots["exceedance_count"].min(), hotspots["exceedance_count"].max()), (12, 56))
    ax_map.scatter(
        hotspots["lon"],
        hotspots["lat"],
        s=hotspot_sizes,
        marker="o",
        color="#D0164D",
        edgecolor="white",
        linewidth=0.28,
        alpha=1.0,
        zorder=3,
    )

    ax_map.scatter(
        endpoints["lon"],
        endpoints["lat"],
        s=13,
        marker="o",
        facecolors="none",
        edgecolors="#008FA0",
        linewidths=0.55,
        alpha=1.0,
        zorder=4,
    )

    lon_pad = (active["lon"].max() - active["lon"].min()) * 0.02
    lat_pad = (active["lat"].max() - active["lat"].min()) * 0.02
    fig2a_xlim = (active["lon"].min() - lon_pad, active["lon"].max() + lon_pad)
    fig2a_ylim = (active["lat"].min() - lat_pad, active["lat"].max() + lat_pad)
    ax_map.set_xlim(*fig2a_xlim)
    ax_map.set_ylim(*fig2a_ylim)
    finish_map_axis(ax_map)
    ax_map.set_facecolor("none")
    add_panel_label(ax_map, "a")

    active_cell = cell[cell["total_points"] > 0].copy()
    ranked_hotspot = active_cell.sort_values(["exceedance_count", "total_points"], ascending=[False, False]).reset_index(drop=True)
    ranked_hotspot["hotspot_rank_pct"] = (np.arange(len(ranked_hotspot)) + 1) / len(ranked_hotspot) * 100.0
    ranked_hotspot["cumulative_exceedance_share"] = ranked_hotspot["exceedance_count"].cumsum() / ranked_hotspot["exceedance_count"].sum() * 100.0

    mask = ranked_hotspot["hotspot_rank_pct"] <= 5
    ax_curve.fill_between(
        ranked_hotspot.loc[mask, "hotspot_rank_pct"],
        0,
        ranked_hotspot.loc[mask, "cumulative_exceedance_share"],
        color="#f2b8ac",
        alpha=0.10,
    )
    ax_curve.plot(
        ranked_hotspot.loc[mask, "hotspot_rank_pct"],
        ranked_hotspot.loc[mask, "cumulative_exceedance_share"],
        color=PALETTE["red"],
        lw=2.2,
    )
    for pct in [0.5, 1.0]:
        idx = max(0, int(np.floor(len(ranked_hotspot) * pct / 100.0)) - 1)
        yy = ranked_hotspot.loc[idx, "cumulative_exceedance_share"]
        ax_curve.axvline(pct, color=PALETTE["grey"], lw=1.0, ls="--")
        ax_curve.scatter(pct, yy, s=24, color=PALETTE["red"], zorder=3)
        offset_y = -7.0 if pct == 0.5 else -5.0

    ax_curve.set_xlim(0, 5)
    ax_curve.set_ylim(0, 102)
    ax_curve.set_xlabel("Active cells ranked by state-sample count (%)")
    ax_curve.set_ylabel("Cumulative state-sample share (%)")
    ax_curve.grid(axis="y")
    add_panel_label(ax_curve, "b")

    wanted = [
        "Station exact",
        "Endpoint exact",
        "Station +1 cell",
        "Endpoint +1 cell",
        "Backbone (top 10%)",
    ]
    anchor_plot = anchors.set_index("anchor_label").loc[wanted].reset_index()
    y = np.arange(len(anchor_plot))[::-1]
    x_area = anchor_plot["active_cells_covered_share"].to_numpy() * 100.0
    x_exc = anchor_plot["exceedance_samples_covered_share"].to_numpy() * 100.0

    for i in range(len(anchor_plot)):
        ax_anchor.plot([x_area[i], x_exc[i]], [y[i], y[i]], color=PALETTE["grey"], lw=1.6, zorder=1)
        ax_anchor.scatter(x_area[i], y[i], s=32, color=PALETTE["navy"], edgecolor="white", linewidth=0.65, zorder=3)
        ax_anchor.scatter(x_exc[i], y[i], s=32, color=PALETTE["red"], edgecolor="white", linewidth=0.65, zorder=3)

    ax_anchor.set_xscale("log")
    ax_anchor.set_xlim(0.2, 120)
    ax_anchor.set_yticks(y)
    ax_anchor.set_yticklabels(anchor_plot["anchor_label"])
    ax_anchor.set_xlabel("Share of active cells or state samples (%)")
    ax_anchor.grid(axis="x", which="major")
    ax_anchor.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:g}"))
    add_panel_label(ax_anchor, "c")

    marker_legend = [
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=6, markerfacecolor=PALETTE["navy"], markeredgecolor="white", label="Footprint share"),
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=6, markerfacecolor=PALETTE["red"], markeredgecolor="white", label="State-sample share"),
    ]
    ax_anchor.legend(handles=marker_legend, frameon=False, loc="lower left")

    write_map_canvas_range(fig, ax_map, "fig2a")
    save_panel(fig, ax_map, "manuscript_fig2a_localized_hotspots_anchors", transparent=True)
    save_panel(fig, ax_curve, "manuscript_fig2b_slowdown_concentration")
    save_panel(fig, ax_anchor, "manuscript_fig2c_footprint_vs_slowdown_share")
    plt.close(fig)

    fig_station = plt.figure(figsize=(1266 / 300, 779 / 300), dpi=300)
    ax_station = fig_station.add_axes([0, 0, 1, 1])
    fig_station.patch.set_alpha(0)
    ax_station.set_facecolor("none")
    ax_station.scatter(
        station["lon"],
        station["lat"],
        s=22,
        marker="^",
        color="#F2AF13",
        edgecolor="#8F741C",
        linewidth=0.35,
        alpha=1.0,
        zorder=5,
    )
    ax_station.set_xlim(*fig2a_xlim)
    ax_station.set_ylim(*fig2a_ylim)
    finish_map_axis(ax_station)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig_station.savefig(
            OUTPUT_DIR / "manuscript_fig2a_mapped_uav_stations_layer.png",
            transparent=True,
            bbox_inches=None,
            pad_inches=0,
        )
        fig_station.savefig(
            OUTPUT_DIR / "manuscript_fig2a_mapped_uav_stations_layer.svg",
            transparent=True,
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig_station)

    fig_table, ax_table = plt.subplots(figsize=(5.8, 1.55), constrained_layout=True)
    station_anchor_sets = station_anchor["anchor_sets"]["operational_sites"]
    station_rows = [
        ("Station exact", station_anchor_sets["exact_anchor_cells"]),
        ("Station +1 cell", station_anchor_sets["buffer1_anchor_cells"]),
        ("Station +2 cells", station_anchor_sets["buffer2_anchor_cells"]),
    ]
    table_rows = []
    for label, row in station_rows:
        table_rows.append(
            [
                label,
                f"{row['active_cells_covered_share'] * 100:.3f}%",
                f"{row['endpoint_cells_covered_share'] * 100:.1f}%",
                f"{row['exceedance_samples_covered_share'] * 100:.1f}%",
                f"{row['sample_rr_exceedance']:.2f}x",
            ]
        )
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Anchor set", "Active cells", "Endpoint cells", "Slowdown samples", "RR"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.25)
    for (row, col), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor(PALETTE["grey_light"])
        cell_obj.set_linewidth(0.35)
        if row == 0:
            cell_obj.set_facecolor("#eef2f5")
            cell_obj.set_text_props(weight="bold", color=PALETTE["navy_dark"])
        elif row % 2 == 0:
            cell_obj.set_facecolor("#f7f9fb")
    add_panel_label(ax_table, "d")
    save_panel(fig_table, ax_table, "manuscript_fig2d_external_station_neighborhoods")
    plt.close(fig_table)
    print("Saved manuscript Figure 2 panels")


if __name__ == "__main__":
    main()
