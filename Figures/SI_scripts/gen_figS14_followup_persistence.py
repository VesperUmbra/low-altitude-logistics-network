from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, finish_map_axis, save_figure


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
FOR_REVIEW = WORKSPACE_ROOT / "for_review"
REVIEW_DATA_DIR = FOR_REVIEW / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

PATHS_CSV = REVIEW_DATA_DIR / "followup_compressed_paths_2025_2026.csv"
FLIGHT_METRICS_CSV = REVIEW_DATA_DIR / "followup_persistence_flight_metrics.csv"
SKELETON_EDGES_CSV = REVIEW_DATA_DIR / "followup_reference_route_skeleton_edges_support100.csv"
FIGURE_CELLS_CSV = REVIEW_DATA_DIR / "figure_cell_metrics.csv"
GRID_INFO_JSON = FOR_REVIEW / "review_code" / "data" / "processed" / "grid_refined_100m" / "grid_info.json"
SUMMARY_JSON = SOURCE_JSON_DIR / "followup_persistence_summary.json"
ROUTE_GROUP_CSV = REVIEW_DATA_DIR / "followup_route_group_metrics.csv"


def cell_center(cell_id: str, grid: dict[str, float]) -> tuple[float, float]:
    row_str, col_str = cell_id.split("_")
    row = int(row_str)
    col = int(col_str)
    lat = grid["min_lat"] + ((row + 0.5) * grid["grid_size"]) / grid["meters_per_degree_lat"]
    lon = grid["min_lon"] + ((col + 0.5) * grid["grid_size"]) / grid["meters_per_degree_lon"]
    return lon, lat


def load_grid() -> dict[str, float]:
    raw = json.loads(GRID_INFO_JSON.read_text(encoding="utf-8"))
    lat_center = (float(raw["min_lat"]) + float(raw["max_lat"])) / 2.0
    return {
        "grid_size": float(raw["grid_size"]),
        "min_lon": float(raw["min_lon"]),
        "max_lon": float(raw["max_lon"]),
        "min_lat": float(raw["min_lat"]),
        "max_lat": float(raw["max_lat"]),
        "meters_per_degree_lat": 111000.0,
        "meters_per_degree_lon": 111000.0 * __import__("math").cos(__import__("math").radians(lat_center)),
    }


def add_route_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["route_group"] = (
        df["flight_date_local"].astype(str).str.slice(0, 10)
        + " | "
        + df["source_folder"].astype(str)
        + " | "
        + df["route_code"].astype(str)
    )
    return df


def route_group_label(route_group: str) -> str:
    date, folder, route = [part.strip() for part in route_group.split("|")]
    folder_map = {
        "A航路": "A",
        "B航路": "B",
        "B航路-2": "B-2",
        "root_logs": "root",
    }
    folder_short = folder_map.get(folder, folder)
    if route == "unassigned":
        return f"{date}\nroot logs"
    return f"{date}\n{folder_short}-{route}"


def group_color(route_group: str) -> str:
    if "2025-11-10" in route_group:
        return PALETTE["grey_dark"]
    if "2025-12-18" in route_group:
        return PALETTE["orange"]
    if "2026-03-16" in route_group:
        return PALETTE["teal"]
    if "2026-03-18" in route_group:
        return PALETTE["grey"]
    return PALETTE["red"]


def main() -> None:
    apply_style()

    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    grid = load_grid()
    paths = add_route_group(pd.read_csv(PATHS_CSV))
    metrics = add_route_group(pd.read_csv(FLIGHT_METRICS_CSV))
    cells = pd.read_csv(FIGURE_CELLS_CSV, usecols=["cell_id", "lon", "lat", "is_backbone_cell"])
    backbone = cells[cells["is_backbone_cell"]].copy()
    skeleton = pd.read_csv(SKELETON_EDGES_CSV)

    route_group_metrics = (
        metrics.groupby("route_group", as_index=False)
        .agg(
            flights=("flight_id", "count"),
            weighted_hits=("edge_hits_on_2023_skeleton", "sum"),
            total_edges=("n_edges", "sum"),
            median_edge_share=("edge_share_on_2023_skeleton", "median"),
        )
        .assign(
            weighted_edge_share=lambda d: d["weighted_hits"] / d["total_edges"],
            label=lambda d: d["route_group"].map(route_group_label),
            color=lambda d: d["route_group"].map(group_color),
        )
        .sort_values(["weighted_edge_share", "route_group"])
        .reset_index(drop=True)
    )
    route_group_metrics.to_csv(ROUTE_GROUP_CSV, index=False)

    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.28)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_diag = fig.add_subplot(gs[1, 1])

    for row in skeleton.itertuples(index=False):
        lon1, lat1 = cell_center(str(row.source_cell), grid)
        lon2, lat2 = cell_center(str(row.target_cell), grid)
        ax_map.plot([lon1, lon2], [lat1, lat2], color=PALETTE["grey_light"], lw=0.55, alpha=0.45, zorder=1)

    ax_map.scatter(backbone["lon"], backbone["lat"], s=2.2, color=PALETTE["blue_light"], alpha=0.45, linewidths=0, zorder=2)

    for route_group, group in paths.groupby("route_group"):
        color = group_color(route_group)
        for _, flight_path in group.groupby("flight_id"):
            ax_map.plot(
                flight_path["lon_center"],
                flight_path["lat_center"],
                color=color,
                lw=1.0,
                alpha=0.82 if "2026" in route_group else 0.72,
                zorder=3,
            )

    finish_map_axis(ax_map)
    ax_map.set_xlim(grid["min_lon"], grid["max_lon"])
    ax_map.set_ylim(grid["min_lat"], grid["max_lat"])
    ax_map.set_title("2024 support-100 route skeleton with 2025-2026 follow-up flights", loc="left", pad=4)
    add_panel_label(ax_map, "a")

    x = range(len(route_group_metrics))
    ax_bar.bar(
        x,
        route_group_metrics["weighted_edge_share"] * 100.0,
        color=route_group_metrics["color"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax_bar.scatter(
        x,
        route_group_metrics["median_edge_share"] * 100.0,
        color=PALETTE["navy_dark"],
        s=22,
        zorder=3,
        label="Median flight edge share",
    )
    ax_bar.axhline(50.0, color=PALETTE["grey_dark"], lw=0.8, ls="--", alpha=0.8)
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(route_group_metrics["label"], rotation=35, ha="right")
    ax_bar.set_ylabel("Edges on 2024 skeleton (%)")
    ax_bar.set_ylim(0, 85)
    ax_bar.set_title("Route-group overlap with the 2024 skeleton", loc="left", pad=4)
    ax_bar.legend(frameon=False, loc="upper left")
    ax_bar.grid(axis="y", alpha=0.45)
    add_panel_label(ax_bar, "b")

    diag_labels = [
        "All-flight\nweighted edge share",
        "Median flight\nedge share",
        "2024 endpoint+1\nmatch of follow-up endpoints",
        "Station+1\nmatch of follow-up endpoints",
        "Slow points in\nown endpoint+1",
        "All points in\nown endpoint+1",
    ]
    diag_values = [
        summary["route_persistence"]["weighted_edge_share_on_2023_skeleton"] * 100.0,
        summary["route_persistence"]["median_flight_edge_share_on_2023_skeleton"] * 100.0,
        summary["interface_persistence"]["endpoint_buffer1_match_share"] * 100.0,
        summary["interface_persistence"]["station_buffer1_match_share"] * 100.0,
        summary["localized_slowing_persistence"]["slow_point_share_in_own_endpoint_buffer1"] * 100.0,
        summary["localized_slowing_persistence"]["all_point_share_in_own_endpoint_buffer1"] * 100.0,
    ]
    diag_colors = [
        PALETTE["navy"],
        PALETTE["blue"],
        PALETTE["grey"],
        PALETTE["grey_dark"],
        PALETTE["red"],
        PALETTE["orange"],
    ]
    ax_diag.barh(range(len(diag_labels)), diag_values, color=diag_colors, edgecolor="white", linewidth=0.6)
    for idx, value in enumerate(diag_values):
        ax_diag.text(value + 1.2, idx, f"{value:.1f}", va="center", ha="left", fontsize=8.3)
    ax_diag.set_yticks(range(len(diag_labels)))
    ax_diag.set_yticklabels(diag_labels)
    ax_diag.invert_yaxis()
    ax_diag.set_xlim(0, 100)
    ax_diag.set_xlabel("Share (%)")
    ax_diag.set_title("Persistence and interface localization diagnostics", loc="left", pad=4)
    ax_diag.grid(axis="x", alpha=0.45)
    ax_diag.text(
        0.98,
        0.05,
        f"Slow-point lift = {summary['localized_slowing_persistence']['slowing_localization_lift']:.2f}x",
        transform=ax_diag.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=PALETTE["grey_dark"],
    )
    add_panel_label(ax_diag, "c")

    save_figure(fig, "S14_followup_persistence")


if __name__ == "__main__":
    main()
