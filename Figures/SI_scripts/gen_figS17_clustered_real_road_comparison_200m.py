from __future__ import annotations

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_figure


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON = REVIEW_DATA / "source_json"

NODES_CSV = REVIEW_DATA / "air_ground_clustered_road_nodes_200m.csv"
EDGES_CSV = REVIEW_DATA / "air_ground_clustered_road_edges_200m.csv"
BIN_CSV = REVIEW_DATA / "air_ground_clustered_road_by_distance_bin_200m.csv"
GEOMETRY_JSON = SOURCE_JSON / "air_ground_clustered_road_geometries_200m.json"
SUMMARY_JSON = SOURCE_JSON / "air_ground_clustered_road_summary_200m.json"

DIST_ORDER = ["0-1", "1-3", "3-5", "5-10", "10-20", "20-80"]


def _sizes(values: pd.Series, lo: float = 14, hi: float = 125) -> np.ndarray:
    clean = np.sqrt(values.to_numpy(dtype=float))
    if np.all(clean == clean[0]):
        return np.full_like(clean, (lo + hi) / 2.0)
    return lo + (clean - clean.min()) / (clean.max() - clean.min()) * (hi - lo)


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = np.argsort(values.to_numpy(dtype=float))
    sorted_values = values.to_numpy(dtype=float)[order]
    sorted_weights = weights.to_numpy(dtype=float)[order]
    cutoff = sorted_weights.sum() / 2.0
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def map_panel(ax: plt.Axes, nodes: pd.DataFrame, edges: pd.DataFrame, geometries: dict[str, object]) -> None:
    top_edges = edges.sort_values("order_support", ascending=False).head(60).copy()
    top_ids = set(top_edges["edge_id"].astype(str))

    for item in geometries.get("routes", []):
        if item.get("edge_id") not in top_ids:
            continue
        geom = item.get("geometry")
        if not geom:
            continue
        coords = np.asarray(geom["coordinates"], dtype=float)
        ax.plot(coords[:, 0], coords[:, 1], color=PALETTE["grey"], lw=0.75, alpha=0.24, zorder=1)

    for row in top_edges.itertuples(index=False):
        ax.plot(
            [row.source_lon, row.target_lon],
            [row.source_lat, row.target_lat],
            color=PALETTE["navy"],
            lw=0.35 + 1.25 * np.sqrt(row.order_support) / np.sqrt(top_edges["order_support"].max()),
            alpha=0.52,
            zorder=2,
        )

    source_strength = edges.groupby("source_cluster", observed=False)["order_support"].sum()
    target_strength = edges.groupby("target_cluster", observed=False)["order_support"].sum()
    route_strength = source_strength.add(target_strength, fill_value=0)
    active_nodes = nodes.loc[nodes["cluster_id"].isin(route_strength.index)].copy()
    active_nodes["route_strength_orders"] = active_nodes["cluster_id"].map(route_strength)
    ax.scatter(
        active_nodes["centroid_lon"],
        active_nodes["centroid_lat"],
        s=_sizes(active_nodes["route_strength_orders"], 12, 105),
        color=PALETTE["orange"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.86,
        zorder=3,
    )
    ax.set_title("200 m endpoint-clustered route network", pad=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", color=PALETTE["grey_light"], alpha=0.8)

    handles = [
        mlines.Line2D([], [], color=PALETTE["navy"], lw=1.5, label="Aerial connection"),
        mlines.Line2D([], [], color=PALETTE["grey"], lw=1.5, label="OSRM road path"),
        mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=PALETTE["orange"], markeredgecolor="white", markersize=6, label="Endpoint cluster"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")


def distance_panel(ax: plt.Axes, edges: pd.DataFrame) -> None:
    weights = edges["order_support"].to_numpy(dtype=float)
    sc = ax.scatter(
        edges["median_air_distance_km"],
        edges["road_distance_km"],
        s=_sizes(edges["order_support"], 10, 120),
        c=edges["time_delta_peak25_min"],
        cmap="RdYlBu",
        vmin=-8,
        vmax=8,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    upper = max(float(edges["median_air_distance_km"].max()), float(edges["road_distance_km"].max())) * 1.05
    ax.plot([0, upper], [0, upper], color=PALETTE["grey_dark"], lw=1.0, ls="--")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Median aerial route distance (km)")
    ax.set_ylabel("OSRM road distance (km)")
    ax.set_title("Road paths are longer than aerial paths", pad=4)
    ax.grid(True, linestyle=":", color=PALETTE["grey_light"], alpha=0.85)

    median_ratio = _weighted_median(edges["road_to_air_distance_ratio"], edges["order_support"])
    ax.text(
        0.04,
        0.93,
        f"Weighted median ratio: {median_ratio:.2f}x",
        transform=ax.transAxes,
        color=PALETTE["navy_dark"],
        fontsize=8.0,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.047, pad=0.02)
    cbar.set_label("Time advantage at 25 km/h\n(road - air, min)")


def duration_panel(ax: plt.Axes, by_bin: pd.DataFrame) -> None:
    data = by_bin.copy()
    data["dist_bin"] = pd.Categorical(data["dist_bin"], categories=DIST_ORDER, ordered=True)
    data = data.sort_values("dist_bin")
    x = np.arange(len(data))

    ax.plot(x, data["weighted_median_air_duration_min"], color=PALETTE["navy"], marker="o", lw=2.1, label="Aerial observed")
    ax.plot(x, data["weighted_median_osrm_duration_min"], color=PALETTE["grey_dark"], marker="s", lw=1.8, label="OSRM road")
    ax.plot(x, data["weighted_median_peak25_duration_min"], color=PALETTE["red"], marker="^", lw=1.8, label="Road at 25 km/h")
    ax.fill_between(
        x,
        data["weighted_median_air_duration_min"],
        data["weighted_median_peak25_duration_min"],
        where=data["weighted_median_peak25_duration_min"] >= data["weighted_median_air_duration_min"],
        color=PALETTE["red"],
        alpha=0.10,
        interpolate=True,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data["dist_bin"])
    ax.set_xlabel("Aerial distance bin (km)")
    ax.set_ylabel("Median duration (min)")
    ax.set_title("Duration comparison by route length", pad=4)
    ax.grid(True, axis="y", linestyle=":", color=PALETTE["grey_light"], alpha=0.85)
    ax.legend(frameon=False, loc="upper left")


def advantage_panel(ax: plt.Axes, by_bin: pd.DataFrame) -> None:
    data = by_bin.copy()
    data["dist_bin"] = pd.Categorical(data["dist_bin"], categories=DIST_ORDER, ordered=True)
    data = data.sort_values("dist_bin")
    x = np.arange(len(data))
    width = 0.36

    ax.bar(
        x - width / 2,
        data["weighted_share_aerial_faster_osrm"] * 100,
        width=width,
        color=PALETTE["grey"],
        alpha=0.72,
        label="vs OSRM duration",
    )
    ax.bar(
        x + width / 2,
        data["weighted_share_aerial_faster_peak25"] * 100,
        width=width,
        color=PALETTE["orange"],
        alpha=0.86,
        label="vs road at 25 km/h",
    )
    ax.axhline(50, color=PALETTE["grey_dark"], ls="--", lw=1.0)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(data["dist_bin"])
    ax.set_xlabel("Aerial distance bin (km)")
    ax.set_ylabel("Aerial faster routes (%)")
    ax.set_title("Where aerial transport has a time advantage", pad=4)
    ax.grid(True, axis="y", linestyle=":", color=PALETTE["grey_light"], alpha=0.85)
    ax.legend(frameon=False, loc="upper left")

    for xpos, val, n_orders in zip(x + width / 2, data["weighted_share_aerial_faster_peak25"] * 100, data["n_orders"]):
        ax.text(xpos, min(val + 3, 101), f"{val:.0f}%", ha="center", va="bottom", fontsize=7.0, color=PALETTE["navy_dark"])
        ax.text(xpos, 3, f"n={int(n_orders):,}", ha="center", va="bottom", fontsize=6.6, color=PALETTE["grey_dark"], rotation=90)


def main() -> None:
    apply_style()
    nodes = pd.read_csv(NODES_CSV)
    edges = pd.read_csv(EDGES_CSV)
    by_bin = pd.read_csv(BIN_CSV)
    geometries = json.loads(GEOMETRY_JSON.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(11.2, 8.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.08, 0.92])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_dur = fig.add_subplot(gs[1, 0])
    ax_adv = fig.add_subplot(gs[1, 1])

    map_panel(ax_map, nodes, edges, geometries)
    distance_panel(ax_dist, edges)
    duration_panel(ax_dur, by_bin)
    advantage_panel(ax_adv, by_bin)

    add_panel_label(ax_map, "a")
    add_panel_label(ax_dist, "b")
    add_panel_label(ax_dur, "c")
    add_panel_label(ax_adv, "d")

    fig.suptitle(
        (
            "Real-road comparison for 200 m endpoint-clustered directed UAV routes "
            f"({summary['unique_directed_route_connections']} routes; {summary['orders_on_nonself_directed_routes']:,} flights)"
        ),
        y=1.01,
        fontsize=11.5,
        fontweight="semibold",
    )
    save_figure(fig, "S17_clustered_air_ground_real_road_200m")


if __name__ == "__main__":
    main()
