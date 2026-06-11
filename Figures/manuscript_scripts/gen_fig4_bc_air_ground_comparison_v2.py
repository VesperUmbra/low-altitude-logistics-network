# 产出Fig4 panel b,c
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

from paper_plot_style import OUTPUT_DIR, PALETTE, add_panel_label, apply_style, save_panel


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA = WORKSPACE_ROOT / "for_review" /"review_data"
SOURCE_JSON = REVIEW_DATA / "source_json"
AIR_GROUND_PACKAGE_JSON = WORKSPACE_ROOT / "preview_air_ground_comparison_v2_data_package" / "data" / "source_json"

TRAJ_CSV = REVIEW_DATA / "trajectory_length_distribution.csv"

CLUSTER_BIN_CSV = REVIEW_DATA / "air_ground_clustered_road_by_distance_bin_200m.csv"
CLUSTER_EDGE_CSV = REVIEW_DATA / "air_ground_clustered_road_edges_200m.csv"
CLUSTER_SUMMARY_JSON = SOURCE_JSON / "air_ground_clustered_road_summary_200m.json"

GROUND_SPEED_GRID = [float(v) for v in np.arange(10.0, 131.0, 2.0)]
DIST_ORDER = ["0-1", "1-3", "3-5", "5-10", "10-20", "20-80"]
DIST_BINS = [0, 1, 3, 5, 10, 20, 80]
ADVANTAGE_CMAP = LinearSegmentedColormap.from_list(
    "aerial_advantage_white_blue",
    ["#fbfcfb", "#e5edf0", "#c6dbe4", "#8ebed3", "#2f78ad"],
)


def apply_air_ground_style() -> None:
    """Compact overrides for a three-panel, double-column figure."""
    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.9,
        }
    )


def load_json(primary: Path, fallback_name: str) -> dict[str, object]:
    path = primary if primary.exists() else AIR_GROUND_PACKAGE_JSON / fallback_name
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    midpoint = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(quantile * weights.sum(), midpoint, values))


def build_advantage_for_detour(detour_factor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the OD speed-reference panel for the route-observed detour factor."""
    traj = pd.read_csv(TRAJ_CSV)
    traj = traj.loc[(traj["duration_minutes"] > 0) & (traj["od_distance_km"] > 0)].copy()
    traj["dist_bin"] = pd.cut(
        traj["path_length_km"],
        bins=DIST_BINS,
        labels=DIST_ORDER,
        right=False,
        include_lowest=True,
    )
    traj = traj.loc[traj["dist_bin"].notna()].copy()
    traj["dist_bin"] = traj["dist_bin"].astype(str)
    traj["road_distance_km"] = traj["od_distance_km"] * detour_factor
    traj["break_even_speed_kmh"] = traj["road_distance_km"] / (traj["duration_minutes"] / 60.0)

    advantage_rows: list[dict[str, object]] = []
    breakeven_rows: list[dict[str, object]] = []
    for dist_bin, sub in traj.groupby("dist_bin", observed=False):
        speeds = sub["break_even_speed_kmh"].to_numpy(dtype=float)
        breakeven_rows.append(
            {
                "dist_bin": str(dist_bin),
                "detour_factor": detour_factor,
                "n": int(len(sub)),
                "median_break_even_speed_kmh": float(np.median(speeds)),
                "q25_break_even_speed_kmh": float(np.quantile(speeds, 0.25)),
                "q75_break_even_speed_kmh": float(np.quantile(speeds, 0.75)),
            }
        )
        for ground_speed in GROUND_SPEED_GRID:
            ground_duration = 60.0 * sub["road_distance_km"] / ground_speed
            delta = ground_duration - sub["duration_minutes"]
            advantage_rows.append(
                {
                    "dist_bin": str(dist_bin),
                    "detour_factor": detour_factor,
                    "ground_speed_kmh": ground_speed,
                    "n": int(len(sub)),
                    "share_aerial_faster": float((delta > 0).mean()),
                }
            )

    return pd.DataFrame(advantage_rows), pd.DataFrame(breakeven_rows)


def advantage_phase_panel(
    ax: plt.Axes,
    advantage: pd.DataFrame,
    breakeven: pd.DataFrame,
    detour_factor: float,
    legend_kwargs: dict[str, object] | None = None,
) -> plt.Axes:
    sub = advantage.loc[
        (np.isclose(advantage["detour_factor"], detour_factor))
        & (advantage["ground_speed_kmh"].isin(GROUND_SPEED_GRID))
        & (advantage["dist_bin"].isin(DIST_ORDER))
    ].copy()
    sub["dist_bin"] = pd.Categorical(sub["dist_bin"], categories=DIST_ORDER, ordered=True)
    sub = sub.sort_values(["dist_bin", "ground_speed_kmh"])

    grid = (
        sub.pivot(index="dist_bin", columns="ground_speed_kmh", values="share_aerial_faster")
        .reindex(index=DIST_ORDER, columns=GROUND_SPEED_GRID)
        .to_numpy(dtype=float)
    )

    im = ax.imshow(
        grid,
        cmap=ADVANTAGE_CMAP,
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=[GROUND_SPEED_GRID[0] - 1, GROUND_SPEED_GRID[-1] + 1, len(DIST_ORDER) - 0.5, -0.5],
    )

    x_grid = np.asarray(GROUND_SPEED_GRID, dtype=float)
    y_grid = np.arange(len(DIST_ORDER), dtype=float)
    ax.contour(x_grid, y_grid, grid, levels=[0.5], colors=[PALETTE["navy_dark"]], linewidths=1.35)
    ax.contour(x_grid, y_grid, grid, levels=[0.8], colors=[PALETTE["teal"]], linewidths=1.1, linestyles="--")

    base = breakeven.loc[np.isclose(breakeven["detour_factor"], detour_factor)].copy()
    base["dist_bin"] = pd.Categorical(base["dist_bin"], categories=DIST_ORDER, ordered=True)
    base = base.sort_values("dist_bin")
    ax.plot(
        base["median_break_even_speed_kmh"],
        np.arange(len(DIST_ORDER)),
        color=PALETTE["orange"],
        lw=1.8,
        marker="o",
        ms=4.2,
        label="Break-even speed",
    )

    ax.axvline(17.57, color=PALETTE["teal"], ls=":", lw=1.15)
    ax.axvline(25.0, color=PALETTE["red"], ls=":", lw=1.15)

    ax.set_xticks([10, 20, 30, 40, 60, 80, 100, 120])
    ax.set_yticks(np.arange(len(DIST_ORDER)))
    ax.set_yticklabels(DIST_ORDER)
    ax.set_xlabel("Surface-speed benchmark (km/h)")
    ax.set_ylabel("Drone-route distance bin (km)")

    handles = [
        plt.Line2D([0], [0], color=PALETTE["orange"], lw=1.8, marker="o", ms=3.8, label="Break-even"),
        plt.Line2D([0], [0], color=PALETTE["navy_dark"], lw=1.35, label="50% faster"),
        plt.Line2D([0], [0], color=PALETTE["teal"], lw=1.1, ls="--", label="80% faster"),
        plt.Line2D([0], [0], color=PALETTE["teal"], lw=1.15, ls=":", label="17.57 km/h"),
        plt.Line2D([0], [0], color=PALETTE["red"], lw=1.15, ls=":", label="25 km/h"),
    ]
    legend_options = {
        "loc": "upper center",
        "bbox_to_anchor": (0.50, 1.16),
        "ncol": 3,
        "handlelength": 1.6,
        "columnspacing": 0.9,
        "borderaxespad": 0,
    }
    if legend_kwargs is not None:
        legend_options.update(legend_kwargs)
    ax.legend(handles=handles, frameon=False, **legend_options)

    cbar = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label("Share with shorter drone-flight duration")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(["0", "25%", "50%", "75%", "100%"])
    return cbar.ax


def _scale_sizes(values: pd.Series, lo: float = 14.0, hi: float = 105.0) -> np.ndarray:
    arr = np.sqrt(values.to_numpy(dtype=float))
    if np.allclose(arr.max(), arr.min()):
        return np.full_like(arr, (lo + hi) / 2.0)
    return lo + (arr - arr.min()) / (arr.max() - arr.min()) * (hi - lo)


def route_time_advantage_panel(
    ax: plt.Axes,
    edges: pd.DataFrame,
    by_bin: pd.DataFrame,
    clustered_summary: dict[str, object],
) -> None:
    routes = edges.loc[edges["status"] == "Ok"].copy()
    routes["dist_bin"] = pd.Categorical(routes["dist_bin"], categories=DIST_ORDER, ordered=True)
    routes = routes.sort_values(["dist_bin", "median_air_distance_km"]).reset_index(drop=True)

    binned = by_bin.copy()
    binned["dist_bin"] = pd.Categorical(binned["dist_bin"], categories=DIST_ORDER, ordered=True)
    binned = binned.sort_values("dist_bin").reset_index(drop=True)

    rng = np.random.default_rng(13)
    routes["y_bin"] = routes["dist_bin"].cat.codes.astype(float)
    routes["y_jitter"] = routes["y_bin"] + rng.uniform(-0.22, 0.22, size=len(routes))

    y_peak = routes["time_delta_peak25_min"].to_numpy(dtype=float)
    y_osrm = routes["time_delta_osrm_min"].to_numpy(dtype=float)
    xmin = min(float(y_peak.min()), float(y_osrm.min()), -5.0) * 1.08
    xmax = max(float(y_peak.max()), float(y_osrm.max()), 5.0) * 1.05
    sizes = _scale_sizes(routes["order_support"], 9, 66)

    ax.axvspan(0, xmax, color=PALETTE["blue_light"], alpha=0.12, zorder=0)
    ax.axvspan(xmin, 0, color=PALETTE["grey_light"], alpha=0.25, zorder=0)
    ax.axvline(0, color=PALETTE["grey_dark"], lw=0.9)

    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=60)
    ax.scatter(
        routes["time_delta_peak25_min"],
        routes["y_jitter"],
        s=sizes,
        c=routes["time_delta_peak25_min"],
        cmap="RdBu_r",
        norm=norm,
        alpha=0.74,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )

    y_bins = np.arange(len(DIST_ORDER))
    ax.plot(
        binned["weighted_median_time_delta_peak25_min"],
        y_bins,
        color=PALETTE["orange"],
        marker="o",
        ms=4.2,
        lw=1.85,
        label="Median, 25 km/h road",
        zorder=4,
    )
    ax.plot(
        binned["weighted_median_time_delta_osrm_min"],
        y_bins,
        color=PALETTE["grey_dark"],
        marker="s",
        ms=3.8,
        lw=1.45,
        label="Median, OSRM",
        zorder=4,
    )

    labels = [f"{row.dist_bin} ({int(row.n_directed_routes)})" for row in binned.itertuples(index=False)]
    ax.set_yticks(y_bins)
    ax.set_yticklabels(labels)
    ax.set_ylabel("Drone-route distance bin (routes)")
    ax.set_xlabel("Road time minus drone-flight time (min)")
    ax.set_ylim(len(DIST_ORDER) - 0.55, -0.55)
    ax.set_xscale("symlog", linthresh=5, linscale=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([-5, 0, 5, 10, 20, 50, 100, 175])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(x)}"))
    ax.grid(True, axis="x", linestyle=":", color=PALETTE["grey_light"], alpha=0.85)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=1, handlelength=1.8)



def main() -> None:
    apply_style()
    apply_air_ground_style()
    clustered_by_bin = pd.read_csv(CLUSTER_BIN_CSV)
    clustered_edges = pd.read_csv(CLUSTER_EDGE_CSV)
    clustered_summary = load_json(CLUSTER_SUMMARY_JSON, "air_ground_clustered_road_summary_200m.json")
    detour_factor = clustered_summary["overall_weighted_by_orders"]["median_road_to_air_distance_ratio"]
    advantage, breakeven = build_advantage_for_detour(detour_factor)

    fig4b, ax4b = plt.subplots(figsize=(4.7, 3.25), constrained_layout=True)
    route_time_advantage_panel(ax4b, clustered_edges, clustered_by_bin, clustered_summary)
    add_panel_label(ax4b, "b")
    save_panel(fig4b, ax4b, "manuscript_fig4b_route_level_time_advantage")
    plt.close(fig4b)

    fig4c = plt.figure(figsize=(7.6867, 3.0180), constrained_layout=False)
    ax4c = fig4c.add_axes([0.077, 0.115, 0.788, 0.815])
    advantage_phase_panel(
        ax4c,
        advantage,
        breakeven,
        detour_factor,
        legend_kwargs={
            "loc": "upper right",
            "bbox_to_anchor": (0.985, 0.985),
            "ncol": 1,
            "handlelength": 1.8,
            "columnspacing": 0.8,
        },
    )
    add_panel_label(ax4c, "c")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas_bbox = Bbox.from_bounds(0, 0, *fig4c.get_size_inches())
    fig4c.savefig(OUTPUT_DIR / "manuscript_fig4c_aerial_advantage_by_distance.png", bbox_inches=canvas_bbox, pad_inches=0)
    fig4c.savefig(OUTPUT_DIR / "manuscript_fig4c_aerial_advantage_by_distance.svg", bbox_inches=canvas_bbox, pad_inches=0)
    plt.close(fig4c)
    print("Saved manuscript Figure 4 road-benchmark panels")


if __name__ == "__main__":
    main()
