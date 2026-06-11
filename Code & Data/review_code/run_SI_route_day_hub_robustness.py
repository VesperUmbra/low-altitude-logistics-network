from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_robustness import RobustnessAnalyzer


ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = ROOT / "data" / "processed" / "full_100m"
RESULTS_ROOT = ROOT / "data" / "results" / "full_100m"
GRID_ROOT = PROCESSED_ROOT / "grid"
GRID_REFINED_ROOT = ROOT / "data" / "processed" / "grid_refined_100m"

STATION_TABLE = ROOT.parent / "review_data" / "real_station_anchor_sites_mapped.csv"

ROUTE_DAY_OUTPUT_CSV = ROOT.parent / "review_data" / "route_day_robustness.csv"
ROUTE_DAY_OUTPUT_JSON = ROOT.parent / "review_data" / "source_json" / "route_day_robustness.json"
HUB_OUTPUT_CSV = ROOT.parent / "review_data" / "hub_robustness.csv"
HUB_OUTPUT_JSON = ROOT.parent / "review_data" / "source_json" / "hub_robustness.json"

EARTH_RADIUS_M = 6_371_008.8
MERGE_THRESHOLD_M = 300.0
TOP_ROUTE_DAYS = 50
ROUTE_DAY_BOOTSTRAPS = 200
TOP_HUBS_UNION = 3


def summarize_collection(collection: dict[str, dict]) -> dict:
    rr_values = [payload["rr"] for payload in collection.values() if np.isfinite(payload["rr"])]
    rho_values = [payload["rho_star"] for payload in collection.values() if np.isfinite(payload["rho_star"])]
    or_values = [payload["or_adj"] for payload in collection.values() if np.isfinite(payload["or_adj"])]
    return {
        "n_runs": int(len(collection)),
        "rr_min": float(np.min(rr_values)),
        "rr_median": float(np.median(rr_values)),
        "rr_max": float(np.max(rr_values)),
        "rho_star_min": float(np.min(rho_values)),
        "rho_star_median": float(np.median(rho_values)),
        "rho_star_max": float(np.max(rho_values)),
        "or_min": float(np.min(or_values)),
        "or_median": float(np.median(or_values)),
        "or_max": float(np.max(or_values)),
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def build_analyzer() -> RobustnessAnalyzer:
    analyzer = RobustnessAnalyzer(
        output_dir=RESULTS_ROOT / "route_day_hub_robustness",
        cell_stats_file=GRID_ROOT / "cell_stats.csv",
        spatiotemporal_file=GRID_ROOT / "spatiotemporal_stats.csv",
        cleaned_data_file=PROCESSED_ROOT / "cleaned_data.csv",
        od_pairs_file=GRID_REFINED_ROOT / "od_pairs.csv",
        fundamental_file=RESULTS_ROOT / "diagram" / "fundamental_data.csv",
        breakpoint_file=RESULTS_ROOT / "diagram" / "breakpoint_results.json",
        grid_info_file=GRID_ROOT / "grid_info.json",
    )
    if not analyzer.load_data():
        raise RuntimeError("Failed to load 100 m processed data for route-day/hub robustness.")
    return analyzer


def load_endpoint_clusters(od_pairs_file: Path, threshold_m: float) -> tuple[dict[str, int], pd.DataFrame]:
    od = pd.read_csv(od_pairs_file)
    cell_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in od.itertuples(index=False):
        cell_points[row.start_cell].append((float(row.start_lat), float(row.start_lon)))
        cell_points[row.end_cell].append((float(row.end_lat), float(row.end_lon)))

    centroids = {
        cell_id: (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for cell_id, points in cell_points.items()
    }

    cell_ids = list(centroids)
    uf = UnionFind(len(cell_ids))
    for idx_left, cell_left in enumerate(cell_ids):
        lat_left, lon_left = centroids[cell_left]
        for idx_right in range(idx_left + 1, len(cell_ids)):
            cell_right = cell_ids[idx_right]
            lat_right, lon_right = centroids[cell_right]
            if haversine_m(lat_left, lon_left, lat_right, lon_right) <= threshold_m:
                uf.union(idx_left, idx_right)

    cluster_map = {cell_id: uf.find(i) for i, cell_id in enumerate(cell_ids)}

    od["date"] = pd.to_datetime(od["start_time"]).dt.strftime("%Y-%m-%d")
    od["start_cluster"] = od["start_cell"].map(cluster_map)
    od["end_cluster"] = od["end_cell"].map(cluster_map)
    od = od.loc[od["start_cluster"] != od["end_cluster"]].copy()
    od["merged_route_id"] = od.apply(
        lambda row: "_".join(map(str, sorted((int(row["start_cluster"]), int(row["end_cluster"]))))),
        axis=1,
    )
    od["route_day_id"] = od["merged_route_id"] + "_" + od["date"].astype(str)
    return cluster_map, od[["order_id", "date", "merged_route_id", "route_day_id"]].copy()


def build_point_level_with_routes(analyzer: RobustnessAnalyzer, route_mapping: pd.DataFrame) -> pd.DataFrame:
    point_level = pd.read_csv(
        analyzer.cleaned_data_file,
        usecols=["datetime", "speed", "longitude", "latitude", "order_id"],
        parse_dates=["datetime"],
    )
    rows, cols = analyzer._map_to_full_grid(point_level["longitude"], point_level["latitude"])
    point_level["cell_id"] = rows.astype(str) + "_" + cols.astype(str)
    point_level["window_start"] = point_level["datetime"].dt.floor(analyzer._window_floor_alias())
    point_level["date"] = point_level["window_start"].dt.strftime("%Y-%m-%d")
    point_level = point_level.merge(route_mapping, on=["order_id", "date"], how="inner", validate="many_to_one")
    point_level["route_day_id"] = point_level["route_day_id"].astype(str)
    return point_level[["cell_id", "window_start", "speed", "route_day_id", "merged_route_id"]].copy()


def build_route_day_contributions(point_level: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    point_level = point_level.copy()
    point_level["speed_sum"] = point_level["speed"]
    contributions = (
        point_level.groupby(["route_day_id", "merged_route_id", "cell_id", "window_start"], observed=True)
        .agg(
            n_points=("speed", "size"),
            speed_sum=("speed_sum", "sum"),
        )
        .reset_index()
    )
    route_day_summary = (
        contributions.groupby(["route_day_id", "merged_route_id"], observed=True)
        .agg(
            point_rows=("n_points", "sum"),
            cell_windows=("cell_id", "size"),
        )
        .reset_index()
        .sort_values(["point_rows", "cell_windows", "route_day_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    by_route_day = {
        route_day: frame[["cell_id", "window_start", "n_points", "speed_sum"]].copy()
        for route_day, frame in contributions.groupby("route_day_id", sort=False)
    }
    return contributions, by_route_day, route_day_summary


def full_aggregated_from_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    return (
        contributions.groupby(["cell_id", "window_start"], observed=True)[["n_points", "speed_sum"]]
        .sum()
        .sort_index()
    )


def leave_top_route_days_out(
    analyzer: RobustnessAnalyzer,
    full_aggregated: pd.DataFrame,
    by_route_day: dict[str, pd.DataFrame],
    route_day_summary: pd.DataFrame,
    *,
    top_n: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    top_route_days = route_day_summary.head(top_n)
    for row in top_route_days.itertuples(index=False):
        route_day = str(row.route_day_id)
        frame = by_route_day[route_day]
        remaining = full_aggregated.copy()
        indexed = frame.set_index(["cell_id", "window_start"]).sort_index()
        remaining.loc[indexed.index, "n_points"] = remaining.loc[indexed.index, "n_points"] - indexed["n_points"]
        remaining.loc[indexed.index, "speed_sum"] = remaining.loc[indexed.index, "speed_sum"] - indexed["speed_sum"]
        remaining = remaining.loc[remaining["n_points"] > 0].copy()
        remaining["mean_speed"] = remaining["speed_sum"] / remaining["n_points"]
        subset = remaining.reset_index()[["cell_id", "window_start", "n_points", "mean_speed"]]
        payload = analyzer._summarize_subset(subset)
        payload["removed_route_day"] = route_day
        payload["merged_route_id"] = str(row.merged_route_id)
        payload["removed_point_rows"] = int(row.point_rows)
        payload["removed_cell_windows"] = int(row.cell_windows)
        results[route_day] = payload
    return results


def route_day_cluster_bootstrap(
    analyzer: RobustnessAnalyzer,
    by_route_day: dict[str, pd.DataFrame],
    route_day_ids: list[str],
    *,
    n_bootstraps: int,
    seed: int = 20260405,
) -> dict:
    rng = np.random.default_rng(seed)
    rr_values: list[float] = []
    or_values: list[float] = []

    for _ in range(n_bootstraps):
        sampled = rng.choice(route_day_ids, size=len(route_day_ids), replace=True)
        multiplicities = Counter(sampled.tolist())

        pieces: list[pd.DataFrame] = []
        for route_day_id, mult in multiplicities.items():
            frame = by_route_day[route_day_id]
            scaled = frame.copy()
            if mult != 1:
                scaled["n_points"] = scaled["n_points"] * mult
                scaled["speed_sum"] = scaled["speed_sum"] * mult
            pieces.append(scaled)

        combined = pd.concat(pieces, ignore_index=True)
        aggregated = (
            combined.groupby(["cell_id", "window_start"], observed=True)[["n_points", "speed_sum"]]
            .sum()
            .reset_index()
        )
        aggregated["mean_speed"] = aggregated["speed_sum"] / aggregated["n_points"]
        summary = analyzer._summarize_subset(
            aggregated[["cell_id", "window_start", "n_points", "mean_speed"]],
            corridor_cells=analyzer.full_corridor_cells,
            reestimate_corridor=False,
            rho_star=float(analyzer.rho_star),
            reestimate_rho=False,
        )
        rr_values.append(float(summary["rr"]))
        or_values.append(float(summary["or_adj"]))

    rr_array = np.asarray(rr_values, dtype=float)
    or_array = np.asarray(or_values, dtype=float)
    observed = analyzer._summarize_subset(
        analyzer.spatiotemporal.copy(),
        corridor_cells=analyzer.full_corridor_cells,
        reestimate_corridor=False,
        rho_star=float(analyzer.rho_star),
        reestimate_rho=False,
    )

    return {
        "n_bootstraps": int(n_bootstraps),
        "n_clusters": int(len(route_day_ids)),
        "cluster_definition": "route-day blocks based on 300 m endpoint-merged undirected route connections and calendar date",
        "rr_point_estimate": float(observed["rr"]),
        "rr_bootstrap_mean": float(np.mean(rr_array)),
        "rr_bootstrap_median": float(np.median(rr_array)),
        "rr_ci_95_lower": float(np.percentile(rr_array, 2.5)),
        "rr_ci_95_upper": float(np.percentile(rr_array, 97.5)),
        "or_point_estimate": float(observed["or_adj"]),
        "or_bootstrap_mean": float(np.mean(or_array)),
        "or_bootstrap_median": float(np.median(or_array)),
        "or_ci_95_lower": float(np.percentile(or_array, 2.5)),
        "or_ci_95_upper": float(np.percentile(or_array, 97.5)),
    }


def chebyshev_buffer(cell_id: str, radius: int = 1) -> set[str]:
    row, col = map(int, cell_id.split("_"))
    return {
        f"{r}_{c}"
        for r in range(row - radius, row + radius + 1)
        for c in range(col - radius, col + radius + 1)
    }


def load_station_hub_buffers() -> dict[str, set[str]]:
    df = pd.read_csv(STATION_TABLE)
    df = df.loc[(df["anchor_set"] == "operational_sites") & (~df["is_test_site"].astype(bool))].copy()
    df = df.drop_duplicates(subset=["site_name"]).reset_index(drop=True)
    return {
        str(row.site_name): chebyshev_buffer(str(row.grid_cell_id), radius=1)
        for row in df.itertuples(index=False)
    }


def leave_one_hub_out(analyzer: RobustnessAnalyzer, hub_buffers: dict[str, set[str]]) -> tuple[dict[str, dict], pd.DataFrame]:
    results: dict[str, dict] = {}
    full_data = analyzer.spatiotemporal.copy()
    full_data["is_congested"] = full_data["n_points"] >= float(analyzer.rho_star)
    total_congested = int(full_data["is_congested"].sum())

    summaries: list[dict[str, object]] = []
    for hub_name, cell_buffer in hub_buffers.items():
        removed_mask = full_data["cell_id"].isin(cell_buffer)
        removed_congested = int(full_data.loc[removed_mask, "is_congested"].sum())
        subset = full_data.loc[~removed_mask, ["cell_id", "window_start", "n_points", "mean_speed"]].copy()
        payload = analyzer._summarize_subset(subset)
        payload["hub_name"] = hub_name
        payload["removed_active_cells"] = int(full_data.loc[removed_mask, "cell_id"].nunique())
        payload["removed_samples"] = int(removed_mask.sum())
        payload["removed_congested_samples"] = removed_congested
        payload["removed_congested_share"] = float(removed_congested / total_congested) if total_congested else float("nan")
        results[hub_name] = payload
        summaries.append(
            {
                "hub_name": hub_name,
                "removed_active_cells": payload["removed_active_cells"],
                "removed_samples": payload["removed_samples"],
                "removed_congested_samples": removed_congested,
                "removed_congested_share": payload["removed_congested_share"],
                "rho_star": payload["rho_star"],
                "speed_drop_percent": payload["speed_drop_percent"],
                "rr": payload["rr"],
                "rr_ci_95_lower": payload["rr_ci_95_lower"],
                "rr_ci_95_upper": payload["rr_ci_95_upper"],
                "or_adj": payload["or_adj"],
                "or_ci_95_lower": payload["or_ci_95_lower"],
                "or_ci_95_upper": payload["or_ci_95_upper"],
            }
        )

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["removed_congested_share", "removed_samples", "hub_name"], ascending=[False, False, True]
    ).reset_index(drop=True)
    return results, summary_frame


def remove_top_hubs_union(
    analyzer: RobustnessAnalyzer,
    hub_ranked_frame: pd.DataFrame,
    hub_buffers: dict[str, set[str]],
    *,
    top_k: int,
) -> dict:
    top_hubs = hub_ranked_frame.head(top_k)["hub_name"].astype(str).tolist()
    union_cells: set[str] = set()
    for hub_name in top_hubs:
        union_cells.update(hub_buffers[hub_name])

    full_data = analyzer.spatiotemporal.copy()
    full_data["is_congested"] = full_data["n_points"] >= float(analyzer.rho_star)
    removed_mask = full_data["cell_id"].isin(union_cells)
    removed_congested = int(full_data.loc[removed_mask, "is_congested"].sum())
    total_congested = int(full_data["is_congested"].sum())
    subset = full_data.loc[~removed_mask, ["cell_id", "window_start", "n_points", "mean_speed"]].copy()
    payload = analyzer._summarize_subset(subset)
    payload["removed_hubs"] = top_hubs
    payload["removed_active_cells"] = int(full_data.loc[removed_mask, "cell_id"].nunique())
    payload["removed_samples"] = int(removed_mask.sum())
    payload["removed_congested_samples"] = removed_congested
    payload["removed_congested_share"] = float(removed_congested / total_congested) if total_congested else float("nan")
    return payload


def main() -> None:
    analyzer = build_analyzer()

    # Route-day robustness
    _, route_mapping = load_endpoint_clusters(analyzer.od_pairs_file, MERGE_THRESHOLD_M)
    point_level = build_point_level_with_routes(analyzer, route_mapping)
    contributions, by_route_day, route_day_summary = build_route_day_contributions(point_level)
    full_aggregated = full_aggregated_from_contributions(contributions)

    leave_route_day = leave_top_route_days_out(
        analyzer,
        full_aggregated,
        by_route_day,
        route_day_summary,
        top_n=TOP_ROUTE_DAYS,
    )
    leave_route_day_summary = summarize_collection(leave_route_day)
    route_day_bootstrap = route_day_cluster_bootstrap(
        analyzer,
        by_route_day,
        route_day_summary["route_day_id"].astype(str).tolist(),
        n_bootstraps=ROUTE_DAY_BOOTSTRAPS,
    )

    route_rows = []
    for route_day, payload in leave_route_day.items():
        route_rows.append(
            {
                "analysis": "leave_top_route_day_out",
                "route_day_id": route_day,
                "merged_route_id": payload["merged_route_id"],
                "removed_point_rows": payload["removed_point_rows"],
                "removed_cell_windows": payload["removed_cell_windows"],
                "samples": payload["samples"],
                "active_cells": payload["active_cells"],
                "rho_star": payload["rho_star"],
                "speed_drop_percent": payload["speed_drop_percent"],
                "congestion_share_percent": payload["congestion_share_percent"],
                "rr": payload["rr"],
                "rr_ci_95_lower": payload["rr_ci_95_lower"],
                "rr_ci_95_upper": payload["rr_ci_95_upper"],
                "or_adj": payload["or_adj"],
                "or_ci_95_lower": payload["or_ci_95_lower"],
                "or_ci_95_upper": payload["or_ci_95_upper"],
            }
        )
    route_frame = pd.DataFrame.from_records(route_rows).sort_values(
        ["removed_point_rows", "route_day_id"], ascending=[False, True]
    ).reset_index(drop=True)
    ROUTE_DAY_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ROUTE_DAY_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    route_frame.to_csv(ROUTE_DAY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    ROUTE_DAY_OUTPUT_JSON.write_text(
        json.dumps(
            {
                "analysis_design": (
                    "Route-day robustness for the retained 100 m archive, using 300 m endpoint-merged "
                    "undirected route connections crossed with calendar date. Report top route-day omissions "
                    f"({TOP_ROUTE_DAYS} largest route-day groups by point rows) together with a route-day cluster bootstrap."
                ),
                "route_merge_threshold_m": MERGE_THRESHOLD_M,
                "n_unique_merged_routes": int(route_mapping["merged_route_id"].nunique()),
                "n_route_day_groups": int(route_day_summary["route_day_id"].nunique()),
                "top_route_day_point_share": float(route_day_summary.head(TOP_ROUTE_DAYS)["point_rows"].sum() / contributions["n_points"].sum()),
                "leave_top_route_day_out": {
                    "summary": leave_route_day_summary,
                    "top_n": TOP_ROUTE_DAYS,
                    "per_group": leave_route_day,
                },
                "route_day_cluster_bootstrap": route_day_bootstrap,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Hub robustness
    hub_buffers = load_station_hub_buffers()
    leave_hub, hub_ranked = leave_one_hub_out(analyzer, hub_buffers)
    top_hub_union = remove_top_hubs_union(
        analyzer,
        hub_ranked,
        hub_buffers,
        top_k=TOP_HUBS_UNION,
    )
    HUB_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    HUB_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    hub_ranked.to_csv(HUB_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    HUB_OUTPUT_JSON.write_text(
        json.dumps(
            {
                "analysis_design": (
                    "Leave-hub-out robustness based on one-cell neighborhoods around mapped operational UAV stations. "
                    "Each station-centered neighborhood is removed in turn, followed by a union removal of the top "
                    f"{TOP_HUBS_UNION} hubs ranked by removed exceedance share."
                ),
                "hub_definition": "one-cell Chebyshev neighborhoods around mapped operational UAV stations",
                "n_hubs": int(len(hub_buffers)),
                "leave_one_hub_out": {
                    "summary": summarize_collection(leave_hub),
                    "per_hub": leave_hub,
                },
                "top_hubs_union": top_hub_union,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Route-day summary:")
    print(json.dumps(leave_route_day_summary, indent=2, ensure_ascii=False))
    print(json.dumps(route_day_bootstrap, indent=2, ensure_ascii=False))
    print(f"Saved route-day CSV: {ROUTE_DAY_OUTPUT_CSV}")
    print(f"Saved route-day JSON: {ROUTE_DAY_OUTPUT_JSON}")
    print("\nHub summary:")
    print(json.dumps(summarize_collection(leave_hub), indent=2, ensure_ascii=False))
    print(json.dumps(top_hub_union, indent=2, ensure_ascii=False))
    print(f"Saved hub CSV: {HUB_OUTPUT_CSV}")
    print(f"Saved hub JSON: {HUB_OUTPUT_JSON}")


if __name__ == "__main__":
    main()
