from __future__ import annotations

import csv
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON = REVIEW_DATA / "source_json"

OD_PAIRS = ROOT / "data" / "processed" / "grid_refined_100m" / "od_pairs.csv"
TRAJ_CSV = REVIEW_DATA / "trajectory_length_distribution.csv"

OUT_NODES_CSV = REVIEW_DATA / "air_ground_clustered_road_nodes_200m.csv"
OUT_EDGES_CSV = REVIEW_DATA / "air_ground_clustered_road_edges_200m.csv"
OUT_CACHE_CSV = REVIEW_DATA / "air_ground_clustered_road_osrm_cache_200m.csv"
OUT_BY_BIN_CSV = REVIEW_DATA / "air_ground_clustered_road_by_distance_bin_200m.csv"
OUT_GEOMETRY_JSON = SOURCE_JSON / "air_ground_clustered_road_geometries_200m.json"
OUT_SUMMARY_JSON = SOURCE_JSON / "air_ground_clustered_road_summary_200m.json"

EARTH_RADIUS_M = 6_371_008.8
MERGE_THRESHOLD_M = 200.0
OFFICIAL_ROUTE_COUNT_2023 = 156
PEAK_ROAD_SPEED_KMH = 25.0
BUS_PEAK_SPEED_KMH = 17.57
DIST_BINS = [0, 1, 3, 5, 10, 20, 80]
DIST_LABELS = ["0-1", "1-3", "3-5", "5-10", "10-20", "20-80"]
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{coords}?{query}"


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


def osrm_route(coords: str, *, overview: bool) -> dict[str, object]:
    query = {"alternatives": "false", "steps": "false", "overview": "full" if overview else "false"}
    if overview:
        query["geometries"] = "geojson"
    url = OSRM_ROUTE_URL.format(coords=coords, query=urllib.parse.urlencode(query))
    with urllib.request.urlopen(url, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("code") != "Ok" or not payload.get("routes"):
        return {"status": payload.get("code", "NoRoute"), "distance_m": np.nan, "duration_s": np.nan}
    route = payload["routes"][0]
    result: dict[str, object] = {
        "status": "Ok",
        "distance_m": float(route["distance"]),
        "duration_s": float(route["duration"]),
    }
    if overview:
        result["geometry"] = route.get("geometry")
    return result


def load_od_pairs() -> pd.DataFrame:
    od = pd.read_csv(OD_PAIRS)
    traj = pd.read_csv(
        TRAJ_CSV,
        usecols=["order_id", "vendor", "duration_minutes", "path_length_km", "od_distance_km"],
    )
    od = od.merge(traj, on="order_id", how="inner")
    return od.loc[(od["duration_minutes"] > 0) & (od["od_distance_km"] > 0)].copy()


def endpoint_cell_centroids(od: pd.DataFrame) -> dict[str, tuple[float, float]]:
    cell_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in od.itertuples(index=False):
        cell_points[str(row.start_cell)].append((float(row.start_lat), float(row.start_lon)))
        cell_points[str(row.end_cell)].append((float(row.end_lat), float(row.end_lon)))
    return {
        cell_id: (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for cell_id, points in cell_points.items()
    }


def cluster_endpoint_cells(
    centroids: dict[str, tuple[float, float]],
    threshold_m: float,
) -> tuple[dict[str, int], dict[int, dict[str, object]]]:
    cell_ids = list(centroids)
    index = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    uf = UnionFind(len(cell_ids))

    for idx_left, cell_left in enumerate(cell_ids):
        lat_left, lon_left = centroids[cell_left]
        for idx_right in range(idx_left + 1, len(cell_ids)):
            cell_right = cell_ids[idx_right]
            lat_right, lon_right = centroids[cell_right]
            if haversine_m(lat_left, lon_left, lat_right, lon_right) <= threshold_m:
                uf.union(idx_left, idx_right)

    raw_map = {cell_id: uf.find(index[cell_id]) for cell_id in cell_ids}
    root_to_cluster: dict[int, int] = {}
    cluster_map: dict[str, int] = {}
    for cell_id, root in raw_map.items():
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_map[cell_id] = root_to_cluster[root]

    members: dict[int, list[str]] = defaultdict(list)
    for cell_id, cluster_id in cluster_map.items():
        members[cluster_id].append(cell_id)

    cluster_info: dict[int, dict[str, object]] = {}
    for cluster_id, cells in members.items():
        lat = sum(centroids[cell][0] for cell in cells) / len(cells)
        lon = sum(centroids[cell][1] for cell in cells) / len(cells)
        cluster_info[cluster_id] = {
            "cluster_id": cluster_id,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "member_count": len(cells),
            "member_cells": sorted(cells),
        }
    return cluster_map, cluster_info


def route_key(source: int, target: int, cluster_info: dict[int, dict[str, object]]) -> str:
    src = cluster_info[source]
    dst = cluster_info[target]
    return (
        f"{float(src['centroid_lon']):.6f},{float(src['centroid_lat']):.6f};"
        f"{float(dst['centroid_lon']):.6f},{float(dst['centroid_lat']):.6f}"
    )


def build_nodes_edges(
    od: pd.DataFrame,
    cluster_map: dict[str, int],
    cluster_info: dict[int, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = od.copy()
    work["source_cluster"] = work["start_cell"].astype(str).map(cluster_map)
    work["target_cluster"] = work["end_cell"].astype(str).map(cluster_map)
    work = work.loc[work["source_cluster"] != work["target_cluster"]].copy()
    work["edge_id"] = work["source_cluster"].astype(str) + "->" + work["target_cluster"].astype(str)

    edge_rows = []
    for (source, target), sub in work.groupby(["source_cluster", "target_cluster"], observed=False):
        source = int(source)
        target = int(target)
        src = cluster_info[source]
        dst = cluster_info[target]
        edge_rows.append(
            {
                "edge_id": f"{source}->{target}",
                "source_cluster": source,
                "target_cluster": target,
                "source_lat": float(src["centroid_lat"]),
                "source_lon": float(src["centroid_lon"]),
                "target_lat": float(dst["centroid_lat"]),
                "target_lon": float(dst["centroid_lon"]),
                "route_key": route_key(source, target, cluster_info),
                "order_support": int(len(sub)),
                "median_air_distance_km": float(sub["path_length_km"].median()),
                "mean_air_distance_km": float(sub["path_length_km"].mean()),
                "median_air_duration_min": float(sub["duration_minutes"].median()),
                "mean_air_duration_min": float(sub["duration_minutes"].mean()),
                "median_od_distance_km": float(sub["od_distance_km"].median()),
                "sf_order_share": float((sub["vendor"].astype(str) == "SF").mean()),
            }
        )
    edges = pd.DataFrame(edge_rows)

    degree_counter: Counter[int] = Counter()
    strength_counter: Counter[int] = Counter()
    for row in edges.itertuples(index=False):
        degree_counter[int(row.source_cluster)] += 1
        strength_counter[int(row.source_cluster)] += int(row.order_support)

    node_rows = []
    for cluster_id, info in sorted(cluster_info.items()):
        node_rows.append(
            {
                "cluster_id": cluster_id,
                "centroid_lat": float(info["centroid_lat"]),
                "centroid_lon": float(info["centroid_lon"]),
                "member_count": int(info["member_count"]),
                "out_degree": int(degree_counter[cluster_id]),
                "out_strength_orders": int(strength_counter[cluster_id]),
                "member_cells": ";".join(info["member_cells"]),
            }
        )
    nodes = pd.DataFrame(node_rows)
    return nodes, edges


def load_cache() -> pd.DataFrame:
    if OUT_CACHE_CSV.exists():
        return pd.read_csv(OUT_CACHE_CSV)
    return pd.DataFrame(columns=["route_key", "status", "distance_m", "duration_s"])


def build_route_cache(edges: pd.DataFrame) -> pd.DataFrame:
    cache = load_cache()
    ok_keys = set(cache.loc[cache["status"].eq("Ok"), "route_key"].astype(str))
    keys = sorted(edges["route_key"].unique())
    missing = [key for key in keys if key not in ok_keys]
    rows = cache.to_dict("records")

    for i, key in enumerate(missing, start=1):
        try:
            result = osrm_route(str(key), overview=False)
        except Exception as exc:
            result = {"status": f"error:{type(exc).__name__}", "distance_m": np.nan, "duration_s": np.nan}
        rows.append({"route_key": key, **result})
        if i % 25 == 0 or i == len(missing):
            pd.DataFrame(rows).drop_duplicates("route_key", keep="last").to_csv(OUT_CACHE_CSV, index=False)
        time.sleep(0.05)

    cache = pd.DataFrame(rows).drop_duplicates("route_key", keep="last")
    cache.to_csv(OUT_CACHE_CSV, index=False)
    return cache


def add_road_metrics(edges: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    routed = edges.merge(cache, on="route_key", how="left")
    routed = routed.loc[routed["status"].eq("Ok")].copy()
    routed["road_distance_km"] = routed["distance_m"] / 1000.0
    routed["road_duration_osrm_min"] = routed["duration_s"] / 60.0
    routed["road_duration_peak25_min"] = 60.0 * routed["road_distance_km"] / PEAK_ROAD_SPEED_KMH
    routed["road_duration_bus1757_min"] = 60.0 * routed["road_distance_km"] / BUS_PEAK_SPEED_KMH
    routed["road_to_air_distance_ratio"] = routed["road_distance_km"] / routed["median_air_distance_km"]
    routed["road_to_od_distance_ratio"] = routed["road_distance_km"] / routed["median_od_distance_km"]
    routed["time_delta_osrm_min"] = routed["road_duration_osrm_min"] - routed["median_air_duration_min"]
    routed["time_delta_peak25_min"] = routed["road_duration_peak25_min"] - routed["median_air_duration_min"]
    routed["time_delta_bus1757_min"] = routed["road_duration_bus1757_min"] - routed["median_air_duration_min"]
    routed["aerial_faster_osrm"] = routed["time_delta_osrm_min"] > 0
    routed["aerial_faster_peak25"] = routed["time_delta_peak25_min"] > 0
    routed["aerial_faster_bus1757"] = routed["time_delta_bus1757_min"] > 0
    routed["dist_bin"] = pd.cut(
        routed["median_air_distance_km"],
        bins=DIST_BINS,
        labels=DIST_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)
    return routed


def write_geometries(edges: pd.DataFrame, top_n: int = 60) -> None:
    top = edges.sort_values("order_support", ascending=False).head(top_n)
    rows = []
    for row in top.itertuples(index=False):
        geometry = None
        try:
            result = osrm_route(str(row.route_key), overview=True)
            geometry = result.get("geometry") if result.get("status") == "Ok" else None
        except Exception:
            geometry = None
        rows.append(
            {
                "edge_id": row.edge_id,
                "source_cluster": int(row.source_cluster),
                "target_cluster": int(row.target_cluster),
                "order_support": int(row.order_support),
                "source_lon": float(row.source_lon),
                "source_lat": float(row.source_lat),
                "target_lon": float(row.target_lon),
                "target_lat": float(row.target_lat),
                "road_distance_km": float(row.road_distance_km),
                "median_air_distance_km": float(row.median_air_distance_km),
                "median_air_duration_min": float(row.median_air_duration_min),
                "road_duration_osrm_min": float(row.road_duration_osrm_min),
                "geometry": geometry,
            }
        )
        time.sleep(0.05)
    OUT_GEOMETRY_JSON.write_text(json.dumps({"routes": rows}, ensure_ascii=False), encoding="utf-8")


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = np.argsort(values.to_numpy(dtype=float))
    sorted_values = values.to_numpy(dtype=float)[order]
    sorted_weights = weights.to_numpy(dtype=float)[order]
    cutoff = sorted_weights.sum() / 2.0
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def summarize_by_bin(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dist_bin, sub in edges.groupby("dist_bin", observed=False):
        if sub.empty:
            continue
        weights = sub["order_support"]
        rows.append(
            {
                "dist_bin": str(dist_bin),
                "n_directed_routes": int(len(sub)),
                "n_orders": int(weights.sum()),
                "weighted_median_air_distance_km": weighted_median(sub["median_air_distance_km"], weights),
                "weighted_median_road_distance_km": weighted_median(sub["road_distance_km"], weights),
                "weighted_median_air_duration_min": weighted_median(sub["median_air_duration_min"], weights),
                "weighted_median_osrm_duration_min": weighted_median(sub["road_duration_osrm_min"], weights),
                "weighted_median_peak25_duration_min": weighted_median(sub["road_duration_peak25_min"], weights),
                "weighted_median_time_delta_osrm_min": weighted_median(sub["time_delta_osrm_min"], weights),
                "weighted_median_time_delta_peak25_min": weighted_median(sub["time_delta_peak25_min"], weights),
                "weighted_share_aerial_faster_osrm": float(np.average(sub["aerial_faster_osrm"], weights=weights)),
                "weighted_share_aerial_faster_peak25": float(np.average(sub["aerial_faster_peak25"], weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REVIEW_DATA.mkdir(parents=True, exist_ok=True)
    SOURCE_JSON.mkdir(parents=True, exist_ok=True)

    od = load_od_pairs()
    centroids = endpoint_cell_centroids(od)
    cluster_map, cluster_info = cluster_endpoint_cells(centroids, MERGE_THRESHOLD_M)
    nodes, edges = build_nodes_edges(od, cluster_map, cluster_info)
    cache = build_route_cache(edges)
    routed_edges = add_road_metrics(edges, cache)

    nodes.to_csv(OUT_NODES_CSV, index=False)
    routed_edges.to_csv(OUT_EDGES_CSV, index=False)
    by_bin = summarize_by_bin(routed_edges)
    by_bin.to_csv(OUT_BY_BIN_CSV, index=False)
    write_geometries(routed_edges)

    total_orders = int(len(od))
    self_orders = int(total_orders - routed_edges["order_support"].sum())
    summary = {
        "analysis_design": (
            "Endpoint cells are merged within 200 m; route count is the number of non-self directed "
            "connections between merged endpoint clusters. Road alternatives are OSRM driving routes "
            "between cluster centroids."
        ),
        "routing_engine": "OSRM public demo server, OSM road network, driving profile.",
        "merge_threshold_m": MERGE_THRESHOLD_M,
        "raw_flight_records_in_trajectory_table": int(pd.read_csv(TRAJ_CSV, usecols=["order_id"])["order_id"].nunique()),
        "od_pair_records_available": total_orders,
        "merged_endpoint_clusters": int(len(nodes)),
        "unique_directed_route_connections": int(len(edges)),
        "unique_undirected_route_connections": int(
            len({tuple(sorted((int(row.source_cluster), int(row.target_cluster)))) for row in edges.itertuples(index=False)})
        ),
        "orders_on_nonself_directed_routes": int(routed_edges["order_support"].sum()),
        "self_merged_orders_excluded": self_orders,
        "official_route_count_2023": OFFICIAL_ROUTE_COUNT_2023,
        "overall_weighted_by_orders": {
            "median_air_distance_km": weighted_median(routed_edges["median_air_distance_km"], routed_edges["order_support"]),
            "median_road_distance_km": weighted_median(routed_edges["road_distance_km"], routed_edges["order_support"]),
            "median_air_duration_min": weighted_median(routed_edges["median_air_duration_min"], routed_edges["order_support"]),
            "median_osrm_road_duration_min": weighted_median(routed_edges["road_duration_osrm_min"], routed_edges["order_support"]),
            "median_peak25_road_duration_min": weighted_median(routed_edges["road_duration_peak25_min"], routed_edges["order_support"]),
            "median_road_to_air_distance_ratio": weighted_median(routed_edges["road_to_air_distance_ratio"], routed_edges["order_support"]),
            "median_time_delta_osrm_min": weighted_median(routed_edges["time_delta_osrm_min"], routed_edges["order_support"]),
            "median_time_delta_peak25_min": weighted_median(routed_edges["time_delta_peak25_min"], routed_edges["order_support"]),
            "share_aerial_faster_osrm": float(np.average(routed_edges["aerial_faster_osrm"], weights=routed_edges["order_support"])),
            "share_aerial_faster_peak25": float(np.average(routed_edges["aerial_faster_peak25"], weights=routed_edges["order_support"])),
            "share_aerial_faster_bus1757": float(np.average(routed_edges["aerial_faster_bus1757"], weights=routed_edges["order_support"])),
        },
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
