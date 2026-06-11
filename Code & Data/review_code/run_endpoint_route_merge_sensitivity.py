from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OD_PAIRS = ROOT / "data" / "processed" / "grid_refined_100m" / "od_pairs.csv"
STATION_TABLE = ROOT.parent / "review_data" / "real_station_anchor_sites_mapped.csv"
OUTPUT_CSV = ROOT.parent / "review_data" / "merged_route_connection_sensitivity.csv"
OUTPUT_JSON = ROOT.parent / "review_data" / "source_json" / "merged_route_connection_sensitivity.json"

EARTH_RADIUS_M = 6_371_008.8
THRESHOLDS_M = [200, 250, 300, 350]
CHOSEN_THRESHOLD_M = 300
OFFICIAL_ROUTE_COUNT_2023 = 156


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


def load_endpoint_cells() -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]]]:
    cell_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    directed_pairs: list[tuple[str, str]] = []
    with OD_PAIRS.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start_cell = row["start_cell"]
            end_cell = row["end_cell"]
            start_lon = float(row["start_lon"])
            start_lat = float(row["start_lat"])
            end_lon = float(row["end_lon"])
            end_lat = float(row["end_lat"])
            cell_points[start_cell].append((start_lat, start_lon))
            cell_points[end_cell].append((end_lat, end_lon))
            directed_pairs.append((start_cell, end_cell))

    centroids = {
        cell_id: (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for cell_id, points in cell_points.items()
    }
    return centroids, directed_pairs


def cluster_endpoint_cells(
    centroids: dict[str, tuple[float, float]], threshold_m: float
) -> dict[str, int]:
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

    return {cell_id: uf.find(index[cell_id]) for cell_id in cell_ids}


def summarize_route_connections(
    cluster_map: dict[str, int], directed_pairs: list[tuple[str, str]]
) -> dict[str, float]:
    directed_connections: set[tuple[int, int]] = set()
    undirected_connections: set[tuple[int, int]] = set()
    self_merged_orders = 0

    for start_cell, end_cell in directed_pairs:
        start_cluster = cluster_map[start_cell]
        end_cluster = cluster_map[end_cell]
        if start_cluster == end_cluster:
            self_merged_orders += 1
            continue
        directed_connections.add((start_cluster, end_cluster))
        undirected_connections.add(tuple(sorted((start_cluster, end_cluster))))

    return {
        "merged_endpoint_clusters": len(set(cluster_map.values())),
        "unique_directed_route_connections": len(directed_connections),
        "unique_undirected_route_connections": len(undirected_connections),
        "self_merged_orders": self_merged_orders,
        "nonself_order_share": (len(directed_pairs) - self_merged_orders) / len(directed_pairs),
    }


def load_station_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_names: set[str] = set()
    with STATION_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["anchor_set"] != "operational_sites":
                continue
            if row["is_test_site"].strip().lower() == "true":
                continue
            site_name = row["site_name"]
            if site_name in seen_names:
                continue
            seen_names.add(site_name)
            rows.append(row)
    return rows


def summarize_station_distances(station_rows: list[dict[str, str]], threshold_m: float) -> dict[str, object]:
    close_pairs: list[dict[str, object]] = []
    nearest_pair: dict[str, object] | None = None

    for idx_left, row_left in enumerate(station_rows):
        for idx_right in range(idx_left + 1, len(station_rows)):
            row_right = station_rows[idx_right]
            distance_m = haversine_m(
                float(row_left["lat"]),
                float(row_left["lon"]),
                float(row_right["lat"]),
                float(row_right["lon"]),
            )
            candidate = {
                "distance_m": round(distance_m, 1),
                "site_a": row_left["site_name"],
                "site_b": row_right["site_name"],
            }
            if nearest_pair is None or distance_m < float(nearest_pair["distance_m"]):
                nearest_pair = candidate
            if distance_m < threshold_m:
                close_pairs.append(candidate)

    close_pairs.sort(key=lambda item: float(item["distance_m"]))
    return {
        "operational_site_count": len(station_rows),
        "pairs_within_threshold": len(close_pairs),
        "threshold_m": int(threshold_m),
        "nearest_pair": nearest_pair,
        "pairs_within_threshold_detail": close_pairs,
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "merge_threshold_m",
                "merged_endpoint_clusters",
                "unique_directed_route_connections",
                "unique_undirected_route_connections",
                "self_merged_orders",
                "nonself_order_share",
                "difference_from_official_156",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(payload: dict[str, object]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    centroids, directed_pairs = load_endpoint_cells()
    threshold_rows: list[dict[str, object]] = []
    threshold_payload: list[dict[str, object]] = []

    for threshold_m in THRESHOLDS_M:
        cluster_map = cluster_endpoint_cells(centroids, threshold_m=threshold_m)
        summary = summarize_route_connections(cluster_map, directed_pairs)
        row = {
            "merge_threshold_m": threshold_m,
            "merged_endpoint_clusters": summary["merged_endpoint_clusters"],
            "unique_directed_route_connections": summary["unique_directed_route_connections"],
            "unique_undirected_route_connections": summary["unique_undirected_route_connections"],
            "self_merged_orders": summary["self_merged_orders"],
            "nonself_order_share": round(float(summary["nonself_order_share"]), 6),
            "difference_from_official_156": int(summary["unique_undirected_route_connections"]) - OFFICIAL_ROUTE_COUNT_2023,
        }
        threshold_rows.append(row)
        threshold_payload.append(row.copy())

    station_rows = load_station_rows()
    station_summary = summarize_station_distances(station_rows, threshold_m=CHOSEN_THRESHOLD_M)

    chosen_summary = next(
        row for row in threshold_payload if int(row["merge_threshold_m"]) == CHOSEN_THRESHOLD_M
    )

    write_csv(threshold_rows)
    write_json(
        {
            "analysis_design": (
                "Merge nearby endpoint cells by centroid distance, then collapse opposite directions "
                "between merged endpoint clusters into one undirected route connection."
            ),
            "input_archive": str(OD_PAIRS.relative_to(ROOT.parent)),
            "raw_unique_endpoint_cells": len(centroids),
            "raw_unique_undirected_route_pairs_100m_cells": len(
                {
                    tuple(sorted((start_cell, end_cell)))
                    for start_cell, end_cell in directed_pairs
                    if start_cell != end_cell
                }
            ),
            "threshold_summaries": threshold_payload,
            "chosen_threshold_m": CHOSEN_THRESHOLD_M,
            "chosen_threshold_summary": chosen_summary,
            "official_route_benchmark": {
                "cumulative_uav_routes_end_2023": OFFICIAL_ROUTE_COUNT_2023,
                "benchmark_description": (
                    "Municipal transport-bureau statistic for Shenzhen's cumulative UAV routes by the end of 2023."
                ),
            },
            "station_distance_check": station_summary,
        },
    )


if __name__ == "__main__":
    main()
