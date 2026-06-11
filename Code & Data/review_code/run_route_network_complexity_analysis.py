from __future__ import annotations

import csv
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OD_PAIRS = ROOT / "data" / "processed" / "grid_refined_100m" / "od_pairs.csv"
SUMMARY_JSON = ROOT.parent / "review_data" / "source_json" / "summary_results.json"

OUTPUT_NODE_CSV = ROOT.parent / "review_data" / "route_network_nodes_300m.csv"
OUTPUT_EDGE_CSV = ROOT.parent / "review_data" / "route_network_edges_300m.csv"
OUTPUT_JSON = ROOT.parent / "review_data" / "source_json" / "route_network_complexity_300m.json"

MERGE_THRESHOLD_M = 300.0
EARTH_RADIUS_M = 6_371_008.8


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

    raw_cluster_map = {cell_id: uf.find(index[cell_id]) for cell_id in cell_ids}
    root_to_cluster: dict[int, int] = {}
    cluster_map: dict[str, int] = {}
    for cell_id, root in raw_cluster_map.items():
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_map[cell_id] = root_to_cluster[root]

    members: dict[int, list[str]] = defaultdict(list)
    for cell_id, cluster_id in cluster_map.items():
        members[cluster_id].append(cell_id)

    cluster_info: dict[int, dict[str, object]] = {}
    for cluster_id, cluster_cells in members.items():
        lat = sum(centroids[cell][0] for cell in cluster_cells) / len(cluster_cells)
        lon = sum(centroids[cell][1] for cell in cluster_cells) / len(cluster_cells)
        cluster_info[cluster_id] = {
            "cluster_id": cluster_id,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "member_cells": cluster_cells,
            "member_count": len(cluster_cells),
        }
    return cluster_map, cluster_info


def build_undirected_weighted_graph(
    cluster_map: dict[str, int], directed_pairs: list[tuple[str, str]]
) -> tuple[dict[int, dict[int, int]], Counter[tuple[int, int]], int]:
    adjacency: dict[int, dict[int, int]] = defaultdict(dict)
    edge_weights: Counter[tuple[int, int]] = Counter()
    self_merged_orders = 0

    for start_cell, end_cell in directed_pairs:
        start_cluster = cluster_map[start_cell]
        end_cluster = cluster_map[end_cell]
        if start_cluster == end_cluster:
            self_merged_orders += 1
            continue
        edge = tuple(sorted((start_cluster, end_cluster)))
        edge_weights[edge] += 1

    for (left, right), weight in edge_weights.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight

    return adjacency, edge_weights, self_merged_orders


def connected_components(adjacency: dict[int, dict[int, int]], nodes: list[int]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for node in nodes:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for nbr in adjacency.get(current, {}):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def local_clustering_coefficients(adjacency: dict[int, dict[int, int]], nodes: list[int]) -> dict[int, float]:
    coeffs: dict[int, float] = {}
    for node in nodes:
        neighbors = list(adjacency.get(node, {}).keys())
        k = len(neighbors)
        if k < 2:
            coeffs[node] = 0.0
            continue
        links = 0
        for i, left in enumerate(neighbors):
            left_neighbors = adjacency.get(left, {})
            for right in neighbors[i + 1 :]:
                if right in left_neighbors:
                    links += 1
        coeffs[node] = (2.0 * links) / (k * (k - 1))
    return coeffs


def brandes_unweighted(
    adjacency: dict[int, dict[int, int]], nodes: list[int]
) -> dict[int, float]:
    cb = {node: 0.0 for node in nodes}
    for source in nodes:
        stack: list[int] = []
        predecessors = {node: [] for node in nodes}
        sigma = {node: 0.0 for node in nodes}
        dist = {node: -1 for node in nodes}
        sigma[source] = 1.0
        dist[source] = 0
        queue = deque([source])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adjacency.get(v, {}):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta = {node: 0.0 for node in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != source:
                cb[w] += delta[w]

    for node in cb:
        cb[node] /= 2.0
    return cb


def brandes_weighted_inverse_flow(
    adjacency: dict[int, dict[int, int]], nodes: list[int]
) -> dict[int, float]:
    cb = {node: 0.0 for node in nodes}
    for source in nodes:
        stack: list[int] = []
        predecessors = {node: [] for node in nodes}
        sigma = {node: 0.0 for node in nodes}
        dist = {node: math.inf for node in nodes}
        sigma[source] = 1.0
        dist[source] = 0.0
        heap: list[tuple[float, int]] = [(0.0, source)]

        while heap:
            dist_v, v = heapq.heappop(heap)
            if dist_v > dist[v]:
                continue
            stack.append(v)
            for w, weight in adjacency.get(v, {}).items():
                cost = 1.0 / float(weight)
                vw_dist = dist[v] + cost
                if vw_dist + 1e-12 < dist[w]:
                    dist[w] = vw_dist
                    heapq.heappush(heap, (vw_dist, w))
                    sigma[w] = sigma[v]
                    predecessors[w] = [v]
                elif abs(vw_dist - dist[w]) <= 1e-12:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta = {node: 0.0 for node in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != source:
                cb[w] += delta[w]

    for node in cb:
        cb[node] /= 2.0
    return cb


def normalize_betweenness(values: dict[int, float], n_nodes: int) -> dict[int, float]:
    if n_nodes <= 2:
        return {node: 0.0 for node in values}
    scale = 2.0 / ((n_nodes - 1) * (n_nodes - 2))
    return {node: value * scale for node, value in values.items()}


def gini(values: list[float]) -> float:
    clean = sorted(float(v) for v in values if float(v) >= 0.0)
    if not clean:
        return 0.0
    total = sum(clean)
    if total == 0.0:
        return 0.0
    weighted = 0.0
    n = len(clean)
    for idx, value in enumerate(clean, start=1):
        weighted += idx * value
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def load_core_skeleton_summary() -> dict[str, float]:
    with SUMMARY_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    core = payload["trajectory_path_persistence"]["core_repeated_route_skeleton"]
    return {
        "skeleton_edges": int(core["skeleton_edges"]),
        "skeleton_nodes": int(core["skeleton_nodes"]),
        "skeleton_components": int(core["skeleton_components"]),
        "mean_node_degree": float(core["mean_node_degree"]),
        "backbone_coverage_share": float(core["backbone_coverage_share"]),
        "skeleton_nodes_in_backbone_share": float(core["skeleton_nodes_in_backbone_share"]),
    }


def write_outputs(
    cluster_info: dict[int, dict[str, object]],
    edge_weights: Counter[tuple[int, int]],
    node_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    OUTPUT_NODE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_NODE_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cluster_id",
            "centroid_lat",
            "centroid_lon",
            "member_count",
            "degree",
            "strength_orders",
            "clustering_coeff",
            "component_id",
            "component_size",
            "betweenness_unweighted",
            "betweenness_unweighted_norm",
            "betweenness_invflow",
            "betweenness_invflow_norm",
            "member_cells",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in node_rows:
            writer.writerow(row)

    with OUTPUT_EDGE_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["source_cluster", "target_cluster", "order_support"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (left, right), weight in sorted(edge_weights.items()):
            writer.writerow(
                {
                    "source_cluster": left,
                    "target_cluster": right,
                    "order_support": weight,
                }
            )

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    centroids, directed_pairs = load_endpoint_cells()
    cluster_map, cluster_info = cluster_endpoint_cells(centroids, MERGE_THRESHOLD_M)
    adjacency, edge_weights, self_merged_orders = build_undirected_weighted_graph(cluster_map, directed_pairs)

    all_nodes = sorted(cluster_info)
    components = connected_components(adjacency, all_nodes)
    component_lookup: dict[int, tuple[int, int]] = {}
    for comp_id, comp_nodes in enumerate(components, start=1):
        comp_size = len(comp_nodes)
        for node in comp_nodes:
            component_lookup[node] = (comp_id, comp_size)

    degrees = {node: len(adjacency.get(node, {})) for node in all_nodes}
    strengths = {node: sum(adjacency.get(node, {}).values()) for node in all_nodes}
    clustering = local_clustering_coefficients(adjacency, all_nodes)
    betweenness_unweighted = brandes_unweighted(adjacency, all_nodes)
    betweenness_invflow = brandes_weighted_inverse_flow(adjacency, all_nodes)
    betweenness_unweighted_norm = normalize_betweenness(betweenness_unweighted, len(all_nodes))
    betweenness_invflow_norm = normalize_betweenness(betweenness_invflow, len(all_nodes))

    node_rows: list[dict[str, object]] = []
    for node in all_nodes:
        info = cluster_info[node]
        comp_id, comp_size = component_lookup[node]
        node_rows.append(
            {
                "cluster_id": node,
                "centroid_lat": round(float(info["centroid_lat"]), 6),
                "centroid_lon": round(float(info["centroid_lon"]), 6),
                "member_count": int(info["member_count"]),
                "degree": degrees[node],
                "strength_orders": strengths[node],
                "clustering_coeff": round(clustering[node], 6),
                "component_id": comp_id,
                "component_size": comp_size,
                "betweenness_unweighted": round(betweenness_unweighted[node], 6),
                "betweenness_unweighted_norm": round(betweenness_unweighted_norm[node], 6),
                "betweenness_invflow": round(betweenness_invflow[node], 6),
                "betweenness_invflow_norm": round(betweenness_invflow_norm[node], 6),
                "member_cells": ";".join(sorted(info["member_cells"])),
            }
        )

    density = 0.0
    if len(all_nodes) > 1:
        density = (2.0 * len(edge_weights)) / (len(all_nodes) * (len(all_nodes) - 1))

    degree_counter = Counter(degrees.values())
    degree_shares = {
        "degree_0_share": sum(1 for value in degrees.values() if value == 0) / len(all_nodes),
        "degree_1_share": sum(1 for value in degrees.values() if value == 1) / len(all_nodes),
        "degree_2_share": sum(1 for value in degrees.values() if value == 2) / len(all_nodes),
        "degree_ge_3_share": sum(1 for value in degrees.values() if value >= 3) / len(all_nodes),
    }

    top_strength_nodes = sorted(node_rows, key=lambda row: (row["strength_orders"], row["degree"]), reverse=True)[:10]
    top_betweenness_nodes = sorted(node_rows, key=lambda row: row["betweenness_invflow_norm"], reverse=True)[:10]

    summary = {
        "analysis_design": (
            "Complex-network analysis of the 300 m endpoint-merged undirected route network, "
            "with edge weights equal to the number of orders linking merged endpoint clusters."
        ),
        "merge_threshold_m": MERGE_THRESHOLD_M,
        "input_archive": str(OD_PAIRS.relative_to(ROOT.parent)),
        "route_network_300m": {
            "nodes": len(all_nodes),
            "edges": len(edge_weights),
            "self_merged_orders_excluded": self_merged_orders,
            "density": density,
            "components": len(components),
            "largest_component_nodes": len(components[0]) if components else 0,
            "largest_component_node_share": (len(components[0]) / len(all_nodes)) if components else 0.0,
            "average_degree": sum(degrees.values()) / len(all_nodes),
            "average_strength_orders": sum(strengths.values()) / len(all_nodes),
            "max_degree": max(degrees.values()) if degrees else 0,
            "max_strength_orders": max(strengths.values()) if strengths else 0,
            "mean_local_clustering": sum(clustering.values()) / len(all_nodes),
            "degree_distribution_counts": {str(key): value for key, value in sorted(degree_counter.items())},
            "degree_distribution_shares": degree_shares,
            "degree_gini": gini(list(degrees.values())),
            "strength_gini": gini(list(strengths.values())),
        },
        "top_nodes_by_strength_orders": top_strength_nodes,
        "top_nodes_by_inverse_flow_betweenness": top_betweenness_nodes,
        "fine_route_skeleton_reference": load_core_skeleton_summary(),
        "interpretive_summary": {
            "best_description": (
                "A sparse, spatially constrained, fragmented route network with a hub-and-spoke / "
                "branched-infrastructure character, rather than a canonical dense small-world or "
                "clear scale-free network."
            ),
            "reasoning": [
                "Low average degree and low density indicate a very sparse network.",
                "Many connected components and a modest giant component imply fragmentation rather than a single integrated system.",
                "Low mean local clustering argues against a strong small-world interpretation.",
                "Degree and strength are skewed, but the network remains spatially embedded and branch-like rather than showing an unambiguous scale-free backbone.",
            ],
        },
    }

    write_outputs(cluster_info, edge_weights, node_rows, summary)


if __name__ == "__main__":
    main()
