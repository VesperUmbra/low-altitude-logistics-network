from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from run_route_network_complexity_analysis import (
    MERGE_THRESHOLD_M,
    OD_PAIRS,
    ROOT,
    brandes_weighted_inverse_flow,
    cluster_endpoint_cells,
    connected_components,
    gini,
    load_endpoint_cells,
    local_clustering_coefficients,
    normalize_betweenness,
)


TRAJ_STATS = ROOT / "data" / "processed" / "full_100m" / "trajectory_stats.csv"
OUTPUT_NODE_CSV = ROOT.parent / "review_data" / "route_network_nodes_300m_by_vendor.csv"
OUTPUT_EDGE_CSV = ROOT.parent / "review_data" / "route_network_edges_300m_by_vendor.csv"
OUTPUT_JSON = ROOT.parent / "review_data" / "source_json" / "route_network_complexity_by_vendor_300m.json"


def load_vendor_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with TRAJ_STATS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["order_id"]] = row["vendor"]
    return mapping


def load_directed_pairs_with_vendor(vendor_map: dict[str, str]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with OD_PAIRS.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            order_id = row["order_id"]
            vendor = vendor_map.get(order_id)
            if vendor is None:
                continue
            rows.append((row["start_cell"], row["end_cell"], vendor))
    return rows


def build_vendor_graph(
    vendor: str,
    cluster_map: dict[str, int],
    directed_pairs_with_vendor: list[tuple[str, str, str]],
) -> tuple[list[int], dict[int, dict[int, int]], Counter[tuple[int, int]], int]:
    edge_weights: Counter[tuple[int, int]] = Counter()
    self_merged_orders = 0
    active_nodes: set[int] = set()

    for start_cell, end_cell, row_vendor in directed_pairs_with_vendor:
        if row_vendor != vendor:
            continue
        start_cluster = cluster_map[start_cell]
        end_cluster = cluster_map[end_cell]
        if start_cluster == end_cluster:
            self_merged_orders += 1
            continue
        edge = tuple(sorted((start_cluster, end_cluster)))
        edge_weights[edge] += 1
        active_nodes.add(edge[0])
        active_nodes.add(edge[1])

    adjacency: dict[int, dict[int, int]] = defaultdict(dict)
    for (left, right), weight in edge_weights.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight

    return sorted(active_nodes), adjacency, edge_weights, self_merged_orders


def summarize_vendor_network(
    vendor: str,
    nodes: list[int],
    adjacency: dict[int, dict[int, int]],
    edge_weights: Counter[tuple[int, int]],
    cluster_info: dict[int, dict[str, object]],
    self_merged_orders: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if not nodes:
        return {"vendor": vendor, "route_network_300m": {"nodes": 0, "edges": 0}}, [], []

    components = connected_components(adjacency, nodes)
    component_lookup: dict[int, tuple[int, int]] = {}
    for comp_id, comp_nodes in enumerate(components, start=1):
        comp_size = len(comp_nodes)
        for node in comp_nodes:
            component_lookup[node] = (comp_id, comp_size)

    degrees = {node: len(adjacency.get(node, {})) for node in nodes}
    strengths = {node: sum(adjacency.get(node, {}).values()) for node in nodes}
    clustering = local_clustering_coefficients(adjacency, nodes)
    betweenness = brandes_weighted_inverse_flow(adjacency, nodes)
    betweenness_norm = normalize_betweenness(betweenness, len(nodes))

    node_rows: list[dict[str, object]] = []
    for node in nodes:
        info = cluster_info[node]
        comp_id, comp_size = component_lookup[node]
        node_rows.append(
            {
                "vendor": vendor,
                "cluster_id": node,
                "centroid_lat": round(float(info["centroid_lat"]), 6),
                "centroid_lon": round(float(info["centroid_lon"]), 6),
                "member_count": int(info["member_count"]),
                "degree": degrees[node],
                "strength_orders": strengths[node],
                "clustering_coeff": round(clustering[node], 6),
                "component_id": comp_id,
                "component_size": comp_size,
                "betweenness_invflow": round(betweenness[node], 6),
                "betweenness_invflow_norm": round(betweenness_norm[node], 6),
                "member_cells": ";".join(sorted(info["member_cells"])),
            }
        )

    edge_rows = [
        {
            "vendor": vendor,
            "source_cluster": left,
            "target_cluster": right,
            "order_support": weight,
        }
        for (left, right), weight in sorted(edge_weights.items())
    ]

    density = 0.0
    if len(nodes) > 1:
        density = (2.0 * len(edge_weights)) / (len(nodes) * (len(nodes) - 1))

    degree_counter = Counter(degrees.values())
    summary = {
        "vendor": vendor,
        "route_network_300m": {
            "nodes": len(nodes),
            "edges": len(edge_weights),
            "self_merged_orders_excluded": self_merged_orders,
            "density": density,
            "components": len(components),
            "largest_component_nodes": len(components[0]) if components else 0,
            "largest_component_node_share": (len(components[0]) / len(nodes)) if components else 0.0,
            "average_degree": sum(degrees.values()) / len(nodes),
            "average_strength_orders": sum(strengths.values()) / len(nodes),
            "max_degree": max(degrees.values()),
            "max_strength_orders": max(strengths.values()),
            "mean_local_clustering": sum(clustering.values()) / len(nodes),
            "degree_distribution_counts": {str(key): value for key, value in sorted(degree_counter.items())},
            "degree_distribution_shares": {
                "degree_1_share": sum(1 for value in degrees.values() if value == 1) / len(nodes),
                "degree_2_share": sum(1 for value in degrees.values() if value == 2) / len(nodes),
                "degree_ge_3_share": sum(1 for value in degrees.values() if value >= 3) / len(nodes),
            },
            "degree_gini": gini(list(degrees.values())),
            "strength_gini": gini(list(strengths.values())),
        },
        "top_nodes_by_strength_orders": sorted(
            node_rows, key=lambda row: (row["strength_orders"], row["degree"]), reverse=True
        )[:10],
        "top_nodes_by_inverse_flow_betweenness": sorted(
            node_rows, key=lambda row: row["betweenness_invflow_norm"], reverse=True
        )[:10],
    }
    return summary, node_rows, edge_rows


def main() -> None:
    centroids, _ = load_endpoint_cells()
    cluster_map, cluster_info = cluster_endpoint_cells(centroids, MERGE_THRESHOLD_M)
    vendor_map = load_vendor_map()
    directed_pairs_with_vendor = load_directed_pairs_with_vendor(vendor_map)

    summaries: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    vendor_edge_sets: dict[str, set[tuple[int, int]]] = {}
    vendor_node_sets: dict[str, set[int]] = {}

    for vendor in ["MT", "SF"]:
        nodes, adjacency, edge_weights, self_merged_orders = build_vendor_graph(vendor, cluster_map, directed_pairs_with_vendor)
        summary, vendor_nodes, vendor_edges = summarize_vendor_network(
            vendor, nodes, adjacency, edge_weights, cluster_info, self_merged_orders
        )
        summaries.append(summary)
        node_rows.extend(vendor_nodes)
        edge_rows.extend(vendor_edges)
        vendor_edge_sets[vendor] = set(edge_weights)
        vendor_node_sets[vendor] = set(nodes)

    mt_edges = vendor_edge_sets.get("MT", set())
    sf_edges = vendor_edge_sets.get("SF", set())
    mt_nodes = vendor_node_sets.get("MT", set())
    sf_nodes = vendor_node_sets.get("SF", set())

    comparison = {
        "edge_overlap_jaccard": (len(mt_edges & sf_edges) / len(mt_edges | sf_edges)) if (mt_edges or sf_edges) else 0.0,
        "node_overlap_jaccard": (len(mt_nodes & sf_nodes) / len(mt_nodes | sf_nodes)) if (mt_nodes or sf_nodes) else 0.0,
        "shared_edge_count": len(mt_edges & sf_edges),
        "shared_node_count": len(mt_nodes & sf_nodes),
        "mt_only_edge_count": len(mt_edges - sf_edges),
        "sf_only_edge_count": len(sf_edges - mt_edges),
    }

    OUTPUT_NODE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_NODE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "vendor",
                "cluster_id",
                "centroid_lat",
                "centroid_lon",
                "member_count",
                "degree",
                "strength_orders",
                "clustering_coeff",
                "component_id",
                "component_size",
                "betweenness_invflow",
                "betweenness_invflow_norm",
                "member_cells",
            ],
        )
        writer.writeheader()
        writer.writerows(node_rows)

    with OUTPUT_EDGE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["vendor", "source_cluster", "target_cluster", "order_support"],
        )
        writer.writeheader()
        writer.writerows(edge_rows)

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "analysis_design": (
                    "Vendor-specific complex-network analysis on the common 300 m endpoint-merged route graph, "
                    "using the same spatial node set but vendor-filtered edges and order-count weights."
                ),
                "merge_threshold_m": MERGE_THRESHOLD_M,
                "vendor_summaries": summaries,
                "vendor_overlap": comparison,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
