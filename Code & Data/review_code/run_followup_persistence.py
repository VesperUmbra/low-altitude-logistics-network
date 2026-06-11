from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"
PROCESSED_ROOT = ROOT / "data" / "processed"

FOLLOWUP_POINTS_CSV = REVIEW_DATA_DIR / "followup_trajectory_points_2025_2026.csv"
FOLLOWUP_SUMMARY_CSV = REVIEW_DATA_DIR / "followup_flight_summary_2025_2026.csv"
FIGURE_CELLS_CSV = REVIEW_DATA_DIR / "figure_cell_metrics.csv"
STATION_CSV = REVIEW_DATA_DIR / "real_station_anchor_sites_mapped.csv"
GRID_INFO_JSON = ROOT / "data" / "processed" / "grid_refined_100m" / "grid_info.json"
MAIN_GRID_INFO_JSON = ROOT / "data" / "processed" / "full_100m" / "grid" / "grid_info.json"
RAW_2023_CLEANED = PROCESSED_ROOT / "full_100m" / "cleaned_data.csv"

OUTPUT_FLIGHT_METRICS_CSV = REVIEW_DATA_DIR / "followup_persistence_flight_metrics.csv"
OUTPUT_PATHS_CSV = REVIEW_DATA_DIR / "followup_compressed_paths_2025_2026.csv"
OUTPUT_DATE_SUMMARY_CSV = REVIEW_DATA_DIR / "followup_date_route_summary.csv"
OUTPUT_SKELETON_EDGES_CSV = REVIEW_DATA_DIR / "followup_reference_route_skeleton_edges_support100.csv"
OUTPUT_JSON = SOURCE_JSON_DIR / "followup_persistence_summary.json"

SKELETON_SUPPORT = 100


@dataclass
class GridConfig:
    grid_size: float
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    n_cols: int
    n_rows: int
    meters_per_degree_lat: float
    meters_per_degree_lon: float

    @classmethod
    def from_json(cls, path: Path) -> "GridConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        lat_center = (float(data["min_lat"]) + float(data["max_lat"])) / 2.0
        return cls(
            grid_size=float(data.get("grid_size", data.get("grid_size_m"))),
            min_lon=float(data["min_lon"]),
            max_lon=float(data["max_lon"]),
            min_lat=float(data["min_lat"]),
            max_lat=float(data["max_lat"]),
            n_cols=int(data["n_cols"]),
            n_rows=int(data["n_rows"]),
            meters_per_degree_lat=111000.0,
            meters_per_degree_lon=111000.0 * np.cos(np.radians(lat_center)),
        )

    def map_to_cell_ids(self, lon: pd.Series, lat: pd.Series) -> pd.Series:
        rows = np.floor(((lat - self.min_lat) * self.meters_per_degree_lat) / self.grid_size).astype(int)
        cols = np.floor(((lon - self.min_lon) * self.meters_per_degree_lon) / self.grid_size).astype(int)
        rows = np.clip(rows, 0, self.n_rows - 1)
        cols = np.clip(cols, 0, self.n_cols - 1)
        return rows.astype(str) + "_" + cols.astype(str)

    def cell_center(self, cell_id: str) -> tuple[float, float]:
        row_str, col_str = cell_id.split("_")
        row = int(row_str)
        col = int(col_str)
        lat = self.min_lat + ((row + 0.5) * self.grid_size) / self.meters_per_degree_lat
        lon = self.min_lon + ((col + 0.5) * self.grid_size) / self.meters_per_degree_lon
        return lon, lat


def expand_buffer(cell_ids: set[str], radius: int = 1) -> set[str]:
    expanded: set[str] = set()
    for cell_id in cell_ids:
        row_str, col_str = cell_id.split("_")
        row = int(row_str)
        col = int(col_str)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                expanded.add(f"{row + dr}_{col + dc}")
    return expanded


def load_reference_sets() -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    figure_cells = pd.read_csv(
        FIGURE_CELLS_CSV,
        usecols=[
            "cell_id",
            "is_backbone_cell",
            "is_hotspot_cell",
            "is_endpoint_cell",
            "endpoint_buffer1_cell",
        ],
    )
    backbone_cells = set(figure_cells.loc[figure_cells["is_backbone_cell"], "cell_id"].astype(str))
    hotspot_cells = set(figure_cells.loc[figure_cells["is_hotspot_cell"], "cell_id"].astype(str))
    endpoint_cells = set(figure_cells.loc[figure_cells["is_endpoint_cell"], "cell_id"].astype(str))
    endpoint_buffer1 = set(figure_cells.loc[figure_cells["endpoint_buffer1_cell"], "cell_id"].astype(str))

    station = pd.read_csv(STATION_CSV, usecols=["anchor_set", "grid_cell_id"])
    if "operational_sites" in set(station["anchor_set"]):
        station = station[station["anchor_set"] == "operational_sites"].copy()
    station_exact = set(station["grid_cell_id"].astype(str))
    station_buffer1 = expand_buffer(station_exact, radius=1)
    return backbone_cells, hotspot_cells, endpoint_cells, endpoint_buffer1, station_buffer1


def load_or_build_skeleton_edges(grid: GridConfig) -> dict[tuple[str, str], int]:
    if OUTPUT_SKELETON_EDGES_CSV.exists():
        df = pd.read_csv(OUTPUT_SKELETON_EDGES_CSV)
        return {
            (str(row.source_cell), str(row.target_cell)): int(row.support)
            for row in df.itertuples(index=False)
        }

    raw = pd.read_csv(
        RAW_2023_CLEANED,
        usecols=["order_id", "datetime", "longitude", "latitude"],
        parse_dates=["datetime"],
        dtype={"order_id": "string"},
    )
    raw["cell_id"] = grid.map_to_cell_ids(raw["longitude"], raw["latitude"])
    raw = raw.sort_values(["order_id", "datetime"]).reset_index(drop=True)

    edge_orders: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for order_id, group in raw.groupby("order_id", sort=False):
        cells = group["cell_id"].to_numpy(dtype=object)
        if len(cells) < 2:
            continue
        keep = np.empty(len(cells), dtype=bool)
        keep[0] = True
        keep[1:] = cells[1:] != cells[:-1]
        sequence = cells[keep].tolist()
        if len(sequence) < 2:
            continue
        unique_edges = {tuple(sorted((a, b))) for a, b in zip(sequence[:-1], sequence[1:]) if a != b}
        for edge in unique_edges:
            edge_orders[edge].add(str(order_id))

    skeleton = {edge: len(order_ids) for edge, order_ids in edge_orders.items() if len(order_ids) >= SKELETON_SUPPORT}
    rows = [
        {"source_cell": edge[0], "target_cell": edge[1], "support": support}
        for edge, support in sorted(skeleton.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_SKELETON_EDGES_CSV, index=False)
    return skeleton


def finalize_flight(
    flight_info: dict[str, str] | None,
    points: list[dict[str, object]],
    backbone_cells: set[str],
    hotspot_cells: set[str],
    endpoint_cells: set[str],
    endpoint_buffer1: set[str],
    station_buffer1: set[str],
    skeleton_edges: set[tuple[str, str]],
    grid: GridConfig,
    main_grid: GridConfig,
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    if not flight_info or not points:
        return {}, [], set()

    n_points = len(points)
    point_cells = [str(p["cell_id"]) for p in points]
    point_speeds = np.asarray([float(p["ground_speed"]) for p in points], dtype=float)
    main_point_cells = main_grid.map_to_cell_ids(
        pd.Series([float(p["longitude"]) for p in points]),
        pd.Series([float(p["latitude"]) for p in points]),
    ).astype(str).tolist()

    sequence: list[str] = []
    prev_cell = None
    for cell_id in point_cells:
        if cell_id != prev_cell:
            sequence.append(cell_id)
            prev_cell = cell_id

    main_sequence: list[str] = []
    prev_main_cell = None
    for cell_id in main_point_cells:
        if cell_id != prev_main_cell:
            main_sequence.append(cell_id)
            prev_main_cell = cell_id

    edges = [tuple(sorted((a, b))) for a, b in zip(sequence[:-1], sequence[1:]) if a != b]
    edge_hits = sum(edge in skeleton_edges for edge in edges)
    unique_cells = set(sequence)
    main_unique_cells = set(main_sequence)
    own_endpoint_buffer1 = expand_buffer({sequence[0], sequence[-1]}, radius=1) if sequence else set()
    slow_mask = np.isfinite(point_speeds) & (point_speeds < 2.0)
    own_endpoint_mask = np.array([cell in own_endpoint_buffer1 for cell in point_cells], dtype=bool)

    path_rows: list[dict[str, object]] = []
    for seq_idx, cell_id in enumerate(sequence, start=1):
        lon, lat = grid.cell_center(cell_id)
        path_rows.append(
            {
                "flight_id": flight_info["flight_id"],
                "source_folder": flight_info["source_folder"],
                "route_code": flight_info["route_code"],
                "flight_date_local": flight_info["flight_date_local"],
                "seq_index": seq_idx,
                "cell_id": cell_id,
                "lon_center": lon,
                "lat_center": lat,
            }
        )

    metric = {
        "flight_id": flight_info["flight_id"],
        "source_folder": flight_info["source_folder"],
        "route_code": flight_info["route_code"],
        "file_name": flight_info["file_name"],
        "flight_date_local": flight_info["flight_date_local"],
        "n_points": n_points,
        "n_unique_sequence_cells": len(sequence),
        "n_unique_cells": len(unique_cells),
        "n_edges": len(edges),
        "edge_hits_on_2023_skeleton": edge_hits,
        "edge_share_on_2023_skeleton": edge_hits / len(edges) if edges else np.nan,
        "point_share_in_2023_backbone": float(np.mean([cell in backbone_cells for cell in main_point_cells])),
        "unique_cell_share_in_2023_backbone": len(main_unique_cells & backbone_cells) / len(main_unique_cells) if main_unique_cells else np.nan,
        "unique_cell_share_in_2023_hotspots": len(main_unique_cells & hotspot_cells) / len(main_unique_cells) if main_unique_cells else np.nan,
        "start_cell": sequence[0] if sequence else "",
        "end_cell": sequence[-1] if sequence else "",
        "start_in_2023_endpoint": int(main_sequence[0] in endpoint_cells) if main_sequence else 0,
        "end_in_2023_endpoint": int(main_sequence[-1] in endpoint_cells) if main_sequence else 0,
        "start_in_2023_endpoint_buffer1": int(main_sequence[0] in endpoint_buffer1) if main_sequence else 0,
        "end_in_2023_endpoint_buffer1": int(main_sequence[-1] in endpoint_buffer1) if main_sequence else 0,
        "start_in_station_buffer1": int(main_sequence[0] in station_buffer1) if main_sequence else 0,
        "end_in_station_buffer1": int(main_sequence[-1] in station_buffer1) if main_sequence else 0,
        "both_endpoints_in_2023_endpoint_buffer1": int(main_sequence[0] in endpoint_buffer1 and main_sequence[-1] in endpoint_buffer1) if main_sequence else 0,
        "both_endpoints_in_station_buffer1": int(main_sequence[0] in station_buffer1 and main_sequence[-1] in station_buffer1) if main_sequence else 0,
        "slow_points_total": int(slow_mask.sum()),
        "slow_points_in_own_endpoint_buffer1": int(np.sum(slow_mask & own_endpoint_mask)),
        "slow_point_share_in_own_endpoint_buffer1": float(np.sum(slow_mask & own_endpoint_mask) / slow_mask.sum()) if slow_mask.sum() else np.nan,
        "point_share_in_own_endpoint_buffer1": float(np.mean(own_endpoint_mask)),
        "slow_localization_lift_vs_point_share": float((np.sum(slow_mask & own_endpoint_mask) / slow_mask.sum()) / np.mean(own_endpoint_mask))
        if slow_mask.sum() and np.mean(own_endpoint_mask) > 0
        else np.nan,
    }
    return metric, path_rows, main_unique_cells


def run() -> None:
    SOURCE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    grid = GridConfig.from_json(GRID_INFO_JSON)
    main_grid = GridConfig.from_json(MAIN_GRID_INFO_JSON)
    backbone_cells, hotspot_cells, endpoint_cells, endpoint_buffer1, station_buffer1 = load_reference_sets()
    skeleton_supports = load_or_build_skeleton_edges(grid)
    skeleton_edges = set(skeleton_supports)

    flight_metrics: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    all_main_followup_cells: set[str] = set()

    current_flight_id: str | None = None
    current_info: dict[str, str] | None = None
    current_points: list[dict[str, object]] = []

    with FOLLOWUP_POINTS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_within_main_grid"] not in {"1", "True", "true"}:
                continue
            flight_id = row["flight_id"]
            if current_flight_id is None:
                current_flight_id = flight_id
                current_info = {
                    "flight_id": flight_id,
                    "source_folder": row["source_folder"],
                    "route_code": row["route_code"],
                    "file_name": row["file_name"],
                    "flight_date_local": row["flight_date_local"],
                }
            if flight_id != current_flight_id:
                metric, flight_path_rows, main_unique_cells = finalize_flight(
                    current_info,
                    current_points,
                    backbone_cells,
                    hotspot_cells,
                    endpoint_cells,
                    endpoint_buffer1,
                    station_buffer1,
                    skeleton_edges,
                    grid,
                    main_grid,
                )
                if metric:
                    flight_metrics.append(metric)
                    path_rows.extend(flight_path_rows)
                    all_main_followup_cells.update(main_unique_cells)
                current_flight_id = flight_id
                current_info = {
                    "flight_id": flight_id,
                    "source_folder": row["source_folder"],
                    "route_code": row["route_code"],
                    "file_name": row["file_name"],
                    "flight_date_local": row["flight_date_local"],
                }
                current_points = []

            current_points.append(
                {
                    "cell_id": row["cell_id"],
                    "longitude": row["longitude"],
                    "latitude": row["latitude"],
                    "ground_speed": float(row["ground_speed"]) if row["ground_speed"] else np.nan,
                }
            )

    metric, flight_path_rows, main_unique_cells = finalize_flight(
        current_info,
        current_points,
        backbone_cells,
        hotspot_cells,
        endpoint_cells,
        endpoint_buffer1,
        station_buffer1,
        skeleton_edges,
        grid,
        main_grid,
    )
    if metric:
        flight_metrics.append(metric)
        path_rows.extend(flight_path_rows)
        all_main_followup_cells.update(main_unique_cells)

    metrics_df = pd.DataFrame(flight_metrics).sort_values(["flight_date_local", "source_folder", "route_code", "file_name"])
    paths_df = pd.DataFrame(path_rows)
    metrics_df.to_csv(OUTPUT_FLIGHT_METRICS_CSV, index=False)
    paths_df.to_csv(OUTPUT_PATHS_CSV, index=False)

    flight_summary = pd.read_csv(FOLLOWUP_SUMMARY_CSV, encoding="gb18030")
    date_route_summary = (
        flight_summary.groupby(["flight_date_local", "source_folder", "route_code"], dropna=False)
        .agg(
            flights=("flight_id", "count"),
            first_start=("flight_start_local", "min"),
            last_end=("flight_end_local", "max"),
        )
        .reset_index()
        .sort_values(["flight_date_local", "source_folder", "route_code"])
    )
    date_route_summary.to_csv(OUTPUT_DATE_SUMMARY_CSV, index=False)

    endpoints_total = int(2 * len(metrics_df))
    endpoint_buffer_matches = int(metrics_df["start_in_2023_endpoint_buffer1"].sum() + metrics_df["end_in_2023_endpoint_buffer1"].sum())
    station_buffer_matches = int(metrics_df["start_in_station_buffer1"].sum() + metrics_df["end_in_station_buffer1"].sum())
    slow_points_total = int(metrics_df["slow_points_total"].sum())
    slow_points_endpoint = int(metrics_df["slow_points_in_own_endpoint_buffer1"].sum())
    total_points = int(metrics_df["n_points"].sum())
    total_endpoint_buffer_points = int(np.sum(metrics_df["point_share_in_own_endpoint_buffer1"] * metrics_df["n_points"]))
    weighted_edge_hits = int(metrics_df["edge_hits_on_2023_skeleton"].sum())
    weighted_edges = int(metrics_df["n_edges"].sum())

    by_year = {}
    metrics_df["year"] = metrics_df["flight_date_local"].astype(str).str.slice(0, 4)
    for year, group in metrics_df.groupby("year"):
        by_year[year] = {
            "flights": int(len(group)),
            "dates": sorted(group["flight_date_local"].unique().tolist()),
            "weighted_edge_share_on_2023_skeleton": float(group["edge_hits_on_2023_skeleton"].sum() / group["n_edges"].sum()) if group["n_edges"].sum() else np.nan,
            "median_edge_share_on_2023_skeleton": float(group["edge_share_on_2023_skeleton"].median()),
            "share_flights_edge_coverage_ge_80pct": float(np.mean(group["edge_share_on_2023_skeleton"] >= 0.8)),
        }

    summary = {
        "design": "Limited 2025-2026 follow-up trajectory archive mapped to the manuscript's 2023 100 m grid and compared with the previously identified backbone, endpoint buffers, mapped stations, and support-100 repeated-route skeleton.",
        "followup_archive": {
            "flights": int(len(metrics_df)),
            "dates": sorted(metrics_df["flight_date_local"].unique().tolist()),
            "date_count": int(metrics_df["flight_date_local"].nunique()),
            "points_in_grid": total_points,
            "named_route_codes": sorted([r for r in metrics_df["route_code"].dropna().astype(str).unique().tolist() if r != "unassigned"]),
        },
        "route_persistence": {
            "reference_skeleton_support": SKELETON_SUPPORT,
            "reference_skeleton_edges": int(len(skeleton_edges)),
            "weighted_edge_share_on_2023_skeleton": float(weighted_edge_hits / weighted_edges) if weighted_edges else np.nan,
            "median_flight_edge_share_on_2023_skeleton": float(metrics_df["edge_share_on_2023_skeleton"].median()),
            "p25_flight_edge_share_on_2023_skeleton": float(metrics_df["edge_share_on_2023_skeleton"].quantile(0.25)),
            "share_flights_edge_coverage_ge_80pct": float(np.mean(metrics_df["edge_share_on_2023_skeleton"] >= 0.8)),
            "point_share_in_2023_backbone": float(np.sum(metrics_df["point_share_in_2023_backbone"] * metrics_df["n_points"]) / total_points),
            "unique_cell_share_in_2023_backbone": float(len(all_main_followup_cells & backbone_cells) / len(all_main_followup_cells)) if all_main_followup_cells else np.nan,
        },
        "interface_persistence": {
            "endpoint_buffer1_matches": endpoint_buffer_matches,
            "endpoint_buffer1_match_share": float(endpoint_buffer_matches / endpoints_total) if endpoints_total else np.nan,
            "share_flights_both_endpoints_in_2023_endpoint_buffer1": float(metrics_df["both_endpoints_in_2023_endpoint_buffer1"].mean()),
            "station_buffer1_matches": station_buffer_matches,
            "station_buffer1_match_share": float(station_buffer_matches / endpoints_total) if endpoints_total else np.nan,
            "share_flights_both_endpoints_in_station_buffer1": float(metrics_df["both_endpoints_in_station_buffer1"].mean()),
        },
        "localized_slowing_persistence": {
            "speed_threshold_mps": 2.0,
            "slow_points_total": slow_points_total,
            "slow_points_in_own_endpoint_buffer1": slow_points_endpoint,
            "slow_point_share_in_own_endpoint_buffer1": float(slow_points_endpoint / slow_points_total) if slow_points_total else np.nan,
            "all_point_share_in_own_endpoint_buffer1": float(total_endpoint_buffer_points / total_points) if total_points else np.nan,
            "slowing_localization_lift": float((slow_points_endpoint / slow_points_total) / (total_endpoint_buffer_points / total_points))
            if slow_points_total and total_points and total_endpoint_buffer_points
            else np.nan,
        },
        "by_year": by_year,
    }

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved flight metrics to {OUTPUT_FLIGHT_METRICS_CSV}")
    print(f"Saved compressed paths to {OUTPUT_PATHS_CSV}")
    print(f"Saved date-route summary to {OUTPUT_DATE_SUMMARY_CSV}")
    print(f"Saved summary JSON to {OUTPUT_JSON}")


if __name__ == "__main__":
    run()
