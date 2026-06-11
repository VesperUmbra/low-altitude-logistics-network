from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_station_anchor import (
    BREAKPOINT_JSON,
    GRID_SIZE_M,
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON,
    MIN_LAT,
    MIN_LON,
    aggregate_from_raw_csv,
    expand_buffer,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

RANKED_CSV = REVIEW_DATA_DIR / "ranked_cell_traffic_table.csv"
HOTSPOT_CSV = REVIEW_DATA_DIR / "hotspot_exceedance_rank_table.csv"
STATION_CSV = REVIEW_DATA_DIR / "real_station_anchor_sites_mapped.csv"
CANONICAL_OD_CSV = ROOT / "data" / "processed" / "grid_refined_100m" / "od_pairs.csv"
CANONICAL_GRID_INFO = ROOT / "data" / "processed" / "full_100m" / "grid" / "grid_info.json"
SUMMARY_JSON = SOURCE_JSON_DIR / "summary_results.json"
CELL_OUTPUT_CSV = REVIEW_DATA_DIR / "figure_cell_metrics.csv"
ANCHOR_OUTPUT_CSV = REVIEW_DATA_DIR / "figure_anchor_summary.csv"
THRESHOLD_OUTPUT_CSV = REVIEW_DATA_DIR / "figure_threshold_subset_summary.csv"


def cell_centroids(cell_id: str) -> tuple[float, float]:
    row_str, col_str = cell_id.split("_")
    row = int(row_str)
    col = int(col_str)
    lat = MIN_LAT + ((row + 0.5) * GRID_SIZE_M / METERS_PER_DEG_LAT)
    lon = MIN_LON + ((col + 0.5) * GRID_SIZE_M / METERS_PER_DEG_LON)
    return lon, lat


def load_canonical_endpoint_cells() -> set[str]:
    """Use the OD-derived recurrent endpoint definition used by the manuscript macros."""
    grid_info = json.loads(CANONICAL_GRID_INFO.read_text(encoding="utf-8"))
    od_pairs = pd.read_csv(CANONICAL_OD_CSV, usecols=["start_lon", "start_lat", "end_lon", "end_lat"])

    def map_cells(lon: pd.Series, lat: pd.Series) -> pd.Series:
        rows = (
            ((lat - float(grid_info["min_lat"])) * float(grid_info["meters_per_degree_lat"]))
            / float(grid_info["grid_size_m"])
        ).astype(int)
        cols = (
            ((lon - float(grid_info["min_lon"])) * float(grid_info["meters_per_degree_lon"]))
            / float(grid_info["grid_size_m"])
        ).astype(int)
        rows = rows.clip(lower=0, upper=int(grid_info["n_rows"]) - 1)
        cols = cols.clip(lower=0, upper=int(grid_info["n_cols"]) - 1)
        return rows.astype(str) + "_" + cols.astype(str)

    start_cells = set(map_cells(od_pairs["start_lon"], od_pairs["start_lat"]))
    end_cells = set(map_cells(od_pairs["end_lon"], od_pairs["end_lat"]))
    return start_cells | end_cells


def summarize_anchor(
    *,
    label: str,
    anchor_cells: set[str],
    active_cells: set[str],
    hotspot_cells: set[str],
    endpoint_cells: set[str],
    total_samples_by_cell: dict[str, int],
    exceedance_samples_by_cell: dict[str, int],
    total_exceedance_samples: int,
) -> dict[str, object]:
    active_covered = len(active_cells & anchor_cells)
    hotspot_covered = len(hotspot_cells & anchor_cells)
    endpoint_covered = len(endpoint_cells & anchor_cells)
    exceedance_covered = sum(exceedance_samples_by_cell.get(cell_id, 0) for cell_id in anchor_cells)

    return {
        "anchor_label": label,
        "anchor_cells": int(len(anchor_cells)),
        "active_cells_covered": int(active_covered),
        "active_cells_covered_share": float(active_covered / len(active_cells)) if active_cells else 0.0,
        "hotspot_cells_covered": int(hotspot_covered),
        "hotspot_cells_covered_share": float(hotspot_covered / len(hotspot_cells)) if hotspot_cells else 0.0,
        "endpoint_cells_covered": int(endpoint_covered),
        "endpoint_cells_covered_share": float(endpoint_covered / len(endpoint_cells)) if endpoint_cells else 0.0,
        "exceedance_samples_covered": int(exceedance_covered),
        "exceedance_samples_covered_share": float(exceedance_covered / total_exceedance_samples)
        if total_exceedance_samples
        else 0.0,
        "sample_count_covered": int(sum(total_samples_by_cell.get(cell_id, 0) for cell_id in anchor_cells)),
    }


def main() -> None:
    rho_star = float(json.loads(BREAKPOINT_JSON.read_text(encoding="utf-8"))["consensus_rho_star"])
    aggregated = aggregate_from_raw_csv(rho_star=rho_star)
    endpoint_cells = load_canonical_endpoint_cells()

    ranked = pd.read_csv(RANKED_CSV).rename(columns={"grid_cell_id": "cell_id"})
    hotspot = pd.read_csv(HOTSPOT_CSV).rename(columns={"grid_cell_id": "cell_id"})
    station = pd.read_csv(STATION_CSV)
    station = station[station["anchor_set"] == "operational_sites"].copy()

    hotspot_small = hotspot[
        [
            "cell_id",
            "exceedance_count",
            "exceedance_share",
            "cumulative_exceedance_share",
        ]
    ].copy()

    active = ranked.merge(hotspot_small, on="cell_id", how="left")
    active["exceedance_count"] = active["exceedance_count"].fillna(0).astype(int)
    active["exceedance_share"] = active["exceedance_share"].fillna(0.0)
    active["cumulative_exceedance_share"] = active["cumulative_exceedance_share"].fillna(0.0)
    active["total_samples"] = active["cell_id"].map(aggregated.total_samples_by_cell).fillna(0).astype(int)
    active_cells = set(active["cell_id"])
    active_count = len(active_cells)
    station_exact = set(station["grid_cell_id"].astype(str))
    station_buffer1 = expand_buffer(station_exact, radius=1)
    station_buffer2 = expand_buffer(station_exact, radius=2)
    endpoint_buffer1 = expand_buffer(endpoint_cells, radius=1)
    active["is_hotspot_cell"] = active["exceedance_count"] > 0
    active["is_endpoint_cell"] = active["cell_id"].isin(endpoint_cells)
    active["station_exact_cell"] = active["cell_id"].isin(station_exact)
    active["station_buffer1_cell"] = active["cell_id"].isin(station_buffer1)
    active["station_buffer2_cell"] = active["cell_id"].isin(station_buffer2)
    active["endpoint_buffer1_cell"] = active["cell_id"].isin(endpoint_buffer1)
    centroids = active["cell_id"].map(cell_centroids)
    active["lon"] = centroids.map(lambda x: x[0])
    active["lat"] = centroids.map(lambda x: x[1])
    active["traffic_rank_pct"] = active["rank"] / active_count
    active["log10_total_points"] = active["total_points"].clip(lower=1).map(lambda x: np.log10(x))

    missing_endpoint_cells = sorted(endpoint_cells - active_cells)
    if missing_endpoint_cells:
        missing_rows = []
        for idx, cell_id in enumerate(missing_endpoint_cells, start=1):
            lon, lat = cell_centroids(cell_id)
            missing_rows.append(
                {
                    "rank": active_count + idx,
                    "cell_id": cell_id,
                    "total_points": 0,
                    "traffic_share": 0.0,
                    "cumulative_traffic_share": 1.0,
                    "active_days": 0,
                    "active_hours": 0,
                    "is_backbone_cell": False,
                    "backbone_definition": "not_active_endpoint_cell",
                    "exceedance_count": 0,
                    "exceedance_share": 0.0,
                    "cumulative_exceedance_share": 0.0,
                    "total_samples": 0,
                    "is_hotspot_cell": False,
                    "is_endpoint_cell": True,
                    "station_exact_cell": cell_id in station_exact,
                    "station_buffer1_cell": cell_id in station_buffer1,
                    "station_buffer2_cell": cell_id in station_buffer2,
                    "endpoint_buffer1_cell": True,
                    "lon": lon,
                    "lat": lat,
                    "traffic_rank_pct": np.nan,
                    "log10_total_points": 0.0,
                }
            )
        active = pd.concat([active, pd.DataFrame(missing_rows)], ignore_index=True)

    active.to_csv(CELL_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    hotspot_cells = set(active.loc[active["is_hotspot_cell"], "cell_id"])
    backbone_cells = set(active.loc[active["is_backbone_cell"] == True, "cell_id"])

    anchor_rows = [
        summarize_anchor(
            label="Backbone (top 10%)",
            anchor_cells=backbone_cells,
            active_cells=active_cells,
            hotspot_cells=hotspot_cells,
            endpoint_cells=endpoint_cells,
            total_samples_by_cell=aggregated.total_samples_by_cell,
            exceedance_samples_by_cell=aggregated.exceedance_samples_by_cell,
            total_exceedance_samples=aggregated.total_exceedance_samples,
        ),
        summarize_anchor(
            label="Endpoint exact",
            anchor_cells=endpoint_cells,
            active_cells=active_cells,
            hotspot_cells=hotspot_cells,
            endpoint_cells=endpoint_cells,
            total_samples_by_cell=aggregated.total_samples_by_cell,
            exceedance_samples_by_cell=aggregated.exceedance_samples_by_cell,
            total_exceedance_samples=aggregated.total_exceedance_samples,
        ),
        summarize_anchor(
            label="Endpoint +1 cell",
            anchor_cells=endpoint_buffer1,
            active_cells=active_cells,
            hotspot_cells=hotspot_cells,
            endpoint_cells=endpoint_cells,
            total_samples_by_cell=aggregated.total_samples_by_cell,
            exceedance_samples_by_cell=aggregated.exceedance_samples_by_cell,
            total_exceedance_samples=aggregated.total_exceedance_samples,
        ),
        summarize_anchor(
            label="Station exact",
            anchor_cells=station_exact,
            active_cells=active_cells,
            hotspot_cells=hotspot_cells,
            endpoint_cells=endpoint_cells,
            total_samples_by_cell=aggregated.total_samples_by_cell,
            exceedance_samples_by_cell=aggregated.exceedance_samples_by_cell,
            total_exceedance_samples=aggregated.total_exceedance_samples,
        ),
        summarize_anchor(
            label="Station +1 cell",
            anchor_cells=station_buffer1,
            active_cells=active_cells,
            hotspot_cells=hotspot_cells,
            endpoint_cells=endpoint_cells,
            total_samples_by_cell=aggregated.total_samples_by_cell,
            exceedance_samples_by_cell=aggregated.exceedance_samples_by_cell,
            total_exceedance_samples=aggregated.total_exceedance_samples,
        ),
        summarize_anchor(
            label="Station +2 cells",
            anchor_cells=station_buffer2,
            active_cells=active_cells,
            hotspot_cells=hotspot_cells,
            endpoint_cells=endpoint_cells,
            total_samples_by_cell=aggregated.total_samples_by_cell,
            exceedance_samples_by_cell=aggregated.exceedance_samples_by_cell,
            total_exceedance_samples=aggregated.total_exceedance_samples,
        ),
    ]
    anchor_df = pd.DataFrame(anchor_rows)
    endpoint_exact = anchor_df["anchor_label"] == "Endpoint exact"
    anchor_df.loc[endpoint_exact, "active_cells_covered"] = len(endpoint_cells)
    anchor_df.loc[endpoint_exact, "active_cells_covered_share"] = len(endpoint_cells) / len(active_cells)
    anchor_df.to_csv(ANCHOR_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    subset_rows = [
        {
            "subset_label": "Full sample",
            "rho_star": summary["interface_masking"]["full_sample"]["rho_star"],
            "speed_drop_percent": summary["interface_masking"]["full_sample"]["speed_drop_percent"],
            "rr": summary["interface_masking"]["full_sample"]["rr"],
            "rr_ci_95_lower": summary["interface_masking"]["full_sample"]["rr_ci_95_lower"],
            "rr_ci_95_upper": summary["interface_masking"]["full_sample"]["rr_ci_95_upper"],
            "samples": summary["interface_masking"]["full_sample"]["samples"],
        },
        {
            "subset_label": "Exclude interface-like windows",
            "rho_star": summary["interface_masking"]["excluding_interface_windows"]["rho_star"],
            "speed_drop_percent": summary["interface_masking"]["excluding_interface_windows"]["speed_drop_percent"],
            "rr": summary["interface_masking"]["excluding_interface_windows"]["rr"],
            "rr_ci_95_lower": summary["interface_masking"]["excluding_interface_windows"]["rr_ci_95_lower"],
            "rr_ci_95_upper": summary["interface_masking"]["excluding_interface_windows"]["rr_ci_95_upper"],
            "samples": summary["interface_masking"]["excluding_interface_windows"]["samples"],
        },
        {
            "subset_label": "Exclude endpoint +1 buffer",
            "rho_star": summary["endpoint_buffer_exclusion"]["excluding_endpoint_cells_and_1cell_buffer"]["rho_star"],
            "speed_drop_percent": summary["endpoint_buffer_exclusion"]["excluding_endpoint_cells_and_1cell_buffer"][
                "speed_drop_percent"
            ],
            "rr": summary["endpoint_buffer_exclusion"]["excluding_endpoint_cells_and_1cell_buffer"]["rr"],
            "rr_ci_95_lower": summary["endpoint_buffer_exclusion"]["excluding_endpoint_cells_and_1cell_buffer"][
                "rr_ci_95_lower"
            ],
            "rr_ci_95_upper": summary["endpoint_buffer_exclusion"]["excluding_endpoint_cells_and_1cell_buffer"][
                "rr_ci_95_upper"
            ],
            "samples": summary["endpoint_buffer_exclusion"]["excluding_endpoint_cells_and_1cell_buffer"]["samples"],
        },
        {
            "subset_label": "Away-from-endpoint, altitude >=180 m",
            "rho_star": summary["phase_altitude_decomposition"]["subsets"]["away_from_endpoint_and_cruise_like"][
                "rho_star"
            ],
            "speed_drop_percent": summary["phase_altitude_decomposition"]["subsets"]["away_from_endpoint_and_cruise_like"][
                "speed_drop_percent"
            ],
            "rr": summary["phase_altitude_decomposition"]["subsets"]["away_from_endpoint_and_cruise_like"]["rr"],
            "rr_ci_95_lower": summary["phase_altitude_decomposition"]["subsets"]["away_from_endpoint_and_cruise_like"][
                "rr_ci_95_lower"
            ],
            "rr_ci_95_upper": summary["phase_altitude_decomposition"]["subsets"]["away_from_endpoint_and_cruise_like"][
                "rr_ci_95_upper"
            ],
            "samples": summary["phase_altitude_decomposition"]["subsets"]["away_from_endpoint_and_cruise_like"][
                "samples"
            ],
        },
    ]
    pd.DataFrame(subset_rows).to_csv(THRESHOLD_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Wrote {CELL_OUTPUT_CSV}")
    print(f"Wrote {ANCHOR_OUTPUT_CSV}")
    print(f"Wrote {THRESHOLD_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
