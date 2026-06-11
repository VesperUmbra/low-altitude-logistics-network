from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

BRANCH_GRID_DIR = ROOT / "data" / "processed" / "full_100m" / "grid"
BRANCH_OD_CSV = ROOT / "data" / "processed" / "grid_refined_100m" / "od_pairs.csv"
BRANCH_BREAKPOINT_JSON = ROOT / "data" / "results" / "full_100m" / "diagram" / "breakpoint_results.json"
BRANCH_BINNED_CSV = ROOT / "data" / "results" / "full_100m" / "diagram" / "fundamental_data.csv"
BRANCH_SUMMARY_DIR = SOURCE_JSON_DIR / "robustness_100m"

CELL_STATS_CSV = BRANCH_GRID_DIR / "cell_stats.csv"
SPATIOTEMPORAL_CSV = BRANCH_GRID_DIR / "spatiotemporal_stats.csv"
GRID_INFO_JSON = BRANCH_GRID_DIR / "grid_info.json"

RANKED_OUT_CSV = REVIEW_DATA_DIR / "ranked_cell_traffic_table.csv"
HOTSPOT_OUT_CSV = REVIEW_DATA_DIR / "hotspot_exceedance_rank_table.csv"
BINNED_OUT_CSV = REVIEW_DATA_DIR / "binned_speed_density_relations.csv"


def expand_buffer(cell_ids: set[str], radius: int, *, n_rows: int, n_cols: int) -> set[str]:
    expanded: set[str] = set()
    for cell_id in cell_ids:
        row_str, col_str = cell_id.split("_")
        row = int(row_str)
        col = int(col_str)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = row + dr
                cc = col + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    expanded.add(f"{rr}_{cc}")
    return expanded


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1.0) / n)


def copy_summary_jsons() -> None:
    SOURCE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    for path in BRANCH_SUMMARY_DIR.glob("*.json"):
        shutil.copy2(path, SOURCE_JSON_DIR / path.name)
    shutil.copy2(BRANCH_BREAKPOINT_JSON, SOURCE_JSON_DIR / "breakpoint_results.json")


def build_ranked_table(cell_stats: pd.DataFrame) -> pd.DataFrame:
    ranked = cell_stats.sort_values(["total_points", "cell_id"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    total_points = float(ranked["total_points"].sum())
    ranked["traffic_share"] = ranked["total_points"] / total_points
    ranked["cumulative_traffic_share"] = ranked["traffic_share"].cumsum()
    n_backbone = max(1, int(len(ranked) * 0.10))
    ranked["is_backbone_cell"] = ranked["rank"] <= n_backbone
    ranked["backbone_definition"] = "top_10_percent_active_cells_by_total_points"
    out = ranked[
        [
            "rank",
            "cell_id",
            "total_points",
            "traffic_share",
            "cumulative_traffic_share",
            "active_days",
            "active_hours",
            "is_backbone_cell",
            "backbone_definition",
        ]
    ].rename(columns={"cell_id": "grid_cell_id"})
    return out


def build_hotspot_table(
    cell_stats: pd.DataFrame,
    spatiotemporal: pd.DataFrame,
    rho_star: float,
    *,
    n_rows: int,
    n_cols: int,
) -> pd.DataFrame:
    ranked_lookup = (
        cell_stats.sort_values(["total_points", "cell_id"], ascending=[False, True])[["cell_id"]]
        .reset_index(drop=True)
        .reset_index(names="traffic_rank")
    )
    ranked_lookup["traffic_rank"] += 1

    exceed = spatiotemporal.loc[spatiotemporal["n_points"] >= rho_star].copy()
    hotspot = (
        exceed.groupby("cell_id", observed=True)
        .size()
        .reset_index(name="exceedance_count")
        .sort_values(["exceedance_count", "cell_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    hotspot["exceedance_rank"] = np.arange(1, len(hotspot) + 1)
    total_exceedance = float(hotspot["exceedance_count"].sum())
    hotspot["exceedance_share"] = hotspot["exceedance_count"] / total_exceedance
    hotspot["cumulative_exceedance_share"] = hotspot["exceedance_share"].cumsum()

    total_samples = (
        spatiotemporal.groupby("cell_id", observed=True)
        .size()
        .reset_index(name="total_cell_window_samples")
    )

    endpoint_pairs = pd.read_csv(BRANCH_OD_CSV, usecols=["start_cell", "end_cell"])
    endpoint_cells = set(endpoint_pairs["start_cell"].astype(str)) | set(endpoint_pairs["end_cell"].astype(str))
    endpoint_buffer = expand_buffer(endpoint_cells, radius=1, n_rows=n_rows, n_cols=n_cols)

    n_backbone = max(1, int(len(cell_stats) * 0.10))
    backbone_cells = set(
        cell_stats.sort_values(["total_points", "cell_id"], ascending=[False, True]).head(n_backbone)["cell_id"].astype(str)
    )

    hotspot = hotspot.merge(total_samples, on="cell_id", how="left")
    hotspot = hotspot.merge(ranked_lookup, on="cell_id", how="left")
    hotspot["is_backbone_cell"] = hotspot["cell_id"].isin(backbone_cells)
    hotspot["is_endpoint_cell"] = hotspot["cell_id"].isin(endpoint_cells)
    hotspot["is_endpoint_buffer1_cell"] = hotspot["cell_id"].isin(endpoint_buffer)
    hotspot["rho_star_reference"] = float(rho_star)

    return hotspot[
        [
            "exceedance_rank",
            "cell_id",
            "exceedance_count",
            "exceedance_share",
            "cumulative_exceedance_share",
            "total_cell_window_samples",
            "traffic_rank",
            "is_backbone_cell",
            "is_endpoint_cell",
            "is_endpoint_buffer1_cell",
            "rho_star_reference",
        ]
    ].rename(columns={"cell_id": "grid_cell_id"})


def build_binned_table(rho_star: float) -> pd.DataFrame:
    binned = pd.read_csv(BRANCH_BINNED_CSV)
    binned["regime_relative_to_rho_star"] = np.where(
        binned["density_center"] >= rho_star,
        "at_or_above_rho_star",
        "below_rho_star",
    )
    binned["rho_star_reference"] = float(rho_star)
    return binned[
        [
            "density_bin",
            "density_min",
            "density_max",
            "density_center",
            "sample_count",
            "density_mean",
            "speed_mean",
            "speed_std",
            "speed_count",
            "regime_relative_to_rho_star",
            "rho_star_reference",
        ]
    ]


def build_metadata_payload(ranked: pd.DataFrame) -> dict[str, float]:
    top1_idx = max(0, int(round(len(ranked) * 0.01)) - 1)
    top10_idx = max(0, int(round(len(ranked) * 0.10)) - 1)
    return {
        "gini": gini(ranked["total_points"].to_numpy()),
        "top1_share": float(ranked.iloc[top1_idx]["cumulative_traffic_share"]),
        "top10_share": float(ranked.iloc[top10_idx]["cumulative_traffic_share"]),
    }


def main() -> None:
    copy_summary_jsons()

    grid_info = json.loads(GRID_INFO_JSON.read_text(encoding="utf-8"))
    rho_star = float(json.loads(BRANCH_BREAKPOINT_JSON.read_text(encoding="utf-8"))["consensus_rho_star"])

    cell_stats = pd.read_csv(CELL_STATS_CSV)
    spatiotemporal = pd.read_csv(SPATIOTEMPORAL_CSV, usecols=["cell_id", "n_points"])

    ranked = build_ranked_table(cell_stats)
    hotspot = build_hotspot_table(
        cell_stats,
        spatiotemporal,
        rho_star,
        n_rows=int(grid_info["n_rows"]),
        n_cols=int(grid_info["n_cols"]),
    )
    binned = build_binned_table(rho_star)

    ranked.to_csv(RANKED_OUT_CSV, index=False, encoding="utf-8-sig")
    hotspot.to_csv(HOTSPOT_OUT_CSV, index=False, encoding="utf-8-sig")
    binned.to_csv(BINNED_OUT_CSV, index=False, encoding="utf-8-sig")

    metadata = build_metadata_payload(ranked)
    (SOURCE_JSON_DIR / "grid_switch_100m_metadata.json").write_text(
        json.dumps(
            {
                "grid_size_m": 100,
                "generated_from": "for_review/review_code/data/*full_100m*",
                "gini": metadata["gini"],
                "top1_share": metadata["top1_share"],
                "top10_share": metadata["top10_share"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {RANKED_OUT_CSV}")
    print(f"Wrote {HOTSPOT_OUT_CSV}")
    print(f"Wrote {BINNED_OUT_CSV}")
    print("Grid metadata:", metadata)


if __name__ == "__main__":
    main()
