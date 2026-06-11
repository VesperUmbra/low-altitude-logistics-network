from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from run_robustness import RobustnessAnalyzer


ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = ROOT / "data" / "processed" / "full_100m"
RESULTS_ROOT = ROOT / "data" / "results" / "full_100m"
GRID_ROOT = PROCESSED_ROOT / "grid"
GRID_REFINED_ROOT = ROOT / "data" / "processed" / "grid_refined_100m"

OUTPUT_CSV = ROOT.parent / "review_data" / "vendor_day_robustness.csv"
OUTPUT_JSON = ROOT.parent / "review_data" / "source_json" / "vendor_day_robustness.json"


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


def build_analyzer() -> RobustnessAnalyzer:
    analyzer = RobustnessAnalyzer(
        output_dir=RESULTS_ROOT / "vendor_day_robustness",
        cell_stats_file=GRID_ROOT / "cell_stats.csv",
        spatiotemporal_file=GRID_ROOT / "spatiotemporal_stats.csv",
        cleaned_data_file=PROCESSED_ROOT / "cleaned_data.csv",
        od_pairs_file=GRID_REFINED_ROOT / "od_pairs.csv",
        fundamental_file=RESULTS_ROOT / "diagram" / "fundamental_data.csv",
        breakpoint_file=RESULTS_ROOT / "diagram" / "breakpoint_results.json",
        grid_info_file=GRID_ROOT / "grid_info.json",
    )
    if not analyzer.load_data():
        raise RuntimeError("Failed to load 100 m processed data for vendor-day robustness.")
    return analyzer


def build_vendor_day_contributions(analyzer: RobustnessAnalyzer) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    point_level = analyzer._load_point_level().copy()
    point_level["vendor_day"] = (
        point_level["vendor"].astype(str)
        + "_"
        + point_level["window_start"].dt.strftime("%Y-%m-%d")
    )

    contributions = (
        point_level.groupby(["vendor_day", "cell_id", "window_start"], observed=True)
        .agg(
            n_points=("speed", "size"),
            speed_sum=("speed", "sum"),
        )
        .reset_index()
    )

    vendor_days = sorted(contributions["vendor_day"].astype(str).unique().tolist())
    by_vendor_day = {
        vendor_day: frame[["cell_id", "window_start", "n_points", "speed_sum"]].copy()
        for vendor_day, frame in contributions.groupby("vendor_day", sort=False)
    }
    return contributions, by_vendor_day, vendor_days


def full_aggregated_from_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    full = (
        contributions.groupby(["cell_id", "window_start"], observed=True)[["n_points", "speed_sum"]]
        .sum()
        .sort_index()
    )
    return full


def leave_one_vendor_day_out(
    analyzer: RobustnessAnalyzer,
    full_aggregated: pd.DataFrame,
    by_vendor_day: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for vendor_day, frame in by_vendor_day.items():
        remaining = full_aggregated.copy()
        indexed = frame.set_index(["cell_id", "window_start"]).sort_index()
        remaining.loc[indexed.index, "n_points"] = remaining.loc[indexed.index, "n_points"] - indexed["n_points"]
        remaining.loc[indexed.index, "speed_sum"] = remaining.loc[indexed.index, "speed_sum"] - indexed["speed_sum"]
        remaining = remaining.loc[remaining["n_points"] > 0].copy()
        remaining["mean_speed"] = remaining["speed_sum"] / remaining["n_points"]
        subset = remaining.reset_index()[["cell_id", "window_start", "n_points", "mean_speed"]]
        payload = analyzer._summarize_subset(subset)
        payload["removed_vendor_day"] = vendor_day
        payload["removed_point_rows"] = int(frame["n_points"].sum())
        payload["removed_cell_windows"] = int(len(frame))
        results[vendor_day] = payload
    return results


def vendor_day_cluster_bootstrap(
    analyzer: RobustnessAnalyzer,
    by_vendor_day: dict[str, pd.DataFrame],
    vendor_days: list[str],
    *,
    n_bootstraps: int = 300,
    seed: int = 20260405,
) -> dict:
    rng = np.random.default_rng(seed)
    rr_values: list[float] = []
    or_values: list[float] = []

    for _ in range(n_bootstraps):
        sampled = rng.choice(vendor_days, size=len(vendor_days), replace=True)
        multiplicities = Counter(sampled.tolist())

        pieces: list[pd.DataFrame] = []
        for vendor_day, mult in multiplicities.items():
            frame = by_vendor_day[vendor_day]
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
        "n_clusters": int(len(vendor_days)),
        "cluster_definition": "vendor-day blocks based on operator label and calendar date",
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


def main() -> None:
    analyzer = build_analyzer()
    contributions, by_vendor_day, vendor_days = build_vendor_day_contributions(analyzer)
    full_aggregated = full_aggregated_from_contributions(contributions)

    leave_out = leave_one_vendor_day_out(analyzer, full_aggregated, by_vendor_day)
    leave_out_summary = summarize_collection(leave_out)
    cluster_bootstrap = vendor_day_cluster_bootstrap(analyzer, by_vendor_day, vendor_days)

    rows = []
    for vendor_day, payload in leave_out.items():
        rows.append(
            {
                "analysis": "leave_one_vendor_day_out",
                "group": vendor_day,
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
                "removed_point_rows": payload["removed_point_rows"],
                "removed_cell_windows": payload["removed_cell_windows"],
            }
        )

    frame = pd.DataFrame.from_records(rows).sort_values("group").reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    payload = {
        "analysis_design": (
            "Vendor-day robustness for the retained 100 m archive, combining leave-one-vendor-day-out "
            "subset re-estimation with a vendor-day cluster bootstrap that keeps the main rho* and backbone fixed."
        ),
        "vendor_day_groups": vendor_days,
        "n_vendor_day_groups": int(len(vendor_days)),
        "leave_one_vendor_day_out": {
            "summary": leave_out_summary,
            "per_group": leave_out,
        },
        "vendor_day_cluster_bootstrap": cluster_bootstrap,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Leave-one-vendor-day-out summary:")
    print(json.dumps(leave_out_summary, indent=2, ensure_ascii=False))
    print("\nVendor-day cluster bootstrap:")
    print(json.dumps(cluster_bootstrap, indent=2, ensure_ascii=False))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
