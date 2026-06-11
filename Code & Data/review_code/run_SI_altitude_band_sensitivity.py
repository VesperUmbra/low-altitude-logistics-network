from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_robustness import RobustnessAnalyzer


ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = ROOT / "data" / "processed" / "full_100m"
RESULTS_ROOT = ROOT / "data" / "results" / "full_100m"
GRID_ROOT = PROCESSED_ROOT / "grid"
GRID_REFINED_ROOT = ROOT / "data" / "processed" / "grid_refined_100m"

OUTPUT_CSV = ROOT.parent / "review_data" / "altitude_band_sensitivity.csv"
OUTPUT_JSON = ROOT.parent / "review_data" / "source_json" / "altitude_band_sensitivity.json"


def time_window_minutes_label(grid_root: Path) -> str:
    grid_info = json.loads((grid_root / "grid_info.json").read_text(encoding="utf-8"))
    return f"{int(grid_info.get('time_window_s', 300)) / 60:g}"


def format_band_label(lower_m: float | None, upper_m: float | None) -> str:
    if lower_m is None and upper_m is None:
        return "all_sample"
    if lower_m is None:
        return f"lt_{int(upper_m)}m"
    if upper_m is None:
        return f"ge_{int(lower_m)}m"
    return f"{int(lower_m)}_{int(upper_m)}m"


def subset_frame(data: pd.DataFrame, lower_m: float | None, upper_m: float | None) -> pd.DataFrame:
    mask = pd.Series(True, index=data.index)
    if lower_m is not None:
        mask &= data["mean_altitude"] >= float(lower_m)
    if upper_m is not None:
        mask &= data["mean_altitude"] < float(upper_m)
    return data.loc[mask].copy()


def summarize_altitude_band(
    analyzer: RobustnessAnalyzer,
    data: pd.DataFrame,
    lower_m: float | None,
    upper_m: float | None,
) -> dict:
    subset = subset_frame(data, lower_m, upper_m)
    label = format_band_label(lower_m, upper_m)

    if subset.empty:
        return {
            "band": label,
            "altitude_lower_m": lower_m,
            "altitude_upper_m": upper_m,
            "samples": 0,
            "sample_share_percent": 0.0,
            "active_cells": 0,
            "full_backbone_overlap_share": float("nan"),
            "rho_star": float("nan"),
            "speed_free": float("nan"),
            "speed_congested": float("nan"),
            "speed_drop_percent": float("nan"),
            "congestion_share_percent": float("nan"),
            "rr": float("nan"),
            "rr_ci_95_lower": float("nan"),
            "rr_ci_95_upper": float("nan"),
            "or_adj": float("nan"),
            "or_ci_95_lower": float("nan"),
            "or_ci_95_upper": float("nan"),
            "mean_altitude_p25_m": float("nan"),
            "mean_altitude_p50_m": float("nan"),
            "mean_altitude_p75_m": float("nan"),
        }

    subset_corridor = analyzer._compute_corridor_cells(subset)
    overlap_share = float(len(subset_corridor & analyzer.full_corridor_cells) / len(subset_corridor))
    summary = analyzer._summarize_subset(subset)

    altitude_quantiles = subset["mean_altitude"].quantile([0.25, 0.50, 0.75])

    return {
        "band": label,
        "altitude_lower_m": lower_m,
        "altitude_upper_m": upper_m,
        "samples": int(summary["samples"]),
        "sample_share_percent": float(len(subset) / len(data) * 100.0),
        "active_cells": int(summary["active_cells"]),
        "subset_backbone_cells": int(summary["corridor_cells"]),
        "full_backbone_overlap_share": overlap_share,
        "rho_star": float(summary["rho_star"]),
        "speed_free": float(summary["speed_free"]),
        "speed_congested": float(summary["speed_congested"]),
        "speed_drop_percent": float(summary["speed_drop_percent"]),
        "congestion_share_percent": float(summary["congestion_share_percent"]),
        "rr": float(summary["rr"]),
        "rr_ci_95_lower": float(summary["rr_ci_95_lower"]),
        "rr_ci_95_upper": float(summary["rr_ci_95_upper"]),
        "or_adj": float(summary["or_adj"]),
        "or_ci_95_lower": float(summary["or_ci_95_lower"]),
        "or_ci_95_upper": float(summary["or_ci_95_upper"]),
        "mean_altitude_p25_m": float(altitude_quantiles.loc[0.25]),
        "mean_altitude_p50_m": float(altitude_quantiles.loc[0.50]),
        "mean_altitude_p75_m": float(altitude_quantiles.loc[0.75]),
    }


def main() -> None:
    tw_label = time_window_minutes_label(GRID_ROOT)
    analyzer = RobustnessAnalyzer(
        output_dir=RESULTS_ROOT / "altitude_band_sensitivity",
        cell_stats_file=GRID_ROOT / "cell_stats.csv",
        spatiotemporal_file=GRID_ROOT / "spatiotemporal_stats.csv",
        cleaned_data_file=PROCESSED_ROOT / "cleaned_data.csv",
        od_pairs_file=GRID_REFINED_ROOT / "od_pairs.csv",
        fundamental_file=RESULTS_ROOT / "diagram" / "fundamental_data.csv",
        breakpoint_file=RESULTS_ROOT / "diagram" / "breakpoint_results.json",
        grid_info_file=GRID_ROOT / "grid_info.json",
    )
    if not analyzer.load_data():
        raise RuntimeError("Failed to load 100 m processed data for altitude-band sensitivity.")

    assert analyzer.spatiotemporal is not None
    data = analyzer.spatiotemporal.copy()

    bands = [
        (None, None),
        (None, 60.0),
        (60.0, 120.0),
        (120.0, 180.0),
        (180.0, 240.0),
        (240.0, None),
    ]

    records = [summarize_altitude_band(analyzer, data, lower, upper) for lower, upper in bands]
    frame = pd.DataFrame.from_records(records)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    payload = {
        "analysis_design": (
            "Re-estimate rho*, speed drop, and backbone amplification on mean-altitude bands "
            f"using the retained 100 m x {tw_label} min planar cell-window archive."
        ),
        "notes": {
            "density_definition": (
                f"rho counts telemetry points per 100 m x {tw_label} min planar cell-window; "
                "altitude is used here only to stratify subsets by mean cell-window altitude."
            ),
            "subset_backbone_definition": (
                "Each altitude band redefines its own top-10% backbone for comparability with existing "
                "subset-specific robustness analyses."
            ),
        },
        "altitude_reference_quantiles_m": {
            "p25": float(data["mean_altitude"].quantile(0.25)),
            "p50": float(data["mean_altitude"].quantile(0.50)),
            "p75": float(data["mean_altitude"].quantile(0.75)),
        },
        "bands": records,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    display_cols = [
        "band",
        "samples",
        "sample_share_percent",
        "active_cells",
        "rho_star",
        "speed_drop_percent",
        "congestion_share_percent",
        "rr",
        "full_backbone_overlap_share",
        "mean_altitude_p50_m",
    ]
    print(frame[display_cols].to_string(index=False))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
