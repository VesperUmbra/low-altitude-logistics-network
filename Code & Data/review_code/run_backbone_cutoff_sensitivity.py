from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from run_station_anchor import AggregatedCounts, aggregate_from_raw_csv


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"
BREAKPOINT_JSON = SOURCE_JSON_DIR / "breakpoint_results.json"

OUTPUT_CSV = REVIEW_DATA_DIR / "backbone_cutoff_sensitivity.csv"
OUTPUT_JSON = SOURCE_JSON_DIR / "backbone_cutoff_sensitivity.json"

CUTOFFS = [0.05, 0.10, 0.20]


def relative_risk_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    rr = (a / (a + b)) / (c / (c + d))
    se = math.sqrt((1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + d)))
    delta = 1.96 * se
    return rr, math.exp(math.log(rr) - delta), math.exp(math.log(rr) + delta)


def odds_ratio_ha_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    aa = a + 0.5
    bb = b + 0.5
    cc = c + 0.5
    dd = d + 0.5
    odds_ratio = (aa * dd) / (bb * cc)
    se = math.sqrt((1 / aa) + (1 / bb) + (1 / cc) + (1 / dd))
    delta = 1.96 * se
    return odds_ratio, math.exp(math.log(odds_ratio) - delta), math.exp(math.log(odds_ratio) + delta)


def build_backbone_cells(aggregated: AggregatedCounts, cutoff: float) -> tuple[list[str], set[str]]:
    sorted_cells = sorted(aggregated.total_points_by_cell.items(), key=lambda item: item[1], reverse=True)
    n_backbone = max(1, int(len(sorted_cells) * cutoff))
    ranked_cells = [cell_id for cell_id, _ in sorted_cells]
    backbone_cells = set(ranked_cells[:n_backbone])
    return ranked_cells, backbone_cells


def summarize_cutoff(aggregated: AggregatedCounts, cutoff: float) -> dict[str, object]:
    ranked_cells, backbone_cells = build_backbone_cells(aggregated, cutoff)
    active_cells = set(ranked_cells)
    hotspot_cells = set(aggregated.exceedance_samples_by_cell)

    total_points = sum(aggregated.total_points_by_cell.values())
    total_samples = sum(aggregated.total_samples_by_cell.values())
    total_exceedance = aggregated.total_exceedance_samples

    backbone_points = sum(aggregated.total_points_by_cell[cell_id] for cell_id in backbone_cells)
    backbone_hotspots = sum(1 for cell_id in hotspot_cells if cell_id in backbone_cells)
    backbone_exceedance = sum(aggregated.exceedance_samples_by_cell.get(cell_id, 0) for cell_id in backbone_cells)
    backbone_samples = sum(aggregated.total_samples_by_cell.get(cell_id, 0) for cell_id in backbone_cells)

    a = int(backbone_exceedance)
    b = int(backbone_samples - backbone_exceedance)
    c = int(total_exceedance - backbone_exceedance)
    d = int((total_samples - backbone_samples) - c)

    rr, rr_lo, rr_hi = relative_risk_ci(a, b, c, d)
    or_adj, or_lo, or_hi = odds_ratio_ha_ci(a, b, c, d)

    return {
        "cutoff_share": float(cutoff),
        "cutoff_label": f"top {int(round(cutoff * 100))}%",
        "active_cells": int(len(active_cells)),
        "backbone_cells": int(len(backbone_cells)),
        "backbone_cell_share": float(len(backbone_cells) / len(active_cells)),
        "traffic_points_in_backbone": int(backbone_points),
        "traffic_share": float(backbone_points / total_points),
        "hotspot_cells_total": int(len(hotspot_cells)),
        "hotspot_cells_covered": int(backbone_hotspots),
        "hotspot_cells_covered_share": float(backbone_hotspots / len(hotspot_cells)) if hotspot_cells else 0.0,
        "total_cell_window_samples": int(total_samples),
        "backbone_cell_window_samples": int(backbone_samples),
        "total_exceedance_samples": int(total_exceedance),
        "exceedance_samples_covered": int(backbone_exceedance),
        "exceedance_samples_covered_share": float(backbone_exceedance / total_exceedance) if total_exceedance else 0.0,
        "risk_backbone": float(a / (a + b)),
        "risk_non_backbone": float(c / (c + d)),
        "rr": float(rr),
        "rr_ci_95_lower": float(rr_lo),
        "rr_ci_95_upper": float(rr_hi),
        "or_adj": float(or_adj),
        "or_ci_95_lower": float(or_lo),
        "or_ci_95_upper": float(or_hi),
        "contingency_a": int(a),
        "contingency_b": int(b),
        "contingency_c": int(c),
        "contingency_d": int(d),
    }


def build_interpretive_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    lookup = {row["cutoff_label"]: row for row in rows}
    row_5 = lookup["top 5%"]
    row_10 = lookup["top 10%"]
    row_20 = lookup["top 20%"]

    return {
        "headline": (
            "The top-10% cutoff balances compactness and coverage: it captures materially more traffic and hotspot "
            "coverage than the top-5% cutoff, while the top-20% cutoff roughly doubles the spatial footprint for only "
            "modest additional hotspot and exceedance capture."
        ),
        "contrast_top5_vs_top10": {
            "traffic_share_gain": float(row_10["traffic_share"] - row_5["traffic_share"]),
            "hotspot_share_gain": float(row_10["hotspot_cells_covered_share"] - row_5["hotspot_cells_covered_share"]),
            "exceedance_share_gain": float(row_10["exceedance_samples_covered_share"] - row_5["exceedance_samples_covered_share"]),
            "rr_ratio": float(row_10["rr"] / row_5["rr"]),
            "or_ratio": float(row_10["or_adj"] / row_5["or_adj"]),
        },
        "contrast_top10_vs_top20": {
            "active_footprint_ratio": float(row_20["backbone_cell_share"] / row_10["backbone_cell_share"]),
            "traffic_share_gain": float(row_20["traffic_share"] - row_10["traffic_share"]),
            "hotspot_share_gain": float(row_20["hotspot_cells_covered_share"] - row_10["hotspot_cells_covered_share"]),
            "exceedance_share_gain": float(row_20["exceedance_samples_covered_share"] - row_10["exceedance_samples_covered_share"]),
            "rr_ratio": float(row_20["rr"] / row_10["rr"]),
            "or_ratio": float(row_20["or_adj"] / row_10["or_adj"]),
        },
    }


def main() -> None:
    rho_star = float(json.loads(BREAKPOINT_JSON.read_text(encoding="utf-8"))["consensus_rho_star"])
    aggregated = aggregate_from_raw_csv(rho_star=rho_star)
    rows = [summarize_cutoff(aggregated, cutoff) for cutoff in CUTOFFS]

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "rho_star_used": rho_star,
        "source_raw_counts": {
            "cleaned_points": int(aggregated.cleaned_points),
            "valid_orders": int(aggregated.valid_orders),
            "active_cells": int(len(aggregated.total_points_by_cell)),
            "hotspot_cells": int(len(aggregated.exceedance_samples_by_cell)),
            "total_cell_window_samples": int(sum(aggregated.total_samples_by_cell.values())),
            "total_exceedance_samples": int(aggregated.total_exceedance_samples),
        },
        "cutoff_rows": rows,
        "interpretive_summary": build_interpretive_summary(rows),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    for row in rows:
        print(
            row["cutoff_label"],
            f"cells={row['backbone_cells']}",
            f"traffic={row['traffic_share']:.4f}",
            f"hotspot={row['hotspot_cells_covered_share']:.4f}",
            f"exceedance={row['exceedance_samples_covered_share']:.4f}",
            f"rr={row['rr']:.2f}",
        )


if __name__ == "__main__":
    main()
