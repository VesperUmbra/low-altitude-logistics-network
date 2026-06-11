from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

RANKED_CELLS_CSV = REVIEW_DATA_DIR / "ranked_cell_traffic_table.csv"
HOTSPOT_CSV = REVIEW_DATA_DIR / "hotspot_exceedance_rank_table.csv"
REAL_STATION_JSON = SOURCE_JSON_DIR / "real_station_anchor_results.json"

OUTPUT_CSV = REVIEW_DATA_DIR / "targeted_monitoring_payoff.csv"
OUTPUT_RANDOM_CSV = REVIEW_DATA_DIR / "targeted_monitoring_random_baseline.csv"
OUTPUT_JSON = SOURCE_JSON_DIR / "targeted_monitoring_payoff.json"


def load_cell_level_data() -> pd.DataFrame:
    ranked = pd.read_csv(RANKED_CELLS_CSV)
    hotspot = pd.read_csv(HOTSPOT_CSV)

    ranked = ranked.rename(columns={"grid_cell_id": "cell_id"})
    hotspot = hotspot.rename(columns={"grid_cell_id": "cell_id"})

    cell = ranked[
        [
            "cell_id",
            "rank",
            "total_points",
            "traffic_share",
            "cumulative_traffic_share",
            "is_backbone_cell",
        ]
    ].copy()

    hotspot_small = hotspot[
        ["cell_id", "exceedance_count", "exceedance_share", "cumulative_exceedance_share"]
    ].copy()
    cell = cell.merge(hotspot_small, on="cell_id", how="left")
    cell["exceedance_count"] = cell["exceedance_count"].fillna(0).astype(int)
    cell["exceedance_share"] = cell["exceedance_share"].fillna(0.0)
    return cell.sort_values("rank").reset_index(drop=True)


def summarize_top_traffic_monitoring(cell: pd.DataFrame) -> list[dict[str, float | int | str | None]]:
    n_active = len(cell)
    total_exceedance = int(cell["exceedance_count"].sum())
    budgets = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

    rows: list[dict[str, float | int | str | None]] = []
    for budget_share in budgets:
        budget_cells = max(1, int(np.floor(n_active * budget_share)))
        subset = cell.iloc[:budget_cells]
        captured_exceedance = int(subset["exceedance_count"].sum())
        captured_exceedance_share = float(captured_exceedance / total_exceedance)
        captured_traffic_share = float(subset["traffic_share"].sum())
        rows.append(
            {
                "strategy": "top_traffic_cells",
                "budget_share_active_cells": float(budget_share),
                "budget_cell_count": int(budget_cells),
                "captured_exceedance_samples": int(captured_exceedance),
                "captured_exceedance_share": captured_exceedance_share,
                "captured_traffic_share": captured_traffic_share,
                "lift_vs_random_exceedance": float(captured_exceedance_share / budget_share),
                "lift_vs_random_traffic": float(captured_traffic_share / budget_share),
            }
        )
    return rows


def summarize_station_monitoring() -> tuple[list[dict[str, float | int | str | None]], dict]:
    with open(REAL_STATION_JSON, "r", encoding="utf-8") as f:
        station_payload = json.load(f)["anchor_sets"]["operational_sites"]

    rows: list[dict[str, float | int | str | None]] = []
    for anchor_key in ["exact_anchor_cells", "buffer1_anchor_cells", "buffer2_anchor_cells"]:
        payload = station_payload[anchor_key]
        rows.append(
            {
                "strategy": f"real_station_{anchor_key}",
                "budget_share_active_cells": float(payload["active_cells_covered_share"]),
                "budget_cell_count": int(payload["active_cells_covered"]),
                "captured_exceedance_samples": int(payload["exceedance_samples_covered"]),
                "captured_exceedance_share": float(payload["exceedance_samples_covered_share"]),
                "captured_traffic_share": None,
                "lift_vs_random_exceedance": float(
                    payload["exceedance_samples_covered_share"] / payload["active_cells_covered_share"]
                ),
                "lift_vs_random_traffic": None,
            }
        )
    return rows, station_payload


def monte_carlo_random_baseline(
    cell: pd.DataFrame,
    station_payload: dict,
    *,
    seed: int = 42,
    n_draws: int = 2000,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    n_active = len(cell)
    total_exceedance = int(cell["exceedance_count"].sum())

    rows: list[dict[str, float | int | str]] = []
    for anchor_key in ["exact_anchor_cells", "buffer1_anchor_cells", "buffer2_anchor_cells"]:
        payload = station_payload[anchor_key]
        budget_cells = int(payload["active_cells_covered"])
        draws: list[float] = []
        for _ in range(n_draws):
            indices = rng.choice(n_active, size=budget_cells, replace=False)
            share = float(cell.iloc[indices]["exceedance_count"].sum() / total_exceedance)
            draws.append(share)

        random_mean = float(np.mean(draws))
        random_p95 = float(np.quantile(draws, 0.95))
        observed = float(payload["exceedance_samples_covered_share"])
        rows.append(
            {
                "strategy": anchor_key,
                "cells_monitored": budget_cells,
                "mean_random_exceedance_share": random_mean,
                "p95_random_exceedance_share": random_p95,
                "observed_exceedance_share": observed,
                "observed_over_random_mean": float(observed / random_mean),
                "observed_over_random_p95": float(observed / random_p95),
            }
        )
    return rows


def build_panel_spec(payoff_rows: pd.DataFrame, random_rows: pd.DataFrame) -> dict[str, object]:
    return {
        "figure_purpose": "Quantify the governance payoff of targeted monitoring under limited spatial budgets.",
        "panel_a": {
            "title": "Budget versus exceedance capture",
            "x": "share_of_active_cells_monitored",
            "y": "share_of_exceedance_samples_captured",
            "series": [
                "random_baseline_y_equals_x",
                "top_traffic_cells_curve",
                "real_station_exact_point",
                "real_station_buffer1_point",
                "real_station_buffer2_point",
            ],
        },
        "panel_b": {
            "title": "Lift over random same-area monitoring",
            "metric": "captured_exceedance_share / monitored_active_cell_share",
            "series": [
                "top_traffic_cells",
                "real_station_exact",
                "real_station_buffer1",
                "real_station_buffer2",
            ],
        },
        "headline_numbers": {
            "top_traffic_1pct_exceedance_share": float(
                payoff_rows.loc[
                    (payoff_rows["strategy"] == "top_traffic_cells")
                    & (np.isclose(payoff_rows["budget_share_active_cells"], 0.01)),
                    "captured_exceedance_share",
                ].iloc[0]
            ),
            "real_station_buffer1_exceedance_share": float(
                payoff_rows.loc[
                    payoff_rows["strategy"] == "real_station_buffer1_anchor_cells",
                    "captured_exceedance_share",
                ].iloc[0]
            ),
            "real_station_buffer1_active_share": float(
                payoff_rows.loc[
                    payoff_rows["strategy"] == "real_station_buffer1_anchor_cells",
                    "budget_share_active_cells",
                ].iloc[0]
            ),
            "real_station_buffer1_lift_over_random_mean": float(
                random_rows.loc[random_rows["strategy"] == "buffer1_anchor_cells", "observed_over_random_mean"].iloc[0]
            ),
        },
    }


def main() -> None:
    cell = load_cell_level_data()
    payoff_rows = summarize_top_traffic_monitoring(cell)
    station_rows, station_payload = summarize_station_monitoring()
    payoff_df = pd.DataFrame(payoff_rows + station_rows)
    random_df = pd.DataFrame(monte_carlo_random_baseline(cell, station_payload))

    OUTPUT_CSV.write_text(payoff_df.to_csv(index=False), encoding="utf-8-sig")
    OUTPUT_RANDOM_CSV.write_text(random_df.to_csv(index=False), encoding="utf-8-sig")

    payload = {
        "payoff_table": payoff_df.to_dict(orient="records"),
        "random_baseline_table": random_df.to_dict(orient="records"),
        "figure_spec": build_panel_spec(payoff_df, random_df),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_RANDOM_CSV}")
    print(f"Wrote {OUTPUT_JSON}")

    one_pct = payoff_df.loc[
        (payoff_df["strategy"] == "top_traffic_cells") & (np.isclose(payoff_df["budget_share_active_cells"], 0.01))
    ].iloc[0]
    station_b1 = payoff_df.loc[payoff_df["strategy"] == "real_station_buffer1_anchor_cells"].iloc[0]
    station_b1_rand = random_df.loc[random_df["strategy"] == "buffer1_anchor_cells"].iloc[0]

    print(
        "Top 1% traffic cells:",
        f"capture {one_pct['captured_exceedance_share']:.4f} exceedances",
        f"and {one_pct['captured_traffic_share']:.4f} traffic",
    )
    print(
        "Station buffer-1:",
        f"budget {station_b1['budget_share_active_cells']:.4f}",
        f"capture {station_b1['captured_exceedance_share']:.4f} exceedances",
        f"lift over random mean {station_b1_rand['observed_over_random_mean']:.2f}x",
    )


if __name__ == "__main__":
    main()
