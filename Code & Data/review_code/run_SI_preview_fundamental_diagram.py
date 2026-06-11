from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FOR_REVIEW = ROOT.parent
REVIEW_DATA_DIR = FOR_REVIEW / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"
PROCESSED_ROOT = ROOT / "data" / "processed" / "full_100m"
GRID_INFO_JSON = PROCESSED_ROOT / "grid" / "grid_info.json"
CLEANED_DATA_CSV = PROCESSED_ROOT / "cleaned_data.csv"
UNIQUE_OCCUPANCY_JSON = SOURCE_JSON_DIR / "summary_results.json"

OUTPUT_CELL_WINDOWS = REVIEW_DATA_DIR / "preview_fundamental_diagram_cell_windows.csv"
OUTPUT_BINNED = REVIEW_DATA_DIR / "preview_fundamental_diagram_binned.csv"
OUTPUT_SUMMARY_JSON = SOURCE_JSON_DIR / "preview_fundamental_diagram_summary.json"


def map_to_cell_ids(df: pd.DataFrame, grid_info: dict) -> pd.Series:
    rows = (
        ((df["latitude"] - float(grid_info["min_lat"])) * float(grid_info["meters_per_degree_lat"]))
        / float(grid_info["grid_size_m"])
    ).astype(int)
    cols = (
        ((df["longitude"] - float(grid_info["min_lon"])) * float(grid_info["meters_per_degree_lon"]))
        / float(grid_info["grid_size_m"])
    ).astype(int)
    rows = rows.clip(lower=0, upper=int(grid_info["n_rows"]) - 1)
    cols = cols.clip(lower=0, upper=int(grid_info["n_cols"]) - 1)
    return rows.astype(str) + "_" + cols.astype(str)


def build_operational_table() -> pd.DataFrame:
    grid_info = json.loads(GRID_INFO_JSON.read_text(encoding="utf-8"))
    time_window_s = int(grid_info.get("time_window_s", 300))
    floor_alias = f"{time_window_s // 60}min" if time_window_s % 60 == 0 else f"{time_window_s}s"
    windows_per_hour = 3600 / time_window_s
    df = pd.read_csv(
        CLEANED_DATA_CSV,
        usecols=["datetime", "speed", "longitude", "latitude", "order_id"],
        parse_dates=["datetime"],
        dtype={
            "speed": "float32",
            "longitude": "float64",
            "latitude": "float64",
            "order_id": "string",
        },
    )
    df["cell_id"] = map_to_cell_ids(df, grid_info)
    df["window_start"] = df["datetime"].dt.floor(floor_alias)
    df = df.sort_values(["order_id", "datetime"]).reset_index(drop=True)

    kv = (
        df.groupby(["cell_id", "window_start"], observed=True)
        .agg(
            density_unique_flights=("order_id", "nunique"),
            speed_mean=("speed", "mean"),
            n_points=("speed", "size"),
        )
        .reset_index()
    )

    prev_order = df["order_id"].shift()
    prev_cell = df["cell_id"].shift()
    is_new_order = df["order_id"] != prev_order
    is_new_cell = df["cell_id"] != prev_cell
    entry_rows = df.loc[is_new_order | is_new_cell, ["cell_id", "window_start", "order_id"]].copy()
    q = (
        entry_rows.groupby(["cell_id", "window_start"], observed=True)
        .agg(flow_entries=("order_id", "nunique"))
        .reset_index()
    )

    table = kv.merge(q, on=["cell_id", "window_start"], how="left")
    table["flow_entries"] = table["flow_entries"].fillna(0).astype(int)
    table["flow_entries_per_hour"] = table["flow_entries"] * windows_per_hour
    table = table.loc[(table["density_unique_flights"] > 0) & (table["speed_mean"] > 0)].copy()
    table["window_date"] = pd.to_datetime(table["window_start"]).dt.date.astype(str)
    table.attrs["time_window_minutes"] = time_window_s / 60.0
    return table


def binned_line(df: pd.DataFrame, x_col: str, y_col: str, *, bin_width: float) -> pd.DataFrame:
    x = df[x_col].to_numpy(dtype=float)
    max_x = float(np.nanmax(x))
    bins = np.arange(0, max_x + bin_width + 1e-9, bin_width)
    if len(bins) < 2:
        bins = np.array([0.0, max_x + bin_width])
    work = df[[x_col, y_col]].copy()
    work["bin"] = pd.cut(work[x_col], bins=bins, right=False, include_lowest=True)
    grouped = (
        work.groupby("bin", observed=False)
        .agg(
            sample_count=(y_col, "count"),
            x_mean=(x_col, "mean"),
            y_mean=(y_col, "mean"),
            y_median=(y_col, "median"),
            y_q25=(y_col, lambda s: float(s.quantile(0.25))),
            y_q75=(y_col, lambda s: float(s.quantile(0.75))),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["sample_count"] >= 30].copy()
    return grouped


def export_binned_relations(table: pd.DataFrame) -> None:
    vk = binned_line(table, "density_unique_flights", "speed_mean", bin_width=1.0)
    qk = binned_line(table, "density_unique_flights", "flow_entries_per_hour", bin_width=1.0)
    qv = binned_line(table, "speed_mean", "flow_entries_per_hour", bin_width=0.5)

    binned_export = pd.concat(
        [
            vk.assign(relation="v_k"),
            qk.assign(relation="q_k"),
            qv.assign(relation="q_v"),
        ],
        ignore_index=True,
    )
    binned_export.to_csv(OUTPUT_BINNED, index=False)


def main() -> None:
    table = build_operational_table()
    table.to_csv(OUTPUT_CELL_WINDOWS, index=False)

    summary = json.loads(UNIQUE_OCCUPANCY_JSON.read_text(encoding="utf-8"))
    rho_unique = float(summary["unique_flight_occupancy"]["rho_star_unique"])

    output_summary = {
        "density_definition": f"distinct flights per 100 m x {table.attrs.get('time_window_minutes', 5.0):g} min cell-window",
        "flow_definition": "distinct order entries into each 100 m cell-window, converted to hourly units",
        "speed_definition": f"mean recorded speed within each 100 m x {table.attrs.get('time_window_minutes', 5.0):g} min cell-window",
        "n_cell_windows": int(len(table)),
        "rho_star_unique_reference": rho_unique,
        "density_range": [float(table["density_unique_flights"].min()), float(table["density_unique_flights"].max())],
        "speed_range": [float(table["speed_mean"].min()), float(table["speed_mean"].max())],
        "flow_entries_per_hour_range": [float(table["flow_entries_per_hour"].min()), float(table["flow_entries_per_hour"].max())],
    }
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(output_summary, indent=2), encoding="utf-8")

    export_binned_relations(table)
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
