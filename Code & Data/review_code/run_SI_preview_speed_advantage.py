from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FOR_REVIEW = ROOT.parent
REVIEW_DATA_DIR = FOR_REVIEW / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"
GRID_DIR = ROOT / "data" / "processed" / "full_100m" / "grid"
PROCESSED_DIR = ROOT / "data" / "processed" / "full_100m"

TRAJ_CSV = REVIEW_DATA_DIR / "trajectory_length_distribution.csv"
CELL_METRICS_CSV = REVIEW_DATA_DIR / "figure_cell_metrics.csv"
GRID_INFO_JSON = GRID_DIR / "grid_info.json"
CLEANED_CSV = PROCESSED_DIR / "cleaned_data.csv"

OUT_SPEED_PHASE_CSV = REVIEW_DATA_DIR / "preview_speed_phase_by_distance.csv"
OUT_DURATION_BIN_CSV = REVIEW_DATA_DIR / "preview_distance_duration_bins.csv"
OUT_SUMMARY_JSON = SOURCE_JSON_DIR / "preview_speed_advantage_summary.json"


DIST_BINS = [0, 1, 3, 5, 10, 20, 80]
DIST_LABELS = ["0-1", "1-3", "3-5", "5-10", "10-20", "20-80"]


def load_inputs() -> tuple[pd.DataFrame, set[str], dict[str, float]]:
    traj = pd.read_csv(
        TRAJ_CSV,
        usecols=["order_id", "path_length_km", "duration_minutes", "start_time", "vendor"],
    )
    traj["dist_bin"] = pd.cut(
        traj["path_length_km"],
        bins=DIST_BINS,
        labels=DIST_LABELS,
        right=False,
        include_lowest=True,
    )
    traj = traj.loc[traj["dist_bin"].notna()].copy()
    traj["dist_bin"] = traj["dist_bin"].astype(str)
    traj["effective_speed_kmh"] = traj["path_length_km"] / (traj["duration_minutes"] / 60.0)

    cell_metrics = pd.read_csv(CELL_METRICS_CSV, usecols=["cell_id", "endpoint_buffer1_cell"])
    endpoint_cells = set(
        cell_metrics.loc[cell_metrics["endpoint_buffer1_cell"], "cell_id"].astype(str)
    )

    grid_info = json.loads(GRID_INFO_JSON.read_text(encoding="utf-8"))
    return traj, endpoint_cells, grid_info


def map_to_cell_ids(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    min_lat: float,
    min_lon: float,
    meters_per_degree_lat: float,
    meters_per_degree_lon: float,
    grid_size_m: float,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    rows = np.floor(((lat - min_lat) * meters_per_degree_lat) / grid_size_m).astype(int)
    cols = np.floor(((lon - min_lon) * meters_per_degree_lon) / grid_size_m).astype(int)
    rows = np.clip(rows, 0, n_rows - 1)
    cols = np.clip(cols, 0, n_cols - 1)
    return np.char.add(np.char.add(rows.astype(str), "_"), cols.astype(str))


def aggregate_speed_by_phase(
    traj: pd.DataFrame,
    endpoint_cells: set[str],
    grid_info: dict[str, float],
) -> pd.DataFrame:
    order_info = traj.set_index("order_id")[["dist_bin", "path_length_km", "duration_minutes", "vendor"]]
    order_ids = set(order_info.index)

    rows: list[pd.DataFrame] = []
    chunk_iter = pd.read_csv(
        CLEANED_CSV,
        usecols=["order_id", "latitude", "longitude", "speed"],
        chunksize=400_000,
    )

    for chunk in chunk_iter:
        chunk = chunk.loc[chunk["order_id"].isin(order_ids)].copy()
        if chunk.empty:
            continue

        chunk["cell_id"] = map_to_cell_ids(
            chunk["latitude"].to_numpy(),
            chunk["longitude"].to_numpy(),
            min_lat=float(grid_info["min_lat"]),
            min_lon=float(grid_info["min_lon"]),
            meters_per_degree_lat=float(grid_info["meters_per_degree_lat"]),
            meters_per_degree_lon=float(grid_info["meters_per_degree_lon"]),
            grid_size_m=float(grid_info["grid_size_m"]),
            n_rows=int(grid_info["n_rows"]),
            n_cols=int(grid_info["n_cols"]),
        )
        endpoint_mask = pd.Series(chunk["cell_id"]).isin(endpoint_cells).to_numpy()
        chunk["phase"] = np.where(
            endpoint_mask,
            "Endpoint-adjacent",
            "Away from endpoint",
        )
        meta = order_info.reindex(chunk["order_id"]).reset_index(drop=True)
        out = chunk[["order_id", "phase", "speed"]].reset_index(drop=True)
        out = out.rename(columns={"speed": "point_speed"})
        out["dist_bin"] = meta["dist_bin"].astype(str).to_numpy()
        out["path_length_km"] = meta["path_length_km"].astype(float).to_numpy()
        out["duration_minutes"] = meta["duration_minutes"].astype(float).to_numpy()
        out["vendor"] = meta["vendor"].astype(str).to_numpy()
        rows.append(out)

    if not rows:
        return pd.DataFrame(
            columns=[
                "order_id",
                "phase",
                "point_speed",
                "dist_bin",
                "path_length_km",
                "duration_minutes",
                "vendor",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def summarise_duration_bins(traj: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        traj.groupby("dist_bin", observed=False)
        .agg(
            n=("order_id", "count"),
            median_dist_km=("path_length_km", "median"),
            q25_dist_km=("path_length_km", lambda s: float(s.quantile(0.25))),
            q75_dist_km=("path_length_km", lambda s: float(s.quantile(0.75))),
            median_duration_min=("duration_minutes", "median"),
            q25_duration_min=("duration_minutes", lambda s: float(s.quantile(0.25))),
            q75_duration_min=("duration_minutes", lambda s: float(s.quantile(0.75))),
            median_effective_speed_kmh=("effective_speed_kmh", "median"),
        )
        .reset_index()
    )
    return grouped


def main() -> None:
    traj, endpoint_cells, grid_info = load_inputs()
    speed_phase = aggregate_speed_by_phase(traj, endpoint_cells, grid_info)
    duration_bins = summarise_duration_bins(traj)

    OUT_SPEED_PHASE_CSV.parent.mkdir(parents=True, exist_ok=True)
    speed_phase.to_csv(OUT_SPEED_PHASE_CSV, index=False)
    duration_bins.to_csv(OUT_DURATION_BIN_CSV, index=False)

    summary_rows = []
    for _, row in duration_bins.iterrows():
        bin_label = str(row["dist_bin"])
        if bin_label == "20-80":
            continue
        median_speed = float(row["median_effective_speed_kmh"])
        summary_rows.append(
            {
                "dist_bin": bin_label,
                "median_effective_speed_kmh": median_speed,
                "faster_than_20kph_ref": bool(median_speed > 20),
                "faster_than_25kph_ref": bool(median_speed > 25),
                "faster_than_30kph_ref": bool(median_speed > 30),
            }
        )

    summary = {
        "interpretation": "Potential time advantage is more plausible at medium distances than at the shortest distances when compared against simple surface-speed reference lines rather than matched road travel times.",
        "distance_bin_summary": summary_rows,
        "notes": [
            "Endpoint proximity in panel a is based on the existing 100 m endpoint-buffer1 cells used elsewhere in the paper.",
            "Surface reference lines in panel b are heuristic speed benchmarks, not matched road travel-time observations.",
        ],
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
