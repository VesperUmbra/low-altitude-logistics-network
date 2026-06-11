from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FOR_REVIEW = ROOT.parent
REVIEW_DATA_DIR = FOR_REVIEW / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"
LOCAL_INPUTS_DIR = FOR_REVIEW / "local_inputs"

ENDPOINT_CSV = REVIEW_DATA_DIR / "endpoint_cell_altitude_summary.csv"
STATION_CSV = REVIEW_DATA_DIR / "real_station_anchor_sites_mapped.csv"
BUILDING_XLSX = LOCAL_INPUTS_DIR / "shenzhen_building_height_roofarea.xlsx"
TERRAIN_CACHE_CSV = REVIEW_DATA_DIR / "endpoint_terrain_elevation_cache.csv"
OUTPUT_CSV = REVIEW_DATA_DIR / "endpoint_relative_height_classification.csv"
OUTPUT_JSON = SOURCE_JSON_DIR / "endpoint_relative_height_summary.json"

TERRAIN_API_BASE = "https://api.opentopodata.org/v1/srtm90m"
TERRAIN_BATCH_SIZE = 80

GROUND_LIKE_THRESHOLD_M = 15.0
BUILDING_SEARCH_RADIUS_M = 150.0
BUILDING_HEIGHT_TOLERANCE_M = 20.0
STATION_ADJACENT_THRESHOLD_M = 150.0


def meters_per_degree(mean_lat: float) -> tuple[float, float]:
    meters_lat = 111_320.0
    meters_lon = 111_320.0 * math.cos(math.radians(mean_lat))
    return meters_lat, meters_lon


def load_endpoint_cells() -> pd.DataFrame:
    df = pd.read_csv(ENDPOINT_CSV)
    required = {
        "cell_id",
        "n",
        "starts",
        "ends",
        "median_altitude",
        "mean_altitude",
        "median_lon",
        "median_lat",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing endpoint columns: {sorted(missing)}")
    return df


def load_buildings() -> pd.DataFrame:
    raw = pd.read_excel(BUILDING_XLSX, sheet_name="Sheet1")
    # The copied workbook preserves the numeric columns reliably even if
    # non-ASCII header labels render differently across environments.
    df = raw.iloc[:, [1, 6, 7]].dropna().copy()
    df.columns = ["building_height_m", "lon", "lat"]
    return df


def load_operational_stations() -> pd.DataFrame:
    df = pd.read_csv(STATION_CSV)
    mask = (df["anchor_set"] == "all_shenzhen_sites") & (~df["is_test_site"])
    cols = ["site_name", "platform", "lon", "lat"]
    return df.loc[mask, cols].drop_duplicates().reset_index(drop=True)


def terrain_key(lat: float, lon: float) -> str:
    return f"{lat:.6f},{lon:.6f}"


def load_terrain_cache() -> dict[str, float]:
    if not TERRAIN_CACHE_CSV.exists():
        return {}
    cached = pd.read_csv(TERRAIN_CACHE_CSV)
    if cached.empty:
        return {}
    return {
        terrain_key(float(row["lat"]), float(row["lon"])): float(row["terrain_m"])
        for _, row in cached.iterrows()
        if pd.notna(row["terrain_m"])
    }


def write_terrain_cache(cache: dict[str, float]) -> None:
    rows = []
    for key, terrain in sorted(cache.items()):
        lat_str, lon_str = key.split(",")
        rows.append(
            {
                "lat": float(lat_str),
                "lon": float(lon_str),
                "terrain_m": float(terrain),
                "source": "OpenTopoData_SRTM90m",
            }
        )
    pd.DataFrame(rows).to_csv(TERRAIN_CACHE_CSV, index=False)


def query_terrain_batch(batch: list[tuple[float, float]]) -> dict[str, float]:
    locations = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in batch)
    url = f"{TERRAIN_API_BASE}?{urllib.parse.urlencode({'locations': locations})}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    return {
        terrain_key(lat, lon): float(result["elevation"])
        for (lat, lon), result in zip(batch, payload["results"], strict=True)
    }


def query_terrain_with_retry(batch: list[tuple[float, float]], attempts: int = 3) -> dict[str, float]:
    for attempt in range(attempts):
        try:
            return query_terrain_batch(batch)
        except Exception:
            if len(batch) == 1:
                if attempt == attempts - 1:
                    raise
                time.sleep(1.0 + attempt)
                continue
            midpoint = len(batch) // 2
            left = query_terrain_with_retry(batch[:midpoint], attempts=attempts)
            right = query_terrain_with_retry(batch[midpoint:], attempts=attempts)
            left.update(right)
            return left
    raise RuntimeError("Terrain query retry loop exhausted unexpectedly.")


def fetch_terrain_elevations(points: pd.DataFrame) -> list[float]:
    cache = load_terrain_cache()
    missing: list[tuple[float, float]] = []
    values: list[float | None] = []

    for _, row in points.iterrows():
        lat = float(row["median_lat"])
        lon = float(row["median_lon"])
        key = terrain_key(lat, lon)
        if key in cache:
            values.append(cache[key])
        else:
            values.append(None)
            missing.append((lat, lon))

    for start in range(0, len(missing), TERRAIN_BATCH_SIZE):
        batch = missing[start : start + TERRAIN_BATCH_SIZE]
        if not batch:
            continue
        cache.update(query_terrain_with_retry(batch))

    write_terrain_cache(cache)

    final = []
    for _, row in points.iterrows():
        key = terrain_key(float(row["median_lat"]), float(row["median_lon"]))
        final.append(float(cache[key]))
    return final


def compute_local_matches(endpoint_df: pd.DataFrame, building_df: pd.DataFrame) -> pd.DataFrame:
    mean_lat = float(endpoint_df["median_lat"].mean())
    meters_lat, meters_lon = meters_per_degree(mean_lat)

    building_lon = building_df["lon"].to_numpy()
    building_lat = building_df["lat"].to_numpy()
    building_h = building_df["building_height_m"].to_numpy()

    match_rows = []
    for _, row in endpoint_df.iterrows():
        dx = (building_lon - float(row["median_lon"])) * meters_lon
        dy = (building_lat - float(row["median_lat"])) * meters_lat
        distances = np.sqrt(dx * dx + dy * dy)

        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])
        nearest_height = float(building_h[nearest_idx])

        within = distances <= BUILDING_SEARCH_RADIUS_M
        local_count = int(np.count_nonzero(within))

        if local_count:
            local_dist = distances[within]
            local_height = building_h[within]
            gaps = np.abs(local_height - float(row["rel_ground_m"]))
            best_idx = int(np.argmin(gaps))
            best_gap = float(gaps[best_idx])
            best_height = float(local_height[best_idx])
            best_dist = float(local_dist[best_idx])
        else:
            best_gap = np.nan
            best_height = np.nan
            best_dist = np.nan

        match_rows.append(
            {
                "nearest_bldg_dist_m": nearest_dist,
                "nearest_bldg_height_m": nearest_height,
                "bldg_count_within_150m": local_count,
                "best_height_gap_150m": best_gap,
                "best_match_height_150m": best_height,
                "best_match_dist_150m": best_dist,
            }
        )

    return pd.DataFrame(match_rows)


def compute_station_distance(endpoint_df: pd.DataFrame, station_df: pd.DataFrame) -> list[float]:
    mean_lat = float(endpoint_df["median_lat"].mean())
    meters_lat, meters_lon = meters_per_degree(mean_lat)

    station_lon = station_df["lon"].to_numpy()
    station_lat = station_df["lat"].to_numpy()

    distances_out = []
    for _, row in endpoint_df.iterrows():
        dx = (station_lon - float(row["median_lon"])) * meters_lon
        dy = (station_lat - float(row["median_lat"])) * meters_lat
        distances = np.sqrt(dx * dx + dy * dy)
        distances_out.append(float(np.min(distances)))
    return distances_out


def classify_endpoint(row: pd.Series) -> str:
    rel_ground = float(row["rel_ground_m"])
    if rel_ground <= GROUND_LIKE_THRESHOLD_M:
        return "ground_like"
    if pd.notna(row["best_height_gap_150m"]) and float(row["best_height_gap_150m"]) <= BUILDING_HEIGHT_TOLERANCE_M:
        return "building_compatible"
    return "elevated_unresolved"


def summarize_subset(df: pd.DataFrame) -> dict[str, float | int | dict[str, float | int]]:
    total_cells = int(len(df))
    total_touches = float(df["n"].sum())
    summary: dict[str, float | int | dict[str, float | int]] = {
        "endpoint_cells": total_cells,
        "endpoint_touches": int(round(total_touches)),
        "median_rel_ground_m": float(df["rel_ground_m"].median()),
        "p25_rel_ground_m": float(df["rel_ground_m"].quantile(0.25)),
        "p75_rel_ground_m": float(df["rel_ground_m"].quantile(0.75)),
        "median_nearest_station_m": float(df["nearest_station_m"].median()),
    }
    for cls in ["ground_like", "building_compatible", "elevated_unresolved"]:
        subset = df[df["height_class"] == cls]
        touches = float(subset["n"].sum())
        summary[cls] = {
            "cells": int(len(subset)),
            "cell_share": float(len(subset) / total_cells) if total_cells else np.nan,
            "touches": int(round(touches)),
            "touch_share": float(touches / total_touches) if total_touches else np.nan,
        }
    return summary


def build_sensitivity(df: pd.DataFrame, building_df: pd.DataFrame) -> list[dict[str, float | int]]:
    mean_lat = float(df["median_lat"].mean())
    meters_lat, meters_lon = meters_per_degree(mean_lat)
    building_lon = building_df["lon"].to_numpy()
    building_lat = building_df["lat"].to_numpy()
    building_h = building_df["building_height_m"].to_numpy()

    rows = []
    for radius in (100.0, 150.0):
        gaps_all = []
        for _, row in df.iterrows():
            dx = (building_lon - float(row["median_lon"])) * meters_lon
            dy = (building_lat - float(row["median_lat"])) * meters_lat
            distances = np.sqrt(dx * dx + dy * dy)
            within = distances <= radius
            if np.count_nonzero(within):
                gaps_all.append(float(np.abs(building_h[within] - float(row["rel_ground_m"])).min()))
            else:
                gaps_all.append(np.nan)

        gap_series = pd.Series(gaps_all, index=df.index)
        for tolerance in (15.0, 20.0, 25.0):
            classes = np.where(
                df["rel_ground_m"] <= GROUND_LIKE_THRESHOLD_M,
                "ground_like",
                np.where(gap_series.notna() & (gap_series <= tolerance), "building_compatible", "elevated_unresolved"),
            )
            class_series = pd.Series(classes, index=df.index)
            for cls in ["ground_like", "building_compatible", "elevated_unresolved"]:
                cls_mask = class_series == cls
                rows.append(
                    {
                        "search_radius_m": int(radius),
                        "height_tolerance_m": int(tolerance),
                        "class_name": cls,
                        "cell_share": float(cls_mask.mean()),
                        "touch_share": float(df.loc[cls_mask, "n"].sum() / df["n"].sum()),
                    }
                )
    return rows


def main() -> None:
    endpoint_df = load_endpoint_cells()
    building_df = load_buildings()
    station_df = load_operational_stations()

    endpoint_df["terrain_m"] = fetch_terrain_elevations(endpoint_df)
    endpoint_df["rel_ground_m"] = endpoint_df["median_altitude"] - endpoint_df["terrain_m"]

    endpoint_df = pd.concat([endpoint_df, compute_local_matches(endpoint_df, building_df)], axis=1)
    endpoint_df["nearest_station_m"] = compute_station_distance(endpoint_df, station_df)
    endpoint_df["station_adjacent_150m"] = endpoint_df["nearest_station_m"] <= STATION_ADJACENT_THRESHOLD_M
    endpoint_df["height_class"] = endpoint_df.apply(classify_endpoint, axis=1)

    endpoint_df.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "analysis_spec": {
            "terrain_source": "OpenTopoData SRTM90m",
            "ground_like_threshold_m": GROUND_LIKE_THRESHOLD_M,
            "building_search_radius_m": BUILDING_SEARCH_RADIUS_M,
            "building_height_tolerance_m": BUILDING_HEIGHT_TOLERANCE_M,
            "station_adjacent_threshold_m": STATION_ADJACENT_THRESHOLD_M,
        },
        "source_files": {
            "endpoint_cells": str(ENDPOINT_CSV.relative_to(FOR_REVIEW)),
            "station_inventory": str(STATION_CSV.relative_to(FOR_REVIEW)),
            "building_inventory": str(BUILDING_XLSX.relative_to(FOR_REVIEW)),
            "terrain_cache": str(TERRAIN_CACHE_CSV.relative_to(FOR_REVIEW)),
        },
        "overall": summarize_subset(endpoint_df),
        "station_adjacent_subset": summarize_subset(endpoint_df[endpoint_df["station_adjacent_150m"]].copy()),
        "sensitivity_radius_tolerance": build_sensitivity(endpoint_df, building_df),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote endpoint classification table: {OUTPUT_CSV}")
    print(f"Wrote summary JSON: {OUTPUT_JSON}")
    print("Overall classification:", summary["overall"])
    print("Station-adjacent subset:", summary["station_adjacent_subset"])


if __name__ == "__main__":
    main()
