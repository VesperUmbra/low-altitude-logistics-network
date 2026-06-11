from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
RAW_CSV = WORKSPACE_ROOT / "for_review" / "ods_sq_flight_dynamic_data.csv"
GRID_INFO_JSON = ROOT / "data" / "processed" / "full_100m" / "grid" / "grid_info.json"
TEMP_DIR = ROOT / "data" / "processed" / "grid_sensitivity"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = WORKSPACE_ROOT / "for_review" / "review_data" / "grid_size_sensitivity.csv"
OUTPUT_JSON = WORKSPACE_ROOT / "for_review" / "review_data" / "source_json" / "grid_size_sensitivity.json"

GRID_SIZES = [50, 80, 100, 150, 200, 300]

MIN_LON = 113.31
MAX_LON = 114.25
MIN_LAT = 22.31
MAX_LAT = 22.87

LAT_CENTER = (MIN_LAT + MAX_LAT) / 2.0
METERS_PER_DEGREE_LAT = 111000.0
METERS_PER_DEGREE_LON = 111000.0 * math.cos(math.radians(LAT_CENTER))
TIME_WINDOW_S = int(json.loads(GRID_INFO_JSON.read_text(encoding="utf-8-sig")).get("time_window_s", 300))
CLEANED_POINTS_CSV = TEMP_DIR / f"cleaned_valid_points_minimal_{TIME_WINDOW_S}s.csv"


def parse_window_key(time_value: str) -> str | None:
    text = str(time_value).strip()
    if not text:
        return None
    text = text.split(".")[0].zfill(14)
    try:
        year = int(text[0:4])
        month = int(text[4:6])
        day = int(text[6:8])
        hour = int(text[8:10])
        minute = int(text[10:12])
        second = int(text[12:14])
        datetime(year, month, day, hour, minute, second)
    except Exception:
        return None
    total_seconds = hour * 3600 + minute * 60 + second
    floored_seconds = (total_seconds // TIME_WINDOW_S) * TIME_WINDOW_S
    floor_hour = floored_seconds // 3600
    floor_minute = (floored_seconds % 3600) // 60
    floor_second = floored_seconds % 60
    return f"{year:04d}{month:02d}{day:02d}{floor_hour:02d}{floor_minute:02d}{floor_second:02d}"


def clean_point(row: dict[str, str]) -> tuple[str, float, float, float] | None:
    try:
        speed = float(row["speed"])
        altitude = float(row["altitude"])
        lon = float(row["longitude"])
        lat = float(row["latitude"])
    except Exception:
        return None

    if not (0.0 <= speed <= 40.0):
        return None
    if not (0.0 <= altitude <= 500.0):
        return None
    if not (MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT):
        return None

    window_key = parse_window_key(row["time"])
    if window_key is None:
        return None
    return window_key, speed, lon, lat


def ensure_cleaned_points() -> dict[str, int]:
    if CLEANED_POINTS_CSV.exists():
        counts = {
            "cleaned_points": 0,
            "valid_orders": 0,
            "noncontiguous_order_reuse": 0,
        }
        seen_orders: set[str] = set()
        with CLEANED_POINTS_CSV.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                counts["cleaned_points"] += 1
                order_id = row["order_id"]
                if order_id not in seen_orders:
                    seen_orders.add(order_id)
                    counts["valid_orders"] += 1
        return counts

    order_counts: dict[str, int] = defaultdict(int)
    seen_orders: set[str] = set()
    previous_order: str | None = None
    noncontiguous_order_reuse = 0

    with RAW_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for idx, row in enumerate(reader, start=1):
            order_id = str(row["order_id"])
            if order_id != previous_order:
                if order_id in seen_orders:
                    noncontiguous_order_reuse += 1
                else:
                    seen_orders.add(order_id)
                previous_order = order_id
            point = clean_point(row)
            if point is not None:
                order_counts[order_id] += 1
            if idx % 500000 == 0:
                print(f"[count] processed {idx:,} raw rows")

    valid_order_set = {order_id for order_id, count in order_counts.items() if count >= 3}
    valid_orders = len(valid_order_set)
    cleaned_points = 0

    with RAW_CSV.open("r", encoding="utf-8-sig", newline="") as source, CLEANED_POINTS_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            target,
            fieldnames=["order_id", "window_key", "speed", "longitude", "latitude"],
        )
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            order_id = str(row["order_id"])
            if order_id not in valid_order_set:
                continue
            point = clean_point(row)
            if point is not None:
                window_key, speed, lon, lat = point
                writer.writerow(
                    {
                        "order_id": order_id,
                        "window_key": window_key,
                        "speed": f"{speed:.6f}",
                        "longitude": f"{lon:.6f}",
                        "latitude": f"{lat:.6f}",
                    }
                )
            if idx % 500000 == 0:
                print(f"[write] processed {idx:,} raw rows")

        cleaned_points = sum(order_counts[order_id] for order_id in valid_order_set)

    return {
        "cleaned_points": cleaned_points,
        "valid_orders": valid_orders,
        "noncontiguous_order_reuse": noncontiguous_order_reuse,
    }


def map_to_grid(lon: float, lat: float, grid_size: int) -> tuple[int, int]:
    n_rows = int(math.ceil(((MAX_LAT - MIN_LAT) * METERS_PER_DEGREE_LAT) / grid_size))
    n_cols = int(math.ceil(((MAX_LON - MIN_LON) * METERS_PER_DEGREE_LON) / grid_size))
    row = int(((lat - MIN_LAT) * METERS_PER_DEGREE_LAT) / grid_size)
    col = int(((lon - MIN_LON) * METERS_PER_DEGREE_LON) / grid_size)
    row = max(0, min(n_rows - 1, row))
    col = max(0, min(n_cols - 1, col))
    return row, col


def relative_risk(a: int, b: int, c: int, d: int) -> float:
    if a <= 0 or c <= 0 or (a + b) <= 0 or (c + d) <= 0:
        return float("nan")
    return (a / (a + b)) / (c / (c + d))


def piecewise_breakpoint_from_bins(
    binned: list[tuple[float, int, float]],
) -> tuple[float, float, float, int]:
    if not binned:
        return float("nan"), float("nan"), float("nan"), 0

    x = np.asarray([item[0] for item in binned], dtype=float)
    weights = np.asarray([item[1] for item in binned], dtype=float)
    y = np.asarray([item[2] for item in binned], dtype=float)

    best_breakpoint = float(x[len(x) // 2])
    best_objective = float("inf")
    max_breakpoint = min(len(binned) - 5, 50)

    for breakpoint_idx in range(1, max_breakpoint):
        x1 = x[:breakpoint_idx]
        x2 = x[breakpoint_idx:]
        y1 = y[:breakpoint_idx]
        y2 = y[breakpoint_idx:]
        if len(x1) < 3 or len(x2) < 3:
            continue

        reg1 = np.polyfit(x1, y1, 1)
        reg2 = np.polyfit(x2, y2, 1)
        pred1 = reg1[0] * x1 + reg1[1]
        pred2 = reg2[0] * x2 + reg2[1]
        objective = float(np.sum((y1 - pred1) ** 2) + np.sum((y2 - pred2) ** 2))
        if objective < best_objective:
            best_objective = objective
            best_breakpoint = float(x[breakpoint_idx])

    free_speeds = [speed for density, _, speed in binned if density < best_breakpoint]
    congested_speeds = [speed for density, _, speed in binned if density >= best_breakpoint]
    speed_free = float(sum(free_speeds) / len(free_speeds)) if free_speeds else float("nan")
    speed_cong = float(sum(congested_speeds) / len(congested_speeds)) if congested_speeds else float("nan")
    speed_drop = (
        float((speed_free - speed_cong) / speed_free * 100.0)
        if speed_free and math.isfinite(speed_free) and math.isfinite(speed_cong)
        else float("nan")
    )
    return best_breakpoint, speed_free, speed_drop, len(binned)


def analyze_grid_size(grid_size: int) -> dict[str, float]:
    n_rows = int(math.ceil(((MAX_LAT - MIN_LAT) * METERS_PER_DEGREE_LAT) / grid_size))
    n_cols = int(math.ceil(((MAX_LON - MIN_LON) * METERS_PER_DEGREE_LON) / grid_size))
    total_cells = n_rows * n_cols

    cell_points: dict[str, int] = defaultdict(int)
    cell_windows_count: dict[tuple[int, int, str], int] = defaultdict(int)
    cell_windows_speed_sum: dict[tuple[int, int, str], float] = defaultdict(float)

    with CLEANED_POINTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            speed = float(row["speed"])
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            window_key = row["window_key"]
            grid_row, grid_col = map_to_grid(lon, lat, grid_size)
            cell_id = f"{grid_row}_{grid_col}"
            cell_points[cell_id] += 1
            key = (grid_row, grid_col, window_key)
            cell_windows_count[key] += 1
            cell_windows_speed_sum[key] += speed
            if idx % 1000000 == 0:
                print(f"[grid {grid_size}] processed {idx:,} cleaned points")

    active_cells = len(cell_points)
    ranked_cells = sorted(cell_points.items(), key=lambda item: (-item[1], item[0]))
    top10_count = max(1, int(active_cells * 0.10))
    backbone_cells = {cell_id for cell_id, _ in ranked_cells[:top10_count]}
    total_points = sum(cell_points.values())
    top10_share = sum(value for _, value in ranked_cells[:top10_count]) / total_points

    density_bins: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for key, count in cell_windows_count.items():
        mean_speed = cell_windows_speed_sum[key] / count
        density_bins[count][0] += 1.0
        density_bins[count][1] += mean_speed

    binned = []
    for density in sorted(density_bins):
        sample_count = int(density_bins[density][0])
        if sample_count < 10:
            continue
        speed_mean = density_bins[density][1] / sample_count
        binned.append((density + 0.5, sample_count, speed_mean))

    rho_star, speed_free, speed_drop, n_bins = piecewise_breakpoint_from_bins(binned)

    hotspot_cells: set[str] = set()
    a = b = c = d = 0
    for (grid_row, grid_col, _window_key), count in cell_windows_count.items():
        cell_id = f"{grid_row}_{grid_col}"
        is_backbone = cell_id in backbone_cells
        is_congested = count >= rho_star if math.isfinite(rho_star) else False
        if is_congested:
            hotspot_cells.add(cell_id)
        if is_backbone and is_congested:
            a += 1
        elif is_backbone and not is_congested:
            b += 1
        elif (not is_backbone) and is_congested:
            c += 1
        else:
            d += 1

    rr = relative_risk(a, b, c, d)

    return {
        "grid_size_m": grid_size,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_cells": total_cells,
        "active_cells": active_cells,
        "active_share_full_grid": active_cells / total_cells,
        "spatiotemporal_samples": len(cell_windows_count),
        "top10_traffic_share": top10_share,
        "rho_star": rho_star,
        "speed_free": speed_free,
        "speed_drop_percent": speed_drop,
        "hotspot_cells": len(hotspot_cells),
        "hotspot_share_active_cells": len(hotspot_cells) / active_cells if active_cells else float("nan"),
        "rr_backbone_vs_nonbackbone": rr,
        "n_binned_density_points": n_bins,
        "corridor_congested": a,
        "corridor_not_congested": b,
        "noncorridor_congested": c,
        "noncorridor_not_congested": d,
    }


def write_outputs(results: list[dict[str, float]], cleaned_summary: dict[str, int]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grid_size_m",
        "active_cells",
        "active_share_full_grid",
        "spatiotemporal_samples",
        "top10_traffic_share",
        "rho_star",
        "speed_drop_percent",
        "hotspot_cells",
        "hotspot_share_active_cells",
        "rr_backbone_vs_nonbackbone",
        "n_binned_density_points",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_design": (
            "Grid-size sensitivity for the canonical point-count pipeline using the same trajectory-cleaning rules, "
            "2-minute windows, top-10% backbone definition, piecewise speed-density breakpoint and full-sample RR."
        ),
        "input_raw_csv": str(RAW_CSV),
        "cleaned_point_summary": cleaned_summary,
        "grid_size_results": results,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cleaned_summary = ensure_cleaned_points()
    print("cleaned summary:", cleaned_summary)
    results = []
    for grid_size in GRID_SIZES:
        print(f"running grid-size sensitivity for {grid_size} m")
        result = analyze_grid_size(grid_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        results.append(result)
    write_outputs(results, cleaned_summary)


if __name__ == "__main__":
    main()
