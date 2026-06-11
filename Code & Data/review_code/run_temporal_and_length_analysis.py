from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CLEANED_DATA = ROOT / "data" / "processed" / "full_100m" / "cleaned_data.csv"
TRAJ_STATS = ROOT / "data" / "processed" / "full_100m" / "trajectory_stats.csv"
REVIEW_DATA = ROOT.parent / "review_data"
OUTPUT_JSON = REVIEW_DATA / "source_json" / "temporal_and_length_analysis.json"
OUTPUT_HOURLY = REVIEW_DATA / "diurnal_activity_hourly.csv"
OUTPUT_HALFHOUR = REVIEW_DATA / "diurnal_activity_halfhour.csv"
OUTPUT_HALFHOUR_DAILY = REVIEW_DATA / "diurnal_activity_halfhour_daily.csv"
OUTPUT_LENGTHS = REVIEW_DATA / "trajectory_length_distribution.csv"
OUTPUT_ALTITUDE = REVIEW_DATA / "altitude_point_distribution.csv"

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def slot_label_from_index(slot_idx: int, width_minutes: int) -> str:
    start_minutes = slot_idx * width_minutes
    end_minutes = start_minutes + width_minutes - 1
    sh, sm = divmod(start_minutes, 60)
    eh, em = divmod(end_minutes, 60)
    return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"


def build_diurnal_rows(
    point_counts: dict[int, int],
    date_slot_sets: dict[tuple[str, int], set[str]],
    width_minutes: int,
    n_days: int,
) -> list[dict[str, object]]:
    slots_per_day = 24 * 60 // width_minutes
    rows: list[dict[str, object]] = []
    for slot_idx in range(slots_per_day):
        active_counts = [
            len(order_set)
            for (date_str, idx), order_set in date_slot_sets.items()
            if idx == slot_idx
        ]
        total_points = point_counts.get(slot_idx, 0)
        rows.append(
            {
                "slot_index": slot_idx,
                "slot_label": slot_label_from_index(slot_idx, width_minutes),
                "total_points": total_points,
                "share_of_points": 0.0,  # fill later
                "mean_active_flights_per_day": (sum(active_counts) / n_days) if n_days else 0.0,
                "median_active_flights_per_day": quantile(sorted(active_counts), 0.5) if active_counts else 0.0,
                "peak_active_flights_in_any_day": max(active_counts) if active_counts else 0,
                "days_with_activity": len(active_counts),
            }
        )
    total_points_all = sum(point_counts.values())
    for row in rows:
        row["share_of_points"] = (row["total_points"] / total_points_all) if total_points_all else 0.0
    return rows


def summarize_distances(rows: list[dict[str, object]]) -> dict[str, object]:
    lengths = sorted(float(r["path_length_km"]) for r in rows)
    durations = sorted(float(r["duration_minutes"]) for r in rows)
    od_lengths = sorted(float(r["od_distance_km"]) for r in rows)
    vendors = defaultdict(list)
    for row in rows:
        vendors[str(row["vendor"])].append(float(row["path_length_km"]))

    def basic_summary(values: list[float]) -> dict[str, float]:
        return {
            "mean": sum(values) / len(values) if values else 0.0,
            "median": quantile(values, 0.5),
            "p25": quantile(values, 0.25),
            "p75": quantile(values, 0.75),
            "p90": quantile(values, 0.9),
            "p95": quantile(values, 0.95),
            "max": max(values) if values else 0.0,
        }

    bins_km = [
        ("lt_2km", 0.0, 2.0),
        ("2_5km", 2.0, 5.0),
        ("5_10km", 5.0, 10.0),
        ("10_15km", 10.0, 15.0),
        ("15_20km", 15.0, 20.0),
        ("ge_20km", 20.0, float("inf")),
    ]
    bin_counts: dict[str, int] = {}
    for name, lo, hi in bins_km:
        count = sum(1 for value in lengths if value >= lo and value < hi)
        bin_counts[name] = count

    return {
        "n_trajectories": len(rows),
        "path_length_km": basic_summary(lengths),
        "od_distance_km": basic_summary(od_lengths),
        "duration_minutes": basic_summary(durations),
        "distance_bins": {
            key: {
                "count": value,
                "share": (value / len(rows)) if rows else 0.0,
            }
            for key, value in bin_counts.items()
        },
        "path_length_km_by_vendor": {
            vendor: basic_summary(sorted(values))
            for vendor, values in vendors.items()
        },
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered) if ordered else 0.0,
        "median": quantile(ordered, 0.5),
        "p25": quantile(ordered, 0.25),
        "p75": quantile(ordered, 0.75),
        "p90": quantile(ordered, 0.9),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": max(ordered) if ordered else 0.0,
        "min": min(ordered) if ordered else 0.0,
    }


def main() -> None:
    REVIEW_DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    hourly_points: dict[int, int] = defaultdict(int)
    halfhour_points: dict[int, int] = defaultdict(int)
    hourly_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    halfhour_sets: dict[tuple[str, int], set[str]] = defaultdict(set)

    per_order_state: dict[str, dict[str, object]] = {}
    dates_seen: set[str] = set()

    with CLEANED_DATA.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date_str = row["date"]
            order_id = row["order_id"]
            hour = int(row["hour"])
            minute = int(row["minute"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            dt = datetime.fromisoformat(row["datetime"])
            dates_seen.add(date_str)

            hour_slot = hour
            halfhour_slot = hour * 2 + (minute // 30)
            hourly_points[hour_slot] += 1
            halfhour_points[halfhour_slot] += 1
            hourly_sets[(date_str, hour_slot)].add(order_id)
            halfhour_sets[(date_str, halfhour_slot)].add(order_id)

            state = per_order_state.get(order_id)
            if state is None:
                per_order_state[order_id] = {
                    "vendor": row["vendor"],
                    "num_points": 1,
                    "start_datetime": dt,
                    "end_datetime": dt,
                    "start_lat": lat,
                    "start_lon": lon,
                    "end_lat": lat,
                    "end_lon": lon,
                    "last_lat": lat,
                    "last_lon": lon,
                    "path_length_m": 0.0,
                }
            else:
                state["num_points"] = int(state["num_points"]) + 1
                if dt < state["start_datetime"]:
                    state["start_datetime"] = dt
                    state["start_lat"] = lat
                    state["start_lon"] = lon
                if dt > state["end_datetime"]:
                    state["end_datetime"] = dt
                    state["end_lat"] = lat
                    state["end_lon"] = lon
                prev_lat = float(state["last_lat"])
                prev_lon = float(state["last_lon"])
                state["path_length_m"] = float(state["path_length_m"]) + haversine_m(prev_lat, prev_lon, lat, lon)
                state["last_lat"] = lat
                state["last_lon"] = lon

    hourly_rows = build_diurnal_rows(hourly_points, hourly_sets, width_minutes=60, n_days=len(dates_seen))
    halfhour_rows = build_diurnal_rows(halfhour_points, halfhour_sets, width_minutes=30, n_days=len(dates_seen))

    # Daily half-hour table for scatter overlays
    date_totals: dict[str, int] = defaultdict(int)
    for (date_str, _slot), orders in halfhour_sets.items():
        # actual point counts are computed from cleaned_data, so accumulate separately below
        pass

    halfhour_point_counts_by_date_slot: dict[tuple[str, int], int] = defaultdict(int)
    with CLEANED_DATA.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date_str = row["date"]
            hour = int(row["hour"])
            minute = int(row["minute"])
            halfhour_slot = hour * 2 + (minute // 30)
            halfhour_point_counts_by_date_slot[(date_str, halfhour_slot)] += 1
            date_totals[date_str] += 1

    halfhour_daily_rows: list[dict[str, object]] = []
    for date_str in sorted(dates_seen):
        for slot_idx in range(48):
            point_count = halfhour_point_counts_by_date_slot.get((date_str, slot_idx), 0)
            active_flights = len(halfhour_sets.get((date_str, slot_idx), set()))
            total_points_for_date = date_totals.get(date_str, 0)
            halfhour_daily_rows.append(
                {
                    "date": date_str,
                    "slot_index": slot_idx,
                    "slot_label": slot_label_from_index(slot_idx, 30),
                    "total_points": point_count,
                    "share_of_points_within_date": (point_count / total_points_for_date) if total_points_for_date else 0.0,
                    "active_flights": active_flights,
                }
            )

    altitude_bins = list(range(0, 401, 20))
    altitude_counts = [0] * (len(altitude_bins) - 1)
    altitude_values: list[float] = []

    trajectory_rows: list[dict[str, object]] = []
    with TRAJ_STATS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            order_id = row["order_id"]
            state = per_order_state.get(order_id)
            if state is None:
                continue
            od_distance_m = haversine_m(
                float(state["start_lat"]),
                float(state["start_lon"]),
                float(state["end_lat"]),
                float(state["end_lon"]),
            )
            duration_min = float(row["duration_seconds"]) / 60.0
            trajectory_rows.append(
                {
                    "order_id": order_id,
                    "vendor": row["vendor"],
                    "num_points": int(row["num_points"]),
                    "duration_minutes": round(duration_min, 3),
                    "path_length_km": round(float(state["path_length_m"]) / 1000.0, 6),
                    "od_distance_km": round(od_distance_m / 1000.0, 6),
                    "circuity_ratio": round(
                        (float(state["path_length_m"]) / od_distance_m) if od_distance_m > 0 else 0.0,
                        6,
                    ),
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                }
            )

    with CLEANED_DATA.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            altitude = float(row["altitude"])
            altitude_values.append(altitude)
            placed = False
            for i in range(len(altitude_bins) - 1):
                lo = altitude_bins[i]
                hi = altitude_bins[i + 1]
                if lo <= altitude < hi:
                    altitude_counts[i] += 1
                    placed = True
                    break
            if not placed:
                altitude_counts[-1] += 1

    summary = {
        "analysis_design": (
            "Temporal profile and trajectory-length analysis for the October 2024 main archive only, "
            "excluding the later 2025-2026 follow-up logs."
        ),
        "n_days": len(dates_seen),
        "n_trajectories": len(trajectory_rows),
        "n_points": sum(hourly_points.values()),
        "hourly": {
            "peak_slot_by_points": max(hourly_rows, key=lambda row: row["total_points"]),
            "peak_slot_by_mean_active_flights": max(hourly_rows, key=lambda row: row["mean_active_flights_per_day"]),
        },
        "halfhour": {
            "peak_slot_by_points": max(halfhour_rows, key=lambda row: row["total_points"]),
            "peak_slot_by_mean_active_flights": max(halfhour_rows, key=lambda row: row["mean_active_flights_per_day"]),
        },
        "altitude_point_summary": {
            "n_points": len(altitude_values),
            "summary_m": summarize_values(altitude_values),
            "band_shares": {
                "lt_60m": sum(1 for v in altitude_values if v < 60.0) / len(altitude_values),
                "60_120m": sum(1 for v in altitude_values if 60.0 <= v < 120.0) / len(altitude_values),
                "120_180m": sum(1 for v in altitude_values if 120.0 <= v < 180.0) / len(altitude_values),
                "180_240m": sum(1 for v in altitude_values if 180.0 <= v < 240.0) / len(altitude_values),
                "ge_240m": sum(1 for v in altitude_values if v >= 240.0) / len(altitude_values),
            },
        },
        "trajectory_length_summary": summarize_distances(trajectory_rows),
    }

    with OUTPUT_HOURLY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slot_index",
                "slot_label",
                "total_points",
                "share_of_points",
                "mean_active_flights_per_day",
                "median_active_flights_per_day",
                "peak_active_flights_in_any_day",
                "days_with_activity",
            ],
        )
        writer.writeheader()
        writer.writerows(hourly_rows)

    with OUTPUT_HALFHOUR.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slot_index",
                "slot_label",
                "total_points",
                "share_of_points",
                "mean_active_flights_per_day",
                "median_active_flights_per_day",
                "peak_active_flights_in_any_day",
                "days_with_activity",
            ],
        )
        writer.writeheader()
        writer.writerows(halfhour_rows)

    with OUTPUT_HALFHOUR_DAILY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "slot_index",
                "slot_label",
                "total_points",
                "share_of_points_within_date",
                "active_flights",
            ],
        )
        writer.writeheader()
        writer.writerows(halfhour_daily_rows)

    with OUTPUT_LENGTHS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "order_id",
                "vendor",
                "num_points",
                "duration_minutes",
                "path_length_km",
                "od_distance_km",
                "circuity_ratio",
                "start_time",
                "end_time",
            ],
        )
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with OUTPUT_ALTITUDE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin_index",
                "bin_label",
                "bin_start_m",
                "bin_end_m",
                "count",
                "share_of_points",
            ],
        )
        writer.writeheader()
        total_altitude_points = len(altitude_values)
        for i, count in enumerate(altitude_counts):
            lo = altitude_bins[i]
            hi = altitude_bins[i + 1]
            writer.writerow(
                {
                    "bin_index": i,
                    "bin_label": f"{lo}-{hi}",
                    "bin_start_m": lo,
                    "bin_end_m": hi,
                    "count": count,
                    "share_of_points": (count / total_altitude_points) if total_altitude_points else 0.0,
                }
            )

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
