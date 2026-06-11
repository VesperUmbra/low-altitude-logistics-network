from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_figure


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
FOR_REVIEW = WORKSPACE_ROOT / "for_review"
REVIEW_DATA = FOR_REVIEW / "review_data"
FIGURES = ROOT.parent / "SI_output"
PROCESSED_DIR = FOR_REVIEW / "review_code" / "data" / "processed" / "full_100m"

HALFHOUR_CSV = REVIEW_DATA / "diurnal_activity_halfhour.csv"
HALFHOUR_DAILY_CSV = REVIEW_DATA / "diurnal_activity_halfhour_daily.csv"
ALTITUDE_CSV = REVIEW_DATA / "altitude_point_distribution.csv"
LENGTH_CSV = REVIEW_DATA / "trajectory_length_distribution.csv"
SPEED_PHASE_CSV = REVIEW_DATA / "preview_speed_phase_by_distance.csv"
GRID_INFO_JSON = PROCESSED_DIR / "grid" / "grid_info.json"
CLEANED_CSV = PROCESSED_DIR / "cleaned_data.csv"


DIST_LABELS = ["0-1", "1-3", "3-5", "5-10", "10-20", "20-80"]
LENGTH_BINS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 80]
DURATION_BINS = [0, 2, 5, 7.5, 10, 12.5, 15, 20, 50]


def qtile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def binned_share_stats(arrays: list[list[int]]) -> tuple[list[float], list[float], list[float]]:
    if not arrays:
        return [], [], []
    n_bins = len(arrays[0])
    per_bin: list[list[float]] = [[] for _ in range(n_bins)]
    for arr in arrays:
        total = sum(arr)
        if total <= 0:
            continue
        for i, count in enumerate(arr):
            per_bin[i].append(count / total * 100.0)
    means = [sum(v) / len(v) if v else 0.0 for v in per_bin]
    q25 = [qtile(v, 0.25) if v else 0.0 for v in per_bin]
    q75 = [qtile(v, 0.75) if v else 0.0 for v in per_bin]
    return means, q25, q75


def load_halfhour():
    halfhour = pd.read_csv(HALFHOUR_CSV)
    halfhour_daily = pd.read_csv(HALFHOUR_DAILY_CSV)
    halfhour = halfhour.loc[halfhour["slot_index"].between(12, 43)].copy()
    halfhour_daily = halfhour_daily.loc[halfhour_daily["slot_index"].between(12, 43)].copy()
    return halfhour, halfhour_daily


def altitude_daily_stats():
    altitude_rows = pd.read_csv(ALTITUDE_CSV)
    grid_info = json.loads(GRID_INFO_JSON.read_text(encoding="utf-8"))
    starts = altitude_rows["bin_start_m"].astype(int).tolist()
    ends = altitude_rows["bin_end_m"].astype(int).tolist()
    daily: dict[str, list[int]] = defaultdict(lambda: [0] * len(starts))
    for chunk in pd.read_csv(CLEANED_CSV, usecols=["date", "altitude"], chunksize=400_000):
        for row in chunk.itertuples(index=False):
            day = str(row.date)
            alt = float(row.altitude)
            for i, (lo, hi) in enumerate(zip(starts, ends)):
                if lo <= alt < hi or (i == len(starts) - 1 and alt <= hi):
                    daily[day][i] += 1
                    break
    means, q25, q75 = binned_share_stats(list(daily.values()))
    return starts, means, q25, q75, grid_info


def length_daily_stats():
    length_rows = pd.read_csv(LENGTH_CSV, usecols=["path_length_km", "start_time"])
    daily: dict[str, list[int]] = defaultdict(lambda: [0] * (len(LENGTH_BINS) - 1))
    for row in length_rows.itertuples(index=False):
        value = float(row.path_length_km)
        day = str(row.start_time)[:10]
        idx = len(LENGTH_BINS) - 2
        for i in range(len(LENGTH_BINS) - 1):
            if LENGTH_BINS[i] <= value < LENGTH_BINS[i + 1]:
                idx = i
                break
        daily[day][idx] += 1
    means, q25, q75 = binned_share_stats(list(daily.values()))
    return LENGTH_BINS, means, q25, q75


def duration_daily_stats():
    duration_rows = pd.read_csv(LENGTH_CSV, usecols=["duration_minutes", "start_time"])
    daily: dict[str, list[int]] = defaultdict(lambda: [0] * (len(DURATION_BINS) - 1))
    for row in duration_rows.itertuples(index=False):
        value = float(row.duration_minutes)
        day = str(row.start_time)[:10]
        idx = len(DURATION_BINS) - 2
        for i in range(len(DURATION_BINS) - 1):
            if DURATION_BINS[i] <= value < DURATION_BINS[i + 1]:
                idx = i
                break
        daily[day][idx] += 1
    means, q25, q75 = binned_share_stats(list(daily.values()))
    return DURATION_BINS, means, q25, q75


def duration_by_distance_stats(length_rows: pd.DataFrame, bins: list[int]) -> pd.DataFrame:
    work = length_rows.copy()
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    work["dist_bin"] = pd.cut(
        work["path_length_km"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return (
        work.groupby("dist_bin", observed=False)
        .agg(
            median_duration_min=("duration_minutes", "median"),
            q25_duration_min=("duration_minutes", lambda s: float(s.quantile(0.25))),
            q75_duration_min=("duration_minutes", lambda s: float(s.quantile(0.75))),
        )
        .reset_index()
    )


def speed_violin_panel(ax: plt.Axes, speed_phase: pd.DataFrame) -> None:
    centers = np.arange(len(DIST_LABELS))
    offset = 0.18
    width = 0.30
    colors = {
        "Endpoint-adjacent": PALETTE["orange"],
        "Away from endpoint": PALETTE["navy"],
    }

    for phase, dx in [("Endpoint-adjacent", -offset), ("Away from endpoint", offset)]:
        datasets = []
        positions = []
        for i, label in enumerate(DIST_LABELS):
            vals = speed_phase.loc[
                (speed_phase["dist_bin"] == label) & (speed_phase["phase"] == phase),
                "point_speed",
            ].to_numpy()
            if len(vals) == 0:
                continue
            datasets.append(vals)
            positions.append(centers[i] + dx)
        if not datasets:
            continue
        vp = ax.violinplot(
            datasets,
            positions=positions,
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(colors[phase])
            body.set_edgecolor(colors[phase])
            body.set_alpha(0.23 if phase == "Endpoint-adjacent" else 0.18)

        for pos, vals in zip(positions, datasets):
            q25, med, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
            ax.vlines(pos, q25, q75, color=colors[phase], lw=2.0)
            ax.scatter([pos], [med], color=colors[phase], s=18, zorder=3)

    ax.set_xticks(centers)
    ax.set_xticklabels(DIST_LABELS)
    ax.set_xlabel("Trajectory distance bin (km)")
    ax.set_ylabel("Flight speed (m/s)")
    ax.grid(True, axis="y", linestyle=":", color=PALETTE["grey"], alpha=0.85)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE["orange"], lw=2.5, label="Endpoint-adjacent"),
            plt.Line2D([0], [0], color=PALETTE["navy"], lw=2.5, label="Away from endpoint"),
        ],
        frameon=False,
        loc="upper left",
    )


def main() -> None:
    apply_style()
    halfhour, halfhour_daily = load_halfhour()
    alt_starts, alt_mean, alt_q25, alt_q75, _ = altitude_daily_stats()
    len_bins, len_mean, len_q25, len_q75 = length_daily_stats()
    dur_bins, dur_mean, dur_q25, dur_q75 = duration_daily_stats()
    length_rows = pd.read_csv(LENGTH_CSV, usecols=["path_length_km", "duration_minutes"])
    speed_phase = pd.read_csv(SPEED_PHASE_CSV)
    duration_stats = duration_by_distance_stats(length_rows, len_bins)

    fig = plt.figure(figsize=(13.2, 7.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 11, height_ratios=[1.03, 1.0])
    ax1 = fig.add_subplot(gs[0, 0:6])
    ax5 = fig.add_subplot(gs[0, 7:11])
    ax2 = fig.add_subplot(gs[1, 0:3])
    ax3 = fig.add_subplot(gs[1, 4:7])
    ax4 = fig.add_subplot(gs[1, 8:11])

    # a: time
    x = np.arange(len(halfhour))
    point_share = halfhour["share_of_points"].to_numpy() * 100.0
    mean_active = halfhour["mean_active_flights_per_day"].to_numpy()
    active_p25 = []
    active_p75 = []
    for slot in halfhour["slot_index"]:
        vals = sorted(
            halfhour_daily.loc[halfhour_daily["slot_index"] == slot, "active_flights"].astype(float).tolist()
        )
        active_p25.append(qtile(vals, 0.25))
        active_p75.append(qtile(vals, 0.75))

    line1 = ax1.plot(x, point_share, color=PALETTE["blue"], lw=2.3, label="Mean point share")[0]
    ax1.set_ylabel("Telemetry points (%)")
    ax1.set_xlabel("Time of day")
    ax1.set_title("Half-hour activity", fontsize=10.3)
    tick_pos = [i for i, slot in enumerate(halfhour["slot_index"]) if int(slot) % 4 == 0]
    tick_lab = [f"{int(slot)//2:02d}:00" for slot in halfhour.iloc[tick_pos]["slot_index"]]
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_lab)
    ax1.grid(True, axis="y", linestyle=":", color=PALETTE["grey"], alpha=0.85)
    ax1.set_ylim(bottom=0)
    ax1b = ax1.twinx()
    band = ax1b.fill_between(x, active_p25, active_p75, color=PALETTE["orange"], alpha=0.20, label="Active-trajectory IQR")
    line2 = ax1b.plot(x, mean_active, color=PALETTE["orange"], lw=2.2, label="Mean active trajectories")[0]
    ax1b.set_ylabel("Mean active trajectories/day", labelpad=6)
    ax1b.set_ylim(bottom=0)
    ax1b.tick_params(axis="y", pad=2)
    ax1.legend(
        [line1, band, line2],
        ["Mean point share", "Active-trajectory IQR", "Mean active trajectories"],
        frameon=False,
        loc="upper left",
        ncol=1,
    )
    add_panel_label(ax1, "a")

    # b: altitude
    xa = np.arange(len(alt_mean))
    ax2.fill_between(xa, alt_q25, alt_q75, color=PALETTE["teal"], alpha=0.18, label="Daily IQR")
    ax2.plot(xa, alt_mean, color=PALETTE["teal"], lw=2.3, label="Mean daily share")
    ax2.set_title("Altitude profile", fontsize=10.3)
    ax2.set_xlabel("Altitude bin (m)")
    ax2.set_ylabel("Daily point share (%)")
    tick_pos2 = [i for i, start in enumerate(alt_starts) if start in [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]]
    tick_lab2 = [str(alt_starts[i]) for i in tick_pos2]
    ax2.set_xticks(tick_pos2)
    ax2.set_xticklabels(tick_lab2)
    ax2.grid(True, axis="y", linestyle=":", color=PALETTE["grey"], alpha=0.85)
    ax2.set_ylim(bottom=0)
    ax2.legend(frameon=False, loc="upper right")
    add_panel_label(ax2, "b")

    # c: length frequency
    xl = np.arange(len(len_mean))
    ax3.fill_between(xl, len_q25, len_q75, color=PALETTE["grey_dark"], alpha=0.18, label="Daily IQR")
    ax3.plot(xl, len_mean, color=PALETTE["grey_dark"], lw=2.3, label="Mean daily share")
    ax3.set_title("Trajectory length", fontsize=10.3)
    ax3.set_xlabel("Path length bin (km)")
    ax3.set_ylabel("Daily trajectory share (%)")
    tick_lab3 = [f"{len_bins[i]}-{len_bins[i+1]}" for i in range(len(len_mean))]
    ax3.set_xticks(xl)
    ax3.set_xticklabels(tick_lab3)
    ax3.grid(True, axis="y", linestyle=":", color=PALETTE["grey"], alpha=0.85)
    ax3.set_ylim(bottom=0)
    ax3.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE["grey_dark"], lw=2.3, label="Mean daily share"),
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["grey_dark"], alpha=0.18, edgecolor="none", label="Daily IQR"),
        ],
        frameon=False,
        loc="upper right",
    )
    add_panel_label(ax3, "c")

    # d: duration frequency
    xd = np.arange(len(dur_mean))
    ax4.fill_between(xd, dur_q25, dur_q75, color=PALETTE["purple"], alpha=0.18, label="Daily IQR")
    ax4.plot(xd, dur_mean, color=PALETTE["purple"], lw=2.3, label="Mean daily share")
    ax4.set_title("Flight duration", fontsize=10.3)
    ax4.set_xlabel("Flight-duration bin (min)")
    ax4.set_ylabel("Daily trajectory share (%)")
    tick_lab4 = []
    for i in range(len(dur_mean)):
        lo = dur_bins[i]
        hi = dur_bins[i + 1]
        lo_txt = f"{lo:g}"
        hi_txt = f"{hi:g}"
        tick_lab4.append(f"{lo_txt}-{hi_txt}")
    ax4.set_xticks(xd)
    ax4.set_xticklabels(tick_lab4)
    ax4.grid(True, axis="y", linestyle=":", color=PALETTE["grey"], alpha=0.85)
    ax4.set_ylim(bottom=0)
    ax4.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE["purple"], lw=2.3, label="Mean daily share"),
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["purple"], alpha=0.18, edgecolor="none", label="Daily IQR"),
        ],
        frameon=False,
        loc="upper right",
    )
    add_panel_label(ax4, "d")

    # e: speed violin
    speed_violin_panel(ax5, speed_phase)
    ax5.set_title("Speed by distance and endpoint proximity", fontsize=10.3)
    add_panel_label(ax5, "e")

    save_figure(fig, "S16_preview_time_altitude_length")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "S16_preview_time_altitude_length.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
