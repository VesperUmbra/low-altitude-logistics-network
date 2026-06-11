# 产出Fig4 panel a
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_panel


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA = WORKSPACE_ROOT / "for_review" / "review_data"
FIGURES = ROOT.parent / "output"
PROCESSED_DIR = WORKSPACE_ROOT / "for_review"/ "review_code" / "data" / "processed" / "full_100m"

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
                "mean_point_speed",
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
    halfhour = halfhour.reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    daily = halfhour_daily.copy()
    active_daily = (
        daily.groupby(["slot_index", "slot_label"], as_index=False)["active_flights"]
        .agg(q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index(drop=True)
    )
    x = np.arange(len(halfhour))
    point_share = halfhour["share_of_points"].to_numpy() * 100.0
    mean_active = halfhour["mean_active_flights_per_day"].to_numpy()
    left_color = PALETTE["blue"]
    right_color = PALETTE["orange"]

    ax1.axvspan(2, 7, color=PALETTE["grey_light"], alpha=0.55, zorder=0)
    ax1.axvspan(23, 27, color=PALETTE["grey_light"], alpha=0.55, zorder=0)
    line1 = ax1.plot(x, point_share, color=left_color, lw=2.0, label="Aerial point share")[0]
    ax1.set_ylabel("Aerial point share (%)", color=left_color)
    ax1.set_xlabel("Time of day")
    tick_labels = ["06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"]
    tick_pos = halfhour.index[halfhour["slot_label"].str.slice(0, 5).isin(tick_labels)]
    tick_lab = halfhour.loc[tick_pos, "slot_label"].str.slice(0, 5)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_lab)
    ax1.grid(True, axis="y", linestyle=":", color=left_color, alpha=0.26)
    ax1.set_ylim(bottom=0)
    ax1.set_yticks(np.arange(0, 11, 2))
    ax1.tick_params(axis="y", colors=left_color)
    ax1.spines["left"].set_color(left_color)
    ax1b = ax1.twinx()
    ax1b.grid(True, axis="y", linestyle="--", color=right_color, alpha=0.18)
    ax1b.patch.set_visible(False)
    ax1b.set_ylim(0, mean_active.max() * 1.15)
    ax1b.set_yticks(np.arange(0, 251, 50))
    ax1b.set_ylabel("Mean active flights/day", labelpad=6, color=right_color)
    band = ax1b.fill_between(
        x,
        active_daily["q25"],
        active_daily["q75"],
        color=right_color,
        alpha=0.18,
        label="Active-trajectory IQR",
    )
    line2 = ax1b.plot(
        x,
        mean_active,
        color=right_color,
        lw=1.85,
        label="Mean active flights/day",
    )[0]
    for side in ("top", "left", "bottom"):
        ax1b.spines[side].set_visible(False)
    ax1b.spines["right"].set_visible(True)
    ax1b.spines["right"].set_color(right_color)
    ax1b.tick_params(axis="y", direction="out", length=3.0, width=0.8, colors=right_color, right=True, labelright=True)
    ax1.legend(
        [
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["grey_light"], alpha=0.55, edgecolor="none"),
            line1,
            band,
            line2,
        ],
        ["Ground commuting windows", "Aerial point share", "Active-trajectory IQR", "Mean active flights/day"],
        frameon=False,
        loc="upper left",
        ncol=1,
        fontsize=6.0,
        handlelength=1.1,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.25,
    )
    add_panel_label(ax1, "a")
    save_panel(fig, [ax1, ax1b], "manuscript_fig4a_half_hour_activity")
    plt.close(fig)
    print("Saved manuscript Figure 4 half-hour panel")


if __name__ == "__main__":
    main()
