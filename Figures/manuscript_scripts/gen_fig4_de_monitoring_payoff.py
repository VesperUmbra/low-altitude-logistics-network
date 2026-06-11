# 产出Fig 4 panel d,e
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_panel

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"

PAYOFF_CSV = REVIEW_DATA_DIR / "targeted_monitoring_payoff.csv"
RANDOM_CSV = REVIEW_DATA_DIR / "targeted_monitoring_random_baseline.csv"


def pct(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def main() -> None:
    apply_style()

    payoff = pd.read_csv(PAYOFF_CSV)
    random_baseline = pd.read_csv(RANDOM_CSV)

    top = payoff[payoff["strategy"] == "top_traffic_cells"].copy().sort_values("budget_share_active_cells")
    station = payoff[payoff["strategy"].str.startswith("real_station_")].copy()
    station = station.assign(
        anchor_label=station["strategy"]
        .map(
            {
                "real_station_exact_anchor_cells": "Exact station cells",
                "real_station_buffer1_anchor_cells": "Station +1 cell",
                "real_station_buffer2_anchor_cells": "Station +2 cells",
            }
        )
    )
    station = station.sort_values("budget_share_active_cells")
    random_key_lookup = {
        "real_station_exact_anchor_cells": "exact_anchor_cells",
        "real_station_buffer1_anchor_cells": "buffer1_anchor_cells",
        "real_station_buffer2_anchor_cells": "buffer2_anchor_cells",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    ax = axes[0]
    x = pct(top["budget_share_active_cells"])
    y = pct(top["captured_exceedance_share"])
    ax.plot(x, y, color=PALETTE["navy"], lw=2.5, marker="o", ms=4.6)

    station_colors = {
        "real_station_exact_anchor_cells": PALETTE["red"],
        "real_station_buffer1_anchor_cells": PALETTE["orange"],
        "real_station_buffer2_anchor_cells": PALETTE["purple"],
    }
    station_x = pct(station["budget_share_active_cells"])
    station_y = pct(station["captured_exceedance_share"])
    ax.plot(station_x, station_y, color=PALETTE["gold"], lw=1.4, ls=":", alpha=0.88, zorder=2)
    for _, row in station.iterrows():
        sx = pct([row["budget_share_active_cells"]])[0]
        sy = pct([row["captured_exceedance_share"]])[0]
        random_key = random_key_lookup[row["strategy"]]
        rand_row = random_baseline[random_baseline["strategy"] == random_key].iloc[0]
        rand_mean = rand_row["mean_random_exceedance_share"] * 100.0
        rand_p95 = rand_row["p95_random_exceedance_share"] * 100.0

        ax.vlines(sx, rand_mean, rand_p95, color=PALETTE["grey"], lw=1.0, alpha=0.8, zorder=1)
        ax.scatter(sx, rand_mean, s=22, facecolors="white", edgecolors=PALETTE["grey"], linewidth=0.8, zorder=2)
        ax.scatter(
            sx,
            sy,
            s=50,
            color=station_colors[row["strategy"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

    baseline_x = np.linspace(0, 5.2, 200)
    ax.plot(baseline_x, baseline_x, ls="--", lw=1.1, color=PALETTE["grey"], alpha=0.95)

    station_b1 = station[station["strategy"] == "real_station_buffer1_anchor_cells"].iloc[0]
    ax.set_xlabel("Active cells monitored (%)")
    ax.set_ylabel("State-sample samples captured (%)")
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 100)
    ax.grid(axis="y")
    add_panel_label(ax, "a")
    legend_a = [
        mlines.Line2D([], [], color=PALETTE["navy"], lw=2.5, marker="o", markersize=4.6, label="Top traffic cells"),
        mlines.Line2D([], [], color=PALETTE["red"], marker="o", linestyle="None", markersize=6.2, label="Station exact"),
        mlines.Line2D([], [], color=PALETTE["orange"], marker="o", linestyle="None", markersize=6.2, label="Station +1 cell"),
        mlines.Line2D([], [], color=PALETTE["purple"], marker="o", linestyle="None", markersize=6.2, label="Station +2 cells"),
        mlines.Line2D([], [], color=PALETTE["grey"], lw=1.1, ls="--", label="Random same-area baseline"),
    ]
    ax.legend(handles=legend_a, frameon=False, loc="lower right", ncol=2, handlelength=1.8, columnspacing=0.95)

    ax = axes[1]
    ax.plot(
        pct(top["budget_share_active_cells"]),
        top["lift_vs_random_exceedance"],
        color=PALETTE["navy"],
        lw=2.5,
        marker="o",
        ms=4.8,
    )

    station_rand = random_baseline.set_index("strategy")
    station_lift = station["strategy"].map(lambda value: station_rand.loc[random_key_lookup[value], "observed_over_random_mean"])
    ax.plot(station_x, station_lift, color=PALETTE["gold"], lw=1.3, ls=":", alpha=0.9, zorder=2)
    for _, row in station.iterrows():
        ax.scatter(
            pct([row["budget_share_active_cells"]])[0],
            station_rand.loc[random_key_lookup[row["strategy"]], "observed_over_random_mean"],
            s=50,
            color=station_colors[row["strategy"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax.axhline(1.0, ls="--", lw=1.1, color=PALETTE["grey"])
    ax.set_xlabel("Active cells monitored (%)")
    ax.set_ylabel("Lift over random baseline")
    ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.grid(axis="y", which="major")
    add_panel_label(ax, "b")

    save_panel(fig, axes[0], "manuscript_fig4d_monitoring_capture_curves")
    save_panel(fig, axes[1], "manuscript_fig4e_lift_over_random_baseline")
    plt.close(fig)
    print("Saved manuscript Figure 4 monitoring panels")


if __name__ == "__main__":
    main()
