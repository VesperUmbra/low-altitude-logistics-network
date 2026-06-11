# 产出Fig 3 panel a,b,c
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_panel


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"

BINNED_CSV = REVIEW_DATA_DIR / "binned_speed_density_relations.csv"
THRESHOLD_CSV = REVIEW_DATA_DIR / "figure_threshold_subset_summary.csv"


def main() -> None:
    apply_style()

    binned = pd.read_csv(BINNED_CSV)
    subsets = pd.read_csv(THRESHOLD_CSV)
    subsets = subsets.iloc[[0, 1, 2, 3]].copy()
    subsets["display_label"] = [
        "Full sample",
        "Exclude interface windows",
        "Exclude endpoint +1",
        "Away from endpoints,\naltitude >=180 m",
    ]

    rho_star = float(binned["rho_star_reference"].iloc[0])
    binned["sample_share_pct"] = binned["sample_count"] / binned["sample_count"].sum() * 100.0
    high_mask = binned["density_mean"] >= rho_star
    share_above = binned.loc[high_mask, "sample_share_pct"].sum()

    curve_xmax = float(binned["density_center"].max()) + 2.0

    fig = plt.figure(figsize=(10.8, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.76, 0.86, 1.0])

    gs_left = gs[0, 0].subgridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.05)
    ax_curve = fig.add_subplot(gs_left[0, 0])
    ax_hist = fig.add_subplot(gs_left[1, 0], sharex=ax_curve)
    ax_rho = fig.add_subplot(gs[0, 1])
    ax_rr = fig.add_subplot(gs[0, 2])

    ax_curve.axvspan(rho_star, curve_xmax, color="#f8d7da", alpha=0.24, zorder=0)
    ax_curve.plot(
        binned.loc[~high_mask, "density_center"],
        binned.loc[~high_mask, "speed_mean"],
        color=PALETTE["navy"],
        lw=2.25,
    )
    ax_curve.plot(
        binned.loc[high_mask, "density_center"],
        binned.loc[high_mask, "speed_mean"],
        color=PALETTE["red"],
        lw=2.25,
    )
    ax_curve.scatter(
        binned.loc[~high_mask, "density_center"],
        binned.loc[~high_mask, "speed_mean"],
        s=7,
        color=PALETTE["navy"],
        alpha=0.55,
    )
    ax_curve.scatter(
        binned.loc[high_mask, "density_center"],
        binned.loc[high_mask, "speed_mean"],
        s=7,
        color=PALETTE["red"],
        alpha=0.62,
    )
    ax_curve.axvline(rho_star, color=PALETTE["red"], lw=1.3, ls="--")
    ax_curve.set_ylabel("Mean binned speed (m/s)")
    ax_curve.set_ylim(0, 19.2)
    ax_curve.tick_params(labelbottom=False)
    add_panel_label(ax_curve, "a")

    ax_hist.axvspan(rho_star, curve_xmax, color="#f8d7da", alpha=0.24, zorder=0)
    hist_colors = np.where(high_mask, "#efb2a8", PALETTE["grey_light"])
    ax_hist.bar(
        binned["density_center"],
        binned["sample_share_pct"],
        width=0.9,
        color=hist_colors,
        edgecolor="none",
    )
    ax_hist.axvline(rho_star, color=PALETTE["red"], lw=1.3, ls="--")
    ax_hist.set_xlabel(r"Record density, $\rho$ (records per cell-window)")
    ax_hist.set_ylabel("Sample\nshare (%)")
    ax_hist.set_ylim(0, max(60, binned["sample_share_pct"].max() * 1.08))
    ax_hist.set_xlim(0, curve_xmax)

    subset_colors = [PALETTE["red"], PALETTE["blue"], PALETTE["teal"], PALETTE["purple"]]
    y_pos = np.arange(len(subsets))[::-1]

    for idx, row in enumerate(subsets.itertuples(index=False)):
        ax_rho.hlines(y_pos[idx], 0, row.rho_star, color=PALETTE["grey_light"], lw=1.6, zorder=1)
        ax_rho.scatter(row.rho_star, y_pos[idx], s=42, color=subset_colors[idx], edgecolor="white", linewidth=0.7, zorder=3)

    ax_rho.set_yticks(y_pos)
    ax_rho.set_yticklabels(subsets["display_label"])
    ax_rho.set_xlabel(r"Estimated state marker, $\rho^\ast$")
    ax_rho.set_xlim(0, max(20, subsets["rho_star"].max() * 1.15))
    ax_rho.grid(axis="x")
    ax_rho.tick_params(axis="y", labelsize=8.0)
    add_panel_label(ax_rho, "b")

    for idx, row in enumerate(subsets.itertuples(index=False)):
        ax_rr.hlines(y_pos[idx], row.rr_ci_95_lower, row.rr_ci_95_upper, color=subset_colors[idx], lw=1.8)
        ax_rr.scatter(row.rr, y_pos[idx], s=42, color=subset_colors[idx], edgecolor="white", linewidth=0.7, zorder=3)

    ax_rr.set_xscale("log")
    ax_rr.set_xlim(20, 2000)
    ax_rr.set_yticks(y_pos)
    ax_rr.set_yticklabels(subsets["display_label"])
    ax_rr.tick_params(axis="y", labelsize=8.0)
    ax_rr.set_xlabel("RR for state samples in backbone cells", x=0.47)
    ax_rr.grid(axis="x")
    ax_rr.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}"))
    add_panel_label(ax_rr, "c")

    fig.canvas.draw()
    fig.set_constrained_layout(False)
    for ax in [ax_rho, ax_rr]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + pos.height * 0.5, pos.width, pos.height * 0.5])

    save_panel(fig, [ax_curve, ax_hist], "manuscript_fig3a_speed_density_breakpoint")
    save_panel(fig, ax_rho, "manuscript_fig3b_breakpoint_across_subsets")
    save_panel(fig, ax_rr, "manuscript_fig3c_backbone_amplification")
    plt.close(fig)
    print("Saved manuscript Figure 3 panels")


if __name__ == "__main__":
    main()
