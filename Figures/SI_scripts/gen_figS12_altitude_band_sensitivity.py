from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_figure


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
FOR_REVIEW = WORKSPACE_ROOT / "for_review"
REVIEW_DATA_DIR = FOR_REVIEW / "review_data"

INPUT_CSV = REVIEW_DATA_DIR / "altitude_band_sensitivity.csv"
FIGURE_STEM = "S12_altitude_band_sensitivity"

BAND_ORDER = ["lt_60m", "60_120m", "120_180m", "180_240m", "ge_240m"]
BAND_LABELS = {
    "lt_60m": "<60 m",
    "60_120m": "60-120 m",
    "120_180m": "120-180 m",
    "180_240m": "180-240 m",
    "ge_240m": ">=240 m",
}
BAND_COLORS = {
    "lt_60m": PALETTE["gold"],
    "60_120m": "#d7a33d",
    "120_180m": PALETTE["teal"],
    "180_240m": PALETTE["navy"],
    "ge_240m": PALETTE["purple"],
}


def format_xticklabels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for _, row in df.iterrows():
        labels.append(f"{BAND_LABELS[row['band']]}\n({row['sample_share_percent']:.1f}%)")
    return labels


def main() -> None:
    apply_style()

    df = pd.read_csv(INPUT_CSV)
    overall = df.loc[df["band"] == "all_sample"].iloc[0]
    bands = df.loc[df["band"].isin(BAND_ORDER)].copy()
    bands["band"] = pd.Categorical(bands["band"], categories=BAND_ORDER, ordered=True)
    bands = bands.sort_values("band").reset_index(drop=True)

    colors = [BAND_COLORS[b] for b in bands["band"]]
    x = np.arange(len(bands))
    xticklabels = format_xticklabels(bands)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.95), gridspec_kw={"wspace": 0.34})
    ax0, ax1, ax2 = axes

    ax0.bar(x, bands["rho_star"], color=colors, edgecolor="white", linewidth=0.8)
    ax0.axhline(overall["rho_star"], color=PALETTE["grey_dark"], linestyle="--", linewidth=1.2)
    ax0.set_ylabel(r"Breakpoint $\rho^\ast$")
    ax0.set_xticks(x)
    ax0.set_xticklabels(xticklabels)
    ax0.set_ylim(0, max(24, bands["rho_star"].max() + 2))
    ax0.grid(axis="y", linestyle="--", alpha=0.5)
    ax0.text(
        0.98,
        0.96,
        f"All-sample = {overall['rho_star']:.1f}",
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color=PALETTE["grey_dark"],
    )

    ax1.bar(x, bands["speed_drop_percent"], color=colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(overall["speed_drop_percent"], color=PALETTE["grey_dark"], linestyle="--", linewidth=1.2)
    ax1.set_ylabel("Speed drop")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticklabels)
    ax1.set_ylim(0, 90)
    ax1.set_yticks([0, 20, 40, 60, 80])
    ax1.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.text(
        0.98,
        0.96,
        f"All-sample = {overall['speed_drop_percent']:.1f}%",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color=PALETTE["grey_dark"],
    )

    rr = bands["rr"].to_numpy(dtype=float)
    rr_lo = bands["rr_ci_95_lower"].to_numpy(dtype=float)
    rr_hi = bands["rr_ci_95_upper"].to_numpy(dtype=float)
    yerr = np.vstack([rr - rr_lo, rr_hi - rr])
    for xi, yi, yerri, color in zip(x, rr, yerr.T, colors, strict=True):
        ax2.errorbar(
            xi,
            yi,
            yerr=[[yerri[0]], [yerri[1]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    ax2.axhline(overall["rr"], color=PALETTE["grey_dark"], linestyle="--", linewidth=1.2)
    ax2.set_yscale("log")
    ax2.set_ylabel("RR (log scale)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels)
    ax2.set_ylim(1, 1200)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.text(
        0.98,
        0.96,
        f"All-sample = {overall['rr']:.1f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color=PALETTE["grey_dark"],
    )

    add_panel_label(ax0, "a")
    add_panel_label(ax1, "b")
    add_panel_label(ax2, "c")

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.95, wspace=0.34)
    save_figure(fig, FIGURE_STEM)


if __name__ == "__main__":
    main()
