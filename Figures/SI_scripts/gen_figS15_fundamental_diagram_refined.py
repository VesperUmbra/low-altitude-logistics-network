from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_figure


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
FOR_REVIEW = WORKSPACE_ROOT / "for_review"
REVIEW_DATA_DIR = FOR_REVIEW / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"

CELL_WINDOWS_CSV = REVIEW_DATA_DIR / "preview_fundamental_diagram_cell_windows.csv"
FIGURE_CELLS_CSV = REVIEW_DATA_DIR / "figure_cell_metrics.csv"
SUMMARY_JSON = SOURCE_JSON_DIR / "summary_results.json"
GRID_INFO_JSON = FOR_REVIEW / "review_code" / "data" / "processed" / "full_100m" / "grid" / "grid_info.json"

OUTPUT_BINNED = REVIEW_DATA_DIR / "preview_fundamental_diagram_refined_binned.csv"
OUTPUT_SUMMARY = SOURCE_JSON_DIR / "preview_fundamental_diagram_refined_summary.json"

def load_data() -> tuple[pd.DataFrame, float]:
    table = pd.read_csv(CELL_WINDOWS_CSV)
    figure_cells = pd.read_csv(FIGURE_CELLS_CSV, usecols=["cell_id", "endpoint_buffer1_cell"])
    endpoint_buffer_cells = set(
        figure_cells.loc[figure_cells["endpoint_buffer1_cell"], "cell_id"].astype(str)
    )
    table["is_endpoint_buffer1"] = table["cell_id"].astype(str).isin(endpoint_buffer_cells)
    table["spacing_surrogate_m"] = 100.0 / table["density_unique_flights"].clip(lower=1.0)

    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    rho_unique = float(summary["unique_flight_occupancy"]["rho_star_unique"])
    return table, rho_unique


def binned_relation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    bin_width: float,
    min_samples: int = 30,
) -> pd.DataFrame:
    bins = np.arange(float(df[x_col].min()), float(df[x_col].max()) + bin_width + 1e-9, bin_width)
    if len(bins) < 2:
        bins = np.array([float(df[x_col].min()), float(df[x_col].max()) + bin_width])

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
        .reset_index(drop=True)
    )
    return grouped.loc[grouped["sample_count"] >= min_samples].copy()


def export_binned(full_df: pd.DataFrame, away_df: pd.DataFrame) -> None:
    specs = [
        ("speed_vs_density", "density_unique_flights", "speed_mean", 1.0),
        ("speed_vs_flow", "flow_entries_per_hour", "speed_mean", 6.0),
        ("flow_vs_density", "density_unique_flights", "flow_entries_per_hour", 1.0),
        ("speed_vs_spacing", "spacing_surrogate_m", "speed_mean", 4.0),
    ]

    rows: list[pd.DataFrame] = []
    for relation, x_col, y_col, bin_width in specs:
        full_binned = binned_relation(full_df, x_col, y_col, bin_width=bin_width)
        full_binned["subset"] = "all_windows"
        full_binned["relation"] = relation
        away_binned = binned_relation(away_df, x_col, y_col, bin_width=bin_width)
        away_binned["subset"] = "away_from_endpoint"
        away_binned["relation"] = relation
        rows.extend([full_binned, away_binned])
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT_BINNED, index=False)


def plot_panel(
    ax: plt.Axes,
    full_df: pd.DataFrame,
    away_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    bin_width: float,
    invert_x: bool = False,
) -> None:
    full_binned = binned_relation(full_df, x_col, y_col, bin_width=bin_width)
    away_binned = binned_relation(away_df, x_col, y_col, bin_width=bin_width)

    ax.fill_between(
        full_binned["x_mean"],
        full_binned["y_q25"],
        full_binned["y_q75"],
        color=PALETTE["orange"],
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(full_binned["x_mean"], full_binned["y_mean"], color=PALETTE["orange"], lw=2.2, label="All windows")

    ax.fill_between(
        away_binned["x_mean"],
        away_binned["y_q25"],
        away_binned["y_q75"],
        color=PALETTE["navy"],
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(away_binned["x_mean"], away_binned["y_mean"], color=PALETTE["navy"], lw=2.1, label="Away from endpoint")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, linestyle=":", color=PALETTE["grey"], alpha=0.85)
    ax.set_ylim(bottom=0)
    if invert_x:
        ax.invert_xaxis()


def make_figure(table: pd.DataFrame, rho_unique: float) -> None:
    apply_style()

    away = table.loc[~table["is_endpoint_buffer1"]].copy()

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    axes = axes.ravel()

    plot_panel(
        axes[0],
        table,
        away,
        x_col="density_unique_flights",
        y_col="speed_mean",
        x_label="density k (distinct flights per cell-window)",
        y_label="speed v (m/s)",
        title="Speed vs density",
        bin_width=1.0,
    )
    add_panel_label(axes[0], "a")

    plot_panel(
        axes[1],
        table,
        away,
        x_col="flow_entries_per_hour",
        y_col="speed_mean",
        x_label="flow q (entries h$^{-1}$)",
        y_label="speed v (m/s)",
        title="Speed vs flow",
        bin_width=6.0,
    )
    add_panel_label(axes[1], "b")

    plot_panel(
        axes[2],
        table,
        away,
        x_col="density_unique_flights",
        y_col="flow_entries_per_hour",
        x_label="density k (distinct flights per cell-window)",
        y_label="flow q (entries h$^{-1}$)",
        title="Flow vs density",
        bin_width=1.0,
    )
    add_panel_label(axes[2], "c")

    plot_panel(
        axes[3],
        table,
        away,
        x_col="spacing_surrogate_m",
        y_col="speed_mean",
        x_label="spacing surrogate s = 100/k (m)",
        y_label="speed v (m/s)",
        title="Speed vs spacing",
        bin_width=4.0,
        invert_x=True,
    )
    add_panel_label(axes[3], "d")

    legend_handles = [
        Line2D([0], [0], color=PALETTE["orange"], lw=2.2, label="All-window mean"),
        Patch(facecolor=PALETTE["orange"], alpha=0.16, edgecolor="none", label="All-window IQR"),
        Line2D([0], [0], color=PALETTE["navy"], lw=2.1, label="Away-from-endpoint mean"),
        Patch(facecolor=PALETTE["navy"], alpha=0.12, edgecolor="none", label="Away-from-endpoint IQR"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    save_figure(fig, "S15_preview_fundamental_diagram_refined")
    plt.close(fig)


def main() -> None:
    table, rho_unique = load_data()
    away = table.loc[~table["is_endpoint_buffer1"]].copy()
    export_binned(table, away)
    grid_info = json.loads(GRID_INFO_JSON.read_text(encoding="utf-8"))
    tw_label = f"{int(grid_info.get('time_window_s', 300)) / 60:g}"

    summary = {
        "density_definition": f"distinct flights per 100 m x {tw_label} min cell-window",
        "flow_definition": "distinct order entries into each 100 m cell-window, converted to hourly units",
        "spacing_surrogate_definition": "100 divided by distinct flights per cell-window",
        "all_windows": int(len(table)),
        "away_from_endpoint_windows": int(len(away)),
        "rho_star_unique_reference": float(rho_unique),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(table, rho_unique)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
