from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import PALETTE, add_panel_label, apply_style, save_figure


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA = WORKSPACE_ROOT / "for_review" / "review_data"

NODES_CSV = REVIEW_DATA / "route_network_nodes_300m_by_vendor.csv"
SUMMARY_JSON = REVIEW_DATA / "source_json" / "route_network_complexity_by_vendor_300m.json"
SUMMARY_CSV = REVIEW_DATA / "vendor_network_summary_300m.csv"


def main() -> None:
    apply_style()

    nodes = pd.read_csv(NODES_CSV)
    payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    summary_rows: list[dict[str, object]] = []
    for item in payload["vendor_summaries"]:
        stats = item["route_network_300m"]
        summary_rows.append(
            {
                "vendor": item["vendor"],
                "nodes": stats["nodes"],
                "edges": stats["edges"],
                "components": stats["components"],
                "largest_component_share_pct": round(stats["largest_component_node_share"] * 100, 1),
                "mean_local_clustering": round(stats["mean_local_clustering"], 3),
                "average_degree": round(stats["average_degree"], 2),
                "max_degree": stats["max_degree"],
                "max_strength_orders": stats["max_strength_orders"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    colors = {"MT": PALETTE["orange"], "SF": PALETTE["navy"]}
    fig, ax = plt.subplots(figsize=(5.9, 4.6))

    for vendor in ["MT", "SF"]:
        subset = nodes.loc[nodes["vendor"] == vendor].copy()
        sizes = 18 + subset["degree"].astype(float) * 10.0
        ax.scatter(
            subset["strength_orders"],
            subset["betweenness_invflow_norm"],
            s=sizes,
            c=colors[vendor],
            alpha=0.48,
            edgecolors="white",
            linewidths=0.5,
            label=f"{vendor} nodes",
        )

        ax.scatter(
            [subset["strength_orders"].median()],
            [subset["betweenness_invflow_norm"].median()],
            s=110,
            c=colors[vendor],
            marker="D",
            edgecolors="white",
            linewidths=0.9,
            alpha=0.95,
            label=f"{vendor} median",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow-weighted degree (strength, orders)")
    ax.set_ylabel("Weighted betweenness\n(inverse-flow shortest paths)")
    ax.set_title("Operator-specific node centrality", fontsize=10.5)
    ax.grid(True, which="major", linestyle=":", color=PALETTE["grey"], alpha=0.85)
    ax.grid(True, which="minor", linestyle=":", color=PALETTE["grey_light"], alpha=0.45)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    add_panel_label(ax, "a")

    save_figure(fig, "S18_vendor_network_centrality")
    plt.close(fig)


if __name__ == "__main__":
    main()
