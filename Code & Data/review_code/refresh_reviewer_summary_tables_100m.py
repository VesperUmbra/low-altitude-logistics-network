from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
REVIEW_DATA_DIR = WORKSPACE_ROOT / "for_review" / "review_data"
SOURCE_JSON_DIR = REVIEW_DATA_DIR / "source_json"
GRID_INFO_JSON = ROOT / "data" / "processed" / "full_100m" / "grid" / "grid_info.json"


def load_json(name: str) -> dict:
    return json.loads((SOURCE_JSON_DIR / name).read_text(encoding="utf-8"))


def normalize_measurement_strings() -> None:
    grid_info = json.loads(GRID_INFO_JSON.read_text(encoding="utf-8"))
    measurement = (
        "distinct order_id counts per cell-window on the full 100 m x "
        f"{int(grid_info.get('time_window_s', 300)) / 60:g} min grid"
    )
    for path in [
        SOURCE_JSON_DIR / "summary_results.json",
        SOURCE_JSON_DIR / "unique_flight_occupancy_results.json",
        SOURCE_JSON_DIR / "robustness_100m" / "summary_results.json",
        SOURCE_JSON_DIR / "robustness_100m" / "unique_flight_occupancy_results.json",
    ]:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        if "unique_flight_occupancy" in obj:
            obj["unique_flight_occupancy"]["measurement"] = measurement
        else:
            obj["measurement"] = measurement
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    normalize_measurement_strings()
    summary = load_json("summary_results.json")
    station = load_json("real_station_anchor_results.json")
    cutoff = load_json("backbone_cutoff_sensitivity.json")

    top10 = next(row for row in cutoff["cutoff_rows"] if row["cutoff_label"] == "top 10%")
    full = summary["interface_masking"]["full_sample"]

    contingency_rows: list[dict] = []
    robustness_rows: list[dict] = []
    breakpoint_rows: list[dict] = []
    bootstrap_rows: list[dict] = []

    def add_contingency(analysis_id: str, analysis_group: str, description: str, exposed_group_label: str, outcome_label: str, item: dict, *, source: str = "review_data/source_json/summary_results.json", rho_key: str = "rho_star", samples_key: str = "samples", a_key: str = "corridor_congested", b_key: str = "corridor_not_congested", c_key: str = "noncorridor_congested", d_key: str = "noncorridor_not_congested", risk_a_key: str = "risk_corridor", risk_c_key: str = "risk_noncorridor", rr_key: str = "rr", rr_lo_key: str = "rr_ci_95_lower", rr_hi_key: str = "rr_ci_95_upper", or_key: str = "or_adj", or_lo_key: str = "or_ci_95_lower", or_hi_key: str = "or_ci_95_upper", fisher_key: str = "fisher_p") -> None:
        contingency_rows.append({
            "analysis_id": analysis_id,
            "analysis_group": analysis_group,
            "description": description,
            "exposed_group_label": exposed_group_label,
            "outcome_label": outcome_label,
            "rho_star": item.get(rho_key),
            "samples": item.get(samples_key),
            "exposed_and_exceedance": item.get(a_key),
            "exposed_and_non_exceedance": item.get(b_key),
            "unexposed_and_exceedance": item.get(c_key),
            "unexposed_and_non_exceedance": item.get(d_key),
            "risk_exposed": item.get(risk_a_key),
            "risk_unexposed": item.get(risk_c_key),
            "rr": item.get(rr_key),
            "rr_ci_95_lower": item.get(rr_lo_key),
            "rr_ci_95_upper": item.get(rr_hi_key),
            "or_adj": item.get(or_key),
            "or_ci_95_lower": item.get(or_lo_key),
            "or_ci_95_upper": item.get(or_hi_key),
            "fisher_p": item.get(fisher_key),
            "source_file": source,
        })

    def add_robustness(**kwargs: object) -> None:
        robustness_rows.append(kwargs)

    add_contingency(
        "main_full_sample",
        "main",
        "Main manuscript backbone versus non-backbone contingency table.",
        "backbone_cells",
        "tipping_exceedance",
        {
            "rho_star": full["rho_star"],
            "samples": top10["contingency_a"] + top10["contingency_b"] + top10["contingency_c"] + top10["contingency_d"],
            "corridor_congested": top10["contingency_a"],
            "corridor_not_congested": top10["contingency_b"],
            "noncorridor_congested": top10["contingency_c"],
            "noncorridor_not_congested": top10["contingency_d"],
            "risk_corridor": top10["risk_backbone"],
            "risk_noncorridor": top10["risk_non_backbone"],
            "rr": full["rr"],
            "rr_ci_95_lower": full["rr_ci_95_lower"],
            "rr_ci_95_upper": full["rr_ci_95_upper"],
            "or_adj": full["or_adj"],
            "or_ci_95_lower": full["or_ci_95_lower"],
            "or_ci_95_upper": full["or_ci_95_upper"],
            "fisher_p": 0.0,
        },
    )

    for split_key, label, desc in [
        ("forward_split", "out_of_sample_forward", "Chronological forward half-split held-out contingency table."),
        ("reverse_split", "out_of_sample_reverse", "Chronological reverse half-split held-out contingency table."),
    ]:
        item = summary["out_of_sample_coupling"][split_key]
        add_contingency(
            label,
            "out_of_sample",
            desc,
            "training_defined_backbone_cells",
            "tipping_exceedance",
            item,
            rho_key=None,
            samples_key=None,
            a_key="corridor_congested_test",
            b_key="corridor_not_test",
            c_key="noncorridor_congested_test",
            d_key="noncorridor_not_test",
            risk_a_key="risk_corridor_test",
            risk_c_key="risk_noncorridor_test",
            rr_key="rr_test",
            rr_lo_key="rr_ci_95_lower",
            rr_hi_key="rr_ci_95_upper",
            or_key="or_adj_test",
            or_lo_key="or_ci_95_lower",
            or_hi_key="or_ci_95_upper",
            fisher_key="fisher_p_test",
        )
        contingency_rows[-1]["rho_star"] = full["rho_star"]
        contingency_rows[-1]["samples"] = item["corridor_congested_test"] + item["corridor_not_test"] + item["noncorridor_congested_test"] + item["noncorridor_not_test"]

    u = summary["unique_flight_occupancy"]
    add_contingency(
        "unique_flight_occupancy",
        "measurement_alternative",
        "Alternative occupancy definition using distinct order counts per cell-window.",
        "backbone_cells",
        "tipping_exceedance",
        {
            "rho_star": u["rho_star_unique"],
            "samples": u["n_samples_total_unique"],
            "corridor_congested": u["corridor_congested_unique"],
            "corridor_not_congested": u["corridor_not_congested_unique"],
            "noncorridor_congested": u["noncorridor_congested_unique"],
            "noncorridor_not_congested": u["noncorridor_not_congested_unique"],
            "risk_corridor": u["corridor_congested_unique"] / (u["corridor_congested_unique"] + u["corridor_not_congested_unique"]),
            "risk_noncorridor": u["noncorridor_congested_unique"] / (u["noncorridor_congested_unique"] + u["noncorridor_not_congested_unique"]),
            "rr": u["rr_unique"],
            "rr_ci_95_lower": u["rr_ci_95_lower_unique"],
            "rr_ci_95_upper": u["rr_ci_95_upper_unique"],
            "or_adj": u["or_adj_unique"],
            "or_ci_95_lower": u["or_ci_95_lower_unique"],
            "or_ci_95_upper": u["or_ci_95_upper_unique"],
            "fisher_p": u["fisher_p_unique"],
        },
    )

    alt = summary["alternative_corridor_definition"]
    add_contingency(
        "alternative_corridor_definition",
        "structure_alternative",
        "Alternative backbone defined by distinct-order counts instead of point counts.",
        "alternative_backbone_cells",
        "tipping_exceedance",
        {
            "rho_star": full["rho_star"],
            "samples": alt["corridor_congested_alt"] + alt["corridor_not_congested_alt"] + alt["noncorridor_congested_alt"] + alt["noncorridor_not_congested_alt"],
            "corridor_congested": alt["corridor_congested_alt"],
            "corridor_not_congested": alt["corridor_not_congested_alt"],
            "noncorridor_congested": alt["noncorridor_congested_alt"],
            "noncorridor_not_congested": alt["noncorridor_not_congested_alt"],
            "risk_corridor": alt["corridor_congested_alt"] / (alt["corridor_congested_alt"] + alt["corridor_not_congested_alt"]),
            "risk_noncorridor": alt["noncorridor_congested_alt"] / (alt["noncorridor_congested_alt"] + alt["noncorridor_not_congested_alt"]),
            "rr": alt["rr_alt"],
            "rr_ci_95_lower": alt["rr_ci_95_lower_alt"],
            "rr_ci_95_upper": alt["rr_ci_95_upper_alt"],
            "or_adj": alt["or_adj_alt"],
            "or_ci_95_lower": alt["or_ci_95_lower_alt"],
            "or_ci_95_upper": alt["or_ci_95_upper_alt"],
            "fisher_p": alt["fisher_p_alt"],
        },
    )

    for excl_key, label, desc in [
        ("excluding_endpoint_cells_only", "endpoint_cells_only_exclusion", "Subset after excluding recurrent endpoint cells only."),
        ("excluding_endpoint_cells_and_1cell_buffer", "endpoint_cells_buffer1_exclusion", "Subset after excluding recurrent endpoint cells and a one-cell buffer."),
    ]:
        add_contingency(
            label,
            "endpoint_exclusion",
            desc,
            "redefined_backbone_cells",
            "tipping_exceedance",
            summary["endpoint_buffer_exclusion"][excl_key],
        )

    for phase_key, label, desc in [
        ("near_endpoint_buffer", "phase_near_endpoint_buffer", "Near-endpoint subset."),
        ("away_from_endpoint_buffer", "phase_away_from_endpoint_buffer", "Away-from-endpoint subset."),
        ("low_altitude_lt_120m", "phase_low_altitude_lt_120m", "Low-altitude subset (mean altitude < 120 m)."),
        ("cruise_like_altitude_ge_180m", "phase_cruise_like_altitude_ge_180m", "Cruise-like subset (mean altitude >= 180 m)."),
        ("away_from_endpoint_and_cruise_like", "phase_away_from_endpoint_and_cruise_like", "Cruise-like subset away from recurrent endpoint buffers."),
    ]:
        add_contingency(
            label,
            "phase_altitude",
            desc,
            "redefined_backbone_cells",
            "tipping_exceedance",
            summary["phase_altitude_decomposition"]["subsets"][phase_key],
        )

    add_contingency(
        "leave_top_endpoint_out",
        "leave_one_group_out",
        "Subset after removing the top 10% endpoint cells.",
        "redefined_backbone_cells",
        "tipping_exceedance",
        summary["leave_one_group_out"]["leave_top_endpoint_out"]["summary"],
    )
    add_contingency(
        "leave_largest_backbone_cluster_out",
        "leave_one_group_out",
        "Subset after removing the largest connected backbone component.",
        "redefined_backbone_cells",
        "tipping_exceedance",
        summary["leave_one_group_out"]["leave_largest_backbone_cluster_out"]["summary"],
    )

    hotspot = summary["spatial_hotspot_localization"]
    endpoint = summary["endpoint_node_anchor"]
    add_robustness(
        check_id="spatial_hotspot_localization",
        category="localization",
        description="Hotspot localization among active grid cells.",
        notes="Summarizes the concentration of tipping-exceedance states across active cells.",
        source_file="review_data/source_json/summary_results.json",
        hotspot_cells=hotspot["hotspot_cells"],
        hotspot_share_active_cells=hotspot["hotspot_share_active_cells"],
        hotspot_share_full_grid=hotspot["hotspot_share_full_grid"],
        top_0_5pct_share=hotspot["top_0_5pct_active_cells_congestion_share"],
        top_1pct_share=hotspot["top_1pct_active_cells_congestion_share"],
    )
    add_robustness(
        check_id="endpoint_node_anchor",
        category="localization",
        description="Recurrent order endpoints mapped to the full 100 m grid.",
        notes="Endpoint cells are derived from order origins and destinations rather than tipping states.",
        source_file="review_data/source_json/summary_results.json",
        endpoint_cells=endpoint["endpoint_cells"],
        endpoint_share_active_cells=endpoint["endpoint_share_active_cells"],
        endpoint_corridor_overlap=endpoint["endpoint_corridor_overlap"],
        congested_samples_in_endpoint_cells_share=endpoint["congested_samples_in_endpoint_cells_share"],
        congested_samples_within_1cell_of_endpoints_share=endpoint["congested_samples_within_1cell_of_endpoints_share"],
    )
    add_robustness(
        check_id="unique_flight_occupancy",
        category="measurement_alternative",
        description="Alternative occupancy metric using distinct order counts per cell-window.",
        samples=u["n_samples_total_unique"],
        rho_star=u["rho_star_unique"],
        rr=u["rr_unique"],
        rr_ci_95_lower=u["rr_ci_95_lower_unique"],
        rr_ci_95_upper=u["rr_ci_95_upper_unique"],
        or_adj=u["or_adj_unique"],
        or_ci_95_lower=u["or_ci_95_lower_unique"],
        or_ci_95_upper=u["or_ci_95_upper_unique"],
        notes="Uses unique order occupancy rather than telemetry-point counts.",
        source_file="review_data/source_json/summary_results.json",
        speed_free=u["speed_free_unique"],
        speed_congested=u["speed_congested_unique"],
        speed_drop_percent=u["speed_drop_percent_unique"],
    )
    add_robustness(
        check_id="alternative_corridor_definition",
        category="structure_alternative",
        description="Backbone defined by distinct-order counts instead of point counts.",
        rho_star=full["rho_star"],
        rr=alt["rr_alt"],
        rr_ci_95_lower=alt["rr_ci_95_lower_alt"],
        rr_ci_95_upper=alt["rr_ci_95_upper_alt"],
        or_adj=alt["or_adj_alt"],
        or_ci_95_lower=alt["or_ci_95_lower_alt"],
        or_ci_95_upper=alt["or_ci_95_upper_alt"],
        notes="Retains the top 10% active-cell backbone size but changes the ranking metric.",
        source_file="review_data/source_json/summary_results.json",
        jaccard_with_main_backbone=alt["jaccard_with_point_count_corridor"],
    )

    for key, cid, desc in [
        ("excluding_interface_windows", "interface_mask_excluding_windows", "Exclude interface-like cell-windows."),
        ("excluding_interface_cells", "interface_mask_excluding_cells", "Exclude interface-like cells entirely."),
    ]:
        item = summary["interface_masking"][key]
        add_robustness(
            check_id=cid,
            category="interface_masking",
            description=desc,
            samples=item["samples"],
            rho_star=item["rho_star"],
            rr=item["rr"],
            rr_ci_95_lower=item["rr_ci_95_lower"],
            rr_ci_95_upper=item["rr_ci_95_upper"],
            or_adj=item["or_adj"],
            or_ci_95_lower=item["or_ci_95_lower"],
            or_ci_95_upper=item["or_ci_95_upper"],
            notes=f"Interface definition: {summary['interface_masking']['interface_definition']}.",
            source_file="review_data/source_json/summary_results.json",
            speed_drop_percent=item["speed_drop_percent"],
        )

    for key, cid, desc in [
        ("excluding_endpoint_cells_only", "endpoint_cells_only_exclusion", "Exclude recurrent endpoint cells only."),
        ("excluding_endpoint_cells_and_1cell_buffer", "endpoint_cells_buffer1_exclusion", "Exclude recurrent endpoint cells and a one-cell buffer."),
    ]:
        item = summary["endpoint_buffer_exclusion"][key]
        add_robustness(
            check_id=cid,
            category="endpoint_exclusion",
            description=desc,
            samples=item["samples"],
            rho_star=item["rho_star"],
            rr=item["rr"],
            rr_ci_95_lower=item["rr_ci_95_lower"],
            rr_ci_95_upper=item["rr_ci_95_upper"],
            or_adj=item["or_adj"],
            or_ci_95_lower=item["or_ci_95_lower"],
            or_ci_95_upper=item["or_ci_95_upper"],
            notes="Backbone is redefined on the retained subset after exclusion.",
            source_file="review_data/source_json/summary_results.json",
            speed_drop_percent=item["speed_drop_percent"],
        )

    for phase_key, cid, desc in [
        ("near_endpoint_buffer", "phase_near_endpoint_buffer", "Near-endpoint subset."),
        ("away_from_endpoint_buffer", "phase_away_from_endpoint_buffer", "Away-from-endpoint subset."),
        ("low_altitude_lt_120m", "phase_low_altitude_lt_120m", "Low-altitude subset (mean altitude < 120 m)."),
        ("cruise_like_altitude_ge_180m", "phase_cruise_like_altitude_ge_180m", "Cruise-like subset (mean altitude >= 180 m)."),
        ("away_from_endpoint_and_cruise_like", "phase_away_from_endpoint_and_cruise_like", "Cruise-like subset away from endpoint buffers."),
    ]:
        item = summary["phase_altitude_decomposition"]["subsets"][phase_key]
        add_robustness(
            check_id=cid,
            category="phase_altitude",
            description=desc,
            samples=item["samples"],
            rho_star=item["rho_star"],
            rr=item["rr"],
            rr_ci_95_lower=item["rr_ci_95_lower"],
            rr_ci_95_upper=item["rr_ci_95_upper"],
            or_adj=item["or_adj"],
            or_ci_95_lower=item["or_ci_95_lower"],
            or_ci_95_upper=item["or_ci_95_upper"],
            notes="Subset-specific backbone redefined after filtering by endpoint distance and/or altitude band.",
            source_file="review_data/source_json/summary_results.json",
            speed_drop_percent=item["speed_drop_percent"],
        )

    lday = summary["leave_one_group_out"]["leave_one_day_out"]["summary"]
    lvendor = summary["leave_one_group_out"]["leave_one_vendor_out"]["summary"]
    add_robustness(check_id="leave_one_day_out_range", category="leave_one_group_out", description="Leave-one-day-out range summary.", notes="Range summary across repeated exclusions.", source_file="review_data/source_json/summary_results.json", rr_min=lday["rr_min"], rr_median=lday["rr_median"], rr_max=lday["rr_max"], rho_star_min=lday["rho_star_min"], rho_star_median=lday["rho_star_median"], rho_star_max=lday["rho_star_max"], runs=lday["n_runs"])
    add_robustness(check_id="leave_one_vendor_out_range", category="leave_one_group_out", description="Leave-one-vendor-out range summary.", notes="Range summary across repeated exclusions.", source_file="review_data/source_json/summary_results.json", rr_min=lvendor["rr_min"], rr_median=lvendor["rr_median"], rr_max=lvendor["rr_max"], rho_star_min=lvendor["rho_star_min"], rho_star_median=lvendor["rho_star_median"], rho_star_max=lvendor["rho_star_max"], runs=lvendor["n_runs"])

    lend = summary["leave_one_group_out"]["leave_top_endpoint_out"]["summary"]
    add_robustness(check_id="leave_top_endpoint_out", category="leave_one_group_out", description="Remove the top 10% endpoint cells.", samples=lend["samples"], rho_star=lend["rho_star"], rr=lend["rr"], rr_ci_95_lower=lend["rr_ci_95_lower"], rr_ci_95_upper=lend["rr_ci_95_upper"], or_adj=lend["or_adj"], or_ci_95_lower=lend["or_ci_95_lower"], or_ci_95_upper=lend["or_ci_95_upper"], notes="Subset-specific backbone redefined after exclusion.", source_file="review_data/source_json/summary_results.json", speed_drop_percent=lend["speed_drop_percent"])
    lcluster = summary["leave_one_group_out"]["leave_largest_backbone_cluster_out"]["summary"]
    add_robustness(check_id="leave_largest_backbone_cluster_out", category="leave_one_group_out", description="Remove the largest connected backbone component.", samples=lcluster["samples"], rho_star=lcluster["rho_star"], rr=lcluster["rr"], rr_ci_95_lower=lcluster["rr_ci_95_lower"], rr_ci_95_upper=lcluster["rr_ci_95_upper"], or_adj=lcluster["or_adj"], or_ci_95_lower=lcluster["or_ci_95_lower"], or_ci_95_upper=lcluster["or_ci_95_upper"], notes="Subset-specific backbone redefined after exclusion.", source_file="review_data/source_json/summary_results.json", speed_drop_percent=lcluster["speed_drop_percent"])

    traj = summary["trajectory_path_persistence"]["core_repeated_route_skeleton"]
    add_robustness(check_id="trajectory_path_persistence", category="route_skeleton", description="Repeated-route skeleton compared with the intensity backbone.", notes="Core repeated-route skeleton uses undirected edges reused by at least 100 distinct orders.", source_file="review_data/source_json/summary_results.json", support_threshold=traj["min_distinct_order_support"], skeleton_edges=traj["skeleton_edges"], skeleton_nodes=traj["skeleton_nodes"], backbone_coverage_share=traj["backbone_coverage_share"], mean_order_edge_coverage=traj["mean_order_edge_coverage"], share_orders_edge_coverage_ge_80pct=traj["share_orders_edge_coverage_ge_80pct"])

    morph = summary["morphological_continuity_directionality"]
    add_robustness(check_id="morphological_continuity", category="morphology", description="Connected-component continuity and directionality diagnostics on backbone cells.", notes="Random active-cell null summary is available via the copied source JSON files.", source_file="review_data/source_json/summary_results.json", backbone_cells=morph["backbone_cells"], n_components=morph["continuity"]["n_components"], largest_component_share=morph["continuity"]["largest_component_share"], degree_ge3_share=morph["continuity"]["degree_ge3_share"], weighted_median_component_axial_resultant=morph["moving_point_directionality_ge5"]["weighted_median_component_axial_resultant"])

    for key, cid, desc in [
        ("exact_anchor_cells", "real_station_exact", "Exact independent UAV-station anchor cells."),
        ("buffer1_anchor_cells", "real_station_buffer1", "Independent UAV-station anchor with one-cell buffer."),
        ("buffer2_anchor_cells", "real_station_buffer2", "Independent UAV-station anchor with two-cell buffer."),
    ]:
        item = station["anchor_sets"]["operational_sites"][key]
        add_robustness(check_id=cid, category="station_anchor", description=desc, rr=item["sample_rr_exceedance"], rr_ci_95_lower=item["sample_rr_exceedance_ci"][0], rr_ci_95_upper=item["sample_rr_exceedance_ci"][1], notes="Derived from the supplied operator station inventory and mapped to the canonical 100 m grid.", source_file="review_data/source_json/real_station_anchor_results.json", active_cells_covered_share=item["active_cells_covered_share"], hotspot_cells_covered_share=item["hotspot_cells_covered_share"], endpoint_cells_covered_share=item["endpoint_cells_covered_share"], exceedance_samples_covered_share=item["exceedance_samples_covered_share"])

    spec = summary["breakpoint_specification"]
    breakpoint_rows.append({
        "specification_id": "main_equal_weight_bins",
        "description": "Operational main-text breakpoint specification.",
        "rho_star": spec["main_text_estimator"]["rho_star"],
        "speed_drop_percent": None,
        "n_bins": None,
        "bin_width": spec["main_text_estimator"]["bin_width"],
        "min_samples": spec["main_text_estimator"]["min_samples"],
        "weighting": spec["main_text_estimator"]["weighting"],
        "source_file": "review_data/source_json/summary_results.json",
    })
    for key, item in spec["alternative_specifications"].items():
        breakpoint_rows.append({
            "specification_id": key,
            "description": "Alternative breakpoint specification retained in the SI.",
            "rho_star": item["rho_star"],
            "speed_drop_percent": item["speed_drop_percent"],
            "n_bins": item["n_bins"],
            "bin_width": item["bin_width"],
            "min_samples": item["min_samples"],
            "weighting": item["weighting"],
            "source_file": "review_data/source_json/summary_results.json",
        })

    cluster = summary["cluster_robust_ci"]
    rho_boot = summary["rho_star_bootstrap_ci"]
    bootstrap_rows.extend([
        {
            "summary_id": "cluster_bootstrap_rr",
            "category": "cluster_bootstrap",
            "description": "Date-hour block bootstrap for relative risk.",
            "n_draws": cluster["n_bootstraps"],
            "observed": cluster["rr_point_estimate"],
            "mean": cluster["rr_bootstrap_mean"],
            "median": cluster["rr_bootstrap_median"],
            "ci_95_lower": cluster["rr_ci_95_lower"],
            "ci_95_upper": cluster["rr_ci_95_upper"],
            "max": None,
            "empirical_p_ge_observed": None,
            "source_file": "review_data/source_json/summary_results.json",
        },
        {
            "summary_id": "cluster_bootstrap_or_adj",
            "category": "cluster_bootstrap",
            "description": "Date-hour block bootstrap for Haldane-Anscombe odds ratio.",
            "n_draws": cluster["n_bootstraps"],
            "observed": cluster["or_point_estimate"],
            "mean": cluster["or_bootstrap_mean"],
            "median": cluster["or_bootstrap_median"],
            "ci_95_lower": cluster["or_ci_95_lower"],
            "ci_95_upper": cluster["or_ci_95_upper"],
            "max": None,
            "empirical_p_ge_observed": None,
            "source_file": "review_data/source_json/summary_results.json",
        },
        {
            "summary_id": "rho_star_bootstrap",
            "category": "rho_star_bootstrap",
            "description": "Resampling diagnostic for the breakpoint estimator on binned speed-density data.",
            "n_draws": rho_boot["n_bootstraps"],
            "observed": rho_boot["original_rho_star"],
            "mean": rho_boot["bootstrap_mean"],
            "median": rho_boot["bootstrap_median"],
            "ci_95_lower": rho_boot["ci_95_lower"],
            "ci_95_upper": rho_boot["ci_95_upper"],
            "max": None,
            "empirical_p_ge_observed": None,
            "source_file": "review_data/source_json/summary_results.json",
        },
    ])
    for prefix, key in [("null_time_shuffle", "time_shuffle_null"), ("null_random_backbone", "random_backbone_label_null")]:
        for stat_key, suffix in [("rr", "rr"), ("or_adj", "or_adj")]:
            item = summary["null_model_suite"][key][stat_key]
            bootstrap_rows.append({
                "summary_id": f"{prefix}_{suffix}",
                "category": "null_model",
                "description": f"{summary['null_model_suite'][key]['design']}: {suffix}.",
                "n_draws": item["n_values"],
                "observed": item["observed"],
                "mean": item["mean"],
                "median": item["median"],
                "ci_95_lower": item["ci_95_lower"],
                "ci_95_upper": item["ci_95_upper"],
                "max": item["max"],
                "empirical_p_ge_observed": item["empirical_p_ge_observed"],
                "source_file": "review_data/source_json/summary_results.json",
            })
    for stat_key, suffix in [("endpoint_cell_share", "exact_share"), ("endpoint_buffer_one_share", "buffer1_share")]:
        item = summary["null_model_suite"]["od_endpoint_reassignment_null"][stat_key]
        bootstrap_rows.append({
            "summary_id": f"null_endpoint_reassignment_{suffix}",
            "category": "null_model",
            "description": f"OD endpoint reassignment: {suffix}.",
            "n_draws": item["n_values"],
            "observed": item["observed"],
            "mean": item["mean"],
            "median": item["median"],
            "ci_95_lower": item["ci_95_lower"],
            "ci_95_upper": item["ci_95_upper"],
            "max": item["max"],
            "empirical_p_ge_observed": item["empirical_p_ge_observed"],
            "source_file": "review_data/source_json/summary_results.json",
        })

    pd.DataFrame(contingency_rows).to_csv(REVIEW_DATA_DIR / "contingency_tables.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(robustness_rows).to_csv(REVIEW_DATA_DIR / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(breakpoint_rows).to_csv(REVIEW_DATA_DIR / "breakpoint_specification_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bootstrap_rows).to_csv(REVIEW_DATA_DIR / "bootstrap_and_null_summaries.csv", index=False, encoding="utf-8-sig")

    print("Refreshed reviewer summary tables for the canonical 100 m branch.")


if __name__ == "__main__":
    main()
