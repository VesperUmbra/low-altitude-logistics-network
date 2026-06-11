# Data Map — `review_data/`

This document describes every CSV and JSON data file in the `Code & Data/review_data/` directory of the GitHub upload package.  
Each entry lists the file name, a brief description, and the column names / top-level keys.

---

## Directory Structure

```
review_data/
├── air_ground_clustered_road_by_distance_bin_200m.csv  # Air/ground comparison by distance bin
├── air_ground_clustered_road_edges_200m.csv            # Per-edge air/ground comparison
├── air_ground_clustered_road_nodes_200m.csv            # Endpoint cluster nodes
├── air_ground_clustered_road_osrm_cache_200m.csv       # OSRM routing cache
├── altitude_band_sensitivity.csv                       # Congestion by altitude band
├── altitude_point_distribution.csv                     # Altitude distribution histogram
├── backbone_cutoff_sensitivity.csv                     # Backbone cutoff sensitivity
├── binned_speed_density_relations.csv                  # Binned fundamental diagram
├── bootstrap_and_null_summaries.csv                    # Bootstrap & null-model summaries
├── breakpoint_specification_summary.csv                # Breakpoint specification summary
├── contingency_tables.csv                              # Aggregated 2×2 contingency tables
├── diurnal_activity_halfhour.csv                       # Diurnal activity (30-min bins)
├── diurnal_activity_halfhour_daily.csv                 # Diurnal activity per date
├── diurnal_activity_hourly.csv                         # Diurnal activity (1-hour bins)
├── endpoint_cell_altitude_summary.csv                  # Altitude per endpoint cell
├── endpoint_relative_height_classification.csv         # Endpoint relative height classification
├── endpoint_terrain_elevation_cache.csv                 # SRTM terrain elevation cache
├── figure_anchor_summary.csv                           # Monitoring anchor coverage
├── figure_cell_metrics.csv                             # Per-cell figure metrics
├── figure_threshold_subset_summary.csv                 # Density-threshold subset RR
├── followup_compressed_paths_2025_2026.csv             # 2025–2026 follow-up paths (compressed)
├── followup_date_route_summary.csv                     # 2025–2026 per-date/route summary
├── followup_flight_summary_2025_2026.csv               # 2025–2026 per-flight metadata
├── followup_persistence_flight_metrics.csv             # 2025–2026 persistence metrics
├── followup_reference_route_skeleton_edges_support100.csv # 2023 route skeleton (support ≥100)
├── followup_route_group_metrics.csv                    # 2025–2026 route group metrics
├── followup_trajectory_points_2025_2026.csv            # 2025–2026 full trajectory points (large)
├── grid_size_sensitivity.csv                           # Grid-size sensitivity
├── hotspot_exceedance_rank_table.csv                   # Cells ranked by exceedance
├── hub_robustness.csv                                  # Leave-one-hub-out robustness
├── merged_route_connection_sensitivity.csv             # Route merge sensitivity
├── preview_distance_duration_bins.csv                  # Distance/duration bins
├── preview_fundamental_diagram_binned.csv              # Binned fundamental diagram
├── preview_fundamental_diagram_cell_windows.csv        # Per-cell-window FD data (large)
├── preview_fundamental_diagram_refined_binned.csv      # Refined binned FD
├── preview_speed_phase_by_distance.csv                 # Speed by phase & distance (large)
├── ranked_cell_traffic_table.csv                       # Cells ranked by traffic
├── real_station_anchor_sites_mapped.csv                # Real station anchor sites
├── robustness_summary.csv                              # Master robustness summary
├── route_day_robustness.csv                            # Leave-one-route-day-out
├── route_network_edges_300m.csv                        # Route network edges
├── route_network_edges_300m_by_vendor.csv              # Route network edges by vendor
├── route_network_nodes_300m.csv                        # Route network nodes
├── route_network_nodes_300m_by_vendor.csv              # Route network nodes by vendor
├── targeted_monitoring_payoff.csv                      # Monitoring strategy payoff
├── targeted_monitoring_random_baseline.csv             # Random monitoring baseline
├── trajectory_length_distribution.csv                  # Trajectory length distribution
├── vendor_day_robustness.csv                           # Leave-one-vendor-day-out
├── vendor_network_summary_300m.csv                     # Vendor network summary
├── source_json/                                        # Structured analysis outputs (values stripped)
│   ├── air_ground_clustered_road_geometries_200m.json
│   ├── air_ground_clustered_road_summary_200m.json
│   ├── alternative_corridor_definition_results.json
│   ├── altitude_band_sensitivity.json
│   ├── backbone_cutoff_sensitivity.json
│   ├── breakpoint_results.json
│   ├── breakpoint_specification_results.json
│   ├── cluster_robust_ci_results.json
│   ├── endpoint_buffer_exclusion_results.json
│   ├── endpoint_node_anchor_results.json
│   ├── endpoint_relative_height_summary.json
│   ├── exclude_terminal_zones_results.json
│   ├── followup_persistence_summary.json
│   ├── grid_size_sensitivity.json
│   ├── grid_switch_100m_metadata.json
│   ├── hub_robustness.json
│   ├── interface_masking_results.json
│   ├── leave_one_group_out_results.json
│   ├── merged_route_connection_sensitivity.json
│   ├── morphological_continuity_directionality_results.json
│   ├── null_model_suite_results.json
│   ├── out_of_sample_coupling_results.json
│   ├── phase_altitude_decomposition_results.json
│   ├── preview_fundamental_diagram_refined_summary.json
│   ├── preview_fundamental_diagram_summary.json
│   ├── preview_speed_advantage_summary.json
│   ├── real_station_anchor_results.json
│   ├── rho_star_bootstrap_ci_results.json
│   ├── route_day_robustness.json
│   ├── route_network_complexity_300m.json
│   ├── route_network_complexity_by_vendor_300m.json
│   ├── spatial_hotspot_localization_results.json
│   ├── summary_results.json
│   ├── support_ratio_boundary_results.json
│   ├── targeted_monitoring_payoff.json
│   ├── temporal_and_length_analysis.json
│   ├── trajectory_path_persistence_results.json
│   ├── unique_flight_occupancy_results.json
│   ├── vendor_day_robustness.json
│   └── robustness_100m/                                 # 100 m grid parallel analyses
│       ├── alternative_corridor_definition_results.json
│       ├── breakpoint_specification_results.json
│       ├── cluster_robust_ci_results.json
│       ├── endpoint_buffer_exclusion_results.json
│       ├── endpoint_node_anchor_results.json
│       ├── exclude_terminal_zones_results.json
│       ├── interface_masking_results.json
│       ├── leave_one_group_out_results.json
│       ├── morphological_continuity_directionality_results.json
│       ├── null_model_suite_results.json
│       ├── out_of_sample_coupling_results.json
│       ├── phase_altitude_decomposition_results.json
│       ├── rho_star_bootstrap_ci_results.json
│       ├── spatial_hotspot_localization_results.json
│       ├── summary_results.json
│       ├── support_ratio_boundary_results.json
│       ├── trajectory_path_persistence_results.json
│       └── unique_flight_occupancy_results.json
└── datamap.md                                          # This file
```

---

## CSV Files

### 1. `air_ground_clustered_road_by_distance_bin_200m.csv`
Air vs. ground travel comparison aggregated by distance bin (200 m endpoint clustering).  
**Columns (12):** `dist_bin, n_directed_routes, n_orders, weighted_median_air_distance_km, weighted_median_road_distance_km, weighted_median_air_duration_min, weighted_median_osrm_duration_min, weighted_median_peak25_duration_min, weighted_median_time_delta_osrm_min, weighted_median_time_delta_peak25_min, weighted_share_aerial_faster_osrm, weighted_share_aerial_faster_peak25`

### 2. `air_ground_clustered_road_edges_200m.csv`
Per-edge air/ground comparison for 200 m endpoint-merged route connections.  
**Columns (31):** `edge_id, source_cluster, target_cluster, source_lat, source_lon, target_lat, target_lon, route_key, order_support, median_air_distance_km, mean_air_distance_km, median_air_duration_min, mean_air_duration_min, median_od_distance_km, sf_order_share, status, distance_m, duration_s, road_distance_km, road_duration_osrm_min, road_duration_peak25_min, road_duration_bus1757_min, road_to_air_distance_ratio, road_to_od_distance_ratio, time_delta_osrm_min, time_delta_peak25_min, time_delta_bus1757_min, aerial_faster_osrm, aerial_faster_peak25, aerial_faster_bus1757, dist_bin`

### 3. `air_ground_clustered_road_nodes_200m.csv`
Endpoint clusters (nodes) used in the air/ground road-routing comparison.  
**Columns (7):** `cluster_id, centroid_lat, centroid_lon, member_count, out_degree, out_strength_orders, member_cells`

### 4. `air_ground_clustered_road_osrm_cache_200m.csv`
OSRM routing cache for air/ground comparison edges.  
**Columns (4):** `route_key, status, distance_m, duration_s`

### 5. `altitude_band_sensitivity.csv`
Congestion statistics stratified by UAV altitude band.  
**Columns (22):** `band, altitude_lower_m, altitude_upper_m, samples, sample_share_percent, active_cells, subset_backbone_cells, full_backbone_overlap_share, rho_star, speed_free, speed_congested, speed_drop_percent, congestion_share_percent, rr, rr_ci_95_lower, rr_ci_95_upper, or_adj, or_ci_95_lower, or_ci_95_upper, mean_altitude_p25_m, mean_altitude_p50_m, mean_altitude_p75_m`

### 6. `altitude_point_distribution.csv`
Overall altitude distribution of trajectory points (1 m bins).  
**Columns (6):** `bin_index, bin_label, bin_start_m, bin_end_m, count, share_of_points`

### 7. `backbone_cutoff_sensitivity.csv`
Sensitivity of backbone definition to traffic-percentile cutoff choice.  
**Columns (28):** `cutoff_share, cutoff_label, active_cells, backbone_cells, backbone_cell_share, traffic_points_in_backbone, traffic_share, hotspot_cells_total, hotspot_cells_covered, hotspot_cells_covered_share, total_cell_window_samples, backbone_cell_window_samples, total_exceedance_samples, exceedance_samples_covered, exceedance_samples_covered_share, risk_backbone, risk_non_backbone, rr, rr_ci_95_lower, rr_ci_95_upper, or_adj, or_ci_95_lower, or_ci_95_upper, contingency_a, contingency_b, contingency_c, contingency_d`

### 8. `binned_speed_density_relations.csv`
Binned speed–density fundamental diagram data.  
**Columns (11):** `density_bin, density_min, density_max, density_center, sample_count, density_mean, speed_mean, speed_std, speed_count, regime_relative_to_rho_star, rho_star_reference`

### 9. `bootstrap_and_null_summaries.csv`
Summary statistics from bootstrap and null-model runs.  
**Columns (12):** `summary_id, category, description, n_draws, observed, mean, median, ci_95_lower, ci_95_upper, max, empirical_p_ge_observed, source_file`

### 10. `breakpoint_specification_summary.csv`
Summary of alternative breakpoint estimation specifications.  
**Columns (9):** `specification_id, description, rho_star, speed_drop_percent, n_bins, bin_width, min_samples, weighting, source_file`

### 11. `contingency_tables.csv`
Aggregated 2×2 contingency tables for all risk-ratio analyses.  
**Columns (21):** `analysis_id, analysis_group, description, exposed_group_label, outcome_label, rho_star, samples, exposed_and_exceedance, exposed_and_non_exceedance, unexposed_and_exceedance, unexposed_and_non_exceedance, risk_exposed, risk_unexposed, rr, rr_ci_95_lower, rr_ci_95_upper, or_adj, or_ci_95_lower, or_ci_95_upper, fisher_p, source_file`

### 12. `diurnal_activity_halfhour.csv`
Aggregate diurnal activity profile (30-minute bins, all days pooled).  
**Columns (8):** `slot_index, slot_label, total_points, share_of_points, mean_active_flights_per_day, median_active_flights_per_day, peak_active_flights_in_any_day, days_with_activity`

### 13. `diurnal_activity_halfhour_daily.csv`
Diurnal activity per date and 30-minute bin.  
**Columns (6):** `date, slot_index, slot_label, total_points, share_of_points_within_date, active_flights`

### 14. `diurnal_activity_hourly.csv`
Aggregate diurnal activity profile (1-hour bins, all days pooled).  
**Columns (8):** `slot_index, slot_label, total_points, share_of_points, mean_active_flights_per_day, median_active_flights_per_day, peak_active_flights_in_any_day, days_with_activity`

### 15. `endpoint_cell_altitude_summary.csv`
Altitude statistics per endpoint grid cell.  
**Columns (10):** `cell_id, n, starts, ends, median_altitude, mean_altitude, p25, p75, median_lon, median_lat`

### 16. `endpoint_relative_height_classification.csv`
Endpoint cells classified by relative height above ground (terrain + buildings + stations).  
**Columns (21):** `cell_id, n, starts, ends, median_altitude, mean_altitude, p25, p75, median_lon, median_lat, terrain_m, rel_ground_m, nearest_bldg_dist_m, nearest_bldg_height_m, bldg_count_within_150m, best_height_gap_150m, best_match_height_150m, best_match_dist_150m, nearest_station_m, station_adjacent_150m, height_class`

### 17. `endpoint_terrain_elevation_cache.csv`
SRTM terrain elevation lookup cache for endpoint coordinates.  
**Columns (4):** `lat, lon, terrain_m, source`

### 18. `figure_anchor_summary.csv`
Coverage statistics for monitoring anchor definitions (used in figures).  
**Columns (11):** `anchor_label, anchor_cells, active_cells_covered, active_cells_covered_share, hotspot_cells_covered, hotspot_cells_covered_share, endpoint_cells_covered, endpoint_cells_covered_share, exceedance_samples_covered, exceedance_samples_covered_share, sample_count_covered`

### 19. `figure_cell_metrics.csv`
Per-cell metrics supporting manuscript figures (traffic, congestion, endpoint/station flags).  
**Columns (23):** `rank, cell_id, total_points, traffic_share, cumulative_traffic_share, active_days, active_hours, is_backbone_cell, backbone_definition, exceedance_count, exceedance_share, cumulative_exceedance_share, total_samples, is_hotspot_cell, is_endpoint_cell, station_exact_cell, station_buffer1_cell, station_buffer2_cell, endpoint_buffer1_cell, lon, lat, traffic_rank_pct, log10_total_points`

### 20. `figure_threshold_subset_summary.csv`
Risk ratio results for density-threshold subsets (figure support).  
**Columns (7):** `subset_label, rho_star, speed_drop_percent, rr, rr_ci_95_lower, rr_ci_95_upper, samples`

### 21. `followup_compressed_paths_2025_2026.csv`
Grid-cell trajectory paths for 2025–2026 follow-up flights (compressed format).  
**Columns (8):** `flight_id, source_folder, route_code, flight_date_local, seq_index, cell_id, lon_center, lat_center`

### 22. `followup_date_route_summary.csv`
Per-date / per-route summary of 2025–2026 follow-up flights.  
**Columns (6):** `flight_date_local, source_folder, route_code, flights, first_start, last_end`

### 23. `followup_flight_summary_2025_2026.csv`
Per-flight metadata for the 2025–2026 follow-up dataset.  
**Columns (17):** `source_folder, route_code, file_name, flight_id, flight_date_local, flight_start_local, flight_end_local, duration_s, n_valid_points, n_waypoints, start_lon, start_lat, end_lon, end_lat, start_alt_agl, end_alt_agl, median_ground_speed`

### 24. `followup_persistence_flight_metrics.csv`
Per-flight persistence metrics comparing 2025–2026 flights against the 2023 backbone/hotspot reference.  
**Columns (30):** `flight_id, source_folder, route_code, file_name, flight_date_local, n_points, n_unique_sequence_cells, n_unique_cells, n_edges, edge_hits_on_2023_skeleton, edge_share_on_2023_skeleton, point_share_in_2023_backbone, unique_cell_share_in_2023_backbone, unique_cell_share_in_2023_hotspots, start_cell, end_cell, start_in_2023_endpoint, end_in_2023_endpoint, start_in_2023_endpoint_buffer1, end_in_2023_endpoint_buffer1, start_in_station_buffer1, end_in_station_buffer1, both_endpoints_in_2023_endpoint_buffer1, both_endpoints_in_station_buffer1, slow_points_total, slow_points_in_own_endpoint_buffer1, slow_point_share_in_own_endpoint_buffer1, point_share_in_own_endpoint_buffer1, slow_localization_lift_vs_point_share`

### 25. `followup_reference_route_skeleton_edges_support100.csv`
2023 reference route skeleton edges with ≥100 order support.  
**Columns (3):** `source_cell, target_cell, support`

### 26. `followup_route_group_metrics.csv`
2025–2026 follow-up metrics aggregated by route group.  
**Columns (8):** `route_group, flights, weighted_hits, total_edges, median_edge_share, weighted_edge_share, label, color`

### 27. `followup_trajectory_points_2025_2026.csv` *(large file)*
Full trajectory point data for 2025–2026 follow-up flights (grid-mapped).  
**Columns (20):** `source_folder, route_code, file_name, flight_id, flight_date_local, point_index, elapsed_s, datetime_local, longitude, latitude, altitude_msl, altitude_agl, vel_x, vel_y, vel_z, ground_speed, yaw_deg, grid_row, grid_col, cell_id, is_within_main_grid`

### 28. `grid_size_sensitivity.csv`
Grid-size sensitivity results (50–200 m tested; 100 m used as main specification).  
**Columns (11):** `grid_size_m, active_cells, active_share_full_grid, spatiotemporal_samples, top10_traffic_share, rho_star, speed_drop_percent, hotspot_cells, hotspot_share_active_cells, rr_backbone_vs_nonbackbone, n_binned_density_points`

### 29. `hotspot_exceedance_rank_table.csv`
Grid cells ranked by exceedance count (most congested first).  
**Columns (11):** `exceedance_rank, grid_cell_id, exceedance_count, exceedance_share, cumulative_exceedance_share, total_cell_window_samples, traffic_rank, is_backbone_cell, is_endpoint_cell, is_endpoint_buffer1_cell, rho_star_reference`

### 30. `hub_robustness.csv`
Leave-one-hub-out robustness results (66 UAV stations).  
**Columns (13):** `hub_name, removed_active_cells, removed_samples, removed_congested_samples, removed_congested_share, rho_star, speed_drop_percent, rr, rr_ci_95_lower, rr_ci_95_upper, or_adj, or_ci_95_lower, or_ci_95_upper`

### 31. `merged_route_connection_sensitivity.csv`
Sensitivity of route-connection counts to endpoint merge distance threshold.  
**Columns (7):** `merge_threshold_m, merged_endpoint_clusters, unique_directed_route_connections, unique_undirected_route_connections, self_merged_orders, nonself_order_share, difference_from_official_156`

### 32. `preview_distance_duration_bins.csv`
Distance and duration statistics binned by flight distance.  
**Columns (9):** `dist_bin, n, median_dist_km, q25_dist_km, q75_dist_km, median_duration_min, q25_duration_min, q75_duration_min, median_effective_speed_kmh`

### 33. `preview_fundamental_diagram_binned.csv`
Binned fundamental diagram (point-count density).  
**Columns (8):** `bin, sample_count, x_mean, y_mean, y_median, y_q25, y_q75, relation`

### 34. `preview_fundamental_diagram_cell_windows.csv` *(large file)*
Per-cell-window fundamental diagram data (unique-flight density).  
**Columns (8):** `cell_id, window_start, density_unique_flights, speed_mean, n_points, flow_entries, flow_entries_per_hour, window_date`

### 35. `preview_fundamental_diagram_refined_binned.csv`
Binned fundamental diagram using refined (unique-flight) density definition.  
**Columns (7):** `sample_count, x_mean, y_mean, y_median, y_q25, y_q75, subset, relation`

### 36. `preview_speed_phase_by_distance.csv` *(large file)*
Per-order speed decomposed by flight phase and distance bin.  
**Columns (6):** `order_id, phase, point_speed, dist_bin, path_length_km, duration_minutes, vendor`

### 37. `ranked_cell_traffic_table.csv`
Grid cells ranked by total traffic volume.  
**Columns (9):** `rank, grid_cell_id, total_points, traffic_share, cumulative_traffic_share, active_days, active_hours, is_backbone_cell, backbone_definition`

### 38. `real_station_anchor_sites_mapped.csv`
Real UAV station locations mapped to the analysis grid.  
**Columns (15):** `anchor_set, site_name, platform, note, open_time, lon, lat, grid_row, grid_col, grid_cell_id, is_test_site, is_active_cell, is_backbone_cell, is_hotspot_cell, is_endpoint_cell`

### 39. `robustness_summary.csv`
Master summary table of all robustness and sensitivity checks (47-column wide format).  
**Columns (47):** `check_id, category, description, notes, source_file, hotspot_cells, hotspot_share_active_cells, hotspot_share_full_grid, top_0_5pct_share, top_1pct_share, endpoint_cells, endpoint_share_active_cells, endpoint_corridor_overlap, congested_samples_in_endpoint_cells_share, congested_samples_within_1cell_of_endpoints_share, samples, rho_star, rr, rr_ci_95_lower, rr_ci_95_upper, or_adj, or_ci_95_lower, or_ci_95_upper, speed_free, speed_congested, speed_drop_percent, jaccard_with_main_backbone, rr_min, rr_median, rr_max, rho_star_min, rho_star_median, rho_star_max, runs, support_threshold, skeleton_edges, skeleton_nodes, backbone_coverage_share, mean_order_edge_coverage, share_orders_edge_coverage_ge_80pct, backbone_cells, n_components, largest_component_share, degree_ge3_share, weighted_median_component_axial_resultant, active_cells_covered_share, hotspot_cells_covered_share, endpoint_cells_covered_share, exceedance_samples_covered_share`

### 40. `route_day_robustness.csv`
Leave-one-route-day-out robustness results.  
**Columns (16):** `analysis, route_day_id, merged_route_id, removed_point_rows, removed_cell_windows, samples, active_cells, rho_star, speed_drop_percent, congestion_share_percent, rr, rr_ci_95_lower, rr_ci_95_upper, or_adj, or_ci_95_lower, or_ci_95_upper`

### 41. `route_network_edges_300m.csv`
Edges of the route network (300 m endpoint merge threshold).  
**Columns (3):** `source_cluster, target_cluster, order_support`

### 42. `route_network_edges_300m_by_vendor.csv`
Route network edges split by vendor (SFexpress / Meituan).  
**Columns (4):** `vendor, source_cluster, target_cluster, order_support`

### 43. `route_network_nodes_300m.csv`
Nodes of the route network with centrality metrics (300 m threshold).  
**Columns (14):** `cluster_id, centroid_lat, centroid_lon, member_count, degree, strength_orders, clustering_coeff, component_id, component_size, betweenness_unweighted, betweenness_unweighted_norm, betweenness_invflow, betweenness_invflow_norm, member_cells`

### 44. `route_network_nodes_300m_by_vendor.csv`
Route network nodes split by vendor (SFexpress / Meituan).  
**Columns (13):** `vendor, cluster_id, centroid_lat, centroid_lon, member_count, degree, strength_orders, clustering_coeff, component_id, component_size, betweenness_invflow, betweenness_invflow_norm, member_cells`

### 45. `targeted_monitoring_payoff.csv`
Payoff of targeted monitoring strategies vs. random baseline.  
**Columns (8):** `strategy, budget_share_active_cells, budget_cell_count, captured_exceedance_samples, captured_exceedance_share, captured_traffic_share, lift_vs_random_exceedance, lift_vs_random_traffic`

### 46. `targeted_monitoring_random_baseline.csv`
Random monitoring baseline statistics for payoff comparison.  
**Columns (7):** `strategy, cells_monitored, mean_random_exceedance_share, p95_random_exceedance_share, observed_exceedance_share, observed_over_random_mean, observed_over_random_p95`

### 47. `trajectory_length_distribution.csv`
Per-order trajectory length and duration statistics.  
**Columns (9):** `order_id, vendor, num_points, duration_minutes, path_length_km, od_distance_km, circuity_ratio, start_time, end_time`

### 48. `vendor_day_robustness.csv`
Leave-one-vendor-day-out robustness results.  
**Columns (15):** `analysis, group, samples, active_cells, rho_star, speed_drop_percent, congestion_share_percent, rr, rr_ci_95_lower, rr_ci_95_upper, or_adj, or_ci_95_lower, or_ci_95_upper, removed_point_rows, removed_cell_windows`

### 49. `vendor_network_summary_300m.csv`
Global network summary statistics per vendor.  
**Columns (9):** `vendor, nodes, edges, components, largest_component_share_pct, mean_local_clustering, average_degree, max_degree, max_strength_orders`

---

## JSON Files (in `source_json/`)

All JSON files reside under `source_json/` (39 files in the root, plus 18 in `robustness_100m/`).  
They store structured analysis outputs consumed by the Python plotting scripts. Key top-level fields are noted below.

### Main directory (39 files)

| File | Description | Key top-level keys |
|------|-------------|-------------------|
| `air_ground_clustered_road_geometries_200m.json` | Geometry features for 200 m endpoint clustering | `type`, `features` (GeoJSON-like) |
| `air_ground_clustered_road_summary_200m.json` | Air/ground road-routing summary at 200 m merge | `analysis`, `merge_radius_m`, `merged_endpoint_clusters`, `directed_connections`, `weighted_statistics` |
| `alternative_corridor_definition_results.json` | Alternative corridor definition (top 10% distinct orders) | `corridor_definition`, `n_corridor_cells`, `jaccard_vs_point_count`, `rr`, `or` |
| `altitude_band_sensitivity.json` | Altitude-band congestion sensitivity | `bands`, `reference_quantiles`, per-band: `rho_star`, `rr`, `or`, `speed_drop_percent` |
| `backbone_cutoff_sensitivity.json` | Backbone definition cutoff sensitivity | `cutoffs`, per-cutoff: `backbone_cells`, `rr`, `or`, `contingency` |
| `breakpoint_results.json` | Dual-method breakpoint estimation | `piecewise_regression`, `inflection_method`, `consensus` |
| `breakpoint_specification_results.json` | Alternative breakpoint specifications | `main_result`, `sensitivity_specs` |
| `cluster_robust_ci_results.json` | Bootstrap CI for RR/OR (n=1000) | `n_bootstraps`, `rr`, `or`, `bootstrap_ci` |
| `endpoint_buffer_exclusion_results.json` | Endpoint & buffer exclusion analysis | `endpoint_cells`, `buffer_exclusion`, `rr_after_exclusion` |
| `endpoint_node_anchor_results.json` | Endpoint–hotspot overlap statistics | `endpoint_cells`, `hotspot_overlap`, `exceedance_concentration` |
| `endpoint_relative_height_summary.json` | Endpoint height classification (terrain + buildings + stations) | `height_classes`, `terrain_stats`, `building_stats`, `station_stats` |
| `exclude_terminal_zones_results.json` | Terminal-zone exclusion analysis | `terminal_cells`, `excluded_samples`, `re_estimated_rho_star` |
| `followup_persistence_summary.json` | 2025–2026 follow-up persistence summary | `flights`, `edge_share`, `point_share`, `slow_localization` |
| `grid_size_sensitivity.json` | Grid-size sensitivity (50–200 m) | `grid_sizes`, per-size: `active_cells`, `rho_star`, `rr` |
| `grid_switch_100m_metadata.json` | 100 m grid metadata | `gini`, `top_1pct_share`, `top_10pct_share` |
| `hub_robustness.json` | Leave-one-hub-out results | `hubs`, per-hub: `rr`, `rho_star`, `removed_congested_share` |
| `interface_masking_results.json` | Interface-zone masking analysis | `interface_windows`, `rr_without_interface` |
| `leave_one_group_out_results.json` | Leave-one-day-out results | `groups`, per-group: `rr`, `rho_star` |
| `merged_route_connection_sensitivity.json` | Route merge sensitivity | `thresholds`, per-threshold: `merged_clusters`, `connections` |
| `morphological_continuity_directionality_results.json` | Backbone component morphology | `n_components`, `largest_component`, `degree_stats`, `pca_elongation` |
| `null_model_suite_results.json` | Null model comparison (4 models) | `null_models`, per-model: `rr_null_distribution`, `empirical_p` |
| `out_of_sample_coupling_results.json` | Forward/reverse temporal splits | `forward_rr`, `reverse_rr`, `asymmetry_note` |
| `phase_altitude_decomposition_results.json` | Phase × altitude decomposition | `phases`, per-phase: `rho_star`, `rr`, `speed_drop` |
| `preview_fundamental_diagram_refined_summary.json` | Refined fundamental diagram summary | `density_definition`, `n_windows`, `rho_star_unique` |
| `preview_fundamental_diagram_summary.json` | Fundamental diagram definitions & summary | `density_def`, `flow_def`, `speed_def`, `n_windows` |
| `preview_speed_advantage_summary.json` | UAV speed advantage by distance bin | `distance_bins`, per-bin: `median_speed`, `advantage_kmh` |
| `real_station_anchor_results.json` | Real station anchor coverage | `stations`, `anchor_cells`, `exceedance_coverage`, `rr` |
| `rho_star_bootstrap_ci_results.json` | Bootstrap CI for rho* (n=1000) | `rho_star_point`, `bootstrap_median`, `ci` |
| `route_day_robustness.json` | Route-day robustness results | `groups`, per-group: `rr`, `rho_star` |
| `route_network_complexity_300m.json` | Route network complexity (300 m merge) | `nodes`, `edges`, `components`, `degree_distribution` |
| `route_network_complexity_by_vendor_300m.json` | Route network complexity by vendor | `MT`, `SF`, per-vendor: `nodes`, `edges`, `degree` |
| `spatial_hotspot_localization_results.json` | Hotspot localization statistics | `hotspot_cells`, `exceedance_coverage` |
| `summary_results.json` | Composite summary of all robustness checks | `out_of_sample`, `unique_occupancy`, `terminal_exclusion`, `cluster_bootstrap` |
| `support_ratio_boundary_results.json` | Spatial boundary / support ratio analysis | `active_cells`, `full_grid`, `convex_hull_ratio` |
| `targeted_monitoring_payoff.json` | Monitoring strategy payoff curves | `strategies`, per-strategy: `exceedance_captured`, `lift` |
| `temporal_and_length_analysis.json` | Temporal and trajectory-length descriptive statistics | `peak_hour`, `trajectory_stats`, `altitude_distribution` |
| `trajectory_path_persistence_results.json` | Route skeleton / path persistence analysis | `skeleton_edges`, `coverage_stats`, `top_edges` |
| `unique_flight_occupancy_results.json` | Unique-flight occupancy fundamental diagram | `rho_star_unique`, `rr`, `ci` |
| `vendor_day_robustness.json` | Vendor-day robustness results | `groups`, per-group: `rr`, `rho_star` |

### `robustness_100m/` subdirectory (18 files)

Parallel robustness analyses on a 100 m grid (sensitivity to grid resolution). Same schema as the main-directory counterparts:

`alternative_corridor_definition_results.json`, `breakpoint_specification_results.json`, `cluster_robust_ci_results.json`, `endpoint_buffer_exclusion_results.json`, `endpoint_node_anchor_results.json`, `exclude_terminal_zones_results.json`, `interface_masking_results.json`, `leave_one_group_out_results.json`, `morphological_continuity_directionality_results.json`, `null_model_suite_results.json`, `out_of_sample_coupling_results.json`, `phase_altitude_decomposition_results.json`, `rho_star_bootstrap_ci_results.json`, `spatial_hotspot_localization_results.json`, `summary_results.json`, `support_ratio_boundary_results.json`, `trajectory_path_persistence_results.json`, `unique_flight_occupancy_results.json`

---

*Generated on 2026-06-11. Total: 49 CSV files + 57 JSON files.*
