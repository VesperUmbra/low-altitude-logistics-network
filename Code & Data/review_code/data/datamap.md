# Data Map — `review_code/data/`

This document describes every data file under `Code & Data/review_code/data/`, which holds the **pipeline-processed intermediate outputs** that feed the analysis and figure-generation scripts.

---

## Directory Structure

```
data/
├── processed/
│   ├── full_100m/                    # Main 100 m grid processing
│   │   ├── cleaned_data.csv          # Cleaned raw trajectory points
│   │   ├── trajectories.pkl          # Serialized trajectory objects (Python pickle)
│   │   ├── trajectory_stats.csv      # Per-order trajectory statistics
│   │   ├── data_summary.txt          # Summary report (Chinese → translated)
│   │   ├── processing_progress.json  # Processing progress metadata
│   │   ├── grid/
│   │   │   ├── cell_stats.csv        # Per-cell aggregate statistics
│   │   │   ├── gridded_data.csv      # Raw points mapped to grid cells
│   │   │   ├── spatiotemporal_stats.csv # Per-window spatiotemporal metrics
│   │   │   ├── grid_info.json        # Grid definition metadata
│   │   │   └── grid_summary.txt      # Gridding summary report (Chinese → translated)
│   │   └── temp/
│   │       └── processed_chunk_000–019.pkl  # Intermediate chunk files
│   ├── grid_refined_100m/            # Refined grid with coordinate precision
│   │   ├── gridded_data.csv          # Gridded data with rounded coords
│   │   ├── grid_info.json            # Refined grid metadata
│   │   ├── od_pairs.csv              # Origin–destination pairs
│   │   └── summary.txt               # Refined grid summary (Chinese → translated)
│   └── grid_sensitivity/
│       └── cleaned_valid_points_minimal_120s.csv  # Minimal cleaned data for sensitivity
└── results/
    └── full_100m/
        └── diagram/
            ├── fundamental_data.csv      # Binned fundamental diagram data
            ├── breakpoint_results.json   # Breakpoint estimation results
            ├── diagram_metrics.json      # Free-flow / congested-flow metrics
            └── diagram_summary.txt       # Diagram analysis summary (Chinese → translated)
```

---

## CSV Files (9)

### 1. `processed/full_100m/cleaned_data.csv` *(large)*
Full cleaned trajectory point dataset. One row per GPS point.  
**Columns (15):** `datetime, date, hour, minute, second, day_of_week, is_weekend, speed, altitude, yaw_clean, flight_time, longitude, latitude, order_id, sn, vendor`

### 2. `processed/full_100m/trajectory_stats.csv`
Per-order summary statistics for each flight.  
**Columns (8):** `order_id, num_points, duration_seconds, avg_speed, avg_altitude, start_time, end_time, vendor`

### 3. `processed/full_100m/grid/cell_stats.csv`
Aggregate statistics per spatial grid cell (traffic rank, speed, altitude, activity span).  
**Columns (13):** `cell_id, row, col, total_points, avg_speed, std_speed, avg_altitude, std_altitude, first_seen, last_seen, active_hours, active_days, rank`

### 4. `processed/full_100m/grid/gridded_data.csv` *(large)*
Raw trajectory points mapped to grid cells (100 m, 120 s windows).  
**Columns (6):** `cell_id, row, col, datetime, window_start, speed, altitude`

### 5. `processed/full_100m/grid/spatiotemporal_stats.csv` *(large)*
Per-cell-window spatiotemporal statistics for fundamental diagram construction.  
**Columns (11):** `cell_id, window_start, n_points, mean_speed, std_speed, min_speed, max_speed, mean_altitude, hour, day_of_week, is_weekend`

### 6. `processed/grid_refined_100m/gridded_data.csv` *(large)*
Gridded data with refined coordinate precision (4 decimal places) and UTM projection.  
**Columns (23):** `datetime, date, hour, minute, second, day_of_week, is_weekend, speed, altitude, yaw_clean, flight_time, longitude, latitude, order_id, sn, vendor, lon_rounded, lat_rounded, easting, northing, col_idx, row_idx, cell_id`

### 7. `processed/grid_refined_100m/od_pairs.csv`
Origin–destination pairs extracted from the refined grid.  
**Columns (10):** `order_id, start_cell, end_cell, start_time, end_time, duration, start_lon, start_lat, end_lon, end_lat`

### 8. `processed/grid_sensitivity/cleaned_valid_points_minimal_120s.csv` *(large)*
Minimal cleaned dataset (120 s valid-point filter) for grid-size sensitivity analysis.  
**Columns (5):** `order_id, window_key, speed, longitude, latitude`

### 9. `results/full_100m/diagram/fundamental_data.csv`
Binned speed–density data for fundamental diagram plotting.  
**Columns (9):** `density_bin, sample_count, density_mean, speed_mean, speed_std, speed_count, density_min, density_max, density_center`

---

## JSON Files (5)

### 1. `processed/full_100m/processing_progress.json`
Pipeline execution log for the main data processing step.  
**Keys:** `timestamp`, `total_rows_processed`, `chunks_processed`, `elapsed_time_seconds`, `rows_per_second`, `completion_percentage`

### 2. `processed/full_100m/grid/grid_info.json`
Grid definition for the main 100 m analysis grid.  
**Keys:** `grid_size_m`, `time_window_s`, `n_rows`, `n_cols`, `total_cells`, `min_lat`, `max_lat`, `min_lon`, `max_lon`, `meters_per_degree_lat`, `meters_per_degree_lon`, `creation_time`

### 3. `processed/grid_refined_100m/grid_info.json`
Refined grid definition (coordinate-rounded, 4 decimal places).  
**Keys:** `grid_size`, `lon_precision`, `lat_precision`, `n_cols`, `n_rows`, `total_cells`, `used_cells`, `min_lon`, `max_lon`, `min_lat`, `max_lat`

### 4. `results/full_100m/diagram/breakpoint_results.json`
Congestion breakpoint (ρ*) estimation from two methods and consensus.  
**Keys:** `estimates` (array of objects with `method`, `rho_star`, `speed_drop`), `consensus_rho_star`, `correlation_coefficient`, `correlation_p_value`, `congestion_share_percent`, `n_samples_total`, `n_congested_samples`, `piecewise_settings`, `analysis_time`

### 5. `results/full_100m/diagram/diagram_metrics.json`
Free-flow and congested-flow regime statistics from the fundamental diagram.  
**Keys:** `rho_star`, `free_flow_samples`, `congested_flow_samples`, `free_flow_density_range`, `congested_flow_density_range`, `free_flow_speed_mean`, `congested_flow_speed_mean`, `free_flow_speed_std`, `congested_flow_speed_std`, `speed_drop_at_rho_star`, `free_flow_slope`, `congested_flow_slope`

---

## TXT Files (4) — Translated to English

### 1. `processed/full_100m/data_summary.txt`
Data processing summary: row count, temporal range, trajectory statistics, and hourly distribution of 3,867,146 points over October 2023 (31 days, 46,409 flights).

### 2. `processed/full_100m/grid/grid_summary.txt`
Gridding summary: 12,159 spatial cells, 1,423,376 spatiotemporal windows, top 10% of cells carry 76.3% of traffic, 100 m grid, 120 s windows, bounding box 22.31°–22.87°N × 113.31°–114.25°E.

### 3. `processed/grid_refined_100m/summary.txt`
Refined-grid summary: coordinate precision to 4 decimal places, 618×961 grid, 12,101 used cells (2.04% utilization), 45,407 OD pairs with 351 unique pairs.

### 4. `results/full_100m/diagram/diagram_summary.txt`
Fundamental diagram analysis: consensus ρ* = 11.5, speed–density correlation −0.418 (p ≈ 0), 3.37% of spatiotemporal samples congested, free-flow mean speed 7.37 m/s vs. congested 1.13 m/s.

---

## PKL Files (21) — Deleted

All 21 pickle files (intermediate serialized Python objects and chunk caches) have been removed:

- `processed/full_100m/trajectories.pkl`
- `processed/full_100m/temp/processed_chunk_000.pkl` through `processed_chunk_019.pkl` (20 chunk files)

---

*Generated on 2026-06-11.*
