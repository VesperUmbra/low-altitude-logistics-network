# A sparse, station-anchored low-altitude logistics network in routine drone delivery

**Code & Data Repository**

---

## Overview

This repository contains the analysis code, processed outputs, and figure-generation scripts accompanying the manuscript:

> *"A sparse, station-anchored low-altitude logistics network in routine drone delivery"*

The study analyzes 46,409 drone delivery trajectories reconstructed from 3,867,146 cleaned telemetry records in Shenzhen, China (October 2024, 22 operating days). Using a 100 m × 2 min gridded framework, it shows that routine drone delivery operates through a sparse, hierarchical, and station-anchored low-altitude logistics network rather than a diffuse collection of flights.

**Key findings:**
- Activity occupies **2.0%** of the administrative-extent analysis grid
- The top **10%** of active cells (intensity backbone) carry **76.3%** of point-weighted traffic
- **210** recurrent endpoint cells and **66** mapped drone stations anchor the high-load network
- Low-speed high-record states are concentrated around station neighborhoods and detectable on busy route segments
- Surface-network time advantages are distance- and speed-dependent

---

## ⚠️ Important: Raw Data Access

**The complete raw telemetry data is excluded from this repository.**

Processed data files (CSV) are provided in truncated form (header + 5 rows) for format inspection only. Structured analysis outputs (JSON) have had values stripped, retaining only the key/schema structure. Full data files can be requested by email:

> **p.liu@buaa.edu.cn**

Please include a brief description of your intended use.

---

## Repository Structure

```
├── requirements.txt                 # Python package dependencies
├── Code & Data/
│   ├── review_code/                 # Analysis pipeline scripts
│   │   ├── *.py                     # 22 Python analysis & robustness scripts
│   │   ├── extract_followup_logs_to_csv.m  # MATLAB: follow-up log parser
│   │   ├── data/                    # Pipeline intermediate outputs (truncated)
│   │   │   ├── datamap.md           #   Documentation of data files
│   │   │   ├── processed/           #   Gridding & trajectory processing outputs
│   │   │   └── results/             #   Fundamental diagram analysis outputs
│   │   └── src/                     # Shared utility modules
│   │       ├── data/                #   Data processing utilities
│   │       └── analysis/            #   Congestion analysis utilities
│   └── review_data/                 # Analysis results consumed by figures
│       ├── datamap.md               #   Documentation of all data files
│       ├── *.csv                    #   49 processed CSV tables (truncated)
│       └── source_json/             #   Structured analysis outputs (values stripped)
│           └── robustness_100m/     #   Parallel 100 m grid robustness checks
└── Figures/
    ├── background/                  # Basemap assets
    ├── manuscript_scripts/          # Python scripts generating manuscript figures
    ├── SI_scripts/                  # Python scripts generating SI figures
```

---

## Code

### Analysis pipeline (`Code & Data/review_code/`)

The analysis is organized as a set of standalone scripts, each performing one coherent analysis task. Key modules:

| Script | Purpose |
|--------|---------|
| `build_figure_inputs.py` | Assembles input datasets consumed by figure-generation scripts |
| `grid_refine.py` | Produces the refined 100 m grid with coordinate-precision control |
| `promote_100m_branch.py` | Promotes the 100 m grid branch as the main analysis framework |
| `refresh_reviewer_summary_tables_100m.py` | Regenerates summary tables at 100 m resolution |

**Core analyses:**
| Script | Analysis |
|--------|----------|
| `run_robustness.py` | Master robustness suite (spatial, temporal, definitional) |
| `run_backbone_cutoff_sensitivity.py` | Sensitivity of backbone definition to traffic-percentile cutoff |
| `run_endpoint_relative_height_analysis.py` | Endpoint relative-height classification (terrain, buildings, stations) |
| `run_endpoint_route_merge_sensitivity.py` | Sensitivity of route connections to endpoint merge distance |
| `run_station_anchor.py` | Real station-to-grid-cell anchoring |
| `run_targeted_monitoring.py` | Targeted monitoring strategy payoff analysis |
| `run_temporal_and_length_analysis.py` | Temporal activity patterns & trajectory-length distributions |
| `run_followup_persistence.py` | 2025–2026 follow-up persistence analysis |
| `run_route_network_complexity_analysis.py` | Route network topology (300 m merge threshold) |
| `run_route_network_complexity_by_vendor.py` | Route network topology split by vendor |
| `run_clustered_real_road_routing_200m.py` | Air vs. ground comparison using real road routing (OSRM) |

**Supplementary Information (SI) analyses:**
| Script | Analysis |
|--------|----------|
| `run_SI_preview_fundamental_diagram.py` | Fundamental diagram estimation |
| `run_SI_preview_speed_advantage.py` | Drone speed advantage by distance bin |
| `run_SI_altitude_band_sensitivity.py` | Congestion stratified by UAV altitude band |
| `run_SI_grid_size_sensitivity.py` | Grid-size sensitivity (50–200 m) |
| `run_SI_route_day_hub_robustness.py` | Leave-one-route-day-out & hub-out robustness |
| `run_SI_vendor_day_robustness.py` | Leave-one-vendor-day-out robustness |

**Shared utilities (`src/`):**
| Module | Purpose |
|--------|---------|
| `src/data/process.py` | Core data processing functions |
| `src/data/grid.py` | Grid construction and cell-mapping utilities |
| `src/analysis/congestion.py` | Congestion diagnostics (ρ*, risk ratios, contingency tables) |

### Figure generation (`Figures/`)

| Directory | Contents |
|-----------|----------|
| `manuscript_scripts/` | Python scripts generating manuscript Figures 1–4 |
| `SI_scripts/` | Python scripts generating SI Figures S3–S18 |
| `background/` | CartoDB Positron basemap (HTML, PNG, SVG) |

A shared `paper_plot_style.py` module (present in both `manuscript_scripts/` and `SI_scripts/`) provides consistent styling, color palettes, panel labeling, and figure-saving utilities.

---

## Data

### Processed results (`Code & Data/review_data/`)

The `review_data/` directory contains 49 CSV files and 57 JSON files (in `source_json/`) that serve as the direct input to the figure-generation scripts. All files are documented in `datamap.md`. CSV files are truncated to header + 5 rows; JSON values have been stripped, retaining only the key/schema structure.

### Pipeline intermediates (`Code & Data/review_code/data/`)

Intermediate outputs from the data processing pipeline (gridding, trajectory stats, fundamental diagram estimation) are documented in `data/datamap.md`.

---

## Requirements

Python dependencies are listed in `requirements.txt`:

```
matplotlib
numpy
pandas
scipy
openpyxl
ruptures
```

Install with:

```bash
pip install -r requirements.txt
```

One script (`extract_followup_logs_to_csv.m`) requires MATLAB.

---

## Citation

If you use the code or data from this repository, please cite the accompanying manuscript (citation details to be added upon publication).

---

## License

This repository is provided for research and reproducibility purposes. Please contact the authors for usage terms.

---

## Contact

For data access requests or questions about the repository, please contact:

**p.liu@buaa.edu.cn**
