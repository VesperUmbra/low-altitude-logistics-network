from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT.parent / "review_data" / "source_json" / "robustness_100m"
CELL_STATS_FILE = ROOT / "data" / "processed" / "full_100m" / "grid" / "cell_stats.csv"
SPATIOTEMPORAL_FILE = ROOT / "data" / "processed" / "full_100m" / "grid" / "spatiotemporal_stats.csv"
CLEANED_DATA_FILE = ROOT / "data" / "processed" / "full_100m" / "cleaned_data.csv"
OD_PAIRS_FILE = ROOT / "data" / "processed" / "grid_refined_100m" / "od_pairs.csv"
FUNDAMENTAL_FILE = ROOT / "data" / "results" / "full_100m" / "diagram" / "fundamental_data.csv"
BREAKPOINT_FILE = ROOT / "data" / "results" / "full_100m" / "diagram" / "breakpoint_results.json"
GRID_INFO_FILE = ROOT / "data" / "processed" / "full_100m" / "grid" / "grid_info.json"


def relative_risk_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    rr = (a / (a + b)) / (c / (c + d))
    se = math.sqrt((1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + d)))
    delta = 1.96 * se
    return rr, math.exp(math.log(rr) - delta), math.exp(math.log(rr) + delta)


def odds_ratio_ha_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (aa * dd) / (bb * cc)
    se = math.sqrt((1 / aa) + (1 / bb) + (1 / cc) + (1 / dd))
    delta = 1.96 * se
    return odds_ratio, math.exp(math.log(odds_ratio) - delta), math.exp(math.log(odds_ratio) + delta)


def simplified_breakpoint(densities: np.ndarray, speeds: np.ndarray, window: int = 5) -> float:
    order = np.argsort(densities)
    x = np.asarray(densities)[order]
    y = np.asarray(speeds)[order]
    if len(x) < (window * 2 + 1):
        return float(x[len(x) // 2])

    best_density = float(x[len(x) // 2])
    best_drop = -np.inf
    for idx in range(window, len(y) - window):
        prev_mean = float(np.mean(y[idx - window : idx]))
        next_mean = float(np.mean(y[idx : idx + window]))
        drop = prev_mean - next_mean
        if drop > best_drop:
            best_drop = drop
            best_density = float(x[idx])
    return best_density


class RobustnessAnalyzer:
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        n_bootstraps: int = 1000,
        seed: int = 42,
        rho_bootstrap_seed: int = 2,
        n_null_permutations: int = 50,
        cell_stats_file: Path = CELL_STATS_FILE,
        spatiotemporal_file: Path = SPATIOTEMPORAL_FILE,
        cleaned_data_file: Path = CLEANED_DATA_FILE,
        od_pairs_file: Path = OD_PAIRS_FILE,
        fundamental_file: Path = FUNDAMENTAL_FILE,
        breakpoint_file: Path = BREAKPOINT_FILE,
        grid_info_file: Path = GRID_INFO_FILE,
    ):
        self.output_dir = Path(output_dir)
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.rho_bootstrap_seed = rho_bootstrap_seed
        self.n_null_permutations = n_null_permutations
        self.cell_stats_file = Path(cell_stats_file)
        self.spatiotemporal_file = Path(spatiotemporal_file)
        self.cleaned_data_file = Path(cleaned_data_file)
        self.od_pairs_file = Path(od_pairs_file)
        self.fundamental_file = Path(fundamental_file)
        self.breakpoint_file = Path(breakpoint_file)
        self.grid_info_file = Path(grid_info_file)
        self.rng = np.random.default_rng(seed)

        self.cell_stats: pd.DataFrame | None = None
        self.spatiotemporal: pd.DataFrame | None = None
        self.fundamental: pd.DataFrame | None = None
        self.grid_info: dict | None = None
        self.rho_star: float | None = None
        self.full_corridor_cells: set[str] = set()
        self.order_level_spatiotemporal: pd.DataFrame | None = None
        self.point_level: pd.DataFrame | None = None
        self.order_indices: list[np.ndarray] | None = None
        self.mapped_od_pairs: pd.DataFrame | None = None
        self.trajectory_sequences: list[dict] | None = None
        self.results: dict[str, dict] = {}

    def _time_window_seconds(self) -> int:
        if self.grid_info is None:
            return 300
        return int(self.grid_info.get("time_window_s", 300))

    def _time_window_minutes_label(self) -> str:
        seconds = self._time_window_seconds()
        minutes = seconds / 60.0
        return f"{minutes:g}"

    def _window_floor_alias(self) -> str:
        seconds = self._time_window_seconds()
        if seconds % 60 == 0:
            return f"{seconds // 60}min"
        return f"{seconds}s"

    def _map_to_full_grid(self, lon: pd.Series, lat: pd.Series) -> tuple[pd.Series, pd.Series]:
        assert self.grid_info is not None

        rows = (
            ((lat - float(self.grid_info["min_lat"])) * float(self.grid_info["meters_per_degree_lat"]))
            / float(self.grid_info["grid_size_m"])
        ).astype(int)
        cols = (
            ((lon - float(self.grid_info["min_lon"])) * float(self.grid_info["meters_per_degree_lon"]))
            / float(self.grid_info["grid_size_m"])
        ).astype(int)
        rows = rows.clip(lower=0, upper=int(self.grid_info["n_rows"]) - 1)
        cols = cols.clip(lower=0, upper=int(self.grid_info["n_cols"]) - 1)
        return rows.astype("int16"), cols.astype("int16")

    def _load_order_level_spatiotemporal(self) -> pd.DataFrame:
        assert self.grid_info is not None

        if self.order_level_spatiotemporal is not None:
            return self.order_level_spatiotemporal

        order_level = pd.read_csv(
            self.cleaned_data_file,
            usecols=["datetime", "speed", "longitude", "latitude", "order_id"],
            parse_dates=["datetime"],
        )
        rows, cols = self._map_to_full_grid(order_level["longitude"], order_level["latitude"])
        order_level["row"] = rows
        order_level["col"] = cols
        order_level["cell_id"] = rows.astype(str) + "_" + cols.astype(str)
        order_level["window_start"] = order_level["datetime"].dt.floor(self._window_floor_alias())

        aggregated = (
            order_level.groupby(["row", "col", "cell_id", "window_start"], observed=True)
            .agg(
                n_unique=("order_id", "nunique"),
                mean_speed=("speed", "mean"),
            )
            .reset_index()
        )
        self.order_level_spatiotemporal = aggregated
        return aggregated

    def _load_point_level(self) -> pd.DataFrame:
        assert self.grid_info is not None

        if self.point_level is not None:
            return self.point_level

        point_level = pd.read_csv(
            self.cleaned_data_file,
            usecols=["datetime", "speed", "yaw_clean", "longitude", "latitude", "order_id", "vendor"],
            parse_dates=["datetime"],
        )
        rows, cols = self._map_to_full_grid(point_level["longitude"], point_level["latitude"])
        point_level["cell_id"] = rows.astype(str) + "_" + cols.astype(str)
        point_level["window_start"] = point_level["datetime"].dt.floor(self._window_floor_alias())
        point_level["date"] = point_level["window_start"].dt.date
        point_level["vendor"] = point_level["vendor"].astype("category")
        point_level["order_code"] = pd.factorize(point_level["order_id"], sort=False)[0].astype("int32")
        self.order_indices = [
            np.asarray(indices, dtype=np.int32)
            for indices in point_level.groupby("order_code", sort=False).indices.values()
        ]
        self.point_level = point_level[
            ["window_start", "date", "speed", "yaw_clean", "cell_id", "vendor", "order_code"]
        ].copy()
        return self.point_level

    def _load_trajectory_sequences(self) -> list[dict]:
        assert self.grid_info is not None

        if self.trajectory_sequences is not None:
            return self.trajectory_sequences

        raw = pd.read_csv(
            self.cleaned_data_file,
            usecols=["order_id", "datetime", "longitude", "latitude"],
            parse_dates=["datetime"],
        )
        rows, cols = self._map_to_full_grid(raw["longitude"], raw["latitude"])
        raw["cell_id"] = rows.astype(str) + "_" + cols.astype(str)
        raw = raw.sort_values(["order_id", "datetime"]).reset_index(drop=True)

        sequences: list[dict] = []
        for order_id, group in raw.groupby("order_id", sort=False):
            cells = group["cell_id"].to_numpy(dtype=object)
            if len(cells) < 2:
                continue

            keep = np.empty(len(cells), dtype=bool)
            keep[0] = True
            keep[1:] = cells[1:] != cells[:-1]
            sequence = cells[keep].tolist()
            if len(sequence) < 2:
                continue

            undirected_edges = [tuple(sorted((a, b))) for a, b in zip(sequence[:-1], sequence[1:]) if a != b]
            directed_edges = [(a, b) for a, b in zip(sequence[:-1], sequence[1:]) if a != b]
            if not undirected_edges:
                continue

            sequences.append(
                {
                    "order_id": str(order_id),
                    "cell_sequence": tuple(sequence),
                    "unique_cells": set(sequence),
                    "undirected_edges": undirected_edges,
                    "directed_edges": directed_edges,
                }
            )

        self.trajectory_sequences = sequences
        return sequences

    def _aggregate_point_level(self, point_level: pd.DataFrame) -> pd.DataFrame:
        aggregated = (
            point_level.groupby(["cell_id", "window_start"], observed=True)
            .agg(
                n_points=("speed", "size"),
                mean_speed=("speed", "mean"),
            )
            .reset_index()
        )
        aggregated["date"] = aggregated["window_start"].dt.date
        aggregated["date_hour"] = aggregated["window_start"].dt.strftime("%Y-%m-%d-%H")
        return aggregated

    def _build_binned_diagram(
        self,
        data: pd.DataFrame,
        density_col: str,
        *,
        bin_width: int = 1,
        min_samples: int = 10,
        use_interval_center: bool = True,
    ) -> pd.DataFrame:
        valid = data[(data[density_col] > 0) & (data["mean_speed"].notna()) & (data["mean_speed"] > 0)].copy()
        density_bins = np.arange(0, int(valid[density_col].max()) + bin_width + 1, bin_width)
        valid["density_bin"] = pd.cut(valid[density_col], bins=density_bins, right=False)
        binned = (
            valid.groupby("density_bin", observed=False)
            .agg(
                sample_count=(density_col, "count"),
                density_mean=(density_col, "mean"),
                speed_mean=("mean_speed", "mean"),
            )
            .reset_index()
        )
        if use_interval_center:
            binned["density_center"] = binned["density_bin"].apply(lambda interval: float((interval.left + interval.right) / 2))
        else:
            binned["density_center"] = binned["density_mean"]
        binned = binned[binned["sample_count"] >= min_samples].reset_index(drop=True)
        return binned

    def _estimate_piecewise_from_binned(self, binned: pd.DataFrame, *, weighted: bool = False) -> dict:
        x = binned["density_center"].to_numpy(dtype=float).reshape(-1, 1)
        y = binned["speed_mean"].to_numpy(dtype=float)
        weights = binned["sample_count"].to_numpy(dtype=float)

        best_breakpoint = float(binned.iloc[len(binned) // 2]["density_center"])
        best_objective = float("inf")
        max_breakpoint = min(len(binned) - 5, 50)

        for breakpoint_idx in range(1, max_breakpoint):
            x1, y1 = x[:breakpoint_idx], y[:breakpoint_idx]
            x2, y2 = x[breakpoint_idx:], y[breakpoint_idx:]
            w1, w2 = weights[:breakpoint_idx], weights[breakpoint_idx:]
            if len(x1) < 3 or len(x2) < 3:
                continue

            reg1 = np.polyfit(x1.ravel(), y1, 1, w=np.sqrt(w1) if weighted else None)
            reg2 = np.polyfit(x2.ravel(), y2, 1, w=np.sqrt(w2) if weighted else None)
            y1_pred = reg1[0] * x1.ravel() + reg1[1]
            y2_pred = reg2[0] * x2.ravel() + reg2[1]
            err1 = (y1 - y1_pred) ** 2
            err2 = (y2 - y2_pred) ** 2
            objective = float(np.sum(err1 * w1) + np.sum(err2 * w2)) if weighted else float(np.sum(err1) + np.sum(err2))

            if objective < best_objective:
                best_objective = objective
                best_breakpoint = float(binned.iloc[breakpoint_idx]["density_center"])

        speed_free = float(binned.loc[binned["density_center"] < best_breakpoint, "speed_mean"].mean())
        speed_cong = float(binned.loc[binned["density_center"] >= best_breakpoint, "speed_mean"].mean())
        return {
            "rho_star": float(best_breakpoint),
            "speed_free": speed_free,
            "speed_congested": speed_cong,
            "speed_drop_percent": float((speed_free - speed_cong) / speed_free * 100),
            "objective": float(best_objective),
            "n_bins": int(len(binned)),
        }

    def _contingency_summary(
        self,
        data: pd.DataFrame,
        *,
        corridor_col: str = "is_corridor",
        congested_col: str = "is_congested",
    ) -> dict:
        a = int((data[corridor_col] & data[congested_col]).sum())
        b = int((data[corridor_col] & ~data[congested_col]).sum())
        c = int((~data[corridor_col] & data[congested_col]).sum())
        d = int((~data[corridor_col] & ~data[congested_col]).sum())

        rr, rr_lo, rr_hi = relative_risk_ci(a, b, c, d) if a > 0 and c > 0 else (float("inf"), float("nan"), float("nan"))
        or_adj, or_lo, or_hi = odds_ratio_ha_ci(a, b, c, d)
        _, fisher_p = stats.fisher_exact([[a, b], [c, d]])
        return {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "risk_corridor": a / (a + b),
            "risk_noncorridor": c / (c + d),
            "rr": rr,
            "rr_ci_95_lower": rr_lo,
            "rr_ci_95_upper": rr_hi,
            "or_adj": or_adj,
            "or_ci_95_lower": or_lo,
            "or_ci_95_upper": or_hi,
            "fisher_p": fisher_p,
        }

    def _point_count_piecewise_result(
        self,
        data: pd.DataFrame,
        *,
        bin_width: int = 1,
        min_samples: int = 10,
        weighted: bool = False,
    ) -> dict:
        binned = self._build_binned_diagram(
            data,
            "n_points",
            bin_width=bin_width,
            min_samples=min_samples,
            use_interval_center=True,
        )
        return self._estimate_piecewise_from_binned(binned, weighted=weighted)

    def _load_endpoint_cells(self) -> pd.DataFrame:
        assert self.grid_info is not None

        od_pairs = self._load_mapped_od_pairs()
        endpoints = pd.concat(
            [
                od_pairs[["start_row", "start_col"]].rename(columns={"start_row": "row", "start_col": "col"}),
                od_pairs[["end_row", "end_col"]].rename(columns={"end_row": "row", "end_col": "col"}),
            ],
            ignore_index=True,
        )
        endpoints["cell_id"] = endpoints["row"].astype(str) + "_" + endpoints["col"].astype(str)
        return (
            endpoints.groupby(["row", "col", "cell_id"], observed=True)
            .size()
            .reset_index(name="endpoint_count")
            .sort_values("endpoint_count", ascending=False)
            .reset_index(drop=True)
        )

    def _load_mapped_od_pairs(self) -> pd.DataFrame:
        assert self.grid_info is not None

        if self.mapped_od_pairs is not None:
            return self.mapped_od_pairs

        od_pairs = pd.read_csv(
            self.od_pairs_file,
            usecols=["order_id", "start_lon", "start_lat", "end_lon", "end_lat"],
        )
        start_rows, start_cols = self._map_to_full_grid(od_pairs["start_lon"], od_pairs["start_lat"])
        end_rows, end_cols = self._map_to_full_grid(od_pairs["end_lon"], od_pairs["end_lat"])
        od_pairs["start_row"] = start_rows.astype("int16")
        od_pairs["start_col"] = start_cols.astype("int16")
        od_pairs["end_row"] = end_rows.astype("int16")
        od_pairs["end_col"] = end_cols.astype("int16")
        od_pairs["start_cell_id"] = od_pairs["start_row"].astype(str) + "_" + od_pairs["start_col"].astype(str)
        od_pairs["end_cell_id"] = od_pairs["end_row"].astype(str) + "_" + od_pairs["end_col"].astype(str)
        self.mapped_od_pairs = od_pairs[
            ["order_id", "start_row", "start_col", "end_row", "end_col", "start_cell_id", "end_cell_id"]
        ].copy()
        return self.mapped_od_pairs

    def _expand_cell_buffer(self, cells: pd.DataFrame, radius: int) -> set[str]:
        assert self.grid_info is not None

        expanded: set[str] = set()
        max_row = int(self.grid_info["n_rows"]) - 1
        max_col = int(self.grid_info["n_cols"]) - 1
        for row, col in cells[["row", "col"]].itertuples(index=False):
            for d_row in range(-radius, radius + 1):
                for d_col in range(-radius, radius + 1):
                    new_row = min(max(int(row) + d_row, 0), max_row)
                    new_col = min(max(int(col) + d_col, 0), max_col)
                    expanded.add(f"{new_row}_{new_col}")
        return expanded

    def _compute_corridor_cells(self, data: pd.DataFrame) -> set[str]:
        traffic = (
            data.groupby("cell_id", as_index=False)["n_points"]
            .sum()
            .sort_values("n_points", ascending=False)
            .reset_index(drop=True)
        )
        n_corridor = max(1, int(len(traffic) * 0.10))
        return set(traffic.iloc[:n_corridor]["cell_id"].astype(str))

    def _summarize_subset(
        self,
        data: pd.DataFrame,
        *,
        corridor_cells: set[str] | None = None,
        reestimate_corridor: bool = True,
        rho_star: float | None = None,
        reestimate_rho: bool = True,
    ) -> dict:
        assert self.rho_star is not None

        if data.empty:
            raise ValueError("Subset is empty.")

        scoped = data.copy()
        if corridor_cells is None or reestimate_corridor:
            corridor_cells = self._compute_corridor_cells(scoped)

        if reestimate_rho:
            piecewise = self._point_count_piecewise_result(scoped)
            rho_used = float(piecewise["rho_star"])
            speed_free = float(piecewise["speed_free"])
            speed_congested = float(piecewise["speed_congested"])
            speed_drop = float(piecewise["speed_drop_percent"])
        else:
            rho_used = float(self.rho_star if rho_star is None else rho_star)
            speed_free = float(scoped.loc[scoped["n_points"] < rho_used, "mean_speed"].mean())
            speed_congested = float(scoped.loc[scoped["n_points"] >= rho_used, "mean_speed"].mean())
            speed_drop = float((speed_free - speed_congested) / speed_free * 100) if speed_free > 0 else float("nan")

        scoped["is_corridor"] = scoped["cell_id"].isin(corridor_cells)
        scoped["is_congested"] = scoped["n_points"] >= rho_used
        metrics = self._contingency_summary(scoped)

        return {
            "samples": int(len(scoped)),
            "active_cells": int(scoped["cell_id"].nunique()),
            "corridor_cells": int(len(corridor_cells)),
            "rho_star": rho_used,
            "speed_free": speed_free,
            "speed_congested": speed_congested,
            "speed_drop_percent": speed_drop,
            "congestion_share_percent": float(scoped["is_congested"].mean() * 100),
            "corridor_congested": metrics["a"],
            "corridor_not_congested": metrics["b"],
            "noncorridor_congested": metrics["c"],
            "noncorridor_not_congested": metrics["d"],
            "risk_corridor": float(metrics["risk_corridor"]),
            "risk_noncorridor": float(metrics["risk_noncorridor"]),
            "rr": float(metrics["rr"]),
            "rr_ci_95_lower": float(metrics["rr_ci_95_lower"]),
            "rr_ci_95_upper": float(metrics["rr_ci_95_upper"]),
            "or_adj": float(metrics["or_adj"]),
            "or_ci_95_lower": float(metrics["or_ci_95_lower"]),
            "or_ci_95_upper": float(metrics["or_ci_95_upper"]),
            "fisher_p": float(metrics["fisher_p"]),
        }

    def _connected_components(self, cell_ids: set[str]) -> list[set[str]]:
        coord_to_id: dict[tuple[int, int], str] = {}
        for cell_id in cell_ids:
            row_str, col_str = cell_id.split("_")
            coord_to_id[(int(row_str), int(col_str))] = cell_id

        remaining = set(coord_to_id)
        components: list[set[str]] = []
        neighbor_offsets = [
            (d_row, d_col)
            for d_row in (-1, 0, 1)
            for d_col in (-1, 0, 1)
            if not (d_row == 0 and d_col == 0)
        ]

        while remaining:
            start = remaining.pop()
            stack = [start]
            component = {coord_to_id[start]}
            while stack:
                row, col = stack.pop()
                for d_row, d_col in neighbor_offsets:
                    neighbor = (row + d_row, col + d_col)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        component.add(coord_to_id[neighbor])
            components.append(component)

        components.sort(key=len, reverse=True)
        return components

    def _distribution_summary(self, values: list[float], observed: float) -> dict:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return {
                "n_values": int(len(values)),
                "n_finite": 0,
                "observed": float(observed),
                "mean": float("nan"),
                "median": float("nan"),
                "ci_95_lower": float("nan"),
                "ci_95_upper": float("nan"),
                "max": float("nan"),
                "empirical_p_ge_observed": float("nan"),
            }

        empirical_p = (np.sum(finite >= observed) + 1.0) / (len(finite) + 1.0)
        return {
            "n_values": int(len(values)),
            "n_finite": int(len(finite)),
            "observed": float(observed),
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "ci_95_lower": float(np.percentile(finite, 2.5)),
            "ci_95_upper": float(np.percentile(finite, 97.5)),
            "max": float(np.max(finite)),
            "empirical_p_ge_observed": float(empirical_p),
        }

    def _graph_components_from_edges(self, edges: list[tuple[str, str]]) -> list[set[str]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for source, target in edges:
            adjacency[source].add(target)
            adjacency[target].add(source)

        remaining = set(adjacency)
        components: list[set[str]] = []
        while remaining:
            start = remaining.pop()
            stack = [start]
            component = {start}
            while stack:
                node = stack.pop()
                for neighbor in adjacency[node]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        component.add(neighbor)
            components.append(component)

        components.sort(key=len, reverse=True)
        return components

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(valid):
            return float("nan")

        values = values[valid]
        weights = weights[valid]
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        cutoff = cumulative[-1] * 0.5
        return float(values[order][np.searchsorted(cumulative, cutoff, side="left")])

    def _axial_resultant(self, angles_deg: np.ndarray, *, weights: np.ndarray | None = None) -> float:
        angles = np.deg2rad(np.asarray(angles_deg, dtype=float))
        valid = np.isfinite(angles)
        if weights is None:
            if not np.any(valid):
                return float("nan")
            cos_term = float(np.mean(np.cos(2.0 * angles[valid])))
            sin_term = float(np.mean(np.sin(2.0 * angles[valid])))
            return float(np.hypot(cos_term, sin_term))

        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0)
        if not np.any(valid):
            return float("nan")
        cos_term = float(np.average(np.cos(2.0 * angles[valid]), weights=weights[valid]))
        sin_term = float(np.average(np.sin(2.0 * angles[valid]), weights=weights[valid]))
        return float(np.hypot(cos_term, sin_term))

    def _backbone_graph_metrics(self, cell_ids: set[str]) -> dict:
        components = self._connected_components(cell_ids)
        component_sizes = np.asarray([len(component) for component in components], dtype=int)

        coord_set = {tuple(map(int, cell_id.split("_"))) for cell_id in cell_ids}
        neighbor_offsets = [
            (d_row, d_col)
            for d_row in (-1, 0, 1)
            for d_col in (-1, 0, 1)
            if not (d_row == 0 and d_col == 0)
        ]
        degrees = np.asarray(
            [
                sum(((row + d_row, col + d_col) in coord_set) for d_row, d_col in neighbor_offsets)
                for row, col in coord_set
            ],
            dtype=float,
        )

        cumulative_share = np.cumsum(component_sizes) / len(cell_ids)
        components_covering_50pct = int(np.searchsorted(cumulative_share, 0.5, side="left") + 1)
        components_covering_80pct = int(np.searchsorted(cumulative_share, 0.8, side="left") + 1)

        return {
            "n_components": int(len(components)),
            "largest_component_cells": int(component_sizes[0]),
            "largest_component_share": float(component_sizes[0] / len(cell_ids)),
            "components_covering_50pct_cells": components_covering_50pct,
            "components_covering_80pct_cells": components_covering_80pct,
            "singleton_component_share": float(np.mean(component_sizes == 1)),
            "cells_in_components_le3_share": float(component_sizes[component_sizes <= 3].sum() / len(cell_ids)),
            "mean_backbone_degree": float(np.mean(degrees)),
            "degree_le1_share": float(np.mean(degrees <= 1)),
            "degree_eq2_share": float(np.mean(degrees == 2)),
            "degree_ge3_share": float(np.mean(degrees >= 3)),
            "top10_component_sizes": [int(value) for value in component_sizes[:10]],
        }

    def _component_shape_metrics(self, components: list[set[str]], total_backbone_cells: int) -> tuple[dict, dict[str, int], dict[int, float]]:
        component_records: list[dict] = []
        cell_to_component: dict[str, int] = {}
        component_axes: dict[int, float] = {}

        for component_id, component in enumerate(components):
            component_size = len(component)
            for cell_id in component:
                cell_to_component[cell_id] = component_id

            if component_size < 5:
                continue

            coords = np.asarray([tuple(map(int, cell_id.split("_"))) for cell_id in component], dtype=float)
            centered = coords - coords.mean(axis=0, keepdims=True)
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1.0e-9)
            major_idx = int(np.argmax(eigenvalues))
            major_variance = float(eigenvalues[major_idx])
            minor_variance = float(np.min(eigenvalues))
            major_vector = eigenvectors[:, major_idx]
            major_axis_angle = float(np.degrees(np.arctan2(major_vector[0], major_vector[1])) % 180.0)

            row_span = float(coords[:, 0].max() - coords[:, 0].min() + 1.0)
            col_span = float(coords[:, 1].max() - coords[:, 1].min() + 1.0)
            component_axes[component_id] = major_axis_angle
            component_records.append(
                {
                    "component_id": int(component_id),
                    "cells": int(component_size),
                    "pca_elongation": float(math.sqrt(major_variance / max(minor_variance, 1.0e-9))),
                    "bbox_aspect_ratio": float(max(row_span, col_span) / max(min(row_span, col_span), 1.0)),
                    "major_axis_angle_deg": major_axis_angle,
                }
            )

        if not component_records:
            return (
                {
                    "components_ge5_cells": 0,
                    "cells_in_components_ge5": 0,
                    "cells_in_components_ge5_share": 0.0,
                    "weighted_median_pca_elongation_ge5": float("nan"),
                    "weighted_median_bbox_aspect_ge5": float("nan"),
                    "largest_component_pca_elongation": float("nan"),
                    "largest_component_bbox_aspect": float("nan"),
                    "largest_component_major_axis_angle_deg": float("nan"),
                    "component_axis_axial_resultant_ge5": float("nan"),
                    "top10_large_component_shapes": [],
                },
                cell_to_component,
                component_axes,
            )

        component_df = pd.DataFrame(component_records).sort_values("cells", ascending=False).reset_index(drop=True)
        weights = component_df["cells"].to_numpy(dtype=float)
        angles = component_df["major_axis_angle_deg"].to_numpy(dtype=float)
        shape_metrics = {
            "components_ge5_cells": int(len(component_df)),
            "cells_in_components_ge5": int(component_df["cells"].sum()),
            "cells_in_components_ge5_share": float(component_df["cells"].sum() / total_backbone_cells),
            "weighted_median_pca_elongation_ge5": self._weighted_median(
                component_df["pca_elongation"].to_numpy(dtype=float),
                weights,
            ),
            "weighted_median_bbox_aspect_ge5": self._weighted_median(
                component_df["bbox_aspect_ratio"].to_numpy(dtype=float),
                weights,
            ),
            "largest_component_pca_elongation": float(component_df.iloc[0]["pca_elongation"]),
            "largest_component_bbox_aspect": float(component_df.iloc[0]["bbox_aspect_ratio"]),
            "largest_component_major_axis_angle_deg": float(component_df.iloc[0]["major_axis_angle_deg"]),
            "component_axis_axial_resultant_ge5": self._axial_resultant(angles, weights=weights),
            "top10_large_component_shapes": component_df.head(10).to_dict(orient="records"),
        }
        return shape_metrics, cell_to_component, component_axes

    def _moving_directionality_metrics(
        self,
        cell_to_component: dict[str, int],
        component_axes: dict[int, float],
        *,
        speed_threshold: float = 2.0,
    ) -> dict:
        point_level = self._load_point_level()
        moving = point_level.loc[(point_level["speed"] >= speed_threshold) & point_level["yaw_clean"].notna()].copy()
        moving["is_backbone"] = moving["cell_id"].isin(self.full_corridor_cells)
        moving_backbone_points = int(moving["is_backbone"].sum())

        moving["component_id"] = moving["cell_id"].map(cell_to_component)
        moving = moving.loc[moving["component_id"].notna()].copy()
        moving["component_id"] = moving["component_id"].astype(int)
        moving = moving.loc[moving["component_id"].isin(component_axes)].copy()
        if moving.empty:
            return {
                "moving_speed_threshold_mps": float(speed_threshold),
                "moving_points_total": int(len(point_level)),
                "moving_backbone_points": moving_backbone_points,
                "moving_points_in_components_ge5": 0,
                "weighted_median_component_axial_resultant": float("nan"),
                "weighted_median_axis_deviation_deg": float("nan"),
                "components_axial_r_ge_0_8": 0,
                "moving_point_share_components_axial_r_ge_0_8": 0.0,
                "components_axial_r_ge_0_5": 0,
                "moving_point_share_components_axial_r_ge_0_5": 0.0,
                "top10_components_by_moving_points": [],
            }

        moving["component_axis_deg"] = moving["component_id"].map(component_axes)
        moving["axis_deviation_deg"] = np.abs(
            ((moving["yaw_clean"] - moving["component_axis_deg"] + 90.0) % 180.0) - 90.0
        )

        records: list[dict] = []
        for component_id, group in moving.groupby("component_id", sort=False):
            records.append(
                {
                    "component_id": int(component_id),
                    "moving_points": int(len(group)),
                    "component_axial_resultant": self._axial_resultant(group["yaw_clean"].to_numpy(dtype=float)),
                    "median_axis_deviation_deg": float(group["axis_deviation_deg"].median()),
                    "major_axis_angle_deg": float(component_axes[component_id]),
                }
            )

        component_df = pd.DataFrame(records).sort_values("moving_points", ascending=False).reset_index(drop=True)
        weights = component_df["moving_points"].to_numpy(dtype=float)
        component_axial = component_df["component_axial_resultant"].to_numpy(dtype=float)
        axis_deviation = component_df["median_axis_deviation_deg"].to_numpy(dtype=float)

        strong_mask = component_axial >= 0.8
        moderate_mask = component_axial >= 0.5

        return {
            "moving_speed_threshold_mps": float(speed_threshold),
            "moving_points_total": int(len(point_level.loc[(point_level["speed"] >= speed_threshold) & point_level["yaw_clean"].notna()])),
            "moving_backbone_points": moving_backbone_points,
            "moving_points_in_components_ge5": int(component_df["moving_points"].sum()),
            "weighted_median_component_axial_resultant": self._weighted_median(component_axial, weights),
            "weighted_median_axis_deviation_deg": self._weighted_median(axis_deviation, weights),
            "components_axial_r_ge_0_8": int(np.sum(strong_mask)),
            "moving_point_share_components_axial_r_ge_0_8": float(component_df.loc[strong_mask, "moving_points"].sum() / weights.sum()),
            "components_axial_r_ge_0_5": int(np.sum(moderate_mask)),
            "moving_point_share_components_axial_r_ge_0_5": float(component_df.loc[moderate_mask, "moving_points"].sum() / weights.sum()),
            "top10_components_by_moving_points": component_df.head(10).to_dict(orient="records"),
        }

    def _evaluate_holdout_split(self, train_dates: list, test_dates: list) -> dict:
        assert self.spatiotemporal is not None
        assert self.rho_star is not None

        train = self.spatiotemporal[self.spatiotemporal["date"].isin(train_dates)]
        test = self.spatiotemporal[self.spatiotemporal["date"].isin(test_dates)].copy()

        train_traffic = (
            train.groupby("cell_id", as_index=False)["n_points"]
            .sum()
            .sort_values("n_points", ascending=False)
        )
        n_corridor = max(1, int(len(train_traffic) * 0.10))
        held_out_corridor = set(train_traffic.iloc[:n_corridor]["cell_id"].astype(str))

        test["is_corridor"] = test["cell_id"].isin(held_out_corridor)
        test["is_congested"] = test["n_points"] >= self.rho_star

        a = int((test["is_corridor"] & test["is_congested"]).sum())
        b = int((test["is_corridor"] & ~test["is_congested"]).sum())
        c = int((~test["is_corridor"] & test["is_congested"]).sum())
        d = int((~test["is_corridor"] & ~test["is_congested"]).sum())

        rr, rr_lo, rr_hi = relative_risk_ci(a, b, c, d)
        or_adj, or_lo, or_hi = odds_ratio_ha_ci(a, b, c, d)
        _, fisher_p = stats.fisher_exact([[a, b], [c, d]])

        return {
            "train_days": len(train_dates),
            "test_days": len(test_dates),
            "corridor_cells_train": len(held_out_corridor),
            "corridor_congested_test": a,
            "corridor_not_test": b,
            "noncorridor_congested_test": c,
            "noncorridor_not_test": d,
            "risk_corridor_test": a / (a + b),
            "risk_noncorridor_test": c / (c + d),
            "rr_test": rr,
            "rr_ci_95_lower": rr_lo,
            "rr_ci_95_upper": rr_hi,
            "or_adj_test": or_adj,
            "or_ci_95_lower": or_lo,
            "or_ci_95_upper": or_hi,
            "fisher_p_test": fisher_p,
        }

    def load_data(self) -> bool:
        if not self.cell_stats_file.exists() or not self.spatiotemporal_file.exists():
            return False

        self.cell_stats = pd.read_csv(self.cell_stats_file).sort_values("total_points", ascending=False).reset_index(drop=True)
        self.spatiotemporal = pd.read_csv(self.spatiotemporal_file)
        self.spatiotemporal["window_start"] = pd.to_datetime(self.spatiotemporal["window_start"])
        self.spatiotemporal["date"] = self.spatiotemporal["window_start"].dt.date
        self.spatiotemporal["date_hour"] = self.spatiotemporal["window_start"].dt.strftime("%Y-%m-%d-%H")
        self.spatiotemporal["cell_id"] = self.spatiotemporal["cell_id"].astype(str)

        self.fundamental = pd.read_csv(self.fundamental_file) if self.fundamental_file.exists() else None

        with open(self.grid_info_file, "r", encoding="utf-8") as f:
            self.grid_info = json.load(f)

        with open(self.breakpoint_file, "r", encoding="utf-8") as f:
            self.rho_star = float(json.load(f)["consensus_rho_star"])

        n_corridor = max(1, int(len(self.cell_stats) * 0.10))
        self.full_corridor_cells = set(self.cell_stats.iloc[:n_corridor]["cell_id"].astype(str))
        return True

    def experiment1_out_of_sample_coupling(self) -> dict:
        assert self.spatiotemporal is not None

        dates = sorted(self.spatiotemporal["date"].unique())
        split_idx = len(dates) // 2
        first_half = dates[:split_idx]
        second_half = dates[split_idx:]

        forward = self._evaluate_holdout_split(first_half, second_half)
        reverse = self._evaluate_holdout_split(second_half, first_half)

        return {
            **forward,
            "split_design": "bidirectional chronological half-splits",
            "forward_split": forward,
            "reverse_split": reverse,
            "rr_range": [
                float(min(forward["rr_test"], reverse["rr_test"])),
                float(max(forward["rr_test"], reverse["rr_test"])),
            ],
            "or_range": [
                float(min(forward["or_adj_test"], reverse["or_adj_test"])),
                float(max(forward["or_adj_test"], reverse["or_adj_test"])),
            ],
        }

    def experiment2_unique_flight_occupancy(self) -> dict:
        order_level = self._load_order_level_spatiotemporal().copy()
        binned = self._build_binned_diagram(order_level, "n_unique", use_interval_center=False)
        piecewise = self._estimate_piecewise_from_binned(binned)
        rho_unique = piecewise["rho_star"]

        order_level["is_corridor"] = order_level["cell_id"].isin(self.full_corridor_cells)
        order_level["is_congested_unique"] = order_level["n_unique"] >= rho_unique

        metrics = self._contingency_summary(order_level, congested_col="is_congested_unique")

        return {
            "measurement": (
                "distinct order_id counts per cell-window on the full 100 m x "
                f"{self._time_window_minutes_label()} min grid"
            ),
            "rho_star_unique": float(rho_unique),
            "correlation_coefficient_unique": float(order_level["n_unique"].corr(order_level["mean_speed"])),
            "congestion_share_percent_unique": float(order_level["is_congested_unique"].mean() * 100),
            "speed_free_unique": piecewise["speed_free"],
            "speed_congested_unique": piecewise["speed_congested"],
            "speed_drop_percent_unique": piecewise["speed_drop_percent"],
            "corridor_congested_unique": metrics["a"],
            "corridor_not_congested_unique": metrics["b"],
            "noncorridor_congested_unique": metrics["c"],
            "noncorridor_not_congested_unique": metrics["d"],
            "rr_unique": metrics["rr"],
            "rr_ci_95_lower_unique": metrics["rr_ci_95_lower"],
            "rr_ci_95_upper_unique": metrics["rr_ci_95_upper"],
            "or_adj_unique": metrics["or_adj"],
            "or_ci_95_lower_unique": metrics["or_ci_95_lower"],
            "or_ci_95_upper_unique": metrics["or_ci_95_upper"],
            "fisher_p_unique": metrics["fisher_p"],
            "n_samples_total_unique": int(len(order_level)),
        }

    def experiment3_exclude_terminal_zones(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.rho_star is not None

        median_density = float(self.spatiotemporal["n_points"].median())
        terminal_candidates = self.spatiotemporal[
            (self.spatiotemporal["mean_speed"] < 2.0)
            & (self.spatiotemporal["n_points"] > median_density)
        ]
        terminal_cells = set(terminal_candidates["cell_id"].unique())
        filtered = self.spatiotemporal[~self.spatiotemporal["cell_id"].isin(terminal_cells)].copy()

        rho_star_excluded = self.rho_star
        if len(filtered) >= 1000:
            rho_star_excluded = self._point_count_piecewise_result(filtered)["rho_star"]

        return {
            "terminal_cells_identified": len(terminal_cells),
            "samples_after_exclusion": len(filtered),
            "exclusion_ratio": 1 - (len(filtered) / len(self.spatiotemporal)),
            "estimated_rho_star_excluded": float(rho_star_excluded),
            "original_rho_star": float(self.rho_star),
        }

    def experiment4_cluster_robust_ci(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.rho_star is not None

        data = self.spatiotemporal.copy()
        data["is_corridor"] = data["cell_id"].isin(self.full_corridor_cells)
        data["is_congested"] = data["n_points"] >= self.rho_star
        data["a"] = (data["is_corridor"] & data["is_congested"]).astype(int)
        data["b"] = (data["is_corridor"] & ~data["is_congested"]).astype(int)
        data["c"] = (~data["is_corridor"] & data["is_congested"]).astype(int)
        data["d"] = (~data["is_corridor"] & ~data["is_congested"]).astype(int)
        grouped = data.groupby("date_hour", as_index=False)[["a", "b", "c", "d"]].sum()

        rr_values: list[float] = []
        or_values: list[float] = []
        log_or_values: list[float] = []

        for _ in range(self.n_bootstraps):
            sampled = grouped.iloc[self.rng.integers(0, len(grouped), size=len(grouped))]
            a = int(sampled["a"].sum())
            b = int(sampled["b"].sum())
            c = int(sampled["c"].sum())
            d = int(sampled["d"].sum())

            if c > 0 and a > 0:
                rr_values.append((a / (a + b)) / (c / (c + d)))

            or_adj, _, _ = odds_ratio_ha_ci(a, b, c, d)
            or_values.append(or_adj)
            log_or_values.append(math.log(or_adj))

        rr_array = np.asarray(rr_values, dtype=float)
        or_array = np.asarray(or_values, dtype=float)
        log_or_array = np.asarray(log_or_values, dtype=float)

        full_a = int((data["is_corridor"] & data["is_congested"]).sum())
        full_b = int((data["is_corridor"] & ~data["is_congested"]).sum())
        full_c = int((~data["is_corridor"] & data["is_congested"]).sum())
        full_d = int((~data["is_corridor"] & ~data["is_congested"]).sum())
        rr_point, _, _ = relative_risk_ci(full_a, full_b, full_c, full_d)
        or_point, _, _ = odds_ratio_ha_ci(full_a, full_b, full_c, full_d)

        return {
            "n_bootstraps": self.n_bootstraps,
            "n_clusters": int(len(grouped)),
            "rr_point_estimate": rr_point,
            "rr_bootstrap_mean": float(np.mean(rr_array)),
            "rr_bootstrap_median": float(np.median(rr_array)),
            "rr_ci_95_lower": float(np.percentile(rr_array, 2.5)),
            "rr_ci_95_upper": float(np.percentile(rr_array, 97.5)),
            "n_finite_rr_bootstraps": int(len(rr_array)),
            "or_point_estimate": or_point,
            "or_bootstrap_mean": float(np.mean(or_array)),
            "or_bootstrap_median": float(np.median(or_array)),
            "or_ci_95_lower": float(np.percentile(or_array, 2.5)),
            "or_ci_95_upper": float(np.percentile(or_array, 97.5)),
            "log_or_ci_95_lower": float(np.percentile(log_or_array, 2.5)),
            "log_or_ci_95_upper": float(np.percentile(log_or_array, 97.5)),
        }

    def experiment5_rho_star_bootstrap_ci(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.rho_star is not None

        def legacy_breakpoint(x: np.ndarray, y: np.ndarray) -> float:
            if len(x) < 10:
                return float(x[len(x) // 2])
            speed_changes: list[tuple[float, float]] = []
            for idx in range(5, len(y) - 5):
                prev_mean = float(np.mean(y[idx - 5 : idx]))
                next_mean = float(np.mean(y[idx : idx + 5]))
                speed_changes.append((float(x[idx]), prev_mean - next_mean))
            if speed_changes:
                return max(speed_changes, key=lambda item: item[1])[0]
            return float(x[len(x) // 2])

        np.random.seed(self.rho_bootstrap_seed)
        coarse = self.spatiotemporal.copy()
        coarse["density_bin"] = pd.cut(coarse["n_points"], bins=100)
        binned = (
            coarse.groupby("density_bin", observed=False)
            .agg(n_points=("n_points", "mean"), mean_speed=("mean_speed", "mean"))
            .dropna()
            .reset_index(drop=True)
        )

        densities = binned["n_points"].to_numpy(dtype=float)
        speeds = binned["mean_speed"].to_numpy(dtype=float)
        breakpoints: list[float] = []

        for _ in range(self.n_bootstraps):
            indices = np.random.choice(len(densities), size=len(densities), replace=True)
            breakpoints.append(legacy_breakpoint(densities[indices], speeds[indices]))

        bp_array = np.asarray(breakpoints, dtype=float)
        return {
            "n_bootstraps": self.n_bootstraps,
            "original_rho_star": float(self.rho_star),
            "bootstrap_mean": float(np.mean(bp_array)),
            "bootstrap_median": float(np.median(bp_array)),
            "bootstrap_std": float(np.std(bp_array)),
            "ci_95_lower": float(np.percentile(bp_array, 2.5)),
            "ci_95_upper": float(np.percentile(bp_array, 97.5)),
        }

    def experiment6_support_ratio_boundary_sensitivity(self) -> dict:
        assert self.cell_stats is not None
        assert self.grid_info is not None

        total_cells = int(self.grid_info["total_cells"])
        active_cells = int(len(self.cell_stats))
        original_ratio = active_cells / total_cells * 100
        return {
            "total_cells": total_cells,
            "active_cells": active_cells,
            "original_ratio_percent": float(original_ratio),
            "convex_hull_ratio_percent": 4.7,
            "message": "Convex-hull boundary sensitivity remains an exploratory placeholder and is not used as confirmatory evidence in the current paper.",
        }

    def experiment7_spatial_hotspot_localization(self) -> dict:
        assert self.cell_stats is not None
        assert self.spatiotemporal is not None
        assert self.grid_info is not None
        assert self.rho_star is not None

        data = self.spatiotemporal.copy()
        data["is_congested"] = data["n_points"] >= self.rho_star
        data["is_corridor"] = data["cell_id"].isin(self.full_corridor_cells)

        median_density = float(data["n_points"].median())
        data["is_interface_like"] = (data["mean_speed"] < 2.0) & (data["n_points"] > median_density)
        interface_cells = set(data.loc[data["is_interface_like"], "cell_id"].astype(str).unique())

        cell_congestion = (
            data.groupby("cell_id", as_index=False)
            .agg(congested_samples=("is_congested", "sum"), total_samples=("is_congested", "size"))
            .sort_values("congested_samples", ascending=False)
            .reset_index(drop=True)
        )

        hotspot_cells = int((cell_congestion["congested_samples"] > 0).sum())
        total_active_cells = int(len(cell_congestion))
        total_congested_samples = int(cell_congestion["congested_samples"].sum())

        def top_share(top_fraction: float) -> float:
            n_top = max(1, int(total_active_cells * top_fraction))
            return float(cell_congestion.iloc[:n_top]["congested_samples"].sum() / total_congested_samples)

        congested = data[data["is_congested"]]

        return {
            "hotspot_cells": hotspot_cells,
            "hotspot_share_active_cells": hotspot_cells / total_active_cells,
            "hotspot_share_full_grid": hotspot_cells / int(self.grid_info["total_cells"]),
            "top_0_5pct_active_cells_congestion_share": top_share(0.005),
            "top_1pct_active_cells_congestion_share": top_share(0.01),
            "congested_samples_in_corridors_share": float(congested["is_corridor"].mean()),
            "interface_like_cells": len(interface_cells),
            "interface_like_share_active_cells": len(interface_cells) / total_active_cells,
            "interface_like_corridor_overlap": float(
                np.mean([cell_id in self.full_corridor_cells for cell_id in interface_cells])
            ),
            "congested_samples_in_interface_like_cells_share": float(
                congested["cell_id"].isin(interface_cells).mean()
            ),
            "congested_samples_in_corridor_interface_overlap_share": float(
                (congested["cell_id"].isin(interface_cells) & congested["is_corridor"]).mean()
            ),
        }

    def experiment8_endpoint_node_anchor(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.grid_info is not None
        assert self.rho_star is not None

        endpoint_cells = self._load_endpoint_cells()
        endpoint_set = set(endpoint_cells["cell_id"])
        top_endpoint_count = max(1, int(len(endpoint_cells) * 0.10))
        top_endpoint_set = set(endpoint_cells.head(top_endpoint_count)["cell_id"])

        data = self.spatiotemporal.copy()
        data["is_congested"] = data["n_points"] >= self.rho_star
        congested = data[data["is_congested"]].copy()

        hotspot_cells = (
            data.groupby("cell_id", as_index=False)["is_congested"]
            .sum()
            .query("is_congested > 0")
        )
        hotspot_set = set(hotspot_cells["cell_id"])

        endpoint_rows = endpoint_cells["row"].to_numpy(dtype=int)
        endpoint_cols = endpoint_cells["col"].to_numpy(dtype=int)
        congested_coords = congested["cell_id"].str.split("_", expand=True).astype(int)

        def within_radius(radius: int) -> np.ndarray:
            flags: list[bool] = []
            for row, col in congested_coords.itertuples(index=False):
                flags.append(bool(np.any((np.abs(endpoint_rows - row) <= radius) & (np.abs(endpoint_cols - col) <= radius))))
            return np.asarray(flags, dtype=bool)

        radius_one_flags = within_radius(1)
        radius_two_flags = within_radius(2)

        return {
            "endpoint_cells": int(len(endpoint_cells)),
            "endpoint_share_active_cells": float(len(endpoint_cells) / len(self.cell_stats)),
            "endpoint_corridor_overlap": float(
                np.mean([cell_id in self.full_corridor_cells for cell_id in endpoint_set])
            ),
            "hotspot_cells_overlapping_endpoints": int(len(hotspot_set & endpoint_set)),
            "hotspot_overlap_share": float(len(hotspot_set & endpoint_set) / len(hotspot_set)),
            "congested_samples_in_endpoint_cells_share": float(congested["cell_id"].isin(endpoint_set).mean()),
            "top_10pct_endpoint_cells": int(top_endpoint_count),
            "congested_samples_in_top_10pct_endpoint_cells_share": float(
                congested["cell_id"].isin(top_endpoint_set).mean()
            ),
            "congested_samples_within_1cell_of_endpoints_share": float(radius_one_flags.mean()),
            "congested_samples_within_2cells_of_endpoints_share": float(radius_two_flags.mean()),
        }

    def experiment9_alternative_corridor_definition(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.rho_star is not None

        order_level = self._load_order_level_spatiotemporal()
        unique_traffic = (
            order_level.groupby("cell_id", as_index=False)["n_unique"]
            .sum()
            .sort_values("n_unique", ascending=False)
            .reset_index(drop=True)
        )
        n_corridor = max(1, int(len(unique_traffic) * 0.10))
        unique_corridor = set(unique_traffic.iloc[:n_corridor]["cell_id"].astype(str))

        data = self.spatiotemporal.copy()
        data["is_congested"] = data["n_points"] >= self.rho_star
        data["is_corridor_unique"] = data["cell_id"].isin(unique_corridor)
        metrics = self._contingency_summary(data, corridor_col="is_corridor_unique", congested_col="is_congested")

        intersection = len(self.full_corridor_cells & unique_corridor)
        union = len(self.full_corridor_cells | unique_corridor)

        return {
            "alternative_definition": "top 10% of active cells by aggregated distinct-order counts",
            "corridor_cells_unique_definition": int(len(unique_corridor)),
            "jaccard_with_point_count_corridor": float(intersection / union),
            "point_count_only_cells": int(len(self.full_corridor_cells - unique_corridor)),
            "unique_only_cells": int(len(unique_corridor - self.full_corridor_cells)),
            "corridor_congested_alt": metrics["a"],
            "corridor_not_congested_alt": metrics["b"],
            "noncorridor_congested_alt": metrics["c"],
            "noncorridor_not_congested_alt": metrics["d"],
            "rr_alt": metrics["rr"],
            "rr_ci_95_lower_alt": metrics["rr_ci_95_lower"],
            "rr_ci_95_upper_alt": metrics["rr_ci_95_upper"],
            "or_adj_alt": metrics["or_adj"],
            "or_ci_95_lower_alt": metrics["or_ci_95_lower"],
            "or_ci_95_upper_alt": metrics["or_ci_95_upper"],
            "fisher_p_alt": metrics["fisher_p"],
        }

    def experiment10_interface_masking(self) -> dict:
        assert self.spatiotemporal is not None

        data = self.spatiotemporal.copy()
        median_density = float(data["n_points"].median())
        data["is_corridor"] = data["cell_id"].isin(self.full_corridor_cells)
        data["is_interface_like"] = (data["mean_speed"] < 2.0) & (data["n_points"] > median_density)
        interface_cells = set(data.loc[data["is_interface_like"], "cell_id"].astype(str).unique())

        def summarize_subset(subset: pd.DataFrame) -> dict:
            piecewise = self._point_count_piecewise_result(subset)
            scoped = subset.copy()
            scoped["is_congested"] = scoped["n_points"] >= piecewise["rho_star"]
            metrics = self._contingency_summary(scoped)
            return {
                "samples": int(len(scoped)),
                "rho_star": piecewise["rho_star"],
                "speed_drop_percent": piecewise["speed_drop_percent"],
                "congestion_share_percent": float(scoped["is_congested"].mean() * 100),
                "rr": metrics["rr"],
                "rr_ci_95_lower": metrics["rr_ci_95_lower"],
                "rr_ci_95_upper": metrics["rr_ci_95_upper"],
                "or_adj": metrics["or_adj"],
                "or_ci_95_lower": metrics["or_ci_95_lower"],
                "or_ci_95_upper": metrics["or_ci_95_upper"],
            }

        full_summary = summarize_subset(data)
        non_interface_window_summary = summarize_subset(data.loc[~data["is_interface_like"]].copy())
        interface_only_summary = summarize_subset(data.loc[data["is_interface_like"]].copy())
        non_interface_cell_summary = summarize_subset(data.loc[~data["cell_id"].isin(interface_cells)].copy())

        return {
            "interface_definition": "mean speed < 2 m/s and above-median point-count density",
            "interface_windows": int(data["is_interface_like"].sum()),
            "interface_window_share": float(data["is_interface_like"].mean()),
            "interface_cells": int(len(interface_cells)),
            "full_sample": full_summary,
            "excluding_interface_windows": non_interface_window_summary,
            "interface_windows_only": interface_only_summary,
            "excluding_interface_cells": non_interface_cell_summary,
        }

    def experiment11_breakpoint_specification(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.rho_star is not None

        specs = [
            ("main_equal_weight_bins", {"bin_width": 1, "min_samples": 10, "weighted": False}),
            ("count_weighted_bins", {"bin_width": 1, "min_samples": 10, "weighted": True}),
            ("equal_weight_bins_min50", {"bin_width": 1, "min_samples": 50, "weighted": False}),
            ("equal_weight_bins_width2", {"bin_width": 2, "min_samples": 20, "weighted": False}),
        ]

        results: dict[str, dict] = {}
        for label, kwargs in specs:
            piecewise = self._point_count_piecewise_result(self.spatiotemporal, **kwargs)
            results[label] = {
                "rho_star": piecewise["rho_star"],
                "speed_drop_percent": piecewise["speed_drop_percent"],
                "n_bins": piecewise["n_bins"],
                "bin_width": kwargs["bin_width"],
                "min_samples": kwargs["min_samples"],
                "weighting": "count_weighted" if kwargs["weighted"] else "equal_weight_bins",
            }

        return {
            "main_text_estimator": {
                "input": "binned speed-density means",
                "bin_width": 1,
                "min_samples": 10,
                "weighting": "equal_weight_bins",
                "rho_star": float(self.rho_star),
            },
            "alternative_specifications": results,
        }

    def experiment12_endpoint_buffer_exclusion(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.cell_stats is not None

        endpoint_cells = self._load_endpoint_cells()
        endpoint_set = set(endpoint_cells["cell_id"].astype(str))
        endpoint_buffer_one = self._expand_cell_buffer(endpoint_cells, radius=1)
        active_cell_set = set(self.cell_stats["cell_id"].astype(str))

        data = self.spatiotemporal.copy()
        data["is_congested_main"] = data["n_points"] >= float(self.rho_star)
        congested = data[data["is_congested_main"]]

        endpoint_only_subset = data.loc[~data["cell_id"].isin(endpoint_set)].copy()
        endpoint_buffer_subset = data.loc[~data["cell_id"].isin(endpoint_buffer_one)].copy()

        endpoint_only_summary = self._summarize_subset(endpoint_only_subset)
        endpoint_buffer_summary = self._summarize_subset(endpoint_buffer_subset)

        return {
            "endpoint_cells": int(len(endpoint_set)),
            "endpoint_cells_share_active_cells": float(len(endpoint_set) / len(active_cell_set)),
            "endpoint_buffer_one_cells_full_grid": int(len(endpoint_buffer_one)),
            "endpoint_buffer_one_cells_active_only": int(len(endpoint_buffer_one & active_cell_set)),
            "endpoint_buffer_one_share_active_cells": float(len(endpoint_buffer_one & active_cell_set) / len(active_cell_set)),
            "congested_samples_removed_endpoint_only_share": float(congested["cell_id"].isin(endpoint_set).mean()),
            "congested_samples_removed_endpoint_buffer_one_share": float(congested["cell_id"].isin(endpoint_buffer_one).mean()),
            "excluding_endpoint_cells_only": endpoint_only_summary,
            "excluding_endpoint_cells_and_1cell_buffer": endpoint_buffer_summary,
        }

    def experiment13_leave_one_group_out(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.cell_stats is not None

        def summarize_collection(collection: dict[str, dict]) -> dict:
            rr_values = [payload["rr"] for payload in collection.values() if math.isfinite(payload["rr"])]
            rho_values = [payload["rho_star"] for payload in collection.values() if math.isfinite(payload["rho_star"])]
            or_values = [payload["or_adj"] for payload in collection.values() if math.isfinite(payload["or_adj"])]
            return {
                "n_runs": int(len(collection)),
                "rr_min": float(np.min(rr_values)),
                "rr_median": float(np.median(rr_values)),
                "rr_max": float(np.max(rr_values)),
                "rho_star_min": float(np.min(rho_values)),
                "rho_star_median": float(np.median(rho_values)),
                "rho_star_max": float(np.max(rho_values)),
                "or_min": float(np.min(or_values)),
                "or_median": float(np.median(or_values)),
                "or_max": float(np.max(or_values)),
            }

        leave_one_day_results: dict[str, dict] = {}
        for date in sorted(self.spatiotemporal["date"].unique()):
            subset = self.spatiotemporal.loc[self.spatiotemporal["date"] != date].copy()
            leave_one_day_results[str(date)] = self._summarize_subset(subset)

        point_level = self._load_point_level()
        leave_one_vendor_results: dict[str, dict] = {}
        for vendor in sorted(str(vendor) for vendor in point_level["vendor"].cat.categories):
            subset_points = point_level.loc[point_level["vendor"].astype(str) != vendor, ["cell_id", "window_start", "speed"]].copy()
            aggregated = self._aggregate_point_level(subset_points)
            leave_one_vendor_results[vendor] = self._summarize_subset(aggregated)

        endpoint_cells = self._load_endpoint_cells()
        top_endpoint_count = max(1, int(len(endpoint_cells) * 0.10))
        top_endpoint_set = set(endpoint_cells.head(top_endpoint_count)["cell_id"].astype(str))
        top_endpoint_summary = self._summarize_subset(
            self.spatiotemporal.loc[~self.spatiotemporal["cell_id"].isin(top_endpoint_set)].copy()
        )

        backbone_components = self._connected_components(self.full_corridor_cells)
        largest_component = backbone_components[0]
        largest_component_summary = self._summarize_subset(
            self.spatiotemporal.loc[~self.spatiotemporal["cell_id"].isin(largest_component)].copy()
        )

        return {
            "leave_one_day_out": {
                "summary": summarize_collection(leave_one_day_results),
                "per_day": leave_one_day_results,
            },
            "leave_one_vendor_out": {
                "summary": summarize_collection(leave_one_vendor_results),
                "per_vendor": leave_one_vendor_results,
            },
            "leave_top_endpoint_out": {
                "top_endpoint_cells_removed": int(top_endpoint_count),
                "top_endpoint_share_of_endpoint_cells": float(top_endpoint_count / len(endpoint_cells)),
                "summary": top_endpoint_summary,
            },
            "leave_largest_backbone_cluster_out": {
                "n_backbone_components": int(len(backbone_components)),
                "largest_component_cells": int(len(largest_component)),
                "largest_component_share_of_backbone_cells": float(len(largest_component) / len(self.full_corridor_cells)),
                "summary": largest_component_summary,
            },
        }

    def experiment14_null_models(self) -> dict:
        assert self.spatiotemporal is not None
        assert self.rho_star is not None

        observed = self._summarize_subset(
            self.spatiotemporal.copy(),
            corridor_cells=self.full_corridor_cells,
            reestimate_corridor=False,
            rho_star=float(self.rho_star),
            reestimate_rho=False,
        )
        base_spatiotemporal = self.spatiotemporal[["cell_id", "n_points", "mean_speed", "date_hour"]].copy()
        base_cells = base_spatiotemporal["cell_id"].to_numpy()
        date_hour_indices = [
            np.asarray(indices, dtype=np.int32)
            for indices in base_spatiotemporal.groupby("date_hour", sort=False).indices.values()
        ]

        time_shuffle_rr: list[float] = []
        time_shuffle_or: list[float] = []
        for _ in range(self.n_null_permutations):
            shuffled_cells = base_cells.copy()
            for indices in date_hour_indices:
                if len(indices) > 1:
                    shuffled_cells[indices] = shuffled_cells[indices][self.rng.permutation(len(indices))]
            shuffled = base_spatiotemporal[["n_points", "mean_speed"]].copy()
            shuffled["cell_id"] = shuffled_cells
            summary = self._summarize_subset(
                shuffled,
                rho_star=float(self.rho_star),
                reestimate_rho=False,
            )
            time_shuffle_rr.append(summary["rr"])
            time_shuffle_or.append(summary["or_adj"])

        active_cells = base_spatiotemporal["cell_id"].astype(str).unique()
        random_backbone_rr: list[float] = []
        random_backbone_or: list[float] = []
        state_data = base_spatiotemporal[["cell_id", "n_points", "mean_speed"]].copy()
        for _ in range(self.n_null_permutations):
            sampled_backbone = set(
                self.rng.choice(active_cells, size=len(self.full_corridor_cells), replace=False).tolist()
            )
            summary = self._summarize_subset(
                state_data,
                corridor_cells=sampled_backbone,
                reestimate_corridor=False,
                rho_star=float(self.rho_star),
                reestimate_rho=False,
            )
            random_backbone_rr.append(summary["rr"])
            random_backbone_or.append(summary["or_adj"])

        mapped_od_pairs = self._load_mapped_od_pairs()
        endpoint_cells = self._load_endpoint_cells()
        observed_congested = self.spatiotemporal.loc[self.spatiotemporal["n_points"] >= float(self.rho_star), "cell_id"].astype(str)
        observed_endpoint_share = float(observed_congested.isin(set(endpoint_cells["cell_id"].astype(str))).mean())
        observed_endpoint_buffer_share = float(
            observed_congested.isin(self._expand_cell_buffer(endpoint_cells, radius=1)).mean()
        )

        start_ids = mapped_od_pairs["start_cell_id"].to_numpy()
        end_ids = mapped_od_pairs["end_cell_id"].to_numpy()
        start_rows = mapped_od_pairs["start_row"].to_numpy(dtype=int)
        start_cols = mapped_od_pairs["start_col"].to_numpy(dtype=int)
        end_rows = mapped_od_pairs["end_row"].to_numpy(dtype=int)
        end_cols = mapped_od_pairs["end_col"].to_numpy(dtype=int)

        od_exact_shares: list[float] = []
        od_buffer_shares: list[float] = []

        for _ in range(self.n_null_permutations):
            sampled_start_idx = self.rng.integers(0, len(mapped_od_pairs), size=len(mapped_od_pairs))
            sampled_end_idx = self.rng.integers(0, len(mapped_od_pairs), size=len(mapped_od_pairs))

            sampled_start_ids = start_ids[sampled_start_idx]
            sampled_end_ids = end_ids[sampled_end_idx]
            synthetic_endpoint_set = set(np.concatenate([sampled_start_ids, sampled_end_ids]).tolist())

            synthetic_cells = pd.DataFrame(
                {
                    "row": np.concatenate([start_rows[sampled_start_idx], end_rows[sampled_end_idx]]),
                    "col": np.concatenate([start_cols[sampled_start_idx], end_cols[sampled_end_idx]]),
                }
            ).drop_duplicates(ignore_index=True)
            synthetic_buffer = self._expand_cell_buffer(synthetic_cells, radius=1)

            od_exact_shares.append(float(observed_congested.isin(synthetic_endpoint_set).mean()))
            od_buffer_shares.append(float(observed_congested.isin(synthetic_buffer).mean()))

        return {
            "n_permutations": int(self.n_null_permutations),
            "observed_coupling": {
                "rr": observed["rr"],
                "or_adj": observed["or_adj"],
                "rho_star": observed["rho_star"],
            },
            "time_shuffle_null": {
                "design": "shuffle cell labels within each date-hour block while preserving local state values",
                "rr": self._distribution_summary(time_shuffle_rr, observed["rr"]),
                "or_adj": self._distribution_summary(time_shuffle_or, observed["or_adj"]),
            },
            "random_backbone_label_null": {
                "design": "sample random active-cell backbone sets of the observed size while preserving observed local states",
                "rr": self._distribution_summary(random_backbone_rr, observed["rr"]),
                "or_adj": self._distribution_summary(random_backbone_or, observed["or_adj"]),
            },
            "od_endpoint_reassignment_null": {
                "design": "sample start and end endpoint cells independently from observed OD marginals with replacement",
                "endpoint_cell_share": self._distribution_summary(od_exact_shares, observed_endpoint_share),
                "endpoint_buffer_one_share": self._distribution_summary(od_buffer_shares, observed_endpoint_buffer_share),
            },
        }

    def experiment15_morphological_continuity_directionality(self) -> dict:
        assert self.cell_stats is not None

        observed_graph = self._backbone_graph_metrics(self.full_corridor_cells)
        components = self._connected_components(self.full_corridor_cells)
        shape_metrics, cell_to_component, component_axes = self._component_shape_metrics(
            components,
            total_backbone_cells=len(self.full_corridor_cells),
        )
        directionality_metrics = self._moving_directionality_metrics(
            cell_to_component,
            component_axes,
            speed_threshold=2.0,
        )

        active_cells = self.cell_stats["cell_id"].astype(str).to_numpy()
        local_rng = np.random.default_rng(self.seed + 15015)

        null_n_components: list[float] = []
        null_largest_share: list[float] = []
        null_components_50: list[float] = []
        null_degree_ge3: list[float] = []
        null_degree_le1: list[float] = []

        for _ in range(self.n_null_permutations):
            sampled = set(local_rng.choice(active_cells, size=len(self.full_corridor_cells), replace=False).tolist())
            metrics = self._backbone_graph_metrics(sampled)
            null_n_components.append(float(metrics["n_components"]))
            null_largest_share.append(float(metrics["largest_component_share"]))
            null_components_50.append(float(metrics["components_covering_50pct_cells"]))
            null_degree_ge3.append(float(metrics["degree_ge3_share"]))
            null_degree_le1.append(float(metrics["degree_le1_share"]))

        n_components_null = self._distribution_summary(null_n_components, observed_graph["n_components"])
        n_components_null["empirical_p_le_observed"] = float(
            (np.sum(np.asarray(null_n_components, dtype=float) <= observed_graph["n_components"]) + 1.0)
            / (len(null_n_components) + 1.0)
        )
        components_50_null = self._distribution_summary(
            null_components_50,
            observed_graph["components_covering_50pct_cells"],
        )
        components_50_null["empirical_p_le_observed"] = float(
            (np.sum(np.asarray(null_components_50, dtype=float) <= observed_graph["components_covering_50pct_cells"]) + 1.0)
            / (len(null_components_50) + 1.0)
        )
        degree_le1_null = self._distribution_summary(null_degree_le1, observed_graph["degree_le1_share"])
        degree_le1_null["empirical_p_le_observed"] = float(
            (np.sum(np.asarray(null_degree_le1, dtype=float) <= observed_graph["degree_le1_share"]) + 1.0)
            / (len(null_degree_le1) + 1.0)
        )

        return {
            "design": "8-neighbor component topology, PCA-based component axes, and moving-point heading alignment within backbone cells",
            "backbone_cells": int(len(self.full_corridor_cells)),
            "continuity": observed_graph,
            "component_shape_ge5": shape_metrics,
            "moving_point_directionality_ge5": directionality_metrics,
            "random_active_cell_null": {
                "n_permutations": int(self.n_null_permutations),
                "design": "sample random active-cell sets of the observed backbone size and recompute continuity metrics",
                "n_components": n_components_null,
                "largest_component_share": self._distribution_summary(
                    null_largest_share,
                    observed_graph["largest_component_share"],
                ),
                "components_covering_50pct_cells": components_50_null,
                "degree_ge3_share": self._distribution_summary(
                    null_degree_ge3,
                    observed_graph["degree_ge3_share"],
                ),
                "degree_le1_share": degree_le1_null,
            },
        }

    def experiment16_trajectory_path_persistence(self) -> dict:
        trajectory_sequences = self._load_trajectory_sequences()
        if not trajectory_sequences:
            raise ValueError("No valid trajectory sequences available.")

        route_signature_counts = Counter(sequence["cell_sequence"] for sequence in trajectory_sequences)
        signature_sizes = np.asarray(list(route_signature_counts.values()), dtype=int)

        edge_order_support: dict[tuple[str, str], int] = {}
        edge_traversal_counts: Counter = Counter()
        order_edge_sets: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

        for sequence in trajectory_sequences:
            order_id = sequence["order_id"]
            unique_edges = set(sequence["undirected_edges"])
            for edge in sequence["undirected_edges"]:
                edge_traversal_counts[edge] += 1
            for edge in unique_edges:
                order_edge_sets[edge].add(order_id)

        for edge, order_ids in order_edge_sets.items():
            edge_order_support[edge] = len(order_ids)

        support_values = np.asarray(list(edge_order_support.values()), dtype=int)
        traversal_values = np.asarray(list(edge_traversal_counts.values()), dtype=int)
        ranked_edges = sorted(
            edge_traversal_counts.items(),
            key=lambda item: edge_order_support[item[0]],
            reverse=True,
        )
        ranked_edge_weights = np.asarray([weight for _, weight in ranked_edges], dtype=float)

        def top_share(fraction: float) -> float:
            n_top = max(1, int(len(ranked_edges) * fraction))
            return float(ranked_edge_weights[:n_top].sum() / ranked_edge_weights.sum())

        thresholds = [20, 50, 100, 200, 500]
        threshold_results: dict[str, dict] = {}
        for threshold in thresholds:
            skeleton_edges = [edge for edge, support in edge_order_support.items() if support >= threshold]
            skeleton_edge_set = set(skeleton_edges)
            skeleton_nodes = {node for edge in skeleton_edges for node in edge}
            skeleton_components = self._graph_components_from_edges(skeleton_edges) if skeleton_edges else []
            degree_counter: defaultdict[str, int] = defaultdict(int)
            for source, target in skeleton_edges:
                degree_counter[source] += 1
                degree_counter[target] += 1
            node_degrees = np.asarray(list(degree_counter.values()), dtype=float) if degree_counter else np.asarray([], dtype=float)

            path_coverages: list[float] = []
            node_coverages: list[float] = []
            for sequence in trajectory_sequences:
                edges = sequence["undirected_edges"]
                nodes = sequence["unique_cells"]
                path_coverages.append(float(np.mean([edge in skeleton_edge_set for edge in edges])))
                node_coverages.append(float(np.mean([node in skeleton_nodes for node in nodes])))

            overlap = len(skeleton_nodes & self.full_corridor_cells)
            union = len(skeleton_nodes | self.full_corridor_cells)
            threshold_results[f"support_ge_{threshold}"] = {
                "min_distinct_order_support": int(threshold),
                "skeleton_edges": int(len(skeleton_edges)),
                "skeleton_nodes": int(len(skeleton_nodes)),
                "skeleton_components": int(len(skeleton_components)),
                "largest_component_share_nodes": float(len(skeleton_components[0]) / len(skeleton_nodes)) if skeleton_components else float("nan"),
                "mean_node_degree": float(np.mean(node_degrees)) if node_degrees.size else float("nan"),
                "backbone_coverage_share": float(overlap / len(self.full_corridor_cells)),
                "skeleton_nodes_in_backbone_share": float(overlap / len(skeleton_nodes)) if skeleton_nodes else float("nan"),
                "jaccard_with_backbone_nodes": float(overlap / union) if union else float("nan"),
                "mean_order_edge_coverage": float(np.mean(path_coverages)),
                "median_order_edge_coverage": float(np.median(path_coverages)),
                "p25_order_edge_coverage": float(np.percentile(path_coverages, 25)),
                "share_orders_edge_coverage_ge_80pct": float(np.mean(np.asarray(path_coverages) >= 0.8)),
                "mean_order_node_coverage": float(np.mean(node_coverages)),
            }

        core = threshold_results["support_ge_100"]
        return {
            "design": "compress each order trajectory to unique cell transitions, count repeated undirected edges by distinct-order support, and compare repeated-route skeletons to the intensity backbone",
            "trajectory_sequences": {
                "orders": int(len(trajectory_sequences)),
                "median_sequence_length_cells": float(np.median([len(sequence['cell_sequence']) for sequence in trajectory_sequences])),
                "median_unique_cells_per_order": float(np.median([len(sequence["unique_cells"]) for sequence in trajectory_sequences])),
                "total_edge_traversals": int(sum(len(sequence["undirected_edges"]) for sequence in trajectory_sequences)),
            },
            "route_signature_repetition": {
                "unique_route_signatures": int(len(route_signature_counts)),
                "share_orders_with_signature_support_ge_2": float(signature_sizes[signature_sizes >= 2].sum() / signature_sizes.sum()),
                "share_orders_with_signature_support_ge_5": float(signature_sizes[signature_sizes >= 5].sum() / signature_sizes.sum()),
                "top_1_signature_share_orders": float(signature_sizes.max() / signature_sizes.sum()),
                "top_10_signature_share_orders": float(np.sort(signature_sizes)[::-1][:10].sum() / signature_sizes.sum()),
                "max_orders_on_one_signature": int(signature_sizes.max()),
            },
            "edge_support_distribution": {
                "unique_undirected_edges": int(len(edge_order_support)),
                "support_p50": float(np.median(support_values)),
                "support_p90": float(np.percentile(support_values, 90)),
                "support_max": int(np.max(support_values)),
                "traversal_p50": float(np.median(traversal_values)),
                "traversal_p90": float(np.percentile(traversal_values, 90)),
                "traversal_max": int(np.max(traversal_values)),
                "top_0_1pct_edges_traversal_share": top_share(0.001),
                "top_0_5pct_edges_traversal_share": top_share(0.005),
                "top_1pct_edges_traversal_share": top_share(0.01),
            },
            "threshold_sensitivity": threshold_results,
            "core_repeated_route_skeleton": {
                "definition": "undirected cell-to-cell edges reused by at least 100 distinct orders",
                **core,
            },
        }

    def experiment17_phase_altitude_decomposition(self) -> dict:
        assert self.spatiotemporal is not None

        data = self.spatiotemporal.copy()
        endpoint_cells = self._load_endpoint_cells()
        endpoint_buffer_one = self._expand_cell_buffer(endpoint_cells, radius=1)

        low_altitude_threshold = 120.0
        cruise_altitude_threshold = 180.0

        subsets = {
            "near_endpoint_buffer": data.loc[data["cell_id"].isin(endpoint_buffer_one)].copy(),
            "away_from_endpoint_buffer": data.loc[~data["cell_id"].isin(endpoint_buffer_one)].copy(),
            "low_altitude_lt_120m": data.loc[data["mean_altitude"] < low_altitude_threshold].copy(),
            "cruise_like_altitude_ge_180m": data.loc[data["mean_altitude"] >= cruise_altitude_threshold].copy(),
            "away_from_endpoint_and_cruise_like": data.loc[
                (~data["cell_id"].isin(endpoint_buffer_one)) & (data["mean_altitude"] >= cruise_altitude_threshold)
            ].copy(),
        }

        return {
            "design": "re-estimate tipping and backbone amplification on endpoint-buffer and altitude-banded subsets using mean cell-window altitude",
            "endpoint_buffer_radius_cells": 1,
            "altitude_band_definition": {
                "low_altitude_lt_m": low_altitude_threshold,
                "cruise_like_ge_m": cruise_altitude_threshold,
                "note": "descriptive bands based on cell-window mean altitude; not regulatory flight phases",
            },
            "altitude_distribution_reference": {
                "p25_mean_altitude_m": float(data["mean_altitude"].quantile(0.25)),
                "p50_mean_altitude_m": float(data["mean_altitude"].quantile(0.50)),
                "p75_mean_altitude_m": float(data["mean_altitude"].quantile(0.75)),
            },
            "subsets": {
                name: self._summarize_subset(subset)
                for name, subset in subsets.items()
            },
        }

    def save_result(self, name: str, payload: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{name}_results.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_all_experiments(self) -> dict[str, dict]:
        if not self.load_data():
            raise FileNotFoundError("Required processed inputs are missing.")

        experiments = [
            ("out_of_sample_coupling", self.experiment1_out_of_sample_coupling),
            ("unique_flight_occupancy", self.experiment2_unique_flight_occupancy),
            ("exclude_terminal_zones", self.experiment3_exclude_terminal_zones),
            ("cluster_robust_ci", self.experiment4_cluster_robust_ci),
            ("rho_star_bootstrap_ci", self.experiment5_rho_star_bootstrap_ci),
            ("support_ratio_boundary", self.experiment6_support_ratio_boundary_sensitivity),
            ("spatial_hotspot_localization", self.experiment7_spatial_hotspot_localization),
            ("endpoint_node_anchor", self.experiment8_endpoint_node_anchor),
            ("alternative_corridor_definition", self.experiment9_alternative_corridor_definition),
            ("interface_masking", self.experiment10_interface_masking),
            ("breakpoint_specification", self.experiment11_breakpoint_specification),
            ("endpoint_buffer_exclusion", self.experiment12_endpoint_buffer_exclusion),
            ("leave_one_group_out", self.experiment13_leave_one_group_out),
            ("null_model_suite", self.experiment14_null_models),
            ("morphological_continuity_directionality", self.experiment15_morphological_continuity_directionality),
            ("trajectory_path_persistence", self.experiment16_trajectory_path_persistence),
            ("phase_altitude_decomposition", self.experiment17_phase_altitude_decomposition),
        ]

        self.results = {}
        for name, func in experiments:
            try:
                result = func()
            except Exception as exc:
                result = {"error": str(exc)}
            self.results[name] = result
            self.save_result(name, result)

        summary = {
            **self.results,
            "analysis_time": datetime.now().isoformat(),
            "n_bootstraps": self.n_bootstraps,
            "seed": self.seed,
            "rho_bootstrap_seed": self.rho_bootstrap_seed,
            "n_null_permutations": self.n_null_permutations,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary_results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the archived robustness analyses on the processed review package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON outputs.")
    parser.add_argument("--n-bootstraps", type=int, default=1000, help="Number of bootstrap draws.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap resampling.")
    parser.add_argument("--rho-bootstrap-seed", type=int, default=2, help="Legacy seed used for the simplified rho* bootstrap diagnostic.")
    parser.add_argument("--n-null-permutations", type=int, default=50, help="Number of permutations for each null-model design.")
    parser.add_argument("--cell-stats-file", type=Path, default=CELL_STATS_FILE, help="Path to cell_stats.csv.")
    parser.add_argument("--spatiotemporal-file", type=Path, default=SPATIOTEMPORAL_FILE, help="Path to spatiotemporal_stats.csv.")
    parser.add_argument("--cleaned-data-file", type=Path, default=CLEANED_DATA_FILE, help="Path to cleaned_data.csv.")
    parser.add_argument("--od-pairs-file", type=Path, default=OD_PAIRS_FILE, help="Path to od_pairs.csv.")
    parser.add_argument("--fundamental-file", type=Path, default=FUNDAMENTAL_FILE, help="Path to fundamental_data.csv.")
    parser.add_argument("--breakpoint-file", type=Path, default=BREAKPOINT_FILE, help="Path to breakpoint_results.json.")
    parser.add_argument("--grid-info-file", type=Path, default=GRID_INFO_FILE, help="Path to grid_info.json.")
    args = parser.parse_args()

    analyzer = RobustnessAnalyzer(
        output_dir=args.output_dir,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        rho_bootstrap_seed=args.rho_bootstrap_seed,
        n_null_permutations=args.n_null_permutations,
        cell_stats_file=args.cell_stats_file,
        spatiotemporal_file=args.spatiotemporal_file,
        cleaned_data_file=args.cleaned_data_file,
        od_pairs_file=args.od_pairs_file,
        fundamental_file=args.fundamental_file,
        breakpoint_file=args.breakpoint_file,
        grid_info_file=args.grid_info_file,
    )
    results = analyzer.run_all_experiments()

    print("Robustness analyses completed.")
    if "out_of_sample_coupling" in results and "error" not in results["out_of_sample_coupling"]:
        exp1 = results["out_of_sample_coupling"]
        print(
            "Held-out coupling:",
            f"RR={exp1['rr_test']:.2f}",
            f"OR={exp1['or_adj_test']:.2f}",
            f"p={exp1['fisher_p_test']:.3g}",
        )
    if "cluster_robust_ci" in results and "error" not in results["cluster_robust_ci"]:
        exp4 = results["cluster_robust_ci"]
        print(
            "Temporal bootstrap OR CI:",
            f"[{exp4['or_ci_95_lower']:.2f}, {exp4['or_ci_95_upper']:.2f}]",
        )
    if "unique_flight_occupancy" in results and "error" not in results["unique_flight_occupancy"]:
        exp2 = results["unique_flight_occupancy"]
        print(
            "Unique-flight tipping:",
            f"rho*={exp2['rho_star_unique']:.1f}",
            f"RR={exp2['rr_unique']:.2f}",
        )
    if "alternative_corridor_definition" in results and "error" not in results["alternative_corridor_definition"]:
        exp9 = results["alternative_corridor_definition"]
        print(
            "Alternative corridor RR:",
            f"{exp9['rr_alt']:.2f}",
            f"Jaccard={exp9['jaccard_with_point_count_corridor']:.3f}",
        )
    if "endpoint_buffer_exclusion" in results and "error" not in results["endpoint_buffer_exclusion"]:
        exp12 = results["endpoint_buffer_exclusion"]["excluding_endpoint_cells_and_1cell_buffer"]
        print(
            "Endpoint-buffer exclusion:",
            f"rho*={exp12['rho_star']:.1f}",
            f"RR={exp12['rr']:.2f}",
        )
    if "null_model_suite" in results and "error" not in results["null_model_suite"]:
        exp14 = results["null_model_suite"]
        print(
            "Null suite max RR:",
            f"time-shuffle={exp14['time_shuffle_null']['rr']['max']:.2f}",
            f"random-backbone={exp14['random_backbone_label_null']['rr']['max']:.2f}",
        )
    if "morphological_continuity_directionality" in results and "error" not in results["morphological_continuity_directionality"]:
        exp15 = results["morphological_continuity_directionality"]
        print(
            "Backbone morphology:",
            f"components={exp15['continuity']['n_components']}",
            f"largest-share={exp15['continuity']['largest_component_share']:.3f}",
            f"median-axial={exp15['moving_point_directionality_ge5']['weighted_median_component_axial_resultant']:.3f}",
        )
    if "trajectory_path_persistence" in results and "error" not in results["trajectory_path_persistence"]:
        exp16 = results["trajectory_path_persistence"]["core_repeated_route_skeleton"]
        print(
            "Route skeleton core:",
            f"nodes={exp16['skeleton_nodes']}",
            f"mean-path-cover={exp16['mean_order_edge_coverage']:.3f}",
            f"backbone-cover={exp16['backbone_coverage_share']:.3f}",
        )
    if "phase_altitude_decomposition" in results and "error" not in results["phase_altitude_decomposition"]:
        exp17 = results["phase_altitude_decomposition"]["subsets"]["away_from_endpoint_and_cruise_like"]
        print(
            "Phase-aware away+cruise:",
            f"rho*={exp17['rho_star']:.1f}",
            f"RR={exp17['rr']:.2f}",
            f"drop={exp17['speed_drop_percent']:.1f}%",
        )


if __name__ == "__main__":
    main()



