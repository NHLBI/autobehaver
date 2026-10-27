"""Transform MMGen positions and summarize behavior-specific occupancy and orientation.

Process animal data in bounded chunks and use bounded samples for radial-density plots.
"""

from __future__ import annotations

from .input_contract import navigation_to_image_angle
from .hdf_store import select_individual_rows

import hashlib
import os
import re
import shutil
from pathlib import Path
from .batching import ordered_worker_results
from .table_metadata import read_individual_metadata

from autobehaver.atomic_io import atomic_write_json, sha256_file

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from autobehaver.progress import progress as tqdm

from .naming import collapse_individual_id

DEFAULT_CHAMBER_DIAMETER_PX: int | None = 530
DEFAULT_GENERATE_QUIVER_PLOTS: bool = True


# Internal bounded sampling + accumulation helpers

class _BottomKSampler:
    """Maintain a bounded sample of values using a bottom-K random-key scheme.

    This approximates uniform sampling without replacement over a stream.
    We keep the K smallest random keys across all seen values.

    Notes
    -----
    - Uses float32 keys/values to keep memory small.
    - Update cost stays small as the stream grows because very few new keys
      beat the current max key in the bottom-K set.
    """

    def __init__(self, k: int, seed: int):
        self.k = int(k)
        self.rng = np.random.default_rng(int(seed) % (2**32))
        self.keys = np.empty((0,), dtype=np.float32)
        self.values = np.empty((0,), dtype=np.float32)

    def update(self, values: np.ndarray) -> None:
        """Retain the values with the smallest random keys while bounding sample memory."""
        if values is None:
            return

        v = np.asarray(values, dtype=np.float32)
        if v.size == 0:
            return

        v = v[np.isfinite(v)]
        if v.size == 0:
            return

        k_new = self.rng.random(v.size, dtype=np.float32)

        # Fill stage
        if self.values.size < self.k:
            self.values = np.concatenate([self.values, v])
            self.keys = np.concatenate([self.keys, k_new])
            if self.values.size > self.k:
                idx = np.argpartition(self.keys, self.k - 1)[: self.k]
                self.values = self.values[idx]
                self.keys = self.keys[idx]
            return

        # Steady state
        max_key = float(self.keys.max())
        mask = k_new < max_key
        if not np.any(mask):
            return

        cand_values = np.concatenate([self.values, v[mask]])
        cand_keys = np.concatenate([self.keys, k_new[mask]])

        if cand_values.size <= self.k:
            self.values = cand_values
            self.keys = cand_keys
            return

        idx = np.argpartition(cand_keys, self.k - 1)[: self.k]
        self.values = cand_values[idx]
        self.keys = cand_keys[idx]

    def get(self) -> np.ndarray:
        return self.values


class _OccupancyAccumulator:
    """Streaming occupancy accumulator with bounded memory."""

    def __init__(
        self,
        *,
        heatmap_bins: int = 50,
        heatmap_range=((0, 600), (0, 600)),
        quiver_grid_size: int = 20,
        kde_max_points: int = 50_000,
        seed: int = 0,
    ):
        """Allocate bounded heatmap, orientation-vector, and radial-sampling
        accumulators.
        """
        self.behavior_cols = ["masked_pose", "masked_orient", "masked_pose_orient"]

        self.heatmap_bins = int(heatmap_bins)
        self.heatmap_range = heatmap_range
        self.quiver_grid_size = int(quiver_grid_size)
        self.kde_max_points = int(kde_max_points)
        self.seed = int(seed)

        # Track which behaviors exist per category
        self.behaviors: dict[str, set[str]] = {c: set() for c in self.behavior_cols}

        # Heatmaps: counts + totals
        self._heat_counts: dict[tuple[str, str, str], np.ndarray] = {}
        self._heat_n: dict[tuple[str, str, str], int] = {}

        # Quiver: sum U/V + counts
        self._q_sum_u: dict[tuple[str, str, str], np.ndarray] = {}
        self._q_sum_v: dict[tuple[str, str, str], np.ndarray] = {}
        self._q_count: dict[tuple[str, str, str], np.ndarray] = {}

        # KDE: bottom-K samplers
        self._kde: dict[tuple[str, str, str], _BottomKSampler] = {}

    @staticmethod
    def _stable_seed(key: tuple[str, str, str], base_seed: int) -> int:
        # Python's built-in hash is salted per process; use MD5 for stability.
        h = hashlib.md5(repr(key).encode("utf-8")).digest()
        return (int.from_bytes(h[:4], "little") + int(base_seed)) % (2**32)

    def _get_kde_sampler(self, key: tuple[str, str, str]) -> _BottomKSampler:
        if key not in self._kde:
            self._kde[key] = _BottomKSampler(self.kde_max_points, self._stable_seed(key, self.seed))
        return self._kde[key]

    # Heatmap accumulation
    def add_heatmap(self, col: str, behavior: str, group: str, x: np.ndarray, y: np.ndarray) -> None:
        """Add positional counts to the behavior/group occupancy histogram."""
        key = (col, behavior, group)
        if key not in self._heat_counts:
            self._heat_counts[key] = np.zeros((self.heatmap_bins, self.heatmap_bins), dtype=np.float32)
            self._heat_n[key] = 0

        # histogram2d returns float64 by default; cast to float32
        h, _, _ = np.histogram2d(
            x,
            y,
            bins=self.heatmap_bins,
            range=self.heatmap_range,
        )
        self._heat_counts[key] += h.astype(np.float32, copy=False)
        self._heat_n[key] += int(len(x))
        self.behaviors[col].add(str(behavior))

    # KDE sampling accumulation
    def add_kde(self, col: str, behavior: str, group: str, r_norm: np.ndarray) -> None:
        key = (col, behavior, group)
        sampler = self._get_kde_sampler(key)
        sampler.update(r_norm)
        self.behaviors[col].add(str(behavior))

    # Quiver accumulation
    def add_quiver(self, col: str, behavior: str, group: str, x: np.ndarray, y: np.ndarray, angle_deg: np.ndarray) -> None:
        """Accumulate image-coordinate orientation vectors by spatial bin."""
        key = (col, behavior, group)
        if key not in self._q_sum_u:
            gs = self.quiver_grid_size
            self._q_sum_u[key] = np.zeros((gs, gs), dtype=np.float32)
            self._q_sum_v[key] = np.zeros((gs, gs), dtype=np.float32)
            self._q_count[key] = np.zeros((gs, gs), dtype=np.int32)

        # Bin coordinates
        gs = self.quiver_grid_size
        edges = np.linspace(self.heatmap_range[0][0], self.heatmap_range[0][1], gs + 1)

        x_idx = np.digitize(x, edges) - 1
        y_idx = np.digitize(y, edges) - 1

        valid = (x_idx >= 0) & (x_idx < gs) & (y_idx >= 0) & (y_idx < gs)
        if not np.any(valid):
            return

        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        ang = angle_deg[valid]

        # Vector components
        # Input uses KeyPre navigation bearing (0=N, 90=E). Plot axes use
        # image coordinates (+x right, +y down), including inverted display y.
        ang_rad = np.deg2rad(navigation_to_image_angle(ang))
        u = np.cos(ang_rad).astype(np.float32, copy=False)
        v = np.sin(ang_rad).astype(np.float32, copy=False)

        flat = (y_idx * gs + x_idx).astype(np.int64, copy=False)
        n_bins = gs * gs

        c = np.bincount(flat, minlength=n_bins)
        su = np.bincount(flat, weights=u, minlength=n_bins)
        sv = np.bincount(flat, weights=v, minlength=n_bins)

        self._q_count[key] += c.reshape(gs, gs).astype(np.int32, copy=False)
        self._q_sum_u[key] += su.reshape(gs, gs).astype(np.float32, copy=False)
        self._q_sum_v[key] += sv.reshape(gs, gs).astype(np.float32, copy=False)
        self.behaviors[col].add(str(behavior))

    # Accessors for plotting
    def heatmap_keys(self):
        return list(self._heat_counts.keys())

    def get_heatmap(self, key: tuple[str, str, str]):
        counts = self._heat_counts.get(key)
        n = self._heat_n.get(key, 0)
        return counts, n

    def get_kde_values(self, key: tuple[str, str, str]) -> np.ndarray:
        sampler = self._kde.get(key)
        if sampler is None:
            return np.empty((0,), dtype=np.float32)
        return sampler.get()

    def get_quiver_arrays(self, key: tuple[str, str, str]):
        return (
            self._q_sum_u.get(key),
            self._q_sum_v.get(key),
            self._q_count.get(key),
        )


# Streaming transform + accumulation

def _category_subfolder_name(col: str) -> str:
    # Keep the current label capitalization behavior (note: .capitalize() lowercases the rest).
    return col.replace("masked_", "").replace("pose_orient", "Pose+Orient").capitalize()


def _stream_transform_write_and_accumulate(
    temp_store_path: str,
    positional_key: str,
    output_dir: str,
    *,
    chamber_diameter_px: int | None = DEFAULT_CHAMBER_DIAMETER_PX,
    generate_quiver_plots: bool = DEFAULT_GENERATE_QUIVER_PLOTS,
    transformed_output_path: str | None = None,
    num_workers: int = 1,
) -> _OccupancyAccumulator | None:
    """Stream per-individual transformations to CSV and update bounded accumulators.

    Returns
    -------
    _OccupancyAccumulator | None
        None if no individuals produced transformed data.
    """

    acc = _OccupancyAccumulator(
        heatmap_bins=50,
        heatmap_range=((0, 600), (0, 600)),
        quiver_grid_size=20,
        kde_max_points=50_000,
        seed=0,
    )

    # Use fixed arena radius derived from user-supplied CHAMBER_DIAMETER_PX (if provided)
    fixed_arena_radius = None
    if chamber_diameter_px is not None:
        try:
            fixed_arena_radius = float(chamber_diameter_px) / 2.0
            if (not np.isfinite(fixed_arena_radius)) or (fixed_arena_radius <= 0):
                raise ValueError
            print(
                f"[Arena] Using fixed chamber diameter {chamber_diameter_px} px (radius {fixed_arena_radius:.3f} px) for r_norm."
            )
        except Exception:
            print(
                f"WARNING: CHAMBER_DIAMETER_PX={chamber_diameter_px} is invalid. Falling back to auto radius estimation."
            )
            fixed_arena_radius = None

    # QC counters (helps validate diameter choice)
    qc_total = 0
    qc_over_1p0 = 0
    qc_over_1p2 = 0

    transformed_path = transformed_output_path or os.path.join(output_dir, "temp_transformed_positional_data.csv")
    os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
    header_written = False

    # Columns we need for downstream feature_matrix + occupancy plots.
    cols_to_load = [
        "group",
        "individual",
        "source_file",
        "frame_idx",
        # Raw labels
        "masked_pose",
        "masked_orient",
        "masked_pose_orient",
        # Valid labels
        "masked_pose_valid",
        "masked_orient_valid",
        "masked_pose_orient_valid",
        # Position + orientation
        "centroid_x",
        "centroid_y",
        "angle",
        # Movement + angle features
        "speed_mm_s",
        "anglefeature_Wings_L_R",
        "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR",
        "anglefeature_Legs_BL_BR",
    ]

    from .finalize_execution import prepare_report_tasks, budgeted_results, report_part, merge_reduction
    with pd.HDFStore(temp_store_path, mode="r") as store:
        cols_for_select = _load_positional_cols_for_select(store, positional_key, cols_to_load)
    tasks, estimates = prepare_report_tasks(temp_store_path, positional_key, output_dir,
                                           cols_for_select, fixed_arena_radius, generate_quiver_plots)
    temporary_output = transformed_path + '.assembling'
    try:
        with open(temporary_output, 'w', encoding='utf-8', newline='') as output:
            for result in tqdm(budgeted_results(report_part, tasks, estimates, num_workers=num_workers),
                               total=len(tasks), desc="Calibrating arenas"):
                qc_n, qc_1, qc_12 = result['qc']
                qc_total += qc_n
                qc_over_1p0 += qc_1
                qc_over_1p2 += qc_12
                if not result['rows']:
                    continue
                merge_reduction(acc, result)
                with open(result['csv'], encoding='utf-8', newline='') as part:
                    if header_written:
                        next(part, None)
                    shutil.copyfileobj(part, output)
                header_written = True
        os.replace(temporary_output, transformed_path)
    except (OSError, KeyError) as e:
        raise RuntimeError('Failed to finalize positional reports; completed recording parts are retained') from e

    if not header_written:
        print("Warning: No data could be transformed.")
        return None

    # QC summary (printed once)
    if qc_total > 0:
        frac_over_1p0 = qc_over_1p0 / qc_total
        frac_over_1p2 = qc_over_1p2 / qc_total
        print(f"[Arena QC] r_norm_raw > 1.0: {frac_over_1p0:.3%}, > 1.2: {frac_over_1p2:.3%} (frames={qc_total})")

    print(f"Transformed data saved to {transformed_path}")

    return acc


# Plot generation from bounded accumulators

def _generate_radial_distribution_plots_from_acc(acc: _OccupancyAccumulator, output_dir: str) -> list[str]:
    """Generate radial KDE plots from bounded samples."""
    base_plot_folder = os.path.join(output_dir, "9_Occupancy_Plots")
    generated = []

    for col in acc.behavior_cols:
        if not acc.behaviors.get(col):
            continue

        subfolder_name = _category_subfolder_name(col)
        plot_output_folder = os.path.join(base_plot_folder, subfolder_name)
        os.makedirs(plot_output_folder, exist_ok=True)

        unique_behaviors = sorted(list(acc.behaviors[col]))
        for behavior in tqdm(unique_behaviors, desc=f"Plotting {col}"):
            # Collect sampled r_norm values per group
            parts = []
            groups_found = set()
            # Find all sampler keys for this behavior
            for (k_col, k_beh, k_group) in list(acc._kde.keys()):
                if k_col == col and k_beh == behavior:
                    vals = acc.get_kde_values((k_col, k_beh, k_group))
                    if vals.size == 0:
                        continue
                    parts.append(pd.DataFrame({"r_norm": vals, "group": k_group}))
                    groups_found.add(k_group)

            if not parts:
                continue

            behavior_df = pd.concat(parts, ignore_index=True)

            plt.figure(figsize=(10, 6))
            sns.kdeplot(
                data=behavior_df,
                x="r_norm",
                hue="group",
                fill=True,
                common_norm=False,
                alpha=0.3,
                warn_singular=False,
            )

            plt.title(f"Radial Distribution for: {behavior}", fontsize=16)
            plt.xlabel("Normalized Distance from Center (0=Center, 1=Wall)", fontsize=12)
            plt.ylabel("Probability Density", fontsize=12)
            plt.xlim(0, 1.2)
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            safe_filename = re.sub(r"[\\/*?:\"<>| +]", "_", behavior) + ".png"
            plot_path = os.path.join(plot_output_folder, f"{safe_filename.replace('.png', '_radial_dist.png')}")

            try:
                plt.savefig(plot_path, dpi=300)
                generated.append(plot_path)
            finally:
                plt.close()
    return generated


def _generate_2d_heatmaps_from_acc(acc: _OccupancyAccumulator, output_dir: str) -> list[str]:
    """Generate 2D binned occupancy heatmaps from accumulated histograms."""
    base_plot_folder = os.path.join(output_dir, "10_Occupancy_Heatmaps")
    generated = []
    bins = acc.heatmap_bins
    plot_range = acc.heatmap_range

    for col in acc.behavior_cols:
        if not acc.behaviors.get(col):
            continue

        subfolder_name = _category_subfolder_name(col)
        plot_output_folder = os.path.join(base_plot_folder, subfolder_name)
        os.makedirs(plot_output_folder, exist_ok=True)

        unique_behaviors = sorted(list(acc.behaviors[col]))
        for behavior in tqdm(unique_behaviors, desc=f"Generating 2D Heatmaps for {col}"):
            # Gather groups for this behavior
            group_keys = [k for k in acc._heat_counts.keys() if k[0] == col and k[1] == behavior]
            unique_groups = sorted({k[2] for k in group_keys})

            histogram_data: dict[str, np.ndarray | None] = {}
            global_vmax = 0.0

            # PASS 1: compute normalized hist + global vmax
            for group in unique_groups:
                key = (col, behavior, group)
                counts, n = acc.get_heatmap(key)
                if counts is None or n <= 0:
                    histogram_data[group] = None
                    continue
                h = counts / float(n)
                histogram_data[group] = h
                vmax = float(np.nanmax(h)) if h.size else 0.0
                if vmax > global_vmax:
                    global_vmax = vmax

            # PASS 2: plot
            for group in unique_groups:
                h = histogram_data.get(group)
                if h is None:
                    continue

                fig, ax = plt.subplots(figsize=(9, 8))
                sns.heatmap(
                    h.T,
                    ax=ax,
                    cmap="viridis",
                    vmin=0,
                    vmax=global_vmax,
                    cbar_kws={"label": "Proportion of Time (Fraction of Behavior)"},
                )

                ax.set_xticks(np.linspace(0, bins, 5))
                ax.set_xticklabels(np.linspace(plot_range[0][0], plot_range[0][1], 5, dtype=int))
                ax.set_yticks(np.linspace(0, bins, 5))
                ax.set_yticklabels(np.linspace(plot_range[1][0], plot_range[1][1], 5, dtype=int))

                # Seaborn already puts image-coordinate y=0 at the top.
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"Occupancy Histogram: {group}\n{behavior}", fontsize=14)
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")

                safe_bh_name = re.sub(r"[\\/*?:\"<>| +]", "_", behavior)
                cartesian_path = os.path.join(plot_output_folder, f"{safe_bh_name}_{group}_heatmap_histogram.png")
                try:
                    plt.savefig(cartesian_path, dpi=150, bbox_inches="tight")
                    generated.append(cartesian_path)
                finally:
                    plt.close()
    return generated


def _generate_orientation_quiver_plots_from_acc(
    acc: _OccupancyAccumulator,
    output_dir: str,
    *,
    min_samples_per_bin: int = 5,
) -> list[str]:
    """Generate orientation quiver plots from accumulated vector sums."""

    base_plot_folder = os.path.join(output_dir, "11_Orientation_Quiver_Plots")
    generated = []

    gs = acc.quiver_grid_size
    bins = np.linspace(acc.heatmap_range[0][0], acc.heatmap_range[0][1], gs + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    Xc, Yc = np.meshgrid(centers, centers)

    for col in acc.behavior_cols:
        if not acc.behaviors.get(col):
            continue

        subfolder_name = _category_subfolder_name(col)
        plot_output_folder = os.path.join(base_plot_folder, subfolder_name)
        os.makedirs(plot_output_folder, exist_ok=True)

        unique_behaviors = sorted(list(acc.behaviors[col]))
        for behavior in tqdm(unique_behaviors, desc=f"Generating Quiver Plots for {col}"):
            # Gather groups for this behavior
            group_keys = [k for k in acc._q_count.keys() if k[0] == col and k[1] == behavior]
            unique_groups = sorted({k[2] for k in group_keys})

            for group in unique_groups:
                key = (col, behavior, group)
                su, sv, cnt = acc.get_quiver_arrays(key)
                if su is None or sv is None or cnt is None:
                    continue

                mask = cnt >= int(min_samples_per_bin)
                if not np.any(mask):
                    continue

                # Avoid divide-by-zero
                cnt_f = cnt.astype(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    u_mean = np.where(mask, su / cnt_f, np.nan)
                    v_mean = np.where(mask, sv / cnt_f, np.nan)

                x_plot = Xc[mask]
                y_plot = Yc[mask]
                u_plot = u_mean[mask]
                v_plot = v_mean[mask]

                # Vector magnitude for coloring
                magnitude = np.sqrt(u_plot**2 + v_plot**2)

                fig, ax = plt.subplots(figsize=(10, 10))
                q = ax.quiver(
                    x_plot,
                    y_plot,
                    u_plot,
                    v_plot,
                    magnitude,
                    cmap="viridis",
                    clim=(0, 1),
                    headwidth=3,
                    headlength=4,
                    scale=25,
                    pivot="mid",
                    angles="xy",  # Follow image axes, including inverted y.
                )

                fig.colorbar(q, ax=ax, label="Orientation Consistency (Vector Magnitude)")

                ax.set_xlim(acc.heatmap_range[0][0], acc.heatmap_range[0][1])
                ax.set_ylim(acc.heatmap_range[1][0], acc.heatmap_range[1][1])
                ax.invert_yaxis()
                ax.set_title(f"Orientation Quiver Plot: {group}\n{behavior}", fontsize=14)
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")
                ax.set_aspect("equal", adjustable="box")

                safe_bh_name = re.sub(r"[\\/*?:\"<>| +]", "_", behavior)
                plot_path = os.path.join(plot_output_folder, f"{safe_bh_name}_{group}_quiver.png")
                try:
                    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                    generated.append(plot_path)
                finally:
                    plt.close()
    return generated


# Matrix-first fast-path helpers (no plot accumulators)

def _resolve_fixed_arena_radius(chamber_diameter_px: int | None) -> float | None:
    """Resolve a positive fixed radius from the configured diameter, or disable fixed
    calibration.
    """
    fixed_arena_radius = None
    if chamber_diameter_px is not None:
        try:
            fixed_arena_radius = float(chamber_diameter_px) / 2.0
            if (not np.isfinite(fixed_arena_radius)) or (fixed_arena_radius <= 0):
                raise ValueError
            print(
                f"[Arena] Using fixed chamber diameter {chamber_diameter_px} px (radius {fixed_arena_radius:.3f} px) for r_norm."
            )
        except Exception:
            print(
                f"WARNING: CHAMBER_DIAMETER_PX={chamber_diameter_px} is invalid. Falling back to auto radius estimation."
            )
            fixed_arena_radius = None
    return fixed_arena_radius


def _load_positional_cols_for_select(store: pd.HDFStore, positional_key: str, cols_to_load: list[str]) -> list[str] | None:
    schema0 = store.select(positional_key, start=0, stop=0)
    existing_cols = set(schema0.columns)
    return [c for c in cols_to_load if c in existing_cols]


def _transform_positional_individual_df(individual_df: pd.DataFrame, *, fixed_arena_radius: float | None) -> tuple[pd.DataFrame | None, int, int, int]:
    """Center image positions and derive radial distance and relative bearing for one
    recording.
    """
    if individual_df is None or individual_df.empty:
        return None, 0, 0, 0
    clean_df = individual_df.dropna(subset=["centroid_x", "centroid_y"])
    if clean_df.empty:
        return None, 0, 0, 0

    center_x = 299.5
    center_y = 299.5
    centered_x = individual_df["centroid_x"] - center_x
    centered_y = individual_df["centroid_y"] - center_y
    r = np.sqrt(centered_x**2 + centered_y**2)

    if fixed_arena_radius is not None:
        arena_radius = float(fixed_arena_radius)
    else:
        arena_radius = np.nanpercentile(r, 99)
    if (not np.isfinite(arena_radius)) or (arena_radius <= 0):
        arena_radius = 1.0

    r_norm_raw = r / arena_radius
    rnr = np.asarray(r_norm_raw, dtype=float)
    finite_mask = np.isfinite(rnr)
    qc_total = int(np.sum(finite_mask))
    qc_over_1p0 = int(np.sum(rnr[finite_mask] > 1.0))
    qc_over_1p2 = int(np.sum(rnr[finite_mask] > 1.2))

    theta_pos_rad = np.arctan2(centered_y, centered_x)
    theta_pos_deg = np.degrees(theta_pos_rad)
    fly_angle_math_deg = navigation_to_image_angle(individual_df["angle"])
    relative_angle = (fly_angle_math_deg - theta_pos_deg + 180) % 360 - 180

    out_df = individual_df.copy()
    out_df["r_norm"] = pd.Series(r_norm_raw, index=out_df.index).clip(lower=0, upper=1.5)
    out_df["positional_angle"] = theta_pos_deg
    out_df["fly_angle_math"] = fly_angle_math_deg
    out_df["relative_angle"] = relative_angle
    out_df["individual"] = out_df["individual"].map(collapse_individual_id)
    return out_df, qc_total, qc_over_1p0, qc_over_1p2


def _ordered_chunks(items: list[str], n_chunks: int) -> list[list[str]]:
    """Split ordered animal identities into contiguous worker chunks."""
    if not items:
        return []
    # Create more tasks than workers so multiprocessing can rebalance heavy
    # animals better while still processing contiguous groups of identities.
    n_chunks = max(1, min(int(n_chunks) * 4, len(items)))
    base, rem = divmod(len(items), n_chunks)
    out: list[list[str]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        stop = start + size
        if stop > start:
            out.append(items[start:stop])
        start = stop
    return out


def _write_csv_parts_to_target(part_paths: list[str], output_path: str) -> None:
    """Concatenate ordered positional CSV parts with one header."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass
    header_written = False
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        for part_path in part_paths:
            if not part_path or not os.path.exists(part_path):
                raise RuntimeError(f'Missing transformed positional chunk: {part_path}')
            with open(part_path, "r", encoding="utf-8", newline="") as in_f:
                if not header_written:
                    shutil.copyfileobj(in_f, out_f)
                    header_written = True
                else:
                    next(in_f, None)
                    shutil.copyfileobj(in_f, out_f)


def _occupancy_transform_chunk_worker(args: tuple[str, str, list[str], list[str] | None, float | None, str]) -> dict:
    """Transform an ordered animal chunk and write its positional CSV part."""
    temp_store_path, positional_key, individual_ids, cols_for_select, fixed_arena_radius, chunk_output_path = args
    header_written = False
    n_rows = 0
    qc_total = 0
    qc_over_1p0 = 0
    qc_over_1p2 = 0

    with pd.HDFStore(temp_store_path, mode="r") as store:
        for individual_id in individual_ids:
            individual_df = select_individual_rows(
                store, positional_key, individual_id, columns=cols_for_select)
            transformed_df, qc_n, qc_1, qc_12 = _transform_positional_individual_df(
                individual_df,
                fixed_arena_radius=fixed_arena_radius,
            )
            qc_total += int(qc_n)
            qc_over_1p0 += int(qc_1)
            qc_over_1p2 += int(qc_12)
            if transformed_df is None or transformed_df.empty:
                continue
            transformed_df.to_csv(
                chunk_output_path,
                index=False,
                mode="a" if header_written else "w",
                header=not header_written,
            )
            header_written = True
            n_rows += int(len(transformed_df))

    return {
        "chunk_output_path": chunk_output_path,
        "n_rows": int(n_rows),
        "qc_total": int(qc_total),
        "qc_over_1p0": int(qc_over_1p0),
        "qc_over_1p2": int(qc_over_1p2),
    }



def _occupancy_transform_chunk_collect_worker(args: tuple[str, str, list[str], list[str] | None, float | None]) -> dict:
    """Transform an ordered animal chunk and return its positional rows."""
    temp_store_path, positional_key, individual_ids, cols_for_select, fixed_arena_radius = args
    frames: list[pd.DataFrame] = []
    qc_total = 0
    qc_over_1p0 = 0
    qc_over_1p2 = 0

    with pd.HDFStore(temp_store_path, mode="r") as store:
        for individual_id in individual_ids:
            individual_df = select_individual_rows(
                store, positional_key, individual_id, columns=cols_for_select)
            transformed_df, qc_n, qc_1, qc_12 = _transform_positional_individual_df(
                individual_df,
                fixed_arena_radius=fixed_arena_radius,
            )
            qc_total += int(qc_n)
            qc_over_1p0 += int(qc_1)
            qc_over_1p2 += int(qc_12)
            if transformed_df is None or transformed_df.empty:
                continue
            frames.append(transformed_df)

    combined_df = pd.concat(frames, ignore_index=True) if frames else None
    n_rows = 0 if combined_df is None else int(len(combined_df))
    return {
        "dataframe": combined_df,
        "n_rows": int(n_rows),
        "qc_total": int(qc_total),
        "qc_over_1p0": int(qc_over_1p0),
        "qc_over_1p2": int(qc_over_1p2),
    }

def _serial_transform_write_without_accumulate(
    temp_store_path: str,
    positional_key: str,
    transformed_output_path: str,
    *,
    fixed_arena_radius: float | None,
) -> bool:
    """Transform animal positions in order and stream them to CSV without plot
    accumulators.
    """
    cols_to_load = [
        "group", "individual", "source_file", "frame_idx",
        "masked_pose", "masked_orient", "masked_pose_orient",
        "masked_pose_valid", "masked_orient_valid", "masked_pose_orient_valid",
        "centroid_x", "centroid_y", "angle", "speed_mm_s",
        "anglefeature_Wings_L_R", "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR", "anglefeature_Legs_BL_BR",
    ]

    os.makedirs(os.path.dirname(transformed_output_path), exist_ok=True)
    if os.path.exists(transformed_output_path):
        try:
            os.remove(transformed_output_path)
        except Exception:
            pass

    header_written = False
    qc_total = qc_over_1p0 = qc_over_1p2 = 0
    with pd.HDFStore(temp_store_path, mode="r") as store:
        individuals = read_individual_metadata(store, positional_key, columns=("individual",))["individual"].dropna().unique()
        individuals = sorted([str(x) for x in individuals])
        cols_for_select = _load_positional_cols_for_select(store, positional_key, cols_to_load)
        for individual_id in tqdm(individuals, desc="Transforming positional data"):
            individual_df = select_individual_rows(
                store, positional_key, individual_id, columns=cols_for_select)
            transformed_df, qc_n, qc_1, qc_12 = _transform_positional_individual_df(individual_df, fixed_arena_radius=fixed_arena_radius)
            qc_total += int(qc_n)
            qc_over_1p0 += int(qc_1)
            qc_over_1p2 += int(qc_12)
            if transformed_df is None or transformed_df.empty:
                continue
            transformed_df.to_csv(
                transformed_output_path,
                index=False,
                mode="a" if header_written else "w",
                header=not header_written,
            )
            header_written = True

    if not header_written:
        return False
    if qc_total > 0:
        print(f"[Arena QC] r_norm_raw > 1.0: {qc_over_1p0 / qc_total:.3%}, > 1.2: {qc_over_1p2 / qc_total:.3%} (frames={qc_total})")
    print(f"Transformed data saved to {transformed_output_path}")
    return True


def _parallel_transform_write_without_accumulate(
    temp_store_path: str,
    positional_key: str,
    transformed_output_path: str,
    *,
    fixed_arena_radius: float | None,
    num_workers: int,
) -> bool:
    """Transform contiguous animal chunks in workers and combine their ordered CSV
    parts.
    """
    cols_to_load = [
        "group", "individual", "source_file", "frame_idx",
        "masked_pose", "masked_orient", "masked_pose_orient",
        "masked_pose_valid", "masked_orient_valid", "masked_pose_orient_valid",
        "centroid_x", "centroid_y", "angle", "speed_mm_s",
        "anglefeature_Wings_L_R", "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR", "anglefeature_Legs_BL_BR",
    ]

    with pd.HDFStore(temp_store_path, mode="r") as store:
        individuals = read_individual_metadata(store, positional_key, columns=("individual",))["individual"].dropna().unique()
        individuals = sorted([str(x) for x in individuals])
        cols_for_select = _load_positional_cols_for_select(store, positional_key, cols_to_load)

    if not individuals:
        return False

    worker_count = max(1, min(int(num_workers), len(individuals)))
    if worker_count <= 1:
        return _serial_transform_write_without_accumulate(
            temp_store_path,
            positional_key,
            transformed_output_path,
            fixed_arena_radius=fixed_arena_radius,
        )

    parts_dir = transformed_output_path + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    chunks = _ordered_chunks(individuals, worker_count)
    tasks = []
    for idx, chunk in enumerate(chunks):
        tasks.append((
            temp_store_path,
            positional_key,
            list(chunk),
            cols_for_select,
            fixed_arena_radius,
            os.path.join(parts_dir, f"part_{idx:03d}.csv"),
        ))

    results = list(ordered_worker_results(
        _occupancy_transform_chunk_worker, tasks, num_workers=worker_count))

    part_paths = [r["chunk_output_path"] for r in results if int(r.get("n_rows", 0)) > 0]
    if not part_paths:
        shutil.rmtree(parts_dir, ignore_errors=True)
        return False

    _write_csv_parts_to_target(part_paths, transformed_output_path)

    qc_total = sum(int(r.get("qc_total", 0)) for r in results)
    qc_over_1p0 = sum(int(r.get("qc_over_1p0", 0)) for r in results)
    qc_over_1p2 = sum(int(r.get("qc_over_1p2", 0)) for r in results)
    if qc_total > 0:
        print(f"[Arena QC] r_norm_raw > 1.0: {qc_over_1p0 / qc_total:.3%}, > 1.2: {qc_over_1p2 / qc_total:.3%} (frames={qc_total})")
    print(f"Transformed data saved to {transformed_output_path}")
    shutil.rmtree(parts_dir, ignore_errors=True)
    return True


def _serial_transform_collect_without_accumulate(
    temp_store_path: str,
    positional_key: str,
    *,
    fixed_arena_radius: float | None,
) -> pd.DataFrame | None:
    """Transform animal positions in order and return the resulting rows without plot
    accumulators.
    """
    cols_to_load = [
        "group", "individual", "source_file", "frame_idx",
        "masked_pose", "masked_orient", "masked_pose_orient",
        "masked_pose_valid", "masked_orient_valid", "masked_pose_orient_valid",
        "centroid_x", "centroid_y", "angle", "speed_mm_s",
        "anglefeature_Wings_L_R", "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR", "anglefeature_Legs_BL_BR",
    ]

    frames: list[pd.DataFrame] = []
    qc_total = qc_over_1p0 = qc_over_1p2 = 0
    with pd.HDFStore(temp_store_path, mode="r") as store:
        individuals = read_individual_metadata(store, positional_key, columns=("individual",))["individual"].dropna().unique()
        individuals = sorted([str(x) for x in individuals])
        cols_for_select = _load_positional_cols_for_select(store, positional_key, cols_to_load)
        for individual_id in tqdm(individuals, desc="Transforming positional data"):
            individual_df = select_individual_rows(
                store, positional_key, individual_id, columns=cols_for_select)
            transformed_df, qc_n, qc_1, qc_12 = _transform_positional_individual_df(individual_df, fixed_arena_radius=fixed_arena_radius)
            qc_total += int(qc_n)
            qc_over_1p0 += int(qc_1)
            qc_over_1p2 += int(qc_12)
            if transformed_df is None or transformed_df.empty:
                continue
            frames.append(transformed_df)

    if not frames:
        return None
    if qc_total > 0:
        print(f"[Arena QC] r_norm_raw > 1.0: {qc_over_1p0 / qc_total:.3%}, > 1.2: {qc_over_1p2 / qc_total:.3%} (frames={qc_total})")
    return pd.concat(frames, ignore_index=True)


def _parallel_transform_collect_without_accumulate(
    temp_store_path: str,
    positional_key: str,
    *,
    fixed_arena_radius: float | None,
    num_workers: int,
) -> pd.DataFrame | None:
    """Transform contiguous animal chunks in workers and return their ordered positional
    rows.
    """
    cols_to_load = [
        "group", "individual", "source_file", "frame_idx",
        "masked_pose", "masked_orient", "masked_pose_orient",
        "masked_pose_valid", "masked_orient_valid", "masked_pose_orient_valid",
        "centroid_x", "centroid_y", "angle", "speed_mm_s",
        "anglefeature_Wings_L_R", "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR", "anglefeature_Legs_BL_BR",
    ]

    with pd.HDFStore(temp_store_path, mode="r") as store:
        individuals = read_individual_metadata(store, positional_key, columns=("individual",))["individual"].dropna().unique()
        individuals = sorted([str(x) for x in individuals])
        cols_for_select = _load_positional_cols_for_select(store, positional_key, cols_to_load)

    if not individuals:
        return None

    worker_count = max(1, min(int(num_workers), len(individuals)))
    if worker_count <= 1:
        return _serial_transform_collect_without_accumulate(
            temp_store_path,
            positional_key,
            fixed_arena_radius=fixed_arena_radius,
        )

    chunks = _ordered_chunks(individuals, worker_count)
    tasks = [
        (temp_store_path, positional_key, list(chunk), cols_for_select, fixed_arena_radius)
        for chunk in chunks
    ]

    results = list(ordered_worker_results(
        _occupancy_transform_chunk_collect_worker, tasks, num_workers=worker_count))

    frames = [r.get("dataframe") for r in results if r.get("dataframe") is not None and int(r.get("n_rows", 0)) > 0]
    if not frames:
        return None

    qc_total = sum(int(r.get("qc_total", 0)) for r in results)
    qc_over_1p0 = sum(int(r.get("qc_over_1p0", 0)) for r in results)
    qc_over_1p2 = sum(int(r.get("qc_over_1p2", 0)) for r in results)
    if qc_total > 0:
        print(f"[Arena QC] r_norm_raw > 1.0: {qc_over_1p0 / qc_total:.3%}, > 1.2: {qc_over_1p2 / qc_total:.3%} (frames={qc_total})")
    return pd.concat(frames, ignore_index=True)


# Public entrypoint used by the mmgen pipeline

def calculate_and_save_occupancy_analysis(
    temp_store_path,
    positional_key,
    output_dir,
    *,
    chamber_diameter_px: int | None = DEFAULT_CHAMBER_DIAMETER_PX,
    generate_quiver_plots: bool = DEFAULT_GENERATE_QUIVER_PLOTS,
    save_outputs: bool = True,
    transformed_output_path: str | None = None,
    num_workers: int = 1,
    return_transformed_df: bool = False,
):
    """Transform positions by animal and generate requested occupancy reports.

    When reports are disabled, return the transformed substrate in memory only if
    requested for matrix finalization.
    """

    print("\nStarting geometric and postural analysis...")

    if save_outputs and bool(return_transformed_df):
        raise ValueError("return_transformed_df=True requires save_outputs=False")

    transformed_path = transformed_output_path or os.path.join(output_dir, "temp_transformed_positional_data.csv")

    if save_outputs:
        print("\nCalibrating arena, transforming coordinates, writing transformed CSV, and accumulating occupancy stats (streaming from HDF5)...")
        acc = _stream_transform_write_and_accumulate(
            temp_store_path,
            positional_key,
            output_dir,
            chamber_diameter_px=chamber_diameter_px,
            generate_quiver_plots=generate_quiver_plots,
            transformed_output_path=transformed_path,
            num_workers=num_workers,
        )

        if acc is None:
            print("No positional data was transformed. Skipping all geometric/postural analysis.")
            return None

        print("\nGenerating and saving occupancy plots...")
        radial = _generate_radial_distribution_plots_from_acc(acc, output_dir)

        print("\nGenerating and saving 2D occupancy heatmaps...")
        heatmaps = _generate_2d_heatmaps_from_acc(acc, output_dir)

        if generate_quiver_plots:
            print("\nGenerating and saving orientation quiver plots...")
            quiver = _generate_orientation_quiver_plots_from_acc(acc, output_dir)
        else:
            quiver = []
        # Record only this run's applicable plots, including valid empty cases.
        # Stale files in the output directory never establish completion.
        artifacts = {Path(p).relative_to(output_dir).as_posix(): sha256_file(p)
                     for p in [transformed_path, *radial, *heatmaps, *quiver]}
        atomic_write_json({'artifacts': artifacts,
            'plot_counts': {'radial': len(radial), 'heatmaps': len(heatmaps), 'quiver': len(quiver)},
            'quiver_requested': bool(generate_quiver_plots)},
            Path(output_dir)/'MMGEN_OCCUPANCY_OUTPUTS.json')
    else:
        fixed_arena_radius = _resolve_fixed_arena_radius(chamber_diameter_px)
        print("\nCalibrating arena and transforming coordinates for matrix-first spatial features...")
        if bool(return_transformed_df):
            if int(num_workers) > 1:
                transformed_df = _parallel_transform_collect_without_accumulate(
                    temp_store_path,
                    positional_key,
                    fixed_arena_radius=fixed_arena_radius,
                    num_workers=int(num_workers),
                )
            else:
                transformed_df = _serial_transform_collect_without_accumulate(
                    temp_store_path,
                    positional_key,
                    fixed_arena_radius=fixed_arena_radius,
                )

            if transformed_df is None or transformed_df.empty:
                print("No positional data was transformed. Skipping all geometric/postural analysis.")
                return None

            print("\nGeometric and postural analysis complete.")
            return transformed_df

        if int(num_workers) > 1:
            transformed_ok = _parallel_transform_write_without_accumulate(
                temp_store_path,
                positional_key,
                transformed_path,
                fixed_arena_radius=fixed_arena_radius,
                num_workers=int(num_workers),
            )
        else:
            transformed_ok = _serial_transform_write_without_accumulate(
                temp_store_path,
                positional_key,
                transformed_path,
                fixed_arena_radius=fixed_arena_radius,
            )

        if not transformed_ok:
            print("No positional data was transformed. Skipping all geometric/postural analysis.")
            return None

    print("\nGeometric and postural analysis complete.")

    return transformed_path



from .feature_matrix import (
    _filter_features_by_bout_count,
    _infer_unit_for_feature_name,
    write_feature_units_manifest,
    create_feature_matrix,
    validate_feature_matrix,
    format_summary_df,
    save_to_excel_sheets,
)
