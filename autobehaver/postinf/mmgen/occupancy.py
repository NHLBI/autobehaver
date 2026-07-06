"""Behavior-specific occupancy / spatial analysis for mmgen.

This module contains the behavior-specific occupancy and spatial analysis
logic used during mmgen finalization.

Implementation note:
  - Stream per-individual positional transformations to disk (CSV) instead of
    concatenating all transformed frames into one in-memory DataFrame.
  - Compute occupancy heatmaps, orientation quivers, and radial KDE plots using
    bounded, streaming accumulators (including capped KDE sampling).

This eliminates the gradual memory growth previously caused by accumulating
"all_transformed_dfs" and then calling pd.concat(...).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from .naming import collapse_individual_id

DEFAULT_CHAMBER_DIAMETER_PX: int | None = 530
DEFAULT_GENERATE_QUIVER_PLOTS: bool = True


# ---------------------------------------------------------------------------
# Internal bounded sampling + accumulation helpers
# ---------------------------------------------------------------------------

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

    # ------------------------
    # Heatmap accumulation
    # ------------------------
    def add_heatmap(self, col: str, behavior: str, group: str, x: np.ndarray, y: np.ndarray) -> None:
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

    # ------------------------
    # KDE sampling accumulation
    # ------------------------
    def add_kde(self, col: str, behavior: str, group: str, r_norm: np.ndarray) -> None:
        key = (col, behavior, group)
        sampler = self._get_kde_sampler(key)
        sampler.update(r_norm)
        self.behaviors[col].add(str(behavior))

    # ------------------------
    # Quiver accumulation
    # ------------------------
    def add_quiver(self, col: str, behavior: str, group: str, x: np.ndarray, y: np.ndarray, angle_deg: np.ndarray) -> None:
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
        ang_rad = np.deg2rad(ang)
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

    # ------------------------
    # Accessors for plotting
    # ------------------------
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


# ---------------------------------------------------------------------------
# Streaming transform + accumulation
# ---------------------------------------------------------------------------

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
    if os.path.exists(transformed_path):
        try:
            os.remove(transformed_path)
        except Exception:
            pass

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

    try:
        with pd.HDFStore(temp_store_path, mode="r") as store:
            individuals = store.select_column(positional_key, "individual").unique()

            # Ensure deterministic iteration order (helps stable sampling)
            individuals = sorted([str(x) for x in individuals])

            # Attempt to figure out which columns exist for safer selection
            existing_cols = None
            try:
                # 0-row schema read is safe and cheap
                schema0 = store.select(positional_key, start=0, stop=0)
                existing_cols = set(schema0.columns)
            except Exception:
                existing_cols = None

            if existing_cols is not None:
                cols_for_select = [c for c in cols_to_load if c in existing_cols]
            else:
                cols_for_select = None

            for individual_id in tqdm(individuals, desc="Calibrating arenas"):
                # Load one individual's data
                try:
                    if cols_for_select:
                        individual_df = store.select(
                            positional_key,
                            where=f"individual == '{individual_id}'",
                            columns=cols_for_select,
                        )
                    else:
                        individual_df = store.select(positional_key, where=f"individual == '{individual_id}'")
                except Exception:
                    individual_df = store.select(positional_key, where=f"individual == '{individual_id}'")

                if individual_df is None or individual_df.empty:
                    continue

                # Drop rows with missing positional data for calibration
                clean_df = individual_df.dropna(subset=["centroid_x", "centroid_y"])
                if clean_df.empty:
                    continue

                # Arena center is fixed in a 600x600 frame
                center_x = 299.5
                center_y = 299.5

                centered_x = individual_df["centroid_x"] - center_x
                centered_y = individual_df["centroid_y"] - center_y

                r = np.sqrt(centered_x**2 + centered_y**2)

                if fixed_arena_radius is not None:
                    arena_radius = fixed_arena_radius
                else:
                    arena_radius = np.nanpercentile(r, 99)

                if (not np.isfinite(arena_radius)) or (arena_radius <= 0):
                    arena_radius = 1.0

                r_norm_raw = r / arena_radius

                rnr = r_norm_raw.to_numpy()
                finite_mask = np.isfinite(rnr)
                qc_total += int(np.sum(finite_mask))
                qc_over_1p0 += int(np.sum(rnr[finite_mask] > 1.0))
                qc_over_1p2 += int(np.sum(rnr[finite_mask] > 1.2))

                # Positional angle
                theta_pos_rad = np.arctan2(centered_y, centered_x)
                theta_pos_deg = np.degrees(theta_pos_rad)

                # Compute columns in-place (no large copy)
                individual_df["r_norm"] = r_norm_raw
                individual_df["positional_angle"] = theta_pos_deg

                fly_angle_math_deg = individual_df["angle"]
                individual_df["fly_angle_math"] = fly_angle_math_deg

                relative_angle = (fly_angle_math_deg - theta_pos_deg + 180) % 360 - 180
                individual_df["relative_angle"] = relative_angle

                # Guard extreme radial values for plots
                individual_df["r_norm"] = individual_df["r_norm"].clip(lower=0, upper=1.5)

                # Collapse individual IDs to animal IDs
                individual_df["individual"] = individual_df["individual"].map(collapse_individual_id)

                # Write transformed chunk to disk
                individual_df.to_csv(
                    transformed_path,
                    index=False,
                    mode="a" if header_written else "w",
                    header=not header_written,
                )
                header_written = True

                # -----------------------------
                # Update bounded accumulators
                # -----------------------------
                # We update from the transformed DF (contains r_norm, etc.)
                for col in acc.behavior_cols:
                    if col not in individual_df.columns:
                        continue

                    # Drop rows where behavior label or group is missing
                    df_bg = individual_df.dropna(subset=[col, "group"]) 
                    if df_bg.empty:
                        continue

                    # Group by behavior+group for this individual
                    for (behavior, group), gdf in df_bg.groupby([col, "group"], observed=False):
                        # Coordinates for heatmap and quiver
                        coords = gdf[["centroid_x", "centroid_y"]].dropna()
                        if not coords.empty:
                            x = coords["centroid_x"].to_numpy(dtype=float)
                            y = coords["centroid_y"].to_numpy(dtype=float)
                            acc.add_heatmap(col, str(behavior), str(group), x, y)

                        # r_norm KDE sample
                        if "r_norm" in gdf.columns:
                            rn = gdf["r_norm"].to_numpy(dtype=float)
                            acc.add_kde(col, str(behavior), str(group), rn)

                        # Quiver update (optional)
                        if generate_quiver_plots and ("angle" in gdf.columns):
                            # Need finite x/y/angle
                            x_all = gdf["centroid_x"].to_numpy(dtype=float)
                            y_all = gdf["centroid_y"].to_numpy(dtype=float)
                            a_all = gdf["angle"].to_numpy(dtype=float)
                            m = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(a_all)
                            if np.any(m):
                                acc.add_quiver(col, str(behavior), str(group), x_all[m], y_all[m], a_all[m])

    except (FileNotFoundError, KeyError, pd.errors.HDFDatabaseError) as e:
        print(f"Error reading positional data from HDF5 store: {e}")
        return None

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


# ---------------------------------------------------------------------------
# Plot generation from bounded accumulators
# ---------------------------------------------------------------------------

def _generate_radial_distribution_plots_from_acc(acc: _OccupancyAccumulator, output_dir: str) -> None:
    """Generate radial KDE plots from bounded samples."""
    base_plot_folder = os.path.join(output_dir, "9_Occupancy_Plots")

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
            except Exception as e:
                print(f"Could not save plot {plot_path}: {e}")
            plt.close()


def _generate_2d_heatmaps_from_acc(acc: _OccupancyAccumulator, output_dir: str) -> None:
    """Generate 2D binned occupancy heatmaps from accumulated histograms."""
    base_plot_folder = os.path.join(output_dir, "10_Occupancy_Heatmaps")
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

                ax.invert_yaxis()
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"Occupancy Histogram: {group}\n{behavior}", fontsize=14)
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")

                safe_bh_name = re.sub(r"[\\/*?:\"<>| +]", "_", behavior)
                cartesian_path = os.path.join(plot_output_folder, f"{safe_bh_name}_{group}_heatmap_histogram.png")
                plt.savefig(cartesian_path, dpi=150, bbox_inches="tight")
                plt.close()


def _generate_orientation_quiver_plots_from_acc(
    acc: _OccupancyAccumulator,
    output_dir: str,
    *,
    min_samples_per_bin: int = 5,
) -> None:
    """Generate orientation quiver plots from accumulated vector sums."""

    base_plot_folder = os.path.join(output_dir, "11_Orientation_Quiver_Plots")

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
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()


# ---------------------------------------------------------------------------
# Matrix-first fast-path helpers (no plot accumulators)
# ---------------------------------------------------------------------------

def _escape_hdf_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _resolve_fixed_arena_radius(chamber_diameter_px: int | None) -> float | None:
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
    try:
        schema0 = store.select(positional_key, start=0, stop=0)
        existing_cols = set(schema0.columns)
        return [c for c in cols_to_load if c in existing_cols]
    except Exception:
        return None


def _transform_positional_individual_df(individual_df: pd.DataFrame, *, fixed_arena_radius: float | None) -> tuple[pd.DataFrame | None, int, int, int]:
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
    fly_angle_math_deg = individual_df["angle"]
    relative_angle = (fly_angle_math_deg - theta_pos_deg + 180) % 360 - 180

    out_df = individual_df.copy()
    out_df["r_norm"] = pd.Series(r_norm_raw, index=out_df.index).clip(lower=0, upper=1.5)
    out_df["positional_angle"] = theta_pos_deg
    out_df["fly_angle_math"] = fly_angle_math_deg
    out_df["relative_angle"] = relative_angle
    out_df["individual"] = out_df["individual"].map(collapse_individual_id)
    return out_df, qc_total, qc_over_1p0, qc_over_1p2


def _ordered_chunks(items: list[str], n_chunks: int) -> list[list[str]]:
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
                continue
            with open(part_path, "r", encoding="utf-8", newline="") as in_f:
                if not header_written:
                    shutil.copyfileobj(in_f, out_f)
                    header_written = True
                else:
                    next(in_f, None)
                    shutil.copyfileobj(in_f, out_f)


def _occupancy_transform_chunk_worker(args: tuple[str, str, list[str], list[str] | None, float | None, str]) -> dict:
    temp_store_path, positional_key, individual_ids, cols_for_select, fixed_arena_radius, chunk_output_path = args
    header_written = False
    n_rows = 0
    qc_total = 0
    qc_over_1p0 = 0
    qc_over_1p2 = 0

    with pd.HDFStore(temp_store_path, mode="r") as store:
        for individual_id in individual_ids:
            ind_safe = _escape_hdf_string(individual_id)
            try:
                if cols_for_select:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'", columns=cols_for_select)
                else:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
            except Exception:
                individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
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
    temp_store_path, positional_key, individual_ids, cols_for_select, fixed_arena_radius = args
    frames: list[pd.DataFrame] = []
    qc_total = 0
    qc_over_1p0 = 0
    qc_over_1p2 = 0

    with pd.HDFStore(temp_store_path, mode="r") as store:
        for individual_id in individual_ids:
            ind_safe = _escape_hdf_string(individual_id)
            try:
                if cols_for_select:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'", columns=cols_for_select)
                else:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
            except Exception:
                individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
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
        individuals = store.select_column(positional_key, "individual").unique()
        individuals = sorted([str(x) for x in individuals])
        cols_for_select = _load_positional_cols_for_select(store, positional_key, cols_to_load)
        for individual_id in tqdm(individuals, desc="Transforming positional data"):
            ind_safe = _escape_hdf_string(individual_id)
            try:
                if cols_for_select:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'", columns=cols_for_select)
                else:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
            except Exception:
                individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
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
    cols_to_load = [
        "group", "individual", "source_file", "frame_idx",
        "masked_pose", "masked_orient", "masked_pose_orient",
        "masked_pose_valid", "masked_orient_valid", "masked_pose_orient_valid",
        "centroid_x", "centroid_y", "angle", "speed_mm_s",
        "anglefeature_Wings_L_R", "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR", "anglefeature_Legs_BL_BR",
    ]

    with pd.HDFStore(temp_store_path, mode="r") as store:
        individuals = store.select_column(positional_key, "individual").unique()
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

    with Pool(processes=worker_count) as pool:
        results = pool.map(_occupancy_transform_chunk_worker, tasks, chunksize=1)

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
        individuals = store.select_column(positional_key, "individual").unique()
        individuals = sorted([str(x) for x in individuals])
        cols_for_select = _load_positional_cols_for_select(store, positional_key, cols_to_load)
        for individual_id in tqdm(individuals, desc="Transforming positional data"):
            ind_safe = _escape_hdf_string(individual_id)
            try:
                if cols_for_select:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'", columns=cols_for_select)
                else:
                    individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
            except Exception:
                individual_df = store.select(positional_key, where=f"individual == '{ind_safe}'")
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
    cols_to_load = [
        "group", "individual", "source_file", "frame_idx",
        "masked_pose", "masked_orient", "masked_pose_orient",
        "masked_pose_valid", "masked_orient_valid", "masked_pose_orient_valid",
        "centroid_x", "centroid_y", "angle", "speed_mm_s",
        "anglefeature_Wings_L_R", "anglefeature_Legs_FL_FR",
        "anglefeature_Legs_ML_MR", "anglefeature_Legs_BL_BR",
    ]

    with pd.HDFStore(temp_store_path, mode="r") as store:
        individuals = store.select_column(positional_key, "individual").unique()
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

    with Pool(processes=worker_count) as pool:
        results = pool.map(_occupancy_transform_chunk_collect_worker, tasks, chunksize=1)

    frames = [r.get("dataframe") for r in results if r.get("dataframe") is not None and int(r.get("n_rows", 0)) > 0]
    if not frames:
        return None

    qc_total = sum(int(r.get("qc_total", 0)) for r in results)
    qc_over_1p0 = sum(int(r.get("qc_over_1p0", 0)) for r in results)
    qc_over_1p2 = sum(int(r.get("qc_over_1p2", 0)) for r in results)
    if qc_total > 0:
        print(f"[Arena QC] r_norm_raw > 1.0: {qc_over_1p0 / qc_total:.3%}, > 1.2: {qc_over_1p2 / qc_total:.3%} (frames={qc_total})")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Public entrypoint used by the mmgen pipeline
# ---------------------------------------------------------------------------

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
    """Performs behavior-specific occupancy analysis by reading from the HDF5 store.

    When ``save_outputs`` is True, this writes the
    transformed positional CSV plus the occupancy/geometry report outputs.

    When ``save_outputs`` is False and ``return_transformed_df`` is True, the
    transformed positional substrate is returned directly in memory so matrix-first
    shard finalization can skip the CSV write/read round-trip.
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
        )

        if acc is None:
            print("No positional data was transformed. Skipping all geometric/postural analysis.")
            return None

        print("\nGenerating and saving occupancy plots...")
        _generate_radial_distribution_plots_from_acc(acc, output_dir)

        print("\nGenerating and saving 2D occupancy heatmaps...")
        _generate_2d_heatmaps_from_acc(acc, output_dir)

        if generate_quiver_plots:
            print("\nGenerating and saving orientation quiver plots...")
            _generate_orientation_quiver_plots_from_acc(acc, output_dir)
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

# ============================================================================
# MASTER FEATURE MATRIX GENERATION
# ============================================================================


# ---------------------------------------------------------------------------
# Feature matrix assembly helpers
# ---------------------------------------------------------------------------
from .feature_matrix import (
    _filter_features_by_bout_count,
    _infer_unit_for_feature_name,
    write_feature_units_manifest,
    create_feature_matrix,
    validate_feature_matrix,
    format_summary_df,
    save_to_excel_sheets,
)
