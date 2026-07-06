"""Transition analysis utilities for mmgen.

This module aggregates per-file transition counts into the user-facing pose and
pose+orientation transition outputs.
"""

from __future__ import annotations

import os
import itertools
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import beta

from .feature_matrix import save_to_excel_sheets
from .naming import collapse_individual_id

DEFAULT_RUN_TRANSITION_STATISTICS: bool = True

def _is_impossible_pose_orient_transition(from_behavior, to_behavior):
    """
    Checks if a pose+orient transition is physically impossible.
    
    Impossible transitions:
    - Wall-Up ↔ Wall-Down (both directions)
    - Floor → Ceiling (instant flip - but Ceiling → Floor is allowed as it could be a quick fall)
    
    Args:
        from_behavior: String like "Walking + Floor" or "Standing + Wall-Up"
        to_behavior: String like "Standing + Ceiling" or "Walking + Wall-Down"
    
    Returns:
        bool: True if transition is impossible, False otherwise
    """
    # Only check pose+orient transitions (those containing " + ")
    if " + " not in str(from_behavior) or " + " not in str(to_behavior):
        return False  # Not a pose+orient transition, skip check
    
    # Extract orientations from behavior labels
    # Format: "PoseName + OrientName"
    try:
        from_orient = str(from_behavior).split(" + ")[1].strip()
        to_orient = str(to_behavior).split(" + ")[1].strip()
    except IndexError:
        # Malformed behavior label, skip check
        return False
    
    # Check for impossible orientation transitions
    # Note: Ceiling → Floor is allowed (could be a quick fall/flip)
    impossible_pairs = [
        ("Wall-Up", "Wall-Down"),
        ("Wall-Down", "Wall-Up"),
        ("Floor", "Ceiling")
    ]
    
    return (from_orient, to_orient) in impossible_pairs


def _run_transition_analysis_for_category(
    df_category,
    behavior_order,
    file_prefix,
    category_name,
    output_dir,
    *,
    save_outputs: bool = True,
    run_transition_statistics: bool = DEFAULT_RUN_TRANSITION_STATISTICS,
):
    """
    Internal helper that calculates transition matrices and returns them in a
    dictionary for saving, while still saving individual heatmaps and raw data.
    """
    # --- Individual-Level Analysis ---
    individual_data_folder = None
    if save_outputs:
        if category_name == "Pose":
            individual_data_folder = os.path.join(output_dir, '6_Pose-Only_Individual_Transitions')
        else: # Pose+Orient
            individual_data_folder = os.path.join(output_dir, '7_Pose+Orient_Individual_Transitions')
        os.makedirs(individual_data_folder, exist_ok=True)

    # Filter dataframe to only include transitions where both behaviors are in behavior_order
    # This is a defensive check to prevent invalid behaviors from entering the analysis
    initial_count = len(df_category)
    df_category = df_category[
        (df_category['from_behavior'].isin(behavior_order)) &
        (df_category['to_behavior'].isin(behavior_order))
    ].copy()
    filtered_count = initial_count - len(df_category)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} transitions with invalid behaviors (not in behavior_order)")

    # Get all behaviors present in the filtered aggregated data
    present_from = df_category['from_behavior'].unique()
    present_to = df_category['to_behavior'].unique()
    present_behaviors = sorted(list(set(present_from) | set(present_to)))

    # Only use behaviors from behavior_order that are actually present in the data
    # This ensures we don't include behaviors that aren't in the valid list
    all_behaviors = [b for b in behavior_order if b in present_behaviors]
    
    # Add any unexpected behaviors found in data (shouldn't happen after filtering, but defensive)
    unexpected = [b for b in present_behaviors if b not in behavior_order]
    if unexpected:
        print(f"  WARNING: Found {len(unexpected)} unexpected behaviors in data after filtering: {unexpected[:5]}")
        # Don't add them to all_behaviors - they should have been filtered out

    # Aggregate counts per individual (this collapses the per-file/movie counts)
    # First collapse individual IDs to animal IDs for consistency with other outputs
    df_category = df_category.copy()
    df_category['animal_id'] = df_category['individual'].map(collapse_individual_id)
    ind_counts_df = df_category.groupby(
        ['animal_id', 'group', 'from_behavior', 'to_behavior'],
        observed=True
    )['count'].sum().reset_index().rename(columns={'animal_id': 'individual'})

    individuals = ind_counts_df['individual'].unique()
    individual_matrices = {}
    print(f"Calculating transition matrices for {len(individuals)} individuals from aggregated counts...")

    # --- Loop through individuals to calculate probability matrices ---
    for individual in individuals:
        ind_df = ind_counts_df[ind_counts_df['individual'] == individual]
        if ind_df.empty:
            continue

        group = ind_df['group'].iloc[0]

        # Pivot to get the count matrix for this individual
        transition_counts = ind_df.pivot_table(
            index='from_behavior',
            columns='to_behavior',
            values='count',
            fill_value=0
        )

        # Reindex to ensure all matrices have the same shape and order
        transition_counts = transition_counts.reindex(index=all_behaviors, columns=all_behaviors, fill_value=0)
        # Set diagonal (self-transitions) to 0 - they should already be filtered out, but ensure it here too
        np.fill_diagonal(transition_counts.values, 0)
        # Calculate row sums (total counts outgoing from each state)
        row_sums = transition_counts.sum(axis=1)

        # Calculate probabilities: prob = count / row_sum
        # Handle division by zero for rows with sum = 0 (set resulting probabilities to NaN)
        with np.errstate(divide='ignore', invalid='ignore'):
            transition_probs = transition_counts.div(row_sums, axis=0)
        zero_rows_mask = row_sums == 0
        transition_probs.loc[zero_rows_mask, :] = np.nan # Apply NaN where denominator was zero

        # Calculate sums of the calculated probabilities along each row
        row_sums_check_probs = transition_probs.sum(axis=1)

        # Keep probability sums only for states with outgoing transitions
        valid_prob_sums = row_sums_check_probs[row_sums > 0]

        # Check if the filtered sums (where transitions were possible) are close to 1.0
        if not valid_prob_sums.empty:
             # Use a slightly looser tolerance (atol) for floating point sums
             assert np.allclose(valid_prob_sums, 1.0, atol=1e-6), \
                    f"Individual {individual} transition probability rows (where outgoing count > 0) do not sum to 1. Problem rows:\n{row_sums_check_probs[row_sums > 0][~np.isclose(valid_prob_sums, 1.0, atol=1e-6)]}"

        # Save individual matrix
        if save_outputs and individual_data_folder is not None:
            transition_probs.to_csv(os.path.join(individual_data_folder, f"{individual}_transition_matrix.csv"))

        # Store for group aggregation
        if group not in individual_matrices:
            individual_matrices[group] = []
        individual_matrices[group].append((individual, transition_probs))

    # Print confirmation message AFTER the loop finishes successfully
    print(" - Assert Check Passed: Individual transition probability rows sum correctly to 1.0 (where calculable).")


    # --- Group-Level Aggregation and Visualization ---
    print("\nAggregating group-level statistics and generating visualizations...")
    sheets_to_save = {}
    for group, matrices_data in individual_matrices.items():
        matrices = [m[1] for m in matrices_data]
        if not matrices: continue

        # Stack as float to preserve NaNs; aggregate with NaN-aware stats.
        # Ensure matrices are aligned before stacking
        aligned_matrices = [m.reindex(index=all_behaviors, columns=all_behaviors) for m in matrices]
        group_stack = np.stack([m.to_numpy(dtype=float) for m in aligned_matrices])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice\\.", category=RuntimeWarning)
            mean_matrix = np.nanmean(group_stack, axis=0)
            std_matrix  = np.nanstd(group_stack,  axis=0)

        mean_df = pd.DataFrame(mean_matrix, index=all_behaviors, columns=all_behaviors)
        std_df  = pd.DataFrame(std_matrix,  index=all_behaviors, columns=all_behaviors)

        # Add DataFrames to the dictionary for later saving
        sheets_to_save[f'{group}_Means'] = mean_df
        sheets_to_save[f'{group}_Stdevs'] = std_df

        # Build readable annotations; blank where mean is NaN.
        def _fmt(mu, sd, wide):
            if np.isnan(mu): return ""
            # Reduce decimal places for wide format, handle zero values cleanly
            mu_fmt = f"{mu:.2f}" if not np.isclose(mu, 0) else "0"
            sd_fmt = f"{sd:.2f}" if not np.isclose(sd, 0) else "0"

            if wide:
                 return mu_fmt if float(mu_fmt) != 0 else "" # Only show mean if wide & non-zero
            else:
                 # Show mean +/- std if std is not NaN and not zero, otherwise just mean
                 return f"{mu_fmt}\n±{sd_fmt}" if (not np.isnan(sd) and float(sd_fmt) != 0) else mu_fmt


        num_behaviors = len(all_behaviors)
        figsize, annot_kws = ((30, 24), {"size": 8}) if num_behaviors > 20 else ((16, 12), {"size": 10})
        annotations = np.array(
            [_fmt(mu, sd, num_behaviors > 20) for mu, sd in zip(mean_matrix.ravel(), std_matrix.ravel())]
        ).reshape(mean_matrix.shape)
        if save_outputs:
            plt.figure(figsize=figsize)
            # Use vmin=0, vmax=1 for probability heatmap
            sns.heatmap(mean_df, annot=annotations, fmt="", cmap="viridis", vmin=0, vmax=1, linewidths=.5, annot_kws=annot_kws, cbar_kws={'label': 'Transition Probability'})
            plt.title(f'Transition Matrix ({category_name}) for Group: {group}', fontsize=16)
            plt.ylabel('From Behavior', fontsize=12)
            plt.xlabel('To Behavior', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{file_prefix}_Transition_Heatmap_{category_name}_{group}.png'), dpi=300)
            plt.close()

    # --- Create Master Tidy DataFrame ---
    all_transitions_tidy = []
    for group, indiv_data in individual_matrices.items():
        for individual_id, matrix_df in indiv_data:
            tidy_df = matrix_df.reset_index().melt(id_vars='from_behavior', var_name='to_behavior', value_name='probability')
            # Filter out self-transitions (where from_behavior == to_behavior)
            tidy_df = tidy_df[tidy_df['from_behavior'] != tidy_df['to_behavior']]
            # individual_id is already collapsed from the groupby above, so use it directly
            tidy_df['individual'] = individual_id
            tidy_df['group'] = group
            all_transitions_tidy.append(tidy_df)

    master_tidy_df = pd.DataFrame(columns=['group', 'individual', 'from_behavior', 'to_behavior', 'probability'])
    if all_transitions_tidy:
        master_tidy_df = pd.concat(all_transitions_tidy, ignore_index=True)
        master_tidy_df = master_tidy_df[['group', 'individual', 'from_behavior', 'to_behavior', 'probability']]

        # Path for the CSV in the main output folder
        if save_outputs:
            tidy_csv_path = os.path.join(output_dir, f'{file_prefix}_{category_name}_All_Transitions_Tidy.csv')

            # Save the huge tidy file as a separate CSV
            master_tidy_df.to_csv(tidy_csv_path, index=False)
            print(f"Saved large tidy transition data to {tidy_csv_path}")

    print(f"Transition analysis for category '{category_name}' complete.")

    # --- Run Comparative Statistical Analysis ---
    if save_outputs and run_transition_statistics and len(individual_matrices) > 1:
        print(f"\n--- Running Comparative Analysis for {category_name} ---")

        # 1. Create a dedicated output folder
        comparison_output_folder = os.path.join(output_dir, "8_Transition_Comparisons", category_name)
        os.makedirs(comparison_output_folder, exist_ok=True)

        # 2. Consolidate mean matrices
        mean_matrices = {}
        for group, matrices_data in individual_matrices.items():
            if matrices_data:
                matrices = [m[1] for m in matrices_data] # Extract DataFrames
                # Ensure matrices are aligned before stacking
                aligned_matrices = [m.reindex(index=all_behaviors, columns=all_behaviors) for m in matrices]
                group_stack = np.stack([m.to_numpy(dtype=float) for m in aligned_matrices])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                    mean_df = pd.DataFrame(np.nanmean(group_stack, axis=0), index=all_behaviors, columns=all_behaviors)
                mean_matrices[group] = mean_df

        # 3. Run the analysis
        if len(mean_matrices) > 1:
            # Assume _run_comparative_analysis is defined elsewhere
             try:
                 _run_comparative_analysis(
                     individual_matrices=individual_matrices,
                     mean_matrices=mean_matrices,
                     behavior_order=all_behaviors,
                     output_folder=comparison_output_folder
                 )
             except Exception as e:
                  print(f"ERROR during comparative transition analysis for {category_name}: {e}")
                  # Optionally re-raise e if you want the script to halt on error

    return {'sheets_to_save': sheets_to_save, 'tidy_df': master_tidy_df}


def calculate_and_save_transition_analysis(aggregated_transitions_df, output_dir, *, run_transition_statistics: bool = DEFAULT_RUN_TRANSITION_STATISTICS, save_outputs: bool = True):
    """
    Controller function that runs transition analysis for pose and pose+orient
    categories, saving consolidated results to separate Excel files.
    
    Reads from the pre-aggregated transition table for memory safety.
    """
    print("\nStarting sequential transition analysis for both categories...")

    # --- 1. Define Behavior Orders ---
    pose_order = ['Walking', 'Standing', 'Scramble', 'Groom-FrontLegs', 'Groom-Head', 'Groom-SideLeg', 'Groom-Wings', 'Groom-Abdomen', 'Groom-BackLegs', 'Jump']
    orient_order = ['Floor', 'Ceiling', 'Wall-Side', 'Wall-Up', 'Wall-Down']
    # Build the combined order list, then add the special, non-oriented poses
    pose_order_for_combo = [p for p in pose_order if p not in ['Jump', 'Scramble']]
    combined_order = [f"{p} + {o}" for p in pose_order_for_combo for o in orient_order] + ['Jump', 'Scramble']

    transition_results = {
        'pose_tidy_df': pd.DataFrame(columns=['group', 'individual', 'from_behavior', 'to_behavior', 'probability']),
        'pose_orient_tidy_df': pd.DataFrame(columns=['group', 'individual', 'from_behavior', 'to_behavior', 'probability']),
    }

    # --- 2. Run Analysis for "Pose"-only Category ---
    print("\n--- Analyzing POSE-ONLY transitions ---")
    
    # Filter the aggregated DataFrame instead of the raw one
    pose_df = aggregated_transitions_df[aggregated_transitions_df['category'] == 'Pose'].copy()
    
    if not pose_df.empty:
        pose_result = _run_transition_analysis_for_category(
            df_category=pose_df,
            behavior_order=pose_order,
            file_prefix="6",
            category_name="Pose",
            output_dir=output_dir,
            save_outputs=save_outputs,
            run_transition_statistics=run_transition_statistics,
        )
        transition_results['pose_tidy_df'] = pose_result['tidy_df']
        if save_outputs:
            excel_path_pose = os.path.join(output_dir, '6_Pose-Only_Transitions.xlsx')
            save_to_excel_sheets(pose_result['sheets_to_save'], excel_path_pose)
    else:
        print("No Pose-only bouts found. Skipping Pose transition analysis.")

    # --- 3. Run Analysis for "Pose+Orient" Category (with Jump/Scramble) ---
    print("\n--- Analyzing POSE+ORIENT transitions (including Jump/Scramble) ---")
    
    # Filter the aggregated DataFrame for 'Pose+Orient'
    po_df = aggregated_transitions_df[aggregated_transitions_df['category'] == 'Pose+Orient'].copy()
    
    # Collect Jump/Scramble transitions from 'Pose' category, but only where
    # the other behavior is a valid Pose+Orient behavior (contains " + ") or is Jump/Scramble
    # This ensures we don't include invalid transitions like "Walking" → "Jump" (both pose-only)
    jump_scramble_transitions = pose_df[
        ((pose_df['from_behavior'].isin(['Jump', 'Scramble'])) & 
         ((pose_df['to_behavior'].isin(['Jump', 'Scramble'])) | 
          (pose_df['to_behavior'].str.contains(' + ', na=False)))) |
        ((pose_df['to_behavior'].isin(['Jump', 'Scramble'])) & 
         ((pose_df['from_behavior'].isin(['Jump', 'Scramble'])) | 
          (pose_df['from_behavior'].str.contains(' + ', na=False))))
    ].copy()
    
    # Combine Pose+Orient transitions with Jump/Scramble transitions
    po_plus_specials_df = pd.concat([po_df, jump_scramble_transitions]).drop_duplicates()
    
    # Filter to only include transitions where both behaviors are valid for Pose+Orient
    # This ensures no pose-only behaviors (except Jump/Scramble) are included
    initial_count = len(po_plus_specials_df)
    po_plus_specials_df = po_plus_specials_df[
        (po_plus_specials_df['from_behavior'].isin(combined_order)) &
        (po_plus_specials_df['to_behavior'].isin(combined_order))
    ].copy()
    filtered_count = initial_count - len(po_plus_specials_df)
    
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} invalid transitions (pose-only behaviors not in Pose+Orient list)")
    
    print(f"  Pose+Orient transitions: {len(po_df)}")
    print(f"  Jump/Scramble transitions from Pose category: {len(jump_scramble_transitions)}")
    print(f"  Total valid transitions after filtering: {len(po_plus_specials_df)}")
    print(f"  Unique behaviors in final dataframe: {len(set(po_plus_specials_df['from_behavior'].unique()) | set(po_plus_specials_df['to_behavior'].unique()))}")

    if not po_plus_specials_df.empty:
        po_result = _run_transition_analysis_for_category(
            df_category=po_plus_specials_df,
            behavior_order=combined_order,
            file_prefix="7",
            category_name="Pose+Orient",
            output_dir=output_dir,
            save_outputs=save_outputs,
            run_transition_statistics=run_transition_statistics,
        )
        transition_results['pose_orient_tidy_df'] = po_result['tidy_df']
        if save_outputs:
            excel_path_po = os.path.join(output_dir, '7_Pose+Orient_Transitions.xlsx')
            save_to_excel_sheets(po_result['sheets_to_save'], excel_path_po)

    else:
        print("No Pose+Orient or special bouts found. Skipping Pose+Orient transition analysis.")

    return transition_results


# 8 Comparative Analysis (between-group comparisons / heatmaps)


def _run_comparative_analysis(individual_matrices, mean_matrices, behavior_order, output_folder):
    """
    Orchestrates the statistical comparison and visualization of transition matrices between groups.
    """
    all_pairs_stats = []
    
    # --- Part 1: Calculate p-values for all pairs ---
    group_names = sorted(list(individual_matrices.keys()))
    for group1, group2 in itertools.combinations(group_names, 2):
        print(f"  Comparing {group1} vs. {group2}...")
        
        indiv_mats1 = [m[1] for m in individual_matrices[group1]]
        indiv_mats2 = [m[1] for m in individual_matrices[group2]]
        
        if not indiv_mats1 or not indiv_mats2:
            continue
            
        p_value_df = _calculate_p_values_for_comparison(indiv_mats1, indiv_mats2, behavior_order)
        diff_df = mean_matrices[group2] - mean_matrices[group1]
        
        # Convert to tidy format
        p_value_df.columns.name = 'to_behavior'
        diff_df.columns.name = 'to_behavior'

        # --- Name the index before resetting it ---
        p_value_df.index.name = 'from_behavior' 
        diff_df.index.name = 'from_behavior'

        tidy_p = p_value_df.reset_index().melt(id_vars='from_behavior', value_name='p_value')
        tidy_diff = diff_df.reset_index().melt(id_vars='from_behavior', value_name='difference_in_mean')
        
        stats_df = pd.merge(tidy_diff, tidy_p, on=['from_behavior', 'to_behavior'])
        stats_df['group1'] = group1
        stats_df['group2'] = group2
        all_pairs_stats.append(stats_df)

    if not all_pairs_stats:
        print("Not enough group pairs to run comparative analysis.")
        return

    # --- Part 2: Multiple comparisons correction ---
    master_stats_df = pd.concat(all_pairs_stats, ignore_index=True)
    result = fdrcorrection(master_stats_df['p_value'], alpha=0.05, method='indep')
    if len(result) == 4:
        reject, pvals_corrected, _, _ = result
    else:
        reject, pvals_corrected = result
    master_stats_df['p_value_corrected'] = pvals_corrected
    master_stats_df.to_csv(os.path.join(output_folder, 'all_transition_stats.csv'), index=False)
    print(f"\nSaved comprehensive statistical results to {output_folder}")

    # --- Part 3: Generate plots with a unified scale ---
    vmax = master_stats_df['difference_in_mean'].abs().max()
    
    for group1, group2 in itertools.combinations(group_names, 2):
        pair_df = master_stats_df[(master_stats_df['group1'] == group1) & (master_stats_df['group2'] == group2)]
        if pair_df.empty:
            continue

        # Build pivoted matrices
        diff_matrix = pair_df.pivot(index='from_behavior', columns='to_behavior', values='difference_in_mean')
        pval_matrix = pair_df.pivot(index='from_behavior', columns='to_behavior', values='p_value_corrected')

        # Enforce the canonical behavior order for both rows and columns
        diff_matrix = diff_matrix.reindex(index=behavior_order, columns=behavior_order)
        pval_matrix = pval_matrix.reindex(index=behavior_order, columns=behavior_order)

        _generate_comparison_heatmap(
            diff_matrix, pval_matrix, vmax, group1, group2,
            os.path.join(output_folder, f'diff_{group1}_vs_{group2}.png')
        )


def _calculate_p_values_for_comparison(indiv_mats1, indiv_mats2, behavior_order,
                                       num_permutations=10000, batch_size=2048,
                                       seed=123, early_stop=True, eps=0.002, min_batches=3):
    """
    Row-aware Monte Carlo permutation test on macro-averaged transition probabilities.
    - Treats each animal equally.
    - For a given 'from' row r, only animals with ≥1 r→* transition are included.
    - Preserves observed row-level group sizes during permutation.
    - Two-sided test on difference of means (group2 - group1).
    - Plus-one correction; optional early stopping via Clopper–Pearson CI (per row).
    """
    rng = np.random.default_rng(seed)
    B = len(behavior_order)

    # Stack groups as float arrays with NaNs preserved
    A = np.stack([m.reindex(index=behavior_order, columns=behavior_order).to_numpy(dtype=float) for m in indiv_mats1], axis=0)
    C = np.stack([m.reindex(index=behavior_order, columns=behavior_order).to_numpy(dtype=float) for m in indiv_mats2], axis=0)
    pooled = np.concatenate([A, C], axis=0)  # (N_total, B, B)
    labels = np.array([0] * len(A) + [1] * len(C))  # 0=group1, 1=group2
    N_total = pooled.shape[0]

    # Validity mask per row: True if this animal contributed any count to that row
    valid_row = ~np.isnan(pooled).all(axis=2)  # (N_total, B)

    # Observed difference (group2 - group1), row-aware
    obs_diff = np.full((B, B), np.nan, dtype=float)
    for r in range(B):
        idx = np.where(valid_row[:, r])[0]
        if idx.size == 0:
            continue
        grp = labels[idx]
        n1 = np.sum(grp == 0)
        n2 = np.sum(grp == 1)
        if n1 == 0 or n2 == 0:
            continue
        row_data = pooled[idx, r, :]  # (k, B), no NaNs expected within valid rows
        mu1 = row_data[grp == 0, :].mean(axis=0)
        mu2 = row_data[grp == 1, :].mean(axis=0)
        obs_diff[r, :] = mu2 - mu1

    exceed = np.zeros((B, B), dtype=np.int64)
    trials = np.zeros(B, dtype=np.int64)
    batches_run = np.zeros(B, dtype=np.int64)

    # Permute per row, preserving observed per-row group sizes
    for r in range(B):
        if np.isnan(obs_diff[r, :]).all():
            continue

        idx = np.where(valid_row[:, r])[0]
        grp_obs = labels[idx]
        n1 = int(np.sum(grp_obs == 0))
        n2 = int(np.sum(grp_obs == 1))
        k  = idx.size

        if n1 == 0 or n2 == 0:
            continue

        row_data = pooled[idx, r, :]  # (k, B)
        # No NaNs expected across columns in valid rows; safe to use as-is.

        while trials[r] < num_permutations:
            K = int(min(batch_size, num_permutations - trials[r]))
            # For each perm, randomly choose n2 of the k animals to assign to group2
            permuted = np.empty((K, k), dtype=np.int32)
            for t in range(K):
                permuted[t] = rng.permutation(k)
            mask2 = np.zeros((K, k), dtype=bool)
            # first n2 indices in each permuted row → group2
            rows = np.arange(K)[:, None]
            mask2[rows, permuted[:, :n2]] = True
            mask1 = ~mask2

            # Compute means for each perm (vectorized over columns)
            sum2 = mask2.astype(float) @ row_data   # (K, B)
            sum1 = mask1.astype(float) @ row_data   # (K, B)
            mu2p = sum2 / n2
            mu1p = sum1 / n1
            diffs = mu2p - mu1p                      # (K, B)

            # Two-sided exceedance, broadcast vs obs_diff[r, :]
            exceed[r, :] += (np.abs(diffs) >= np.abs(obs_diff[r, :])[None, :]).sum(axis=0)
            trials[r] += K
            batches_run[r] += 1

            if early_stop and batches_run[r] >= min_batches:
                a = exceed[r, :] + 1
                b_ = (trials[r] - exceed[r, :]) + 1
                lo = beta.ppf(0.025, a, b_)
                hi = beta.ppf(0.975, a, b_)
                if np.nanmax(hi - lo) <= eps:
                    break

    # Plus-one correction; rows never tested remain NaN
    pvals = np.full((B, B), np.nan, dtype=float)
    for r in range(B):
        if trials[r] > 0:
            pvals[r, :] = (exceed[r, :] + 1) / (trials[r] + 1)

    return pd.DataFrame(pvals, index=behavior_order, columns=behavior_order)


def _generate_comparison_heatmap(diff_matrix, pval_matrix, vmax, group1, group2, output_path):
    """
    Generates and saves a single difference heatmap with significance annotations.
    """
    # Create annotation matrix ('*' for significant, '' for non-significant)
    annot_df = pval_matrix.map(lambda p: '*' if p < 0.05 else '')
    
    num_behaviors = len(diff_matrix)
    figsize, annot_kws = ((30, 24), {"size": 12}) if num_behaviors > 20 else ((16, 12), {"size": 14})
    
    plt.figure(figsize=figsize)
    sns.heatmap(diff_matrix, annot=annot_df, fmt="", cmap="coolwarm",
                vmin=-vmax, vmax=vmax, center=0,
                linewidths=.5, annot_kws=annot_kws)
    
    plt.title(f'Transition Difference: {group2} vs. {group1}\n(Red = {group2} > {group1}, Blue = {group1} > {group2})', fontsize=16)
    plt.ylabel('From Behavior', fontsize=12)
    plt.xlabel('To Behavior', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# 9 + 10 + 11 Spatial and Geometric Analysis
