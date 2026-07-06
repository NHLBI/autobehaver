"""Time budget summaries (mmgen).

This module contains the time-budget Excel summary routine used during mmgen
finalization.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .feature_matrix import format_summary_df, save_to_excel_sheets



def calculate_and_save_time_budgets(master_df, animal_total_frames, output_dir, *, save_outputs: bool = True):
    """
    Calculates standard, conditional, and sex-stratified time budget summaries
    and saves them as multiple tabs in a single Excel file.
    """
    print("\nCalculating and saving consolidated time budget summaries...")

    # --- 1. Define Column Orders ---
    pose_order = ['Walking', 'Standing', 'Scramble', 'Groom-FrontLegs', 'Groom-Head', 'Groom-SideLeg', 'Groom-Wings', 'Groom-Abdomen', 'Groom-BackLegs', 'Jump']
    orient_order = ['Floor', 'Ceiling', 'Wall-Side', 'Wall-Up', 'Wall-Down']
    combined_order = [f"{p} + {o}" for p in pose_order if p != 'Jump' for o in orient_order]
    if 'Jump' in pose_order:
        combined_order.insert(pose_order.index('Jump'), 'Jump')
        
    # --- 2. Data Preparation ---
    df = master_df.copy()
    
    
    df['pose'] = df['bout_type'].apply(lambda x: x.split(' + ')[0])
    df['orient'] = df['bout_type'].apply(lambda x: x.split(' + ')[1] if ' + ' in x else None)

    # For Orient-only rows, bout_type IS the orientation label (e.g., "Floor")
    df.loc[df['category'] == 'Orient', 'orient'] = df.loc[df['category'] == 'Orient', 'bout_type']
    df.loc[df['category'] == 'Orient', 'pose'] = None

    df['analysis_bout_type'] = df['bout_type']
    df.loc[df['pose'] == 'Jump', 'analysis_bout_type'] = 'Jump'

    
    # --- 3. Calculate Denominators for Conditional Budgets ---
    # Denominator for Pose given Orient should be based on Pose+Orient data ONLY
    # (so numerator and denominator refer to the same filtered frame set)
    poseorient_df_totals = df[df['category'] == 'Pose+Orient'].dropna(subset=['animal_id', 'orient', 'total_length_frames'])
    total_time_per_orient = poseorient_df_totals.groupby(['animal_id', 'orient'], observed=True)['total_length_frames'] \
                             .sum().reset_index().rename(columns={'total_length_frames': 'total_orient_frames', 'animal_id': 'individual'})

    # Orient-only totals for true "Orientation (Overall)" + "Orientation (by Sex)"
    orient_df_totals = df[df['category'] == 'Orient'].dropna(subset=['animal_id', 'orient', 'total_length_frames'])


    # --- 4. Calculate Conditional Budgets ---
    # First, filter to only Pose+Orient data
    summary_po = df[df['category'] == 'Pose+Orient'].copy()

    if not summary_po.empty:
        # Aggregate the Pose+Orient data to get the 'length' of each specific combined bout type
        # Use animal_id for consistency with overall budgets (aggregate per animal, not per movie)
        summary_po = summary_po.groupby(['animal_id', 'group', 'sex', 'pose', 'orient'], observed=True)['total_length_frames'] \
                      .sum().reset_index().rename(columns={'total_length_frames': 'length', 'animal_id': 'individual'})

        # --- Calculate Denominator for Orient given Pose using ONLY Pose+Orient data ---
        # Sum the 'length' column (derived ONLY from Pose+Orient bouts) grouped by individual and pose
        total_pose_orient_frames = summary_po.groupby(['individual', 'pose'], observed=True)['length'] \
                                    .sum().reset_index().rename(columns={'length': 'total_pose_orient_len'})

        # Merge the necessary denominators back onto the summary_po DataFrame
        summary_po = pd.merge(summary_po, total_time_per_orient, on=['individual', 'orient'], how='left')
        summary_po = pd.merge(summary_po, total_pose_orient_frames, on=['individual', 'pose'], how='left') # Merge the CORRECT denominator for OGP

        # --- Calculate the conditional budget values ---
        # Handle potential division by zero by replacing inf with NaN
        summary_po['pose_given_orient_budget'] = (summary_po['length'] / summary_po['total_orient_frames']).replace([np.inf, -np.inf], np.nan)
        # *** USE THE CORRECT DENOMINATOR FOR OGP ***
        summary_po['orient_given_pose_budget'] = (summary_po['length'] / summary_po['total_pose_orient_len']).replace([np.inf, -np.inf], np.nan)

        # *** ASSERT CHECK 1: No Infinities in Conditional Budgets ***
        assert not np.isinf(summary_po['pose_given_orient_budget'].dropna()).any(), "Infinite values found in 'pose_given_orient_budget'"
        assert not np.isinf(summary_po['orient_given_pose_budget'].dropna()).any(), "Infinite values found in 'orient_given_pose_budget'"

        # *** ASSERT CHECK 2: Sum of Conditional Budgets (Pose given Orient) ***
        # For a given individual/orientation, sum probabilities across all poses
        cond_sum_pgo = summary_po.groupby(['individual', 'orient'], observed=True)['pose_given_orient_budget'].sum()
        # Check if sums are close to 1.0 (allow small tolerance). Ignore NaNs (where total_orient_frames might be 0 or no data)
        assert np.allclose(cond_sum_pgo.dropna(), 1.0, atol=1e-6), \
               f"Sum of 'Pose given Orient' budgets not close to 1.0. Problematic sums:\n{cond_sum_pgo[~np.isclose(cond_sum_pgo.dropna(), 1.0, atol=1e-6)]}"

        # *** ASSERT CHECK 3: Sum of Conditional Budgets (Orient given Pose) ***
        # For a given individual/pose, sum probabilities across all orientations
        cond_sum_ogp = summary_po.groupby(['individual', 'pose'], observed=True)['orient_given_pose_budget'].sum()
        # Check if sums are close to 1.0. Ignore NaNs (where total_pose_orient_len might be 0 or no data)
        assert np.allclose(cond_sum_ogp.dropna(), 1.0, atol=1e-6), \
               f"Sum of 'Orient given Pose' budgets not close to 1.0. Problematic sums:\n{cond_sum_ogp[~np.isclose(cond_sum_ogp.dropna(), 1.0, atol=1e-6)]}"

    # Note: 'summary_po' now contains the correctly calculated conditional budgets
    # The rest of section 5 (preparing sheets for Excel) will use this corrected summary_po


    # --- 5. Prepare All DataFrames for Excel Sheets ---
    all_sheets = {}
    # Create maps using animal_id (collapsed IDs) since all pivot tables now use collapsed animal IDs
    individual_group_map = df.drop_duplicates('animal_id').set_index('animal_id')['group'].to_dict()
    individual_info_map = df.drop_duplicates('animal_id').set_index('animal_id')[['group', 'sex']].to_dict('index')

    ### PART A: OVERALL BUDGETS ###
    # (Existing calculations for Pose, Orientation, and Pose-Orient Overall)
    pose_df_orig = df[df['category'] == 'Pose'].copy()
    
    pose_summary_orig = pose_df_orig.groupby(['animal_id', 'pose'], observed=True)['total_length_frames'].sum().reset_index().rename(columns={'total_length_frames': 'length', 'animal_id': 'individual'})
    
    pose_summary_orig['total_frames'] = pose_summary_orig['individual'].map(animal_total_frames)
    pose_summary_orig['time_budget'] = pose_summary_orig['length'] / pose_summary_orig['total_frames']
    pose_pivot_orig = pose_summary_orig.pivot_table(index='individual', columns='pose', values='time_budget', observed=True).reset_index().fillna(0)
    
    
    pose_pivot_orig.insert(1, 'group', pose_pivot_orig['individual'].map(individual_group_map))
    
    # *** ASSERT CHECK 4: Sum of Overall Pose Budgets ***
    pose_budget_cols = [col for col in pose_pivot_orig.columns if col not in ['individual', 'group']]
    pose_sums = pose_pivot_orig[pose_budget_cols].sum(axis=1)
    assert (pose_sums >= 0).all() and (pose_sums <= 1.05).all(), f"Overall Pose budget sums out of range [0, 1.05]. Min: {pose_sums.min()}, Max: {pose_sums.max()}"

    all_sheets['Pose (Overall)'] = format_summary_df(pose_pivot_orig, pose_order)

    if not orient_df_totals.empty:
        orient_summary_orig = orient_df_totals.groupby(['animal_id', 'orient'], observed=True)['total_length_frames'].sum().reset_index().rename(columns={'total_length_frames': 'length', 'animal_id': 'individual'})
        orient_summary_orig['total_frames'] = orient_summary_orig['individual'].map(animal_total_frames)
        orient_summary_orig['time_budget'] = orient_summary_orig['length'] / orient_summary_orig['total_frames']
        orient_pivot_orig = orient_summary_orig.pivot_table(index='individual', columns='orient', values='time_budget', observed=True).reset_index().fillna(0)
        orient_pivot_orig.insert(1, 'group', orient_pivot_orig['individual'].map(individual_group_map))

        # *** ASSERT CHECK 5: Sum of Overall Orient Budgets ***
        orient_budget_cols = [col for col in orient_pivot_orig.columns if col not in ['individual', 'group']]
        orient_sums = orient_pivot_orig[orient_budget_cols].sum(axis=1)
        assert (orient_sums >= 0).all() and (orient_sums <= 1.05).all(), f"Overall Orient budget sums out of range [0, 1.05]. Min: {orient_sums.min()}, Max: {orient_sums.max()}"

        all_sheets['Orientation (Overall)'] = format_summary_df(orient_pivot_orig, orient_order)

    combined_df_orig = df[df['category'] == 'Pose+Orient'].copy()
    if not combined_df_orig.empty:
        combined_summary_orig = combined_df_orig.groupby(['animal_id', 'analysis_bout_type'], observed=True)['total_length_frames'].sum().reset_index().rename(columns={'total_length_frames': 'length', 'animal_id': 'individual'})
        combined_summary_orig['total_frames'] = combined_summary_orig['individual'].map(animal_total_frames)
        combined_summary_orig['time_budget'] = combined_summary_orig['length'] / combined_summary_orig['total_frames']
        combined_pivot_orig = combined_summary_orig.pivot_table(index='individual', columns='analysis_bout_type', values='time_budget', observed=True).reset_index().fillna(0)
        combined_pivot_orig.insert(1, 'group', combined_pivot_orig['individual'].map(individual_group_map))

        # *** ASSERT CHECK 6: Sum of Overall Pose+Orient Budgets ***
        combined_budget_cols = [col for col in combined_pivot_orig.columns if col not in ['individual', 'group']]
        combined_sums = combined_pivot_orig[combined_budget_cols].sum(axis=1)
        assert (combined_sums >= 0).all() and (combined_sums <= 1.05).all(), f"Overall Pose+Orient budget sums out of range [0, 1.05]. Min: {combined_sums.min()}, Max: {combined_sums.max()}"

        all_sheets['Pose-Orient (Overall)'] = format_summary_df(combined_pivot_orig, combined_order)
    
    # Pose given Orient (Overall)
    if not summary_po.empty:
        pgo_pivot = summary_po.pivot_table(index=['individual', 'group'], columns=['pose', 'orient'], values='pose_given_orient_budget', observed=True).fillna(0)
        pgo_pivot.columns = [f"{p} | {o}" for p, o in pgo_pivot.columns]
        
        pgo_ordered_cols, separators = [], []
        for i, pose in enumerate(pose_order):
            pose_cols = [col for col in pgo_pivot.columns if col.startswith(f"{pose} |")]
            if pose_cols:
                pose_cols.sort(key=lambda x: orient_order.index(x.split(' | ')[1]) if x.split(' | ')[1] in orient_order else 99)
                pgo_ordered_cols.extend(pose_cols)
                sep = f'---_{i}'
                pgo_ordered_cols.append(sep)
                separators.append(sep)
        if pgo_ordered_cols: pgo_ordered_cols.pop() # Remove last separator
        for sep in separators: pgo_pivot[sep] = ''
        all_sheets['Pose given Orient (Overall)'] = format_summary_df(pgo_pivot.reset_index(), pgo_ordered_cols)
        
    # Orient given Pose (Overall)
    if not summary_po.empty:
        ogp_pivot = summary_po.pivot_table(index=['individual', 'group'], columns=['orient', 'pose'], values='orient_given_pose_budget', observed=True).fillna(0)
        ogp_pivot.columns = [f"{o} | {p}" for o, p in ogp_pivot.columns]
        
        ogp_ordered_cols, separators = [], []
        for i, orient in enumerate(orient_order):
            orient_cols = [col for col in ogp_pivot.columns if col.startswith(f"{orient} |")]
            if orient_cols:
                orient_cols.sort(key=lambda x: pose_order.index(x.split(' | ')[1]) if x.split(' | ')[1] in pose_order else 99)
                ogp_ordered_cols.extend(orient_cols)
                sep = f'---_{i+100}'
                ogp_ordered_cols.append(sep)
                separators.append(sep)
        if ogp_ordered_cols: ogp_ordered_cols.pop()
        for sep in separators: ogp_pivot[sep] = ''
        all_sheets['Orient given Pose (Overall)'] = format_summary_df(ogp_pivot.reset_index(), ogp_ordered_cols)

    ### PART B: BY SEX BUDGETS ###
    # (Existing calculations for Pose, Orientation, and Pose-Orient by Sex)
    pose_summary_sex = pose_df_orig.groupby(['animal_id', 'sex', 'pose'], observed=True)['total_length_frames'].sum().reset_index().rename(columns={'total_length_frames': 'length', 'animal_id': 'individual'})
    pose_summary_sex['total_frames'] = pose_summary_sex['individual'].map(animal_total_frames)
    pose_summary_sex['time_budget'] = pose_summary_sex['length'] / pose_summary_sex['total_frames']
    pose_pivot_sex = pose_summary_sex.pivot_table(index=['individual', 'sex'], columns='pose', values='time_budget', observed=True).reset_index().fillna(0)
    pose_pivot_sex.insert(2, 'group', pose_pivot_sex['individual'].map(lambda x: individual_info_map.get(x, {}).get('group')))
    all_sheets['Pose (by Sex)'] = format_summary_df(pose_pivot_sex, pose_order, id_cols=['individual', 'group', 'sex'])
    if not orient_df_totals.empty:
        orient_summary_sex = orient_df_totals.groupby(['animal_id', 'sex', 'orient'], observed=True)['total_length_frames'].sum().reset_index().rename(columns={'total_length_frames': 'length', 'animal_id': 'individual'})
        orient_summary_sex['total_frames'] = orient_summary_sex['individual'].map(animal_total_frames)
        orient_summary_sex['time_budget'] = orient_summary_sex['length'] / orient_summary_sex['total_frames']
        orient_pivot_sex = orient_summary_sex.pivot_table(index=['individual', 'sex'], columns='orient', values='time_budget', observed=True).reset_index().fillna(0)
        orient_pivot_sex.insert(2, 'group', orient_pivot_sex['individual'].map(lambda x: individual_info_map.get(x, {}).get('group')))
        all_sheets['Orientation (by Sex)'] = format_summary_df(orient_pivot_sex, orient_order, id_cols=['individual', 'group', 'sex'])
    if not combined_df_orig.empty:
        combined_summary_sex = combined_df_orig.groupby(['animal_id', 'sex', 'analysis_bout_type'], observed=True)['total_length_frames'].sum().reset_index().rename(columns={'total_length_frames': 'length', 'animal_id': 'individual'})
        combined_summary_sex['total_frames'] = combined_summary_sex['individual'].map(animal_total_frames)
        combined_summary_sex['time_budget'] = combined_summary_sex['length'] / combined_summary_sex['total_frames']
        combined_pivot_sex = combined_summary_sex.pivot_table(index=['individual', 'sex'], columns='analysis_bout_type', values='time_budget', observed=True).reset_index().fillna(0)
        combined_pivot_sex.insert(2, 'group', combined_pivot_sex['individual'].map(lambda x: individual_info_map.get(x, {}).get('group')))
        all_sheets['Pose-Orient (by Sex)'] = format_summary_df(combined_pivot_sex, combined_order, id_cols=['individual', 'group', 'sex'])

    # Pose given Orient (by Sex)
    if not summary_po.empty:
        pgo_pivot_sex = summary_po.pivot_table(index=['individual', 'group', 'sex'], columns=['pose', 'orient'], values='pose_given_orient_budget', observed=True).fillna(0)
        pgo_pivot_sex.columns = [f"{p} | {o}" for p, o in pgo_pivot_sex.columns]
        pgo_ordered_cols, separators = [], [] # Reuse variable from above
        for i, pose in enumerate(pose_order):
            pose_cols = [col for col in pgo_pivot_sex.columns if col.startswith(f"{pose} |")]
            if pose_cols:
                pose_cols.sort(key=lambda x: orient_order.index(x.split(' | ')[1]) if x.split(' | ')[1] in orient_order else 99)
                pgo_ordered_cols.extend(pose_cols)
                sep = f'---_{i}'
                pgo_ordered_cols.append(sep)
                separators.append(sep)
        if pgo_ordered_cols: pgo_ordered_cols.pop()
        for sep in separators: pgo_pivot_sex[sep] = ''
        all_sheets['Pose given Orient (by Sex)'] = format_summary_df(pgo_pivot_sex.reset_index(), pgo_ordered_cols, id_cols=['individual', 'group', 'sex'])

    # Orient given Pose (by Sex)
    if not summary_po.empty:
        ogp_pivot_sex = summary_po.pivot_table(index=['individual', 'group', 'sex'], columns=['orient', 'pose'], values='orient_given_pose_budget', observed=True).fillna(0)
        ogp_pivot_sex.columns = [f"{o} | {p}" for o, p in ogp_pivot_sex.columns]
        ogp_ordered_cols, separators = [], [] # Reuse variable
        for i, orient in enumerate(orient_order):
            orient_cols = [col for col in ogp_pivot_sex.columns if col.startswith(f"{orient} |")]
            if orient_cols:
                orient_cols.sort(key=lambda x: pose_order.index(x.split(' | ')[1]) if x.split(' | ')[1] in pose_order else 99)
                ogp_ordered_cols.extend(orient_cols)
                sep = f'---_{i+100}'
                ogp_ordered_cols.append(sep)
                separators.append(sep)
        if ogp_ordered_cols: ogp_ordered_cols.pop()
        for sep in separators: ogp_pivot_sex[sep] = ''
        all_sheets['Orient given Pose (by Sex)'] = format_summary_df(ogp_pivot_sex.reset_index(), ogp_ordered_cols, id_cols=['individual', 'group', 'sex'])

    # --- 6. Finalize and Save the Excel File ---
    final_sheet_order = [
        'Pose (Overall)', 'Orientation (Overall)', 'Pose-Orient (Overall)', 'Pose given Orient (Overall)', 'Orient given Pose (Overall)',
        'Pose (by Sex)', 'Orientation (by Sex)', 'Pose-Orient (by Sex)', 'Pose given Orient (by Sex)', 'Orient given Pose (by Sex)'
    ]
    sheets_to_save = {name: all_sheets[name] for name in final_sheet_order if name in all_sheets}
    if save_outputs:
        excel_output_path = os.path.join(output_dir, '2_Time_Budgets.xlsx')
        save_to_excel_sheets(sheets_to_save, excel_output_path)
    return sheets_to_save


# 3 Standardized Measurements (Body Measurements)
