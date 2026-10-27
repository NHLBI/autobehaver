"""Define FeatureImp command-line arguments and runtime defaults.

Parser construction does not import optional model or analysis dependencies.
"""

from __future__ import annotations

import argparse

import os

RECOGNIZED_FEATURE_CATEGORY_PREFIXES = (
    'BoutSummary_',
    'TimeBudget_',
    'StdMeasure_',
    'Kinematic_',
    'KinematicBB_',
    'Postural_',
    'Transition_',
    'Spatial_',
    'SleepFeat_',
    'Global_',
)

DEFAULT_SETTINGS = {
    # Warning handling
    "SUPPRESS_RUNTIME_WARNINGS": False,
    "SUPPRESS_FUTURE_WARNINGS": False,
    # Matplotlib's `tight_layout()` commonly emits a noisy (but harmless) warning
    # when figures include colorbars / inset axes. Suppress by default to keep HPC
    # logs readable. Use `--no-SUPPRESS_TIGHT_LAYOUT_WARNINGS` to re-enable.
    "SUPPRESS_TIGHT_LAYOUT_WARNINGS": True,


    # Post-analysis stage error handling (PCA/FSEA/baselines)
    # If True, raise immediately when a post-analysis stage fails (instead of warning+continue).
    "postanalysis_strict": False,
    # If True, print full tracebacks for post-analysis stage failures to stdout/stderr.
    "postanalysis_debug": False,

    # Feature Analysis Run Settings
    "analysis_run_name": "featureimp_run",

    # Staged execution / sharding
    # Default CLI behavior is stage=all, and stage=all executes the same
    # staged engine in serial with num_shards forced to 1.
    # The staged planner (stage=plan) freezes row/feature/fold/subset/task state
    # for later shard execution.
    "stage": "all",
    "num_shards": 1,
    "shard_index": 0,
    "analysis_folder_path": "",

    # Standalone FSEA ranking-file mode (stage=fsea). When both ranked_list_path
    # and output_dir are provided, stage=fsea runs the global-importance FSEA
    # using that file.
    "fsea_ranked_list_path": "",
    "fsea_output_dir": "",
    "fsea_feature_col": "feature_name",
    "fsea_rank_col": "Global_Importance",
    "fsea_label": "",

    # Paths
    "main_dir": "",

    # Folder containing the MMGen outputs (including MASTER_FEATURE_MATRIX.csv).
    # In the standardized Autobehaver pipeline, this is typically "8_master_matrix".
    "matrix_dir": "8_master_matrix",

    # Where FeatureImp analysis runs should be written.
    # If blank, it auto-derives a sibling folder next to matrix_dir named:
    #   "9_feature_importance"
    "results_dir": "",


    # Leave blank to auto-derive from matrix_dir:
    "matrix_path": "",

    # Define the string used to identify the control group for Z-score normalization.
    # If left blank, it will default to searching for "CONTROL".
    "control_group_tag": "",

    # The script trains a BINARY model on the two groups defined below.
    # All SHAP analysis will be performed on ALL groups using this binary model.
    "training_group_a": "",
    "training_group_b": "",

    # Feature category filtering (applied after loading the feature matrix).
    # None = all recognized feature categories, or list of prefixes to include.
    # Available categories: 'BoutSummary_', 'TimeBudget_', 'StdMeasure_', 'Postural_',
    #                       'Kinematic_', 'KinematicBB_', 'Transition_',
    #                       'Spatial_', 'SleepFeat_', 'Global_'
    # Default is None so all recognized feature-style columns are eligible unless
    # the user explicitly narrows the surface with --include_feature_categories.
    "include_feature_categories": None,

    # Sex filtering
    # Standard pooled analysis remains the default. To run a male-only or female-only
    # analysis, use sex_filter.
    "sex_filter": "all",


    # Feature Sparsity Filtering
    # Minimum nonmissing fraction required in each training group.
    # A feature is dropped if it's below this threshold in EITHER training group.
    # Set to 0.0 to disable sparsity filtering.
    "min_feature_completeness_per_group": 0.25,

    # Feature Variance Filtering
    # Filter out features with insufficient variance (after NaN imputation, before training).
    # Variance filtering is always enabled (zero variance and near-zero variance).
    "MIN_VARIANCE_THRESHOLD": 1e-6,        # Minimum std threshold for near-zero variance filtering

    # XGBoost Model Randomness Settings
    "model_subsample": 0.8,                # Fraction of training rows sampled per tree

    # XGBoost Model Complexity Settings
    "n_estimators": 1000,                   # Total number of trees
    "learning_rate": 0.01,                  # Learning rate / eta
    "max_depth": 6,                        # Max depth

    # Committee/Specialist Model Settings
    "n_splits": 5,                         # Number of K-Fold splits (committee design)
    "H_hits_per_feature": 100,               # Target number of times each feature should be tested
    "n_features_per_model": 100,          # Number of features per specialist model

    # Reproducibility / Runtime
    "GLOBAL_RANDOM_SEED": 7,
    "num_workers": 16,

    # Validation Settings
    "VALIDATION_ERROR_ON_FAIL": False,

    # Permutation Test Settings
    "run_permutation_test": True,
    "n_permutations": 1,

    # Subset weighting
    # Committee members (feature subsets) are weighted by their out-of-fold skill
    # relative to the permuted-label subset distribution.
    "subset_weight_tau_quantile": 0.95,   # quantile of permuted delta_norm used as dead-zone tau
    "subset_weight_gamma": 2.0,           # exponent applied after dead-zone: w = max(0, delta_norm - tau)^gamma

    # Automatic reference post-merge analyses
    "run_fsea_analysis": True,
    "run_pca_analysis": True,

    # PCA settings
    "pca_fixed_k": 50,

    # If <=0, save all selected PCA components to PCA_PROJECTED_COORDINATES*.csv.
    "n_pca_components_to_save": 0,

    # FSEA Analysis Settings
    "fsea_permutations": 100000,
    "fsea_min_size": 15,
    "fsea_max_size": 500,
    "fsea_fdr_threshold": 0.15,

    # FSEA Visualization Settings
    "fsea_viz_top_n": 1000,
    # Default OFF; pass --fsea_viz_plot_all_groups to plot all groups instead of only significant ones.
    "fsea_viz_plot_all_groups": False,
    "fsea_viz_color_percentile": 100,
}

# Runtime setting keys accepted by the public settings facade.
# Special cases (derived / normalized): matrix_path, include_feature_categories.

def _int_or_none(v):
    """Parse an integer or the case-insensitive token 'none'."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nil"):
        return None
    return int(v)

def build_arg_parser(defaults=None):
    """Build the FeatureImp argument parser from runtime defaults."""
    if defaults is None:
        defaults = DEFAULT_SETTINGS

    parser = argparse.ArgumentParser(
        description='FeatureImp specialist XGBoost ensemble, SHAP summaries, FSEA, and reference PCA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Run / Paths
    g = parser.add_argument_group("Run / Paths")
    g.add_argument("--analysis_run_name", "--run", dest="analysis_run_name", type=str,
                   default=defaults["analysis_run_name"],
                   help="Run name appended to the analysis folder name.")
    g.add_argument(
        "--main_dir",
        dest="main_dir",
        type=str,
        default=defaults.get("main_dir", ""),
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For FeatureImp this fills matrix_dir=<main_dir>/8_master_matrix and results_dir=<main_dir>/9_feature_importance unless you specify them explicitly."
        ),
    )
    g.add_argument(
        "--matrix_dir",
        dest="matrix_dir",
        type=str,
        default=defaults["matrix_dir"],
        help=(
            "Folder containing the MMGen outputs (including MASTER_FEATURE_MATRIX.csv). "
            "In the standardized Autobehaver pipeline, this is typically '8_master_matrix'."
        ),
    )
    g.add_argument(
        "--results_dir",
        dest="results_dir",
        type=str,
        default=defaults["results_dir"],
        help=(
            "Where FeatureImp analysis run folders should be written. "
            "If blank, auto-derives a sibling folder next to matrix_dir named '9_feature_importance'."
        ),
    )
    g.add_argument("--matrix_path", dest="matrix_path", type=str,
                   default=defaults["matrix_path"],
                   help="Path to MASTER_FEATURE_MATRIX.csv. Leave blank to auto-derive from matrix_dir.")

    # Staged execution / sharding
    g = parser.add_argument_group("Staged execution / sharding")
    g.add_argument(
        "--stage",
        type=str,
        choices=[
            "all",
            "plan",
            "process_shard",
            "merge_shards",
            "plain_xgboost_baseline",
            "plain_xgboost_baseline_plan",
            "plain_xgboost_baseline_process_shard",
            "plain_xgboost_baseline_merge_shards",
            "pca",
            "fsea",
        ],
        default=str(defaults.get("stage", "all")),
        help=(
            "Execution stage. Default is 'all' (serial wrapper over the staged core engine). "
            "Use 'plan' to create a frozen staged/sharded manifest for later HPC execution."
        ),
    )
    g.add_argument(
        "--run_fsea_analysis",
        dest="run_fsea_analysis",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("run_fsea_analysis", True),
        help=(
            "Run FSEA automatically after the main merge_shards/stage=all path and after supported baseline stages."
        ),
    )
    g.add_argument(
        "--run_pca_analysis",
        dest="run_pca_analysis",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("run_pca_analysis", True),
        help=(
            "Run the reference PCA axis analysis automatically after the main merge_shards/stage=all path."
        ),
    )
    g.add_argument(
        "--num_shards",
        type=int,
        default=int(defaults.get("num_shards", 1)),
        help=(
            "Requested total shard count for staged execution. Also used by plain_xgboost_baseline_plan. Ignored when --stage all, which always forces a single serial shard through the staged core path."
        ),
    )
    g.add_argument(
        "--shard_index",
        type=int,
        default=int(defaults.get("shard_index", 0)),
        help="Shard index for stage=process_shard or stage=plain_xgboost_baseline_process_shard.",
    )
    g.add_argument(
        "--analysis_folder_path",
        type=str,
        default=str(defaults.get("analysis_folder_path", "")),
        help=(
            "Existing FeatureImp analysis folder to reuse for non-plan stages. "
            "Required for staged rerun phases such as process_shard, merge_shards, plain_xgboost_baseline_plan/process/merge stages, and standalone postanalysis stages. "
            "For stage=fsea, you may alternatively omit this and instead provide --fsea_ranked_list_path plus --fsea_output_dir to run a global-only preranked FSEA."
        ),
    )

    # Training groups / normalization
    g = parser.add_argument_group("Training groups / normalization")
    g.add_argument("--control_group_tag", "--control", dest="control_group_tag", type=str,
                   default=defaults["control_group_tag"],
                   help="Tag identifying the control group for normalization (blank defaults to searching for 'CONTROL').")
    g.add_argument("--training_group_a", "--a", dest="training_group_a", type=str,
                   default=defaults["training_group_a"],
                   help="Training group A (encoded as class 0 / NEGATIVE_CLASS_INDEX).")
    g.add_argument("--training_group_b", "--b", dest="training_group_b", type=str,
                   default=defaults["training_group_b"],
                   help="Training group B (encoded as class 1 / POSITIVE_CLASS_INDEX).")

    # Feature inclusion / filtering
    g = parser.add_argument_group("Feature inclusion / filtering")
    g.add_argument("--include_feature_categories", nargs="*", metavar="PREFIX",
                   default=defaults["include_feature_categories"],
                   help=("List of feature category prefixes to include. "
                         "Use NONE/ALL/* (or pass no values) to use all recognized feature categories."))
    g.add_argument(
        "--sex_filter",
        choices=["all", "male", "female"],
        default=str(defaults.get("sex_filter", "all")),
        help=(
            "Subset the standard analysis to one sex before training. "
            "Use 'all' (default), 'male', or 'female'."
        ),
    )
    g.add_argument("--min_feature_completeness_per_group", type=float,
                   default=defaults["min_feature_completeness_per_group"],
                   help="Minimum fraction of non-NaN values required in EACH training group to keep a feature.")
    g.add_argument("--min_variance_threshold", dest="MIN_VARIANCE_THRESHOLD", type=float,
                   default=defaults["MIN_VARIANCE_THRESHOLD"],
                   help="Minimum std threshold for near-zero variance filtering.")

    # XGBoost hyperparameters
    g = parser.add_argument_group("XGBoost hyperparameters")
    g.add_argument("--model_subsample", type=float, default=defaults["model_subsample"],
                   help="Row subsample fraction for each tree.")
    g.add_argument("--n_estimators", "--trees", dest="n_estimators", type=int, default=defaults["n_estimators"],
                   help="Number of boosting rounds (trees).")
    g.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=defaults["learning_rate"],
                   help="Learning rate / eta.")
    g.add_argument("--max_depth", type=int, default=defaults["max_depth"],
                   help="Maximum tree depth.")

    # Committee design
    g = parser.add_argument_group("Committee design")
    g.add_argument("--n_splits", "--folds", dest="n_splits", type=int, default=defaults["n_splits"],
                   help="Number of StratifiedKFold splits.")
    g.add_argument("--H_hits_per_feature", "--hits", dest="H_hits_per_feature", type=int,
                   default=defaults["H_hits_per_feature"],
                   help="Target number of times each feature should be tested across specialists.")
    g.add_argument("--n_features_per_model", "--features_per_model", dest="n_features_per_model", type=int,
                   default=defaults["n_features_per_model"],
                   help="Number of features each specialist model will use.")

    # Reproducibility / runtime
    g = parser.add_argument_group("Reproducibility / runtime")
    g.add_argument("--random_seed", "--seed", dest="GLOBAL_RANDOM_SEED", type=_int_or_none,
                   default=defaults["GLOBAL_RANDOM_SEED"],
                   help='Integer random seed, or NONE to generate and record a seed when planning.')
    g.add_argument("--num_workers", "--workers", dest="num_workers", type=int,
                   default=defaults["num_workers"],
                   help="Number of multiprocessing worker processes.")


    # Diagnostics / logging
    g = parser.add_argument_group("Diagnostics / logging")

    g.add_argument(
        "--postanalysis_strict",
        dest="postanalysis_strict",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("postanalysis_strict", False)),
        help=(
            "If enabled, any failure in an optional post-analysis stage (PCA/FSEA/baselines)"
            " will raise immediately instead of printing a warning and continuing."
        ),
    )

    g.add_argument(
        "--postanalysis_debug",
        dest="postanalysis_debug",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("postanalysis_debug", False)),
        help=(
            "If enabled, print full Python tracebacks for optional post-analysis stage failures."
        ),
    )

    # Validation
    g = parser.add_argument_group("Validation")
    g.add_argument("--VALIDATION_ERROR_ON_FAIL",
                   action=argparse.BooleanOptionalAction,
                   default=defaults["VALIDATION_ERROR_ON_FAIL"],
                   help="If enabled, validation failures raise instead of warning.")

    # Permutation tests
    g = parser.add_argument_group("Permutation tests")
    g.add_argument("--run_permutation_test",
                   action=argparse.BooleanOptionalAction,
                   default=defaults["run_permutation_test"],
                   help="If enabled, train models on shuffled labels for sanity checking.")
    g.add_argument("--n_permutations", type=int, default=defaults["n_permutations"],
                   help="Number of permutation iterations.")

    # Subset weighting
    g = parser.add_argument_group("Subset weighting")
    g.add_argument(
        "--subset_weight_tau_quantile",
        type=float,
        default=defaults.get("subset_weight_tau_quantile", 0.95),
        help=(
            "Quantile of permuted subset skill (delta_norm) used as the dead-zone threshold tau. "
            "If permutation results are unavailable, tau defaults to 0."
        ),
    )
    g.add_argument(
        "--subset_weight_gamma",
        type=float,
        default=defaults.get("subset_weight_gamma", 2.0),
        help=(
            "Exponent for subset weighting: w = max(0, delta_norm - tau)^gamma. "
            "Higher values emphasize only the strongest subsets."
        ),
    )

    # PCA reference-axis analysis
    g = parser.add_argument_group("PCA reference-axis analysis")
    g.add_argument("--pca_fixed_k", type=int, default=defaults["pca_fixed_k"],
                   help="Number of PCA components to retain for reference-axis scoring.")
    g.add_argument(
        "--n_pca_components_to_save",
        type=int,
        default=defaults["n_pca_components_to_save"],
        help=(
            "Number of PCA components to write to PCA_PROJECTED_COORDINATES*.csv. "
            "If <=0, save all selected PCA components."
        ),
    )
    # FSEA
    g = parser.add_argument_group("FSEA")
    g.add_argument("--fsea_ranked_list_path", type=str,
                   default=str(defaults.get("fsea_ranked_list_path", "")),
                   help=("Optional preranked global-importance CSV for stage=fsea. "
                         "When provided with --fsea_output_dir, FeatureImp runs a single global-only FSEA from this file instead of reusing an analysis folder."))
    g.add_argument("--fsea_output_dir", type=str,
                   default=str(defaults.get("fsea_output_dir", "")),
                   help=("Output directory for stage=fsea ranking-file mode. "
                         "Required together with --fsea_ranked_list_path."))
    g.add_argument("--fsea_feature_col", type=str,
                   default=str(defaults.get("fsea_feature_col", "feature_name")),
                   help="Feature-name column to read in stage=fsea ranking-file mode.")
    g.add_argument("--fsea_rank_col", type=str,
                   default=str(defaults.get("fsea_rank_col", "Global_Importance")),
                   help="Ranking-value column to read in stage=fsea ranking-file mode.")
    g.add_argument("--fsea_label", type=str,
                   default=str(defaults.get("fsea_label", "")),
                   help=("Optional label for stage=fsea ranking-file mode outputs. "
                         "Defaults to 'GlobalImportance'."))
    g.add_argument("--fsea_permutations", type=int, default=defaults["fsea_permutations"],
                   help="Number of permutations for FSEA.")
    g.add_argument("--fsea_min_size", type=int, default=defaults["fsea_min_size"],
                   help="Minimum features in a group.")
    g.add_argument("--fsea_max_size", type=int, default=defaults["fsea_max_size"],
                   help="Maximum features in a group.")
    g.add_argument("--fsea_fdr_threshold", type=float, default=defaults["fsea_fdr_threshold"],
                   help="FDR threshold for significance.")
    g.add_argument("--fsea_viz_top_n", type=int, default=defaults["fsea_viz_top_n"],
                   help="Top N groups to visualize per analysis.")
    g.add_argument(
        "--fsea_viz_plot_all_groups",
        action="store_true",
        default=defaults["fsea_viz_plot_all_groups"],
        help="If enabled, plot all groups regardless of significance."
    )
    g.add_argument(
        "--no-fsea_viz_plot_all_groups",
        dest="fsea_viz_plot_all_groups",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    g.add_argument("--fsea_viz_color_percentile", type=float, default=defaults["fsea_viz_color_percentile"],
                   help="Color percentile for plots (100 = full range).")

    # Warning suppression toggles
    g = parser.add_argument_group("Warnings")
    g.add_argument("--SUPPRESS_RUNTIME_WARNINGS",
                   action=argparse.BooleanOptionalAction,
                   default=defaults["SUPPRESS_RUNTIME_WARNINGS"],
                   help="Suppress RuntimeWarning.")
    g.add_argument("--SUPPRESS_FUTURE_WARNINGS",
                   action=argparse.BooleanOptionalAction,
                   default=defaults["SUPPRESS_FUTURE_WARNINGS"],
                   help="Suppress FutureWarning.")
    g.add_argument("--SUPPRESS_TIGHT_LAYOUT_WARNINGS",
                   action=argparse.BooleanOptionalAction,
                   default=defaults["SUPPRESS_TIGHT_LAYOUT_WARNINGS"],
                   help="Suppress tight_layout UserWarnings from matplotlib.")

    return parser
