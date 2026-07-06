"""Final merge for the staged FeatureImp workflow."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .committee import FixedBinaryLabelEncoder
from .config import DEFAULT_SETTINGS
from .compat import (
    _combine_insample_outofsample_summaries,
    _generate_kfold_shap_summaries,
    _generate_male_female_comparison,
    create_formatted_predictions_for_graphpad,
    save_predictions_with_formatting,
)
from .finalize import (
    _aggregate_and_summarize_permuted_results,
    _aggregate_and_summarize_results,
)
from .io_utils import _sanitize_excel_sheet_name, save_to_excel_sheets
from .logit_ensemble import load_subset_weights_mapping
from .margin_tasks import (
    WEIGHTED_REAL_MARGIN_PARTIAL_FILENAME,
    compute_logit_ensemble_predictions_from_margin_task_archives,
    finalize_weighted_margin_partial,
    merge_weighted_margin_partials,
    reduce_weighted_margin_task_archives_to_partial,
)
from .plan import load_featureimp_plan_manifest
from .reference import write_featureimp_reference_bundle
from .shap_tasks import (
    WEIGHTED_PERM_SHAP_PARTIAL_FILENAME_TEMPLATE,
    WEIGHTED_REAL_SHAP_PARTIAL_FILENAME,
    finalize_weighted_shap_partial,
    merge_weighted_shap_partials,
    reduce_weighted_shap_task_archives_to_partial,
)
from .staged_shared import (
    MERGE_DIRNAME,
    MERGED_PERFORMANCE_PERM_FILENAME,
    MERGED_PERFORMANCE_REAL_FILENAME,
    MERGED_PERM_PARTIALS_FILENAME_TEMPLATE,
    MERGED_SUBSET_REAL_FILENAME,
    _build_hybrid_shap_views,
    _build_predictions_export,
    _directory_lock,
    _discover_completed_shards,
    _finalize_partial_averages,
    _json_dump,
    _json_load,
    _jsonl_load,
    _load_featureimp_context,
    _load_partial_npz,
    _load_settings_snapshot,
    _safe_relpath,
    _write_ensemble_metrics,
    _write_committee_pooled_performance,
)
from .subset_weight_merge import WEIGHTS_DONE_FILENAME

FINAL_MERGE_RUNNING_FILENAME = "FINAL_MERGE_RUNNING.json"
FINAL_MERGE_DONE_FILENAME = "FINAL_MERGE_DONE.json"
FINAL_MERGE_FAILED_FILENAME = "FINAL_MERGE_FAILED.json"
FINAL_MERGE_LOCK_DIRNAME = "_featureimp_final_merge.lock"
FINAL_POSTANALYSIS_BUNDLE_FILENAME = "FINAL_POSTANALYSIS_BUNDLE.json"
WEIGHTED_REDUCTION_DONE_FILENAME = "WEIGHTED_REDUCTION_DONE.json"


def _reduce_weighted_partials_for_completed_shards(
    *,
    analysis_folder_path: str,
    manifest: Mapping[str, Any],
    completed_training_shards: Sequence[Tuple[int, str, str, Dict[str, Any]]],
    context: Mapping[str, Any],
) -> List[Tuple[int, str, str, Dict[str, Any]]]:
    """Compute weighted partials directly during the final merge.

    This step reduces each completed shard's saved task archives using the subset
    weights calculated from the completed training shards. This finalization step
    folds that bookkeeping into ``merge_shards`` so the public workflow is simply
    ``plan -> process_shard -> merge_shards``.
    """

    merge_dir = os.path.join(analysis_folder_path, MERGE_DIRNAME)
    weights_done_path = os.path.join(merge_dir, WEIGHTS_DONE_FILENAME)
    if not os.path.exists(weights_done_path):
        raise RuntimeError(
            f"Global weight calculation has not completed; missing required file: {weights_done_path}"
        )

    weights_csv_path = os.path.join(analysis_folder_path, "COMMITTEE_SUBSETS", "SUBSET_WEIGHTS_REAL.csv")
    if not os.path.exists(weights_csv_path):
        raise FileNotFoundError(f"Subset weights CSV not found: {weights_csv_path}")
    subset_weights = load_subset_weights_mapping(weights_csv_path)

    n_train = int(len(context["df_for_training"]))
    n_all = int(len(context["df_for_shap_and_agg"]))
    n_splits = int(manifest.get("n_splits", 0) or 0)
    selected_features = list(context["selected_features"])
    run_permutation_test = bool(manifest.get("run_permutation_test", False))
    n_permutations = int(manifest.get("n_permutations", 0) or 0)

    done_entries: List[Tuple[int, str, str, Dict[str, Any]]] = []
    print(f"\n--- Weighted reduction inside final merge across {len(completed_training_shards)} shard(s) ---")

    for shard_index, shard_label, shard_dir, training_done in completed_training_shards:
        done_path = os.path.join(shard_dir, WEIGHTED_REDUCTION_DONE_FILENAME)
        if os.path.exists(done_path):
            done_payload = _json_load(done_path)
            artifacts = dict(done_payload.get("artifacts") or {}) if isinstance(done_payload, Mapping) else {}
            required = [
                artifacts.get("weighted_real_margin_partial"),
                artifacts.get("weighted_real_shap_partial"),
            ]
            if isinstance(done_payload, Mapping) and str(done_payload.get("status", "")).strip().lower() == "completed" and all(
                rel and os.path.exists(os.path.join(analysis_folder_path, str(rel))) for rel in required
            ):
                print(f"[merge_shards] Reusing weighted reduction for {shard_label}: {done_path}")
                done_entries.append((int(shard_index), str(shard_label), str(shard_dir), dict(done_payload)))
                continue

        print(f"[merge_shards] Reducing weighted artifacts for {shard_label}")
        artifacts = dict(training_done.get("artifacts") or {})
        real_margin_tasks = os.path.join(analysis_folder_path, str(artifacts.get("real_margin_tasks_npz", "")))
        real_shap_tasks = os.path.join(analysis_folder_path, str(artifacts.get("real_shap_tasks_npz", "")))
        real_perf_path = os.path.join(analysis_folder_path, str(artifacts.get("performance_records_real_jsonl", "")))
        perm_perf_path = os.path.join(analysis_folder_path, str(artifacts.get("performance_records_perm_jsonl", "")))
        if not os.path.exists(real_margin_tasks):
            raise FileNotFoundError(f"Real margin task archive not found for shard {shard_label}: {real_margin_tasks}")
        if not os.path.exists(real_shap_tasks):
            raise FileNotFoundError(f"Real SHAP task archive not found for shard {shard_label}: {real_shap_tasks}")
        if not os.path.exists(real_perf_path):
            raise FileNotFoundError(f"Real performance records not found for shard {shard_label}: {real_perf_path}")

        real_perf_rows = _jsonl_load(real_perf_path)
        real_margin_partial, real_margin_diag = reduce_weighted_margin_task_archives_to_partial(
            archive_paths=[real_margin_tasks],
            performance_records=real_perf_rows,
            subset_weights=subset_weights,
            n_train=n_train,
            n_all=n_all,
            n_splits=n_splits,
            include_permuted=False,
            verbose=True,
        )
        weighted_real_margin_path = os.path.join(shard_dir, WEIGHTED_REAL_MARGIN_PARTIAL_FILENAME)
        np.savez(weighted_real_margin_path, **real_margin_partial)

        real_shap_partial, real_shap_diag = reduce_weighted_shap_task_archives_to_partial(
            archive_paths=[real_shap_tasks],
            performance_records=real_perf_rows,
            subset_weights=subset_weights,
            all_feature_names=selected_features,
            n_train=n_train,
            n_all=n_all,
            n_splits=n_splits,
            include_permuted=False,
            require_full=True,
            verbose=True,
        )
        weighted_real_shap_path = os.path.join(shard_dir, WEIGHTED_REAL_SHAP_PARTIAL_FILENAME)
        np.savez(weighted_real_shap_path, **real_shap_partial)

        perm_outputs: Dict[int, str] = {}
        perm_diags: Dict[int, Mapping[str, Any]] = {}
        if run_permutation_test:
            perm_perf_rows = _jsonl_load(perm_perf_path) if os.path.exists(perm_perf_path) else []
            perm_shap_tasks_map = {
                int(k): os.path.join(analysis_folder_path, str(v))
                for k, v in dict(artifacts.get("perm_shap_tasks") or {}).items()
            }
            for perm_idx in range(n_permutations):
                perm_path = perm_shap_tasks_map.get(int(perm_idx))
                if not perm_path or not os.path.exists(perm_path):
                    continue
                perm_partial, perm_diag = reduce_weighted_shap_task_archives_to_partial(
                    archive_paths=[perm_path],
                    performance_records=perm_perf_rows,
                    subset_weights=subset_weights,
                    all_feature_names=selected_features,
                    n_train=n_train,
                    n_all=n_all,
                    n_splits=n_splits,
                    include_permuted=True,
                    permutation_idx=int(perm_idx),
                    require_full=True,
                    verbose=False,
                )
                out_path = os.path.join(
                    shard_dir,
                    WEIGHTED_PERM_SHAP_PARTIAL_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)),
                )
                np.savez(out_path, **perm_partial)
                perm_outputs[int(perm_idx)] = os.path.relpath(out_path, analysis_folder_path)
                perm_diags[int(perm_idx)] = perm_diag

        done_payload = {
            "status": "completed",
            "analysis_folder": analysis_folder_path,
            "shard_index": int(shard_index),
            "shard_label": str(shard_label),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": {
                "weighted_real_margin_partial": os.path.relpath(weighted_real_margin_path, analysis_folder_path),
                "weighted_real_shap_partial": os.path.relpath(weighted_real_shap_path, analysis_folder_path),
                "weighted_perm_shap_partials": perm_outputs,
            },
            "diagnostics": {
                "real_margin": real_margin_diag,
                "real_shap": real_shap_diag,
                "perm_shap": {str(k): dict(v) for k, v in perm_diags.items()},
            },
        }
        _json_dump(done_payload, done_path)
        done_entries.append((int(shard_index), str(shard_label), str(shard_dir), dict(done_payload)))

    done_entries.sort(key=lambda x: x[0])
    return done_entries


def _prefix_perm_prediction_columns(df: pd.DataFrame, perm_idx: int) -> pd.DataFrame:
    keep_cols = ["individual", "group"]
    out = df.copy()
    rename: Dict[str, str] = {}
    for col in out.columns:
        if col in keep_cols:
            continue
        rename[col] = f"Perm{int(perm_idx)}_{col}"
    return out.rename(columns=rename)


def _merge_prediction_views(*, base_df: pd.DataFrame, permuted_dfs: Sequence[Tuple[int, pd.DataFrame]]) -> pd.DataFrame:
    merged = base_df.copy()
    key_cols = ["individual", "group"]
    if not permuted_dfs:
        return merged

    merged["___SEPARATOR___"] = np.nan
    perm_prefixed = [_prefix_perm_prediction_columns(perm_df, int(perm_idx)) for perm_idx, perm_df in permuted_dfs]
    combined_permuted = perm_prefixed[0].copy()
    for perm_df in perm_prefixed[1:]:
        perm_df_no_base = perm_df.drop(columns=key_cols, errors="ignore")
        combined_permuted = pd.concat([combined_permuted, perm_df_no_base], axis=1)

    merged = pd.merge(
        merged,
        combined_permuted,
        on=key_cols,
        how="left",
        suffixes=("", "_perm"),
    )
    return merged.drop(columns=["individual_perm", "group_perm"], errors="ignore")


def finalize_featureimp_outputs_run(analysis_folder_path: str) -> str:
    analysis_folder_path = os.path.abspath(str(analysis_folder_path))
    if not os.path.isdir(analysis_folder_path):
        raise FileNotFoundError(f"FeatureImp analysis folder not found: {analysis_folder_path}")

    manifest = load_featureimp_plan_manifest(analysis_folder_path)
    merge_dir = os.path.join(analysis_folder_path, MERGE_DIRNAME)
    os.makedirs(merge_dir, exist_ok=True)
    done_path = os.path.join(merge_dir, FINAL_MERGE_DONE_FILENAME)
    if os.path.exists(done_path):
        print(f"Core merge already completed successfully; reusing existing outputs: {done_path}")
        return analysis_folder_path

    running_path = os.path.join(merge_dir, FINAL_MERGE_RUNNING_FILENAME)
    failed_path = os.path.join(merge_dir, FINAL_MERGE_FAILED_FILENAME)
    lock_dir = os.path.join(merge_dir, FINAL_MERGE_LOCK_DIRNAME)
    if os.path.exists(failed_path):
        try:
            os.remove(failed_path)
        except Exception:
            pass

    _json_dump(
        {
            "status": "running",
            "analysis_folder": analysis_folder_path,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
        running_path,
    )

    try:
        with _directory_lock(lock_dir):
            settings_snapshot = _load_settings_snapshot(analysis_folder_path)
            completed_training_shards, training_shard_errors = _discover_completed_shards(analysis_folder_path, manifest)
            if training_shard_errors:
                raise RuntimeError(
                    "Cannot finalize FeatureImp because shard validation failed:\n- "
                    + "\n- ".join(training_shard_errors)
                )

            context = _load_featureimp_context(analysis_folder_path, manifest)
            completed_weighted_reductions = _reduce_weighted_partials_for_completed_shards(
                analysis_folder_path=analysis_folder_path,
                manifest=manifest,
                completed_training_shards=completed_training_shards,
                context=context,
            )
            df_for_training = context["df_for_training"].reset_index(drop=True)
            df_for_shap_and_agg = context["df_for_shap_and_agg"].reset_index(drop=True)
            X_train_full = context["X_train_full"].reset_index(drop=True)
            y_encoded_train = np.asarray(context["y_encoded_train"], dtype=np.int32)
            le: FixedBinaryLabelEncoder = context["le"]
            num_classes = int(context["num_classes"])
            selected_features = list(context["selected_features"])
            n_train = int(len(df_for_training))
            n_splits = int(manifest.get("n_splits", 0) or 0)
            n_all = int(len(df_for_shap_and_agg))
            H_hits_per_feature = int(manifest.get("H_hits_per_feature", 0) or 0)
            n_features_per_model = int(manifest.get("n_features_per_model", 0) or 0)
            run_permutation_test = bool(manifest.get("run_permutation_test", False))
            n_permutations = int(manifest.get("n_permutations", 0) or 0)
            train_groups_list = [str(manifest["training_groups"][0]), str(manifest["training_groups"][1])]

            # Primary weighted shard partials.
            weighted_real_margin_paths: List[str] = []
            weighted_real_shap_paths: List[str] = []
            weighted_perm_shap_paths: Dict[int, List[str]] = {int(p): [] for p in range(n_permutations)}
            for _, _, _, done in completed_weighted_reductions:
                artifacts = dict(done.get("artifacts") or {})
                weighted_real_margin_paths.append(os.path.join(analysis_folder_path, str(artifacts["weighted_real_margin_partial"])))
                weighted_real_shap_paths.append(os.path.join(analysis_folder_path, str(artifacts["weighted_real_shap_partial"])))
                for perm_key, rel_path in dict(artifacts.get("weighted_perm_shap_partials") or {}).items():
                    weighted_perm_shap_paths.setdefault(int(perm_key), []).append(os.path.join(analysis_folder_path, str(rel_path)))

            # Training-shard archives still matter for committee stability and permuted weighted predictions.
            real_shap_task_paths: List[str] = []
            perm_margin_task_paths: Dict[int, List[str]] = {int(p): [] for p in range(n_permutations)}
            for _, _, _, done in completed_training_shards:
                artifacts = dict(done.get("artifacts") or {})
                real_shap_task_rel = artifacts.get("real_shap_tasks_npz")
                if real_shap_task_rel:
                    real_shap_task_paths.append(os.path.join(analysis_folder_path, str(real_shap_task_rel)))
                for perm_key, rel_path in dict(artifacts.get("perm_margin_tasks") or {}).items():
                    perm_margin_task_paths.setdefault(int(perm_key), []).append(os.path.join(analysis_folder_path, str(rel_path)))

            weighted_margin_partial = merge_weighted_margin_partials(weighted_real_margin_paths)
            logit_preds_real = finalize_weighted_margin_partial(partial=weighted_margin_partial, num_classes=num_classes)
            insample_probas = np.asarray(logit_preds_real.insample_probas, dtype=np.float32)
            oof_probas = np.asarray(logit_preds_real.oof_probas, dtype=np.float32)
            all_groups_avg_probas = np.asarray(logit_preds_real.all_groups_probas, dtype=np.float32)

            weighted_shap_partial = merge_weighted_shap_partials(weighted_real_shap_paths, require_full=True)
            oof_shap_values, insample_shap_values, all_groups_avg_shap_values, feature_weight_sums, weighted_shap_diag = finalize_weighted_shap_partial(
                partial=weighted_shap_partial,
                n_splits=n_splits,
                require_full=True,
            )
            assert insample_shap_values is not None
            assert all_groups_avg_shap_values is not None

            subset_weights_path = os.path.join(analysis_folder_path, "COMMITTEE_SUBSETS", "SUBSET_WEIGHTS_REAL.csv")
            subset_weights_map: Dict[Tuple[int, int], float] = {}
            if os.path.exists(subset_weights_path):
                subset_weights_map = load_subset_weights_mapping(subset_weights_path)
            else:
                subset_weights_path = None
            weighting_manifest_path = os.path.join(analysis_folder_path, "COMMITTEE_SUBSETS", "WEIGHTING_MANIFEST.json")
            if not os.path.exists(weighting_manifest_path):
                weighting_manifest_path = None

            merged_real_partial_path = os.path.join(merge_dir, "MERGED_REAL_PARTIALS.npz")
            insample_probas_unweighted = insample_probas.copy()
            oof_probas_unweighted = oof_probas.copy()
            all_groups_probas_unweighted = all_groups_avg_probas.copy()
            oof_shap_values_unweighted: Optional[np.ndarray] = None
            all_groups_avg_shap_values_unweighted: Optional[np.ndarray] = None
            weighted_feature_usage_fraction: Optional[np.ndarray] = None
            global_importance_overrides = None
            if os.path.exists(merged_real_partial_path):
                real_partial = _load_partial_npz(merged_real_partial_path)
                real_final = _finalize_partial_averages(partial=real_partial, n_splits=n_splits)
                insample_probas_unweighted = np.asarray(real_final["insample_probas"], dtype=np.float32)
                oof_probas_unweighted = np.asarray(real_final["oof_probas"], dtype=np.float32)
                all_groups_probas_unweighted = np.asarray(real_final["all_groups_probas"], dtype=np.float32)
                oof_shap_values_unweighted = np.asarray(real_final["oof_shap"], dtype=np.float32)
                all_groups_avg_shap_values_unweighted = np.asarray(real_final["all_groups_shap"], dtype=np.float32)

            gi_weighted_raw = np.mean(np.abs(oof_shap_values), axis=0).astype(float)
            gi_unweighted = (
                np.mean(np.abs(oof_shap_values_unweighted), axis=0).astype(float)
                if oof_shap_values_unweighted is not None
                else gi_weighted_raw.copy()
            )
            perm_baseline = np.zeros_like(gi_weighted_raw, dtype=float)
            perm_baseline_mode = "none"
            if run_permutation_test:
                perm_importances: List[np.ndarray] = []
                for _, paths in sorted(weighted_perm_shap_paths.items()):
                    if not paths:
                        continue
                    perm_partial = merge_weighted_shap_partials(paths, require_full=False)
                    perm_oof, _, _, _, _ = finalize_weighted_shap_partial(
                        partial=perm_partial,
                        n_splits=n_splits,
                        require_full=False,
                    )
                    perm_importances.append(np.mean(np.abs(perm_oof), axis=0).astype(float))
                if perm_importances:
                    perm_baseline = np.mean(np.stack(perm_importances, axis=0), axis=0)
                    perm_baseline_mode = "weighted_from_weighted_partials"
            gi_weighted_excess = np.clip(gi_weighted_raw - perm_baseline, a_min=0.0, a_max=None)
            global_importance_overrides = {
                "Global_Importance": gi_weighted_excess,
                "Global_Importance_Unweighted": gi_unweighted,
            }
            total_weight_sum = float(weighted_shap_diag.get("total_weight_sum", 0.0) or 0.0)
            if total_weight_sum > 0:
                weighted_feature_usage_fraction = feature_weight_sums.astype(float) / total_weight_sum
                weighted_shap_diag["feature_usage_fraction_min"] = float(np.min(weighted_feature_usage_fraction))
                weighted_shap_diag["feature_usage_fraction_max"] = float(np.max(weighted_feature_usage_fraction))
                weighted_shap_diag["feature_usage_fraction_mean"] = float(np.mean(weighted_feature_usage_fraction))

            committee_subset_dir = os.path.join(analysis_folder_path, "COMMITTEE_SUBSETS")
            os.makedirs(committee_subset_dir, exist_ok=True)
            try:
                with open(os.path.join(committee_subset_dir, "SHAP_WEIGHTING_MANIFEST.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "weighted_shap": weighted_shap_diag,
                            "n_permutations": int(n_permutations),
                            "permutation_baseline_used": bool(run_permutation_test and perm_baseline_mode != "none"),
                            "permutation_baseline_mode": perm_baseline_mode,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception:
                pass

            # Core analysis paths and optional summaries derived from the completed core outputs.
            pipeline_folder = os.path.join(analysis_folder_path, "SAVED_PIPELINE")
            shap_values_folder = os.path.join(pipeline_folder, "temp_shap_values")
            models_folder = os.path.join(pipeline_folder, "models")
            os.makedirs(shap_values_folder, exist_ok=True)
            paths = {
                "analysis_folder": analysis_folder_path,
                "pipeline_folder": pipeline_folder,
                "shap_values_folder": shap_values_folder,
                "models_folder": models_folder,
                "file_suffix": "",
                "permuted_base_folder": os.path.join(analysis_folder_path, "PERMUTED_RESULTS") if run_permutation_test else None,
            }

            print("\n--- Generating SHAP summaries for in-sample vs out-of-sample (staged final merge) ---")
            outofsample_result = _generate_kfold_shap_summaries(
                shap_values=oof_shap_values,
                df=df_for_training,
                selected_features=selected_features,
                le=le,
                paths={"analysis_folder": analysis_folder_path, "file_suffix": ""},
                suffix="outofsample",
                train_groups=train_groups_list,
            )
            insample_result = _generate_kfold_shap_summaries(
                shap_values=insample_shap_values,
                df=df_for_training,
                selected_features=selected_features,
                le=le,
                paths={"analysis_folder": analysis_folder_path, "file_suffix": ""},
                suffix="insample",
                train_groups=train_groups_list,
            )
            if outofsample_result is not None and insample_result is not None:
                outofsample_sheets, outofsample_df = outofsample_result
                insample_sheets, insample_df = insample_result
                if outofsample_sheets is not None and insample_sheets is not None:
                    combined_individual_sheets: Dict[str, pd.DataFrame] = {}
                    for sheet_name, sheet_df in outofsample_sheets.items():
                        combined_individual_sheets[_sanitize_excel_sheet_name(f"OutOfSample_{sheet_name}")] = sheet_df
                    for sheet_name, sheet_df in insample_sheets.items():
                        combined_individual_sheets[_sanitize_excel_sheet_name(f"InSample_{sheet_name}")] = sheet_df
                    individual_path = os.path.join(analysis_folder_path, "SHAP_PER_INDIVIDUAL_SUMMARY.xlsx")
                    save_to_excel_sheets(combined_individual_sheets, individual_path)
                    print(f"Per-individual SHAP summary saved to {individual_path}")
                if outofsample_df is not None and insample_df is not None:
                    _combine_insample_outofsample_summaries(
                        insample_df,
                        outofsample_df,
                        {"analysis_folder": analysis_folder_path, "file_suffix": ""},
                    )

            all_shap_file_paths, extra_consolidated_shap_views, hybrid_shap_values, extra_consolidated_shap_arrays = _build_hybrid_shap_views(
                paths=paths,
                n_train=n_train,
                oof_shap_values=oof_shap_values,
                all_groups_avg_shap_values=all_groups_avg_shap_values,
                oof_shap_values_unweighted=oof_shap_values_unweighted,
                all_groups_avg_shap_values_unweighted=all_groups_avg_shap_values_unweighted,
                weighted_feature_usage_fraction=weighted_feature_usage_fraction,
            )

            real_perf_rows = _jsonl_load(os.path.join(merge_dir, MERGED_PERFORMANCE_REAL_FILENAME))
            perm_perf_rows = _jsonl_load(os.path.join(merge_dir, MERGED_PERFORMANCE_PERM_FILENAME))
            if real_perf_rows and all_shap_file_paths:
                _aggregate_and_summarize_results(
                    df=df_for_shap_and_agg,
                    selected_features=selected_features,
                    performance_results=real_perf_rows,
                    all_shap_values=all_shap_file_paths,
                    le=le,
                    paths=paths,
                    global_importance_overrides=global_importance_overrides,
                    extra_consolidated_shap_views=extra_consolidated_shap_views,
                    avg_shap_values_override=hybrid_shap_values,
                    extra_consolidated_shap_arrays=extra_consolidated_shap_arrays,
                )
                try:
                    _generate_male_female_comparison(
                        df=df_for_shap_and_agg,
                        selected_features=selected_features,
                        all_shap_file_paths=all_shap_file_paths,
                        le=le,
                        paths=paths,
                        avg_shap_values_override=hybrid_shap_values,
                    )
                except Exception as e:
                    print(f"WARNING: Male/female SHAP comparison failed during staged final merge: {e}")

            # Combined performance details payload.
            perf_path = os.path.join(analysis_folder_path, "BOOTSTRAP_PERFORMANCE_DETAILS.json")
            payload = {
                "main_models": real_perf_rows,
                "permuted_models": perm_perf_rows,
            }
            with open(perf_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

            # Predictions export (weighted primary, unweighted comparison), with optional per-permutation views.
            _, _, all_predictions_weighted = _build_predictions_export(
                df_for_training=df_for_training,
                df_for_shap_and_agg=df_for_shap_and_agg,
                training_insample=insample_probas,
                training_oof=oof_probas,
                all_groups_probas=all_groups_avg_probas,
                le=le,
                export_label="weighted",
            )
            _, _, all_predictions_unweighted = _build_predictions_export(
                df_for_training=df_for_training,
                df_for_shap_and_agg=df_for_shap_and_agg,
                training_insample=insample_probas_unweighted,
                training_oof=oof_probas_unweighted,
                all_groups_probas=all_groups_probas_unweighted,
                le=le,
                export_label="unweighted",
            )

            perm_weighted_dfs: List[Tuple[int, pd.DataFrame]] = []
            perm_unweighted_dfs: List[Tuple[int, pd.DataFrame]] = []

            if run_permutation_test:
                for perm_idx in range(n_permutations):
                    weighted_perm_paths = list(weighted_perm_shap_paths.get(int(perm_idx), []) or [])
                    perm_perf = [row for row in perm_perf_rows if int(row.get("permutation_idx", -1)) == int(perm_idx)]
                    
                    perm_insample_shap = None
                    perm_oof_shap = None
                    perm_all_shap = None
                    perm_insample_probas = None
                    perm_oof_probas = None
                    perm_all_probas = None

                    if weighted_perm_paths:
                        perm_partial = merge_weighted_shap_partials(weighted_perm_paths, require_full=True)
                        perm_oof_shap, perm_insample_shap, perm_all_shap, _, _ = finalize_weighted_shap_partial(
                            partial=perm_partial,
                            n_splits=n_splits,
                            require_full=True,
                        )
                    else:
                        merged_perm_partial_path = os.path.join(merge_dir, MERGED_PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)))
                        if not os.path.exists(merged_perm_partial_path):
                            continue
                        perm_summary = _finalize_partial_averages(partial=_load_partial_npz(merged_perm_partial_path), n_splits=n_splits)
                        perm_insample_shap = np.asarray(perm_summary["insample_shap"], dtype=np.float32)
                        perm_oof_shap = np.asarray(perm_summary["oof_shap"], dtype=np.float32)
                        perm_all_shap = np.asarray(perm_summary["all_groups_shap"], dtype=np.float32)
                        perm_insample_probas = np.asarray(perm_summary["insample_probas"], dtype=np.float32)
                        perm_oof_probas = np.asarray(perm_summary["oof_probas"], dtype=np.float32)
                        perm_all_probas = np.asarray(perm_summary["all_groups_probas"], dtype=np.float32)

                    _aggregate_and_summarize_permuted_results(
                        df=df_for_shap_and_agg,
                        selected_features=selected_features,
                        all_permuted_performance=perm_perf,
                        all_permuted_shap_values=[],
                        le=le,
                        paths=paths,
                        permutation_idx=int(perm_idx),
                        permuted_insample_shap=np.asarray(perm_insample_shap, dtype=np.float32),
                        permuted_oof_shap=np.asarray(perm_oof_shap, dtype=np.float32),
                        df_for_training=df_for_training,
                        train_groups_list=train_groups_list,
                        n_splits=n_splits,
                        permuted_all_groups_avg_probas=None if perm_all_probas is None else np.asarray(perm_all_probas, dtype=np.float32),
                        permuted_all_groups_avg_shap_values=None if perm_all_shap is None else np.asarray(perm_all_shap, dtype=np.float32),
                        df_for_shap_and_agg=df_for_shap_and_agg,
                        permuted_insample_probas=None if perm_insample_probas is None else np.asarray(perm_insample_probas, dtype=np.float32),
                        permuted_oof_probas=None if perm_oof_probas is None else np.asarray(perm_oof_probas, dtype=np.float32),
                        all_fold_indices=None,
                    )

                    perm_margin_archives = perm_margin_task_paths.get(int(perm_idx), [])
                    if perm_margin_archives:
                        perm_logit_preds = compute_logit_ensemble_predictions_from_margin_task_archives(
                            archive_paths=perm_margin_archives,
                            performance_records=perm_perf,
                            n_train=n_train,
                            n_all=n_all,
                            num_classes=num_classes,
                            n_splits=n_splits,
                            subset_weights=subset_weights_map,
                            include_permuted=True,
                            permutation_idx=int(perm_idx),
                            verbose=False,
                        )

                        perm_ins_w = (
                            np.asarray(perm_logit_preds.insample_probas_weighted, dtype=np.float32)
                            if getattr(perm_logit_preds, "insample_probas_weighted", None) is not None
                            else np.asarray(perm_logit_preds.insample_probas, dtype=np.float32)
                        )
                        perm_oof_w = (
                            np.asarray(perm_logit_preds.oof_probas_weighted, dtype=np.float32)
                            if getattr(perm_logit_preds, "oof_probas_weighted", None) is not None
                            else np.asarray(perm_logit_preds.oof_probas, dtype=np.float32)
                        )
                        perm_all_w = (
                            np.asarray(perm_logit_preds.all_groups_probas_weighted, dtype=np.float32)
                            if getattr(perm_logit_preds, "all_groups_probas_weighted", None) is not None
                            else np.asarray(perm_logit_preds.all_groups_probas, dtype=np.float32)
                        )
                        _, _, perm_df_weighted = _build_predictions_export(
                            df_for_training=df_for_training,
                            df_for_shap_and_agg=df_for_shap_and_agg,
                            training_insample=perm_ins_w,
                            training_oof=perm_oof_w,
                            all_groups_probas=perm_all_w,
                            le=le,
                            export_label=f"perm{int(perm_idx)}_weighted",
                        )
                        _, _, perm_df_unweighted = _build_predictions_export(
                            df_for_training=df_for_training,
                            df_for_shap_and_agg=df_for_shap_and_agg,
                            training_insample=np.asarray(perm_logit_preds.insample_probas, dtype=np.float32),
                            training_oof=np.asarray(perm_logit_preds.oof_probas, dtype=np.float32),
                            all_groups_probas=np.asarray(perm_logit_preds.all_groups_probas, dtype=np.float32),
                            le=le,
                            export_label=f"perm{int(perm_idx)}_unweighted",
                        )
                        perm_weighted_dfs.append((int(perm_idx), perm_df_weighted))
                        perm_unweighted_dfs.append((int(perm_idx), perm_df_unweighted))

                        perm_pred_path = os.path.join(
                            analysis_folder_path,
                            f"PERMUTED_{int(perm_idx)}_RESULTS",
                            f"AVERAGED_PREDICTIONS_PERMUTED_{int(perm_idx)}.xlsx",
                        )
                        save_predictions_with_formatting(
                            perm_df_weighted,
                            perm_pred_path,
                            comparison_df=perm_df_unweighted,
                            primary_sheet_name="WeightedPredictions",
                            comparison_sheet_name="UnweightedPredictions",
                        )
                        print(f"Permuted predictions workbook saved to {perm_pred_path}")

            final_weighted_df = _merge_prediction_views(base_df=all_predictions_weighted, permuted_dfs=perm_weighted_dfs)
            final_unweighted_df = _merge_prediction_views(base_df=all_predictions_unweighted, permuted_dfs=perm_unweighted_dfs)

            all_predictions_path = os.path.join(analysis_folder_path, "AVERAGED_PREDICTIONS_FULL.xlsx")
            save_predictions_with_formatting(
                final_weighted_df,
                all_predictions_path,
                comparison_df=final_unweighted_df,
                primary_sheet_name="WeightedPredictions",
                comparison_sheet_name="UnweightedPredictions",
            )
            try:
                create_formatted_predictions_for_graphpad(
                    all_predictions_df=final_weighted_df,
                    training_group_a=str(manifest["training_groups"][0]),
                    training_group_b=str(manifest["training_groups"][1]),
                    analysis_folder=analysis_folder_path,
                    file_suffix="",
                    perm_idx=0,
                )
            except Exception as e:
                print(f"WARNING: Could not create formatted predictions for GraphPad: {e}")

            _write_ensemble_metrics(
                analysis_folder=analysis_folder_path,
                y_true=y_encoded_train,
                insample_probas=insample_probas,
                oof_probas=oof_probas,
                num_classes=num_classes,
                le=le,
                H_hits_per_feature=H_hits_per_feature,
                aggregated_performance_results=real_perf_rows,
            )
            committee_pooled_performance_path = _write_committee_pooled_performance(
                analysis_folder=analysis_folder_path,
                y_true=y_encoded_train,
                insample_probas=insample_probas,
                oof_probas=oof_probas,
                H_hits_per_feature=H_hits_per_feature,
                aggregated_performance_results=real_perf_rows,
            )

            group_label_mapping_path = os.path.join(pipeline_folder, "GROUP_LABEL_MAPPING.json")
            if not os.path.exists(group_label_mapping_path):
                group_label_mapping_path = None
            normalization_params_path = os.path.join(analysis_folder_path, "NORMALIZATION_PARAMS.json")
            if not os.path.exists(normalization_params_path):
                normalization_params_path = None
            from . import core as _core

            current_runtime_settings = dict(_core.get_current_settings() or {})
            reference_settings = {
                "analysis_run_name": settings_snapshot.get("analysis_run_name", manifest.get("analysis_folder_name", "")),
                "control_group_tag": settings_snapshot.get("control_group_tag"),
                "training_group_a": manifest["training_groups"][0],
                "training_group_b": manifest["training_groups"][1],
                "include_feature_categories": settings_snapshot.get("include_feature_categories"),
                "sex_filter": manifest.get("sex_filter", "all"),
                "n_splits": int(n_splits),
                "H_hits_per_feature": int(H_hits_per_feature),
                "n_features_per_model": int(n_features_per_model),
                "num_workers": int(current_runtime_settings.get("num_workers", DEFAULT_SETTINGS.get("num_workers", 1)) or DEFAULT_SETTINGS.get("num_workers", 1) or 1),
                "n_estimators": int(settings_snapshot.get("n_estimators", 200) or 200),
                "learning_rate": float(settings_snapshot.get("learning_rate", 0.1) or 0.1),
                "max_depth": int(settings_snapshot.get("max_depth", 6) or 6),
                "run_permutation_test": bool(run_permutation_test),
                "n_permutations": int(n_permutations),
                "min_feature_completeness_per_group": float(settings_snapshot.get("min_feature_completeness_per_group", 0.25) or 0.25),
                "min_variance_threshold": float(settings_snapshot.get("MIN_VARIANCE_THRESHOLD", 1e-6) or 1e-6),
                "GLOBAL_RANDOM_SEED": int(manifest.get("resolved_global_random_seed", 0) or 0),
                "stage": "merge_shards",
            }
            write_featureimp_reference_bundle(
                analysis_folder=analysis_folder_path,
                normalized_matrix_path=str(manifest.get("normalized_matrix_path") or ""),
                selected_feature_columns=selected_features,
                id_columns=["individual", "group", "sex", "source_files", "total_frames_analyzed"],
                control_group_tag=settings_snapshot.get("control_group_tag"),
                training_group_a=manifest["training_groups"][0],
                training_group_b=manifest["training_groups"][1],
                settings=reference_settings,
                performance_records=list(real_perf_rows) + list(perm_perf_rows),
                subset_weights_path=subset_weights_path,
                weighting_manifest_path=weighting_manifest_path,
                group_label_mapping_path=group_label_mapping_path,
                normalization_params_path=normalization_params_path,
            )

            final_bundle_path = os.path.join(merge_dir, FINAL_POSTANALYSIS_BUNDLE_FILENAME)
            _json_dump(
                {
                    "status": "completed",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "analysis_folder": analysis_folder_path,
                    "hybrid_shap_mmap": _safe_relpath(all_shap_file_paths[0], analysis_folder_path) if all_shap_file_paths else None,
                    "n_train": int(n_train),
                    "n_all": int(hybrid_shap_values.shape[0]),
                    "n_features": int(hybrid_shap_values.shape[1]),
                    "selected_feature_count": int(len(selected_features)),
                    "training_groups": [str(manifest["training_groups"][0]), str(manifest["training_groups"][1])],
                },
                final_bundle_path,
            )

            done_payload = {
                "status": "completed",
                "analysis_folder": analysis_folder_path,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "real_weighted_reduction_count": int(len(completed_weighted_reductions)),
                "real_model_count": int(len(real_perf_rows)),
                "perm_model_count": int(len(perm_perf_rows)),
                "artifacts": {
                    "shap_consolidated_summary_xlsx": _safe_relpath(os.path.join(analysis_folder_path, "SHAP_CONSOLIDATED_SUMMARY.xlsx"), analysis_folder_path),
                    "shap_consolidated_summary_insample_vs_outofsample_xlsx": _safe_relpath(os.path.join(analysis_folder_path, "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx"), analysis_folder_path),
                    "shap_per_individual_summary_xlsx": _safe_relpath(os.path.join(analysis_folder_path, "SHAP_PER_INDIVIDUAL_SUMMARY.xlsx"), analysis_folder_path),
                    "shap_male_vs_female_xlsx": _safe_relpath(os.path.join(analysis_folder_path, "SHAP_CONSOLIDATED_SUMMARY_MALE_VS_FEMALE.xlsx"), analysis_folder_path),
                    "shap_weighting_manifest_json": _safe_relpath(os.path.join(committee_subset_dir, "SHAP_WEIGHTING_MANIFEST.json"), analysis_folder_path),
                    "global_importance_ranking_csv": _safe_relpath(os.path.join(analysis_folder_path, "GLOBAL_IMPORTANCE_RANKING.csv"), analysis_folder_path),
                    "averaged_predictions_xlsx": _safe_relpath(os.path.join(analysis_folder_path, "AVERAGED_PREDICTIONS_FULL.xlsx"), analysis_folder_path),
                    "committee_pooled_performance_csv": _safe_relpath(committee_pooled_performance_path, analysis_folder_path),
                    "weights_done_json": _safe_relpath(os.path.join(merge_dir, "WEIGHTS_DONE.json"), analysis_folder_path),
                    "final_postanalysis_bundle_json": _safe_relpath(final_bundle_path, analysis_folder_path),
                },
            }
            _json_dump(done_payload, done_path)
            try:
                os.remove(running_path)
            except Exception:
                pass
            return analysis_folder_path
    except Exception as e:
        _json_dump(
            {
                "status": "failed",
                "analysis_folder": analysis_folder_path,
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            failed_path,
        )
        try:
            os.remove(running_path)
        except Exception:
            pass
        raise


__all__ = [
    "FINAL_MERGE_RUNNING_FILENAME",
    "FINAL_MERGE_DONE_FILENAME",
    "FINAL_MERGE_FAILED_FILENAME",
    "FINAL_MERGE_LOCK_DIRNAME",
    "finalize_featureimp_outputs_run",
]
