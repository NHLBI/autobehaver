from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _make_fold_plan_records(labels: np.ndarray, *, n_splits: int, seeds: list[int]) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    X_dummy = np.zeros((len(labels), 1), dtype=np.float32)
    for h_hit, seed in enumerate(seeds, start=1):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        records[int(h_hit)] = {
            "h_hit": int(h_hit),
            "split_seed": int(seed),
            "folds": [
                {
                    "fold_idx": int(fold_idx),
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                }
                for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_dummy, labels))
            ],
        }
    return records


def test_plain_xgboost_baseline_writes_expected_outputs(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp.plain_xgboost_baseline import run_plain_xgboost_baseline

    rows = []
    rng = np.random.default_rng(23)
    for i in range(24):
        group = "A" if i < 12 else "B"
        sex = "M" if i % 2 == 0 else "F"
        signal = (-2.25 if group == "A" else 2.25) + float(rng.normal(0.0, 0.25))
        rows.append({
            "individual": f"train_{i}",
            "group": group,
            "sex": sex,
            "Signal_feat": signal,
            "Noise_feat1": float(rng.normal(0.0, 1.0)),
            "Noise_feat2": float(rng.normal(0.0, 1.0)),
        })
    for i in range(4):
        rows.append({
            "individual": f"other_{i}",
            "group": "C",
            "sex": "M" if i % 2 == 0 else "F",
            "Signal_feat": float(rng.normal(0.0, 0.5)),
            "Noise_feat1": float(rng.normal(0.0, 1.0)),
            "Noise_feat2": float(rng.normal(0.0, 1.0)),
        })

    df_all = pd.DataFrame(rows)
    df_train = df_all.loc[df_all["group"].isin(["A", "B"])].reset_index(drop=True).copy()
    X_train = df_train[["Signal_feat", "Noise_feat1", "Noise_feat2"]].copy()
    X_all = df_all[["Signal_feat", "Noise_feat1", "Noise_feat2"]].copy()
    stratify = (df_train["sex"].astype(str) + "_" + df_train["group"].astype(str)).to_numpy(copy=False)
    fold_plan_records = _make_fold_plan_records(stratify, n_splits=2, seeds=[101, 202])

    out_dir = run_plain_xgboost_baseline(
        X_train_full=X_train,
        df_for_training=df_train,
        training_group_a="A",
        training_group_b="B",
        analysis_folder_path=str(tmp_path),
        n_splits=2,
        random_state=23,
        df_for_shap_and_agg=df_all,
        X_shap_full=X_all,
        settings_snapshot={"n_estimators": 8, "learning_rate": 0.2, "max_depth": 2, "model_subsample": 1.0},
        fold_plan_records=fold_plan_records,
    )

    out_path = Path(out_dir)
    assert (out_path / "PLAIN_XGBOOST_FOLD_PERFORMANCE.csv").exists()
    assert (out_path / "PLAIN_XGBOOST_REPEAT_PERFORMANCE.csv").exists()
    assert (out_path / "PLAIN_XGBOOST_POOLED_PERFORMANCE.csv").exists()
    assert (out_path / "PLAIN_XGBOOST_CONFIG.json").exists()
    assert (out_path / "AVERAGED_PREDICTIONS_FULL.xlsx").exists()
    assert (out_path / "AVERAGED_PREDICTIONS_FORMATTED_FOR_GRAPHPAD.xlsx").exists()
    assert (out_path / "SHAP_CONSOLIDATED_SUMMARY.xlsx").exists()
    assert (out_path / "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx").exists()

    perf_df = pd.read_csv(out_path / "PLAIN_XGBOOST_FOLD_PERFORMANCE.csv")
    assert len(perf_df) == 4
    assert perf_df["balanced_accuracy"].mean() > 0.75

    repeat_df = pd.read_csv(out_path / "PLAIN_XGBOOST_REPEAT_PERFORMANCE.csv")
    assert len(repeat_df) == 2
    assert repeat_df["oof_balanced_accuracy"].mean() > 0.75

    pooled_df = pd.read_csv(out_path / "PLAIN_XGBOOST_POOLED_PERFORMANCE.csv")
    assert int(pooled_df.loc[0, "n_repeats"]) == 2
    assert int(pooled_df.loc[0, "n_outer_models"]) == 4
    assert int(pooled_df.loc[0, "oof_prediction_count_per_training_row"]) == 2
    assert int(pooled_df.loc[0, "insample_prediction_count_per_training_row"]) == 2

    preds_df = pd.read_excel(out_path / "AVERAGED_PREDICTIONS_FULL.xlsx")
    c_rows = preds_df.loc[preds_df["group"] == "C"].reset_index(drop=True)
    assert len(c_rows) == 4
    assert c_rows["Prob_A_insample"].isna().all()
    assert c_rows["Prob_A_outofsample"].isna().all()
    assert c_rows["Prob_A"].notna().all()
    assert c_rows["Prob_B"].notna().all()

    shap_df = pd.read_excel(out_path / "SHAP_CONSOLIDATED_SUMMARY.xlsx", header=[0, 1])
    feature_col = next(col for col in shap_df.columns if str(col[1]).strip() == "feature_name")
    assert str(shap_df.iloc[0][feature_col]) == "Signal_feat"

    combined_df = pd.read_excel(out_path / "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx", header=[0, 1])
    combined_columns = {(str(col[0]).strip(), str(col[1]).strip()) for col in combined_df.columns}
    assert ("C (POOLED)", "MeanSHAP") in combined_columns
    assert ("C (OUT-OF-SAMPLE)", "MeanSHAP") not in combined_columns



def test_plain_xgboost_baseline_restores_comparison_outputs(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp.plain_xgboost_baseline import run_plain_xgboost_baseline

    rows = []
    rng = np.random.default_rng(41)
    for i in range(20):
        group = "A" if i < 10 else "B"
        sex = "M" if i % 2 == 0 else "F"
        signal = (-2.0 if group == "A" else 2.0) + float(rng.normal(0.0, 0.3))
        rows.append({
            "individual": f"train2_{i}",
            "group": group,
            "sex": sex,
            "Signal_feat": signal,
            "Noise_feat1": float(rng.normal(0.0, 1.0)),
            "Noise_feat2": float(rng.normal(0.0, 1.0)),
        })
    for i in range(4):
        rows.append({
            "individual": f"other2_{i}",
            "group": "C",
            "sex": "M" if i % 2 == 0 else "F",
            "Signal_feat": float(rng.normal(0.0, 0.5)),
            "Noise_feat1": float(rng.normal(0.0, 1.0)),
            "Noise_feat2": float(rng.normal(0.0, 1.0)),
        })

    df_all = pd.DataFrame(rows)
    df_train = df_all.loc[df_all["group"].isin(["A", "B"])].reset_index(drop=True).copy()
    X_train = df_train[["Signal_feat", "Noise_feat1", "Noise_feat2"]].copy()
    X_all = df_all[["Signal_feat", "Noise_feat1", "Noise_feat2"]].copy()
    stratify = (df_train["sex"].astype(str) + "_" + df_train["group"].astype(str)).to_numpy(copy=False)
    fold_plan_records = _make_fold_plan_records(stratify, n_splits=2, seeds=[111, 222])

    out_dir = run_plain_xgboost_baseline(
        X_train_full=X_train,
        df_for_training=df_train,
        training_group_a="A",
        training_group_b="B",
        analysis_folder_path=str(tmp_path),
        n_splits=2,
        random_state=41,
        df_for_shap_and_agg=df_all,
        X_shap_full=X_all,
        settings_snapshot={
            "n_estimators": 6,
            "learning_rate": 0.2,
            "max_depth": 2,
            "model_subsample": 1.0,
            "run_permutation_test": True,
            "n_permutations": 1,
        },
        fold_plan_records=fold_plan_records,
    )

    out_path = Path(out_dir)
    assert (out_path / "GLOBAL_IMPORTANCE_RANKING.csv").exists()
    assert (out_path / "SHAP_FEATURE_STABILITY_ACROSS_REPEATS.csv").exists()
    assert (out_path / "SHAP_PER_INDIVIDUAL_SUMMARY.xlsx").exists()
    assert (out_path / "SHAP_CONSOLIDATED_SUMMARY_MALE_VS_FEMALE.xlsx").exists()
    assert (out_path / "PLAIN_XGBOOST_SHAP_MANIFEST.json").exists()
    assert (out_path / "PERMUTED_0_RESULTS" / "SHAP_CONSOLIDATED_SUMMARY_PERMUTED_0.xlsx").exists()

    ranking_df = pd.read_csv(out_path / "GLOBAL_IMPORTANCE_RANKING.csv")
    assert {"Global_Importance", "Global_Importance_Raw", "Global_Importance_PermBaseline"}.issubset(ranking_df.columns)
    assert str(ranking_df.iloc[0]["feature_name"]) == "Signal_feat"

    stability_df = pd.read_csv(out_path / "SHAP_FEATURE_STABILITY_ACROSS_REPEATS.csv")
    assert {"AcrossRepeats_GlobalMeanAbsSHAP", "AcrossRepeats_GlobalStdAbsSHAP", "AcrossRepeats_GlobalStabilityAbsSHAP"}.issubset(stability_df.columns)
    assert "Signal_feat" in set(stability_df["feature_name"].astype(str))



def test_plain_xgboost_global_importance_cumulative_matches_sorted_ranking(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp.plain_xgboost_baseline import run_plain_xgboost_baseline

    rows = []
    rng = np.random.default_rng(77)
    for i in range(24):
        group = "A" if i < 12 else "B"
        sex = "M" if i % 2 == 0 else "F"
        signal = (-2.0 if group == "A" else 2.0) + float(rng.normal(0.0, 0.35))
        secondary = (-0.8 if group == "A" else 0.8) + float(rng.normal(0.0, 0.45))
        rows.append({
            "individual": f"train3_{i}",
            "group": group,
            "sex": sex,
            "Noise_first": float(rng.normal(0.0, 1.0)),
            "Signal_feat": signal,
            "Secondary_feat": secondary,
        })

    df_all = pd.DataFrame(rows)
    df_train = df_all.copy()
    X_train = df_train[["Noise_first", "Signal_feat", "Secondary_feat"]].copy()
    stratify = (df_train["sex"].astype(str) + "_" + df_train["group"].astype(str)).to_numpy(copy=False)
    fold_plan_records = _make_fold_plan_records(stratify, n_splits=2, seeds=[313, 414])

    out_dir = run_plain_xgboost_baseline(
        X_train_full=X_train,
        df_for_training=df_train,
        training_group_a="A",
        training_group_b="B",
        analysis_folder_path=str(tmp_path),
        n_splits=2,
        random_state=77,
        df_for_shap_and_agg=df_all,
        X_shap_full=df_all[["Noise_first", "Signal_feat", "Secondary_feat"]].copy(),
        settings_snapshot={"n_estimators": 10, "learning_rate": 0.15, "max_depth": 2, "model_subsample": 1.0},
        fold_plan_records=fold_plan_records,
    )

    ranking_df = pd.read_csv(Path(out_dir) / "GLOBAL_IMPORTANCE_RANKING.csv")
    assert str(ranking_df.iloc[0]["feature_name"]) == "Signal_feat"
    gi = pd.to_numeric(ranking_df["Global_Importance"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = float(gi.sum())
    if total > 1e-12:
        expected_norm = gi / total
        expected_cum = np.cumsum(expected_norm)
        expected_cum[-1] = 1.0
        got_norm = pd.to_numeric(ranking_df["Global_Importance_Normalized"], errors="coerce").to_numpy(dtype=float)
        got_cum = pd.to_numeric(ranking_df["Global_Importance_Cumulative"], errors="coerce").to_numpy(dtype=float)
        assert np.allclose(got_norm, expected_norm, atol=1e-9, rtol=0.0)
        assert np.allclose(got_cum, expected_cum, atol=1e-9, rtol=0.0)


def test_plain_xgboost_rerun_cleans_stale_permutation_outputs(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp.plain_xgboost_baseline import run_plain_xgboost_baseline

    rows = []
    rng = np.random.default_rng(88)
    for i in range(20):
        group = "A" if i < 10 else "B"
        sex = "M" if i % 2 == 0 else "F"
        signal = (-1.8 if group == "A" else 1.8) + float(rng.normal(0.0, 0.25))
        rows.append({
            "individual": f"train4_{i}",
            "group": group,
            "sex": sex,
            "Signal_feat": signal,
            "Noise_feat1": float(rng.normal(0.0, 1.0)),
            "Noise_feat2": float(rng.normal(0.0, 1.0)),
        })

    df_all = pd.DataFrame(rows)
    df_train = df_all.copy()
    X_train = df_train[["Signal_feat", "Noise_feat1", "Noise_feat2"]].copy()
    stratify = (df_train["sex"].astype(str) + "_" + df_train["group"].astype(str)).to_numpy(copy=False)
    fold_plan_records = _make_fold_plan_records(stratify, n_splits=2, seeds=[515, 616])

    out_dir = run_plain_xgboost_baseline(
        X_train_full=X_train,
        df_for_training=df_train,
        training_group_a="A",
        training_group_b="B",
        analysis_folder_path=str(tmp_path),
        n_splits=2,
        random_state=88,
        df_for_shap_and_agg=df_all,
        X_shap_full=df_all[["Signal_feat", "Noise_feat1", "Noise_feat2"]].copy(),
        settings_snapshot={
            "n_estimators": 6,
            "learning_rate": 0.2,
            "max_depth": 2,
            "model_subsample": 1.0,
            "run_permutation_test": True,
            "n_permutations": 2,
        },
        fold_plan_records=fold_plan_records,
    )
    out_path = Path(out_dir)
    assert (out_path / "PERMUTED_0_RESULTS").exists()
    assert (out_path / "PERMUTED_1_RESULTS").exists()

    run_plain_xgboost_baseline(
        X_train_full=X_train,
        df_for_training=df_train,
        training_group_a="A",
        training_group_b="B",
        analysis_folder_path=str(tmp_path),
        n_splits=2,
        random_state=88,
        df_for_shap_and_agg=df_all,
        X_shap_full=df_all[["Signal_feat", "Noise_feat1", "Noise_feat2"]].copy(),
        settings_snapshot={
            "n_estimators": 6,
            "learning_rate": 0.2,
            "max_depth": 2,
            "model_subsample": 1.0,
            "run_permutation_test": False,
        },
        fold_plan_records=fold_plan_records,
    )

    assert not (out_path / "PERMUTED_0_RESULTS").exists()
    assert not (out_path / "PERMUTED_1_RESULTS").exists()


def test_plain_xgboost_rerun_respects_saved_fsea_toggle(monkeypatch, tmp_path: Path) -> None:
    from tests.test_featureimp_baseline_reruns import _prepare_small_finalized_analysis
    import autobehaver.postinf.featureimp.plain_xgboost_baseline as px

    analysis_dir = _prepare_small_finalized_analysis(tmp_path)
    seen = {}
    monkeypatch.setattr(px, 'run_plain_xgboost_baseline', lambda **kwargs: seen.update(kwargs) or str(Path(kwargs['analysis_folder_path']) / 'PLAIN_XGBOOST_BASELINE'))

    out_dir = px.rerun_plain_xgboost_baseline_from_analysis_folder(analysis_folder_path=str(analysis_dir))
    assert out_dir.endswith('PLAIN_XGBOOST_BASELINE')
    assert seen['run_fsea_analysis'] is False
