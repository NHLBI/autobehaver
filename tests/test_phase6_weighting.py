import os
import sys
import types
import unittest
import tempfile
import shutil

import numpy as np
import pandas as pd


class TestSubsetWeighting(unittest.TestCase):
    def test_tau_from_permutation_quantile_and_nonneg(self):
        from autobehaver.postinf.featureimp.subset_weighting import (
            SubsetWeightingConfig,
            compute_tau_from_permuted,
        )

        df = pd.DataFrame(
            {
                "delta_norm": [0.0, 0.1, 0.2, 0.3, 0.4],
                "subset_complete": [True, True, True, True, True],
            }
        )
        cfg = SubsetWeightingConfig(tau_quantile=0.8, gamma=2.0)
        tau, stats = compute_tau_from_permuted(df, cfg)
        # 80th percentile of [0,0.1,0.2,0.3,0.4] is 0.32
        self.assertTrue(tau >= 0.0)
        self.assertAlmostEqual(tau, float(np.quantile(df["delta_norm"].values, 0.8)), places=8)
        self.assertEqual(stats["tau_source"], "permutation_quantile")

    def test_tau_floor_at_zero_when_null_is_negative(self):
        from autobehaver.postinf.featureimp.subset_weighting import (
            SubsetWeightingConfig,
            compute_tau_from_permuted,
        )

        df = pd.DataFrame(
            {
                "delta_norm": [-0.5, -0.2, -0.1],
                "subset_complete": [True, True, True],
            }
        )
        cfg = SubsetWeightingConfig(tau_quantile=0.95, gamma=2.0)
        tau, _ = compute_tau_from_permuted(df, cfg)
        self.assertEqual(tau, 0.0)

    def test_compute_subset_weights_respects_tau_gamma_and_reasons(self):
        from autobehaver.postinf.featureimp.subset_weighting import (
            SubsetWeightingConfig,
            compute_subset_weights,
        )

        real = pd.DataFrame(
            {
                "h_hit": [0, 0, 0, 0],
                "subset_idx": [0, 1, 2, 3],
                "delta_norm": [0.5, 0.2, np.nan, 0.9],
                "subset_complete": [True, True, True, False],
            }
        )
        cfg = SubsetWeightingConfig(tau_quantile=0.95, gamma=2.0)
        out = compute_subset_weights(real, tau=0.3, cfg=cfg)

        w = out["subset_weight"].values
        # row0: (0.5-0.3)^2 = 0.04
        self.assertAlmostEqual(float(w[0]), 0.04, places=8)
        # row1: below tau -> 0
        self.assertAlmostEqual(float(w[1]), 0.0, places=12)
        # row2: missing metric -> 0
        self.assertAlmostEqual(float(w[2]), 0.0, places=12)
        # row3: incomplete -> 0
        self.assertAlmostEqual(float(w[3]), 0.0, places=12)

        reasons = list(out["subset_weight_reason"].values)
        self.assertEqual(reasons[0], "ok")
        self.assertEqual(reasons[1], "below_tau")
        self.assertEqual(reasons[2], "missing_metric")
        self.assertEqual(reasons[3], "incomplete_subset")


class TestLogitEnsemblePhysics(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ab_phase6_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_equal_weights_matches_unweighted(self):
        # We monkeypatch xgboost inside the module so we don't require real models.
        from autobehaver.postinf.featureimp import logit_ensemble as le

        class DummyModel:
            def __init__(self):
                self._margin = 0.0

            def load_model(self, path):
                # Encode a deterministic margin from the filename.
                base = os.path.basename(str(path))
                # expected patterns like h0_s1_f0.json
                margin = 0.0
                for part in base.replace(".", "_").split("_"):
                    if part.startswith("s") and part[1:].isdigit():
                        margin += float(int(part[1:]))
                    if part.startswith("f") and part[1:].isdigit():
                        margin += float(int(part[1:])) / 10.0
                self._margin = margin

            def predict(self, X, output_margin=False):
                # output_margin ignored because this dummy only returns margins.
                return np.full((len(X),), self._margin, dtype=np.float64)

        orig_cls = le.xgb.XGBClassifier
        le.xgb.XGBClassifier = DummyModel
        try:
            # Tiny dataset
            X_train = pd.DataFrame(
                {
                    "f0": [0, 1, 2, 3, 4, 5],
                    "f1": [1, 1, 1, 1, 1, 1],
                    "f2": [5, 4, 3, 2, 1, 0],
                }
            )
            X_all = pd.concat(
                [X_train, pd.DataFrame({"f0": [6, 7], "f1": [1, 1], "f2": [-1, -2]})],
                ignore_index=True,
            )
            y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
            sex = np.zeros(len(y), dtype=int)

            # Build fake saved model files
            def touch(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("dummy")

            records = []
            for subset_idx, feats in [(0, ["f0", "f1"]), (1, ["f1", "f2"])]:
                for fold_idx in [0, 1]:
                    mp = os.path.join(self.tmpdir, f"h0_s{subset_idx}_f{fold_idx}.json")
                    touch(mp)
                    records.append(
                        {
                            "h_hit": 0,
                            "subset_idx": subset_idx,
                            "fold_idx": fold_idx,
                            "model_path": mp,
                            "specialist_features": feats,
                            "is_permuted": False,
                        }
                    )

            # Run with equal weights
            weights = {(0, 0): 1.0, (0, 1): 1.0}
            out = le.compute_logit_ensemble_predictions_from_saved_models(
                performance_records=records,
                X_train_full=X_train,
                X_all_groups=X_all,
                y_encoded_train=y,
                sex_group_encoded=sex,
                num_classes=2,
                n_splits=2,
                GLOBAL_RANDOM_SEED=123,
                subset_weights=weights,
                verbose=False,
            )

            # Weighted should match unweighted when all subset weights are equal.
            self.assertIsNotNone(out.oof_probas_weighted)
            self.assertIsNotNone(out.insample_probas_weighted)
            self.assertIsNotNone(out.all_groups_probas_weighted)

            np.testing.assert_allclose(out.oof_probas_weighted, out.oof_probas, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(out.insample_probas_weighted, out.insample_probas, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(out.all_groups_probas_weighted, out.all_groups_probas, rtol=1e-12, atol=1e-12)

        finally:
            le.xgb.XGBClassifier = orig_cls


class TestWeightedSHAPPhysics(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ab_phase6_shap_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scaling_all_subset_weights_does_not_change_weighted_shap(self):
        # This test monkeypatches shap + xgboost to avoid heavy deps and training.
        from autobehaver.postinf.featureimp import shap_ensemble as se

        class DummyModel:
            def __init__(self):
                self._value = 0.0

            def load_model(self, path):
                base = os.path.basename(str(path))
                val = 0.0
                for part in base.replace(".", "_").split("_"):
                    if part.startswith("f") and part[1:].isdigit():
                        val += float(int(part[1:]))
                self._value = val

        # Fake shap module
        dummy_shap = types.ModuleType("shap")

        class DummyTreeExplainer:
            def __init__(self, model):
                self.model = model

            def shap_values(self, X):
                # return deterministic per-model SHAP values
                arr = np.ones((len(X), X.shape[1]), dtype=np.float32) * float(getattr(self.model, "_value", 0.0))
                return arr

        dummy_shap.TreeExplainer = DummyTreeExplainer

        # Patch sys.modules so the local import inside compute_weighted_shap_from_saved_models uses it.
        orig_shap_mod = sys.modules.get("shap", None)
        sys.modules["shap"] = dummy_shap

        orig_cls = se.xgb.XGBClassifier
        se.xgb.XGBClassifier = DummyModel

        try:
            # Create dummy model files
            def touch(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("dummy")

            # Minimal dataset
            X_all = pd.DataFrame({"f0": [0, 1, 2, 3], "f1": [1, 1, 1, 1]})
            all_feats = ["f0", "f1"]
            n_train = 4
            n_splits = 2

            # Fold indices (simple deterministic)
            folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
            fold_map = {0: folds}

            # One subset with two folds
            records = []
            for fold_idx in [0, 1]:
                mp = os.path.join(self.tmpdir, f"h0_s0_f{fold_idx}.json")
                touch(mp)
                records.append(
                    {
                        "h_hit": 0,
                        "subset_idx": 0,
                        "fold_idx": fold_idx,
                        "features": ["f0", "f1"],
                        "model_path": mp,
                        "is_permuted": False,
                    }
                )

            # weights scaled by constant should not change output after normalization
            w1 = {(0, 0): 1.0}
            w2 = {(0, 0): 5.0}

            oof1, ins1, all1, denom1, _ = se.compute_weighted_shap_from_saved_models(
                performance_records=records,
                subset_weights=w1,
                fold_indices_by_h_hit=fold_map,
                X_all_groups=X_all,
                all_feature_names=all_feats,
                n_train=n_train,
                n_splits=n_splits,
                verbose=False,
            )
            oof2, ins2, all2, denom2, _ = se.compute_weighted_shap_from_saved_models(
                performance_records=records,
                subset_weights=w2,
                fold_indices_by_h_hit=fold_map,
                X_all_groups=X_all,
                all_feature_names=all_feats,
                n_train=n_train,
                n_splits=n_splits,
                verbose=False,
            )

            np.testing.assert_allclose(oof1, oof2, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(ins1, ins2, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(all1, all2, rtol=1e-12, atol=1e-12)

            # Denominator (feature_weight_sums) should scale with weights
            self.assertTrue(np.allclose(denom2, denom1 * 5.0))

        finally:
            se.xgb.XGBClassifier = orig_cls
            if orig_shap_mod is None:
                sys.modules.pop("shap", None)
            else:
                sys.modules["shap"] = orig_shap_mod


if __name__ == "__main__":
    unittest.main()
