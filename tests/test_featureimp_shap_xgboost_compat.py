from __future__ import annotations

import numpy as np

from autobehaver.postinf.featureimp.shap_compat import (
    _normalize_base_score_value,
    _normalize_tree_model_dict,
    ensure_shap_xgboost_base_score_patch,
    make_tree_explainer,
)


def test_normalize_base_score_value_variants() -> None:
    assert _normalize_base_score_value("[5E-1]") == 0.5
    assert _normalize_base_score_value("0.5") == 0.5
    assert _normalize_base_score_value([0.5]) == 0.5
    assert _normalize_base_score_value((0.5,)) == 0.5
    assert _normalize_base_score_value(np.asarray([0.5], dtype=np.float32)) == 0.5
    assert _normalize_base_score_value("not-a-number") == "not-a-number"


def test_normalize_tree_model_dict_mutates_base_score() -> None:
    decoded = {
        "learner": {
            "learner_model_param": {
                "base_score": "[5E-1]",
                "num_class": "0",
            }
        }
    }
    out = _normalize_tree_model_dict(decoded)
    assert out["learner"]["learner_model_param"]["base_score"] == 0.5


def test_make_tree_explainer_smoke() -> None:
    import shap.explainers._tree as tree_mod
    import xgboost as xgb
    from sklearn.datasets import make_classification

    ensure_shap_xgboost_base_score_patch()
    assert getattr(tree_mod, "_autobehaver_xgb_base_score_patch", False) is True

    X, y = make_classification(
        n_samples=20,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
    )
    model = xgb.XGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=0,
    )
    model.fit(X, y)

    explainer = make_tree_explainer(model)
    sv = explainer.shap_values(X[:2])
    arr = np.asarray(sv[1] if isinstance(sv, list) and len(sv) > 1 else sv)
    assert arr.shape[0] == 2
