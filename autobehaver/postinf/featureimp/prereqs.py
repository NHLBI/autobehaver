"""autobehaver.postinf.featureimp.prereqs

Runtime prerequisite checks for the feature importance pipeline.

Why this exists
---------------
Some environments can import :mod:`shap` successfully but still fail at runtime
when constructing a TreeExplainer (e.g., due to incompatible NumPy versions).
Those failures otherwise occur deep inside multiprocessing workers, making them
hard to diagnose.

These checks are intentionally:
  - fast (tiny synthetic dataset),
  - deterministic, and
  - executed before the expensive committee training begins.
"""

from __future__ import annotations


def check_shap_xgboost_treeexplainer_compatibility() -> None:
    """Fail fast if SHAP + NumPy + XGBoost are incompatible at runtime.

    This function should be called *before* launching multiprocessing workers.
    It trains a tiny XGBoost model and attempts to compute SHAP values.

    Raises
    ------
    RuntimeError
        If SHAP cannot compute TreeExplainer SHAP values for an XGBoost model.
    """

    # Import heavy deps only inside the function so importing featureimp.cli
    # remains lightweight.
    try:
        import numpy as np
        import shap
        import xgboost as xgb
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Feature importance analysis requires optional dependencies that are not installed. "
            f"Missing module: {missing}."
        ) from e

    # sklearn is already a core dependency for featureimp.
    try:
        from sklearn.datasets import make_classification
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "scikit-learn is required for feature importance analysis (missing sklearn)."
        ) from e

    try:
        X, y = make_classification(
            n_samples=20,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            random_state=0,
        )

        # Keep model as small as possible.
        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=0,
        )
        model.fit(X, y)

        from .shap_compat import make_tree_explainer

        explainer = make_tree_explainer(model)

        # Call shap_values to surface common runtime incompatibilities.
        sv = explainer.shap_values(X[:2])
        # Some SHAP versions return a list for binary; ensure we touch the array.
        if isinstance(sv, list):
            _ = np.asarray(sv[1] if len(sv) > 1 else sv[0])
        else:
            _ = np.asarray(sv)

    except Exception as e:
        # Provide a clear diagnostic including versions.
        np_ver = getattr(np, "__version__", "unknown")
        shap_ver = getattr(shap, "__version__", "unknown")
        xgb_ver = getattr(xgb, "__version__", "unknown")
        raise RuntimeError(
            "SHAP TreeExplainer failed on a tiny XGBoost model. "
            "This usually indicates an incompatible combination of NumPy/SHAP/XGBoost. "
            f"Versions: numpy={np_ver}, shap={shap_ver}, xgboost={xgb_ver}. "
            "Autobehaver applies a dependency guard for the known SHAP/XGBoost "
            "base_score serialization issue, so persistent failures here usually reflect "
            "a different environment problem. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
