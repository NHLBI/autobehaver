"""Compatibility helpers for SHAP + XGBoost interactions in FeatureImp.

Why this exists
---------------
SHAP versions prior to the upstream fix for XGBoost >= 3.1 can fail when
``TreeExplainer`` inspects a booster whose serialized ``base_score`` is stored
in list-like form, e.g. ``"[5E-1]"``.  FeatureImp constructs SHAP explainers in
multiple places (prereq checks, worker-time SHAP, and saved-model weighted SHAP
reaggregation), so we centralize the workaround here.

The workaround is intentionally narrow:
- patch only SHAP's UBJSON decode path used by the XGBoost tree loader;
- normalize only ``learner.learner_model_param.base_score``;
- leave newer SHAP versions effectively unchanged (the patch is idempotent and
  harmless when the upstream parser already supports list-like values).
"""

from __future__ import annotations

import ast
from typing import Any


_PATCH_FLAG = "_autobehaver_xgb_base_score_patch"


def _normalize_base_score_value(value: Any) -> Any:
    """Normalize XGBoost ``base_score`` values into a float when possible.

    Older SHAP releases expect ``base_score`` to be directly castable via
    ``float(value)``.  Newer XGBoost releases may serialize it as list-like
    values (for example ``[5E-1]``) or strings containing such structures.

    Parameters
    ----------
    value:
        The decoded value from XGBoost's UBJSON model representation.

    Returns
    -------
    Any
        A float when the value can be safely normalized, otherwise the original
        value is returned unchanged.
    """

    try:
        import numpy as np
    except Exception:  # pragma: no cover - NumPy is available in real runs
        np = None  # type: ignore

    if isinstance(value, (int, float)):
        return float(value)

    if np is not None and isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _normalize_base_score_value(value[0])
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value

        try:
            return float(text)
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return value
        return _normalize_base_score_value(parsed)

    return value



def _normalize_tree_model_dict(decoded: Any) -> Any:
    """Normalize decoded XGBoost UBJSON dictionaries in-place when possible."""

    if not isinstance(decoded, dict):
        return decoded

    try:
        learner = decoded.get("learner")
        if not isinstance(learner, dict):
            return decoded
        learner_model_param = learner.get("learner_model_param")
        if not isinstance(learner_model_param, dict):
            return decoded
        if "base_score" not in learner_model_param:
            return decoded
        learner_model_param["base_score"] = _normalize_base_score_value(
            learner_model_param["base_score"]
        )
    except Exception:
        # Never let optional alias installation break SHAP.
        return decoded
    return decoded



def ensure_shap_xgboost_base_score_patch() -> None:
    """Install the SHAP/XGBoost base-score compatibility patch once."""

    try:
        import shap.explainers._tree as tree_mod
    except Exception:
        return

    if getattr(tree_mod, _PATCH_FLAG, False):
        return

    decode_fn = getattr(tree_mod, "decode_ubjson_buffer", None)
    if decode_fn is None:
        setattr(tree_mod, _PATCH_FLAG, True)
        return

    def _patched_decode_ubjson_buffer(*args: Any, **kwargs: Any) -> Any:
        decoded = decode_fn(*args, **kwargs)
        return _normalize_tree_model_dict(decoded)

    tree_mod.decode_ubjson_buffer = _patched_decode_ubjson_buffer
    setattr(tree_mod, _PATCH_FLAG, True)



def ensure_shap_numpy_alias_patch() -> None:
    """Install minimal NumPy aliases expected by some SHAP releases.

    SHAP 0.39 still references ``np.int`` inside ``TreeExplainer`` for some
    XGBoost code paths. Newer NumPy releases removed that alias. We set the
    narrowest alias needed at runtime so FeatureImp remains usable without
    forcing global environment downgrades.
    """

    try:
        import numpy as np
    except Exception:
        return

    if not hasattr(np, "int"):
        setattr(np, "int", int)


def make_tree_explainer(model: Any) -> Any:
    """Create ``shap.TreeExplainer`` after installing required NumPy aliases."""

    ensure_shap_xgboost_base_score_patch()
    ensure_shap_numpy_alias_patch()
    import shap

    return shap.TreeExplainer(model)
