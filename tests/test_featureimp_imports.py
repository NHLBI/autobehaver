"""Smoke tests for the featureimp module.

These tests are intentionally lightweight:
- They should not require SHAP/XGBoost at import time.
- They should validate that the featureimp CLI module can be imported.

If you run tests in an environment without optional deps, these should still pass.
"""


def test_featureimp_cli_imports() -> None:
    from autobehaver.postinf.featureimp import cli  # noqa: F401


def test_featureimp_fsea_imports() -> None:
    # fsea.py should be importable with base scientific stack.
    from autobehaver.postinf.featureimp import fsea  # noqa: F401



def test_featureimp_plain_xgboost_baseline_imports() -> None:
    from autobehaver.postinf.featureimp import plain_xgboost_baseline  # noqa: F401


def test_featureimp_staged_merge_module_imports() -> None:
    from autobehaver.postinf.featureimp import merge_shards  # noqa: F401
