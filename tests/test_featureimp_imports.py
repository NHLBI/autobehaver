"""Check that FeatureImp entrypoints import without loading their analysis dependencies."""


def test_featureimp_cli_imports() -> None:
    from autobehaver.postinf.featureimp import cli  # noqa: F401


def test_featureimp_fsea_imports() -> None:
    # Group construction uses only the base scientific dependencies.
    from autobehaver.postinf.featureimp import fsea  # noqa: F401



def test_featureimp_plain_xgboost_baseline_imports() -> None:
    from autobehaver.postinf.featureimp import plain_xgboost_baseline  # noqa: F401


def test_featureimp_staged_merge_module_imports() -> None:
    from autobehaver.postinf.featureimp import merge_shards  # noqa: F401
