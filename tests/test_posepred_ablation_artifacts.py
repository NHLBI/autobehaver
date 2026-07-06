from pathlib import Path

from autobehaver.cli import _write_feature_plan_artifacts, build_parser
from autobehaver.feature_selection import resolve_feature_plan


def test_write_feature_plan_artifacts_writes_single_human_readable_report(tmp_path: Path):
    records_dir = tmp_path / "Records"
    records_dir.mkdir()

    headers = [
        "centroid_x",
        "centroid_y",
        "angle_sin",
        "angle_cos",
        "speed",
        "wing_angle",
    ]
    plan = resolve_feature_plan(headers, preset="no_pos")

    out_dir = Path(_write_feature_plan_artifacts(str(records_dir), "demo_run", plan))

    assert out_dir == records_dir / "feature_columns"
    report_path = out_dir / "demo_run_feature_columns.txt"
    assert report_path.exists()

    report = report_path.read_text(encoding="utf-8")

    assert "INCLUDED COLUMNS" in report
    assert "EXCLUDED COLUMNS" in report
    assert "speed" in report
    assert "wing_angle" in report
    assert "centroid_x" in report
    assert "centroid_y" in report
    assert "angle_sin" in report
    assert "angle_cos" in report


def test_cli_no_longer_exposes_ablation_report_flag():
    parser = build_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert "--ablation_report" not in option_strings
