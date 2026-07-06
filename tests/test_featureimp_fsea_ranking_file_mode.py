from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_featureimp_cli_routes_fsea_ranking_file_mode(monkeypatch, tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import cli
    import autobehaver.postinf.featureimp.core as core

    ranking_path = tmp_path / "GLOBAL_IMPORTANCE_RANKING.csv"
    pd.DataFrame(
        {
            "feature_name": ["BoutSummary_Pose_Walking_bout_frequency"],
            "Global_Importance": [1.0],
        }
    ).to_csv(ranking_path, index=False)
    out_dir = tmp_path / "fsea_out"

    seen = {}
    monkeypatch.setattr(
        core,
        "run_featureimp_global_fsea_from_file",
        lambda **kwargs: seen.update(kwargs) or str(out_dir),
    )

    cli.main(
        [
            "--stage",
            "fsea",
            "--fsea_ranked_list_path",
            str(ranking_path),
            "--fsea_output_dir",
            str(out_dir),
        ]
    )

    assert seen["ranked_list_path"] == str(ranking_path)
    assert seen["output_dir"] == str(out_dir)
    assert seen["feature_col"] == "feature_name"
    assert seen["rank_col"] == "Global_Importance"
    assert seen["label"] == ""



def test_featureimp_cli_routes_fsea_analysis_folder_mode(monkeypatch, tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import cli
    import autobehaver.postinf.featureimp.core as core

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    monkeypatch.setattr(
        core,
        "run_featureimp_postanalysis_stage",
        lambda **kwargs: seen.update(kwargs) or str(analysis_dir),
    )

    cli.main(["--stage", "fsea", "--analysis_folder_path", str(analysis_dir)])

    assert seen == {
        "analysis_folder_path": str(analysis_dir),
        "stage_name": "fsea",
    }



def test_featureimp_cli_fsea_rejects_mixed_inputs(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp import cli

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = tmp_path / "GLOBAL_IMPORTANCE_RANKING.csv"
    pd.DataFrame(
        {
            "feature_name": ["BoutSummary_Pose_Walking_bout_frequency"],
            "Global_Importance": [1.0],
        }
    ).to_csv(ranking_path, index=False)
    out_dir = tmp_path / "fsea_out"

    with pytest.raises(SystemExit, match="accepts either --analysis_folder_path or the pair"):
        cli.main(
            [
                "--stage",
                "fsea",
                "--analysis_folder_path",
                str(analysis_dir),
                "--fsea_ranked_list_path",
                str(ranking_path),
                "--fsea_output_dir",
                str(out_dir),
            ]
        )



def test_featureimp_global_fsea_from_file_writes_outputs_and_uses_grouped_features(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from autobehaver.postinf.featureimp import core
    import autobehaver.postinf.featureimp.fsea_runner as fsea_runner

    ranking_path = tmp_path / "GLOBAL_IMPORTANCE_RANKING.csv"
    pd.DataFrame(
        {
            "feature_name": [
                "BoutSummary_Pose_Walking_bout_frequency",
                "TimeBudget_Pose_Walking",
                "Spatial_Frac_Overall_Center",
            ],
            "Global_Importance": [3.0, 2.0, 1.0],
        }
    ).to_csv(ranking_path, index=False)
    out_dir = tmp_path / "FSEA_FROM_RANKING"

    calls = []

    class _FakeResult:
        def __init__(self) -> None:
            self.res2d = pd.DataFrame(
                {
                    "Term": ["PoseStats_Walking", "Spatial_Frac"],
                    "FDR": [0.01, 0.20],
                    "NES": [1.4, 1.1],
                }
            )

        def plot(self, terms, ofname):
            Path(ofname).write_text("plot", encoding="utf-8")

    def _fake_prerank(**kwargs):
        calls.append(kwargs)
        return _FakeResult()

    monkeypatch.setitem(sys.modules, "gseapy", SimpleNamespace(prerank=_fake_prerank))
    monkeypatch.setattr(fsea_runner, "_orchestrate_fsea_visualization", lambda **kwargs: None)

    core.apply_settings(
        {
            "analysis_run_name": "fsea_ranked_file_test",
            "num_workers": 1,
            "fsea_permutations": 1,
            "fsea_min_size": 1,
            "fsea_max_size": 500,
            "fsea_viz_top_n": 5,
        },
        propagate=False,
    )

    out = core.run_featureimp_global_fsea_from_file(
        ranked_list_path=str(ranking_path),
        output_dir=str(out_dir),
        label="CustomGlobal",
    )

    assert out == str(out_dir)
    assert len(calls) == 1
    gene_sets = calls[0]["gene_sets"]
    assert "PoseStats_Walking" in gene_sets
    assert "Spatial_Frac" in gene_sets
    assert gene_sets["PoseStats_Walking"] == [
        "BoutSummary_Pose_Walking_bout_frequency",
        "TimeBudget_Pose_Walking",
    ]
    assert gene_sets["Spatial_Frac"] == ["Spatial_Frac_Overall_Center"]

    assert (out_dir / "FSEA_FEATURE_GROUP_MEMBERSHIP.csv").exists()
    assert (out_dir / "FSEA_GROUP_DEFINITIONS.txt").exists()
    assert (out_dir / "FSEA_RESULTS_CustomGlobal.xlsx").exists()
    assert (out_dir / "FSEA_CustomGlobal").exists()
