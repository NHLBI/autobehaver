import math
import pytest
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from autobehaver.schedulers import (
    CosineAnnealingScheduler,
    InverseSquareRootDecayScheduler,
    LinearAnnealingScheduler,
)


@pytest.mark.parametrize(
    ("scheduler_factory", "label"),
    [
        (lambda opt: CosineAnnealingScheduler(opt, t_0_batches=8, t_mult=1, eta_min=0.0), "cosine"),
        (lambda opt: InverseSquareRootDecayScheduler(opt), "sqrt"),
        (lambda opt: LinearAnnealingScheduler(opt, t_0_batches=4, t_mult=1, eta_min=0.0), "linear"),
    ],
)
def test_custom_schedulers_work_with_sequentiallr_warmup_handoff(scheduler_factory, label):
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=2)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, scheduler_factory(optimizer)],
        milestones=[2],
    )

    lrs = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert len(lrs) == 8, label
    assert all(math.isfinite(v) for v in lrs), label
    assert lrs[1] == pytest.approx(0.1), label
    assert lrs[2] < lrs[1], label


def test_linear_annealing_scheduler_restarts_to_base_lr():
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=2)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, LinearAnnealingScheduler(optimizer, t_0_batches=4, t_mult=1, eta_min=0.0)],
        milestones=[2],
    )

    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert lrs[:6] == pytest.approx([0.055, 0.1, 0.075, 0.05, 0.025, 0.1], rel=1e-6, abs=1e-9)


def test_posepred_training_namespace_imports():
    import autobehaver.training as training_namespace
    import autobehaver.training.posepred_training as posepred_training
    from autobehaver.training.posepred_training import cli as ppt_cli
    from autobehaver.training.posepred_training import core as ppt_core

    assert callable(training_namespace.main_worker)
    assert callable(posepred_training.main)
    assert callable(ppt_cli.build_parser)
    assert callable(ppt_core.prepare_data)


def test_posepred_training_module_entrypoints_work_from_module_cli():
    from autobehaver.training.posepred_training.cli import build_parser

    help_text = build_parser().format_help()
    assert "usage:" in help_text.lower()
    assert "--directory" in help_text
    assert "--num_epochs" in help_text

def test_root_autobehaver_module_entrypoint_provides_pipeline_roadmap():
    from autobehaver.__main__ import build_parser

    help_text = build_parser().format_help()
    assert "pipeline roadmap" in help_text.lower()
    assert "--version" in help_text

