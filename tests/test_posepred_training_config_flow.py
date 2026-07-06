import inspect

import torch

from autobehaver import _training_core as core
from autobehaver.pretraining import run_pretraining


def test_model_setup_uses_explicit_optimizer_and_scheduler_args_not_module_globals():
    assert not hasattr(core, "weight_decay")
    assert not hasattr(core, "scheduler_type")
    assert not hasattr(core, "warmup_steps")

    _model, optimizer, scheduler, _scaler = core.model_setup(
        sequence_length=5,
        num_features=4,
        d_model=8,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout_rate=0.1,
        frame_representation="cls",
        num_cls=1,
        prediction_head="fc",
        device=torch.device("cpu"),
        base_lr=0.005,
        pose_pred=True,
        orient_pred=False,
        pose_dict={"a": 0, "b": 1},
        orient_dict=None,
        recon=False,
        positional_encoding_type="sinusoidal",
        patience=5,
        plateau_factor=0.5,
        create_scheduler=True,
        scheduler_type="plateau",
        weight_decay=0.123,
        warmup_steps=0,
        t_0=12,
        t_mult=1,
        eta_min=1e-6,
    )

    assert optimizer.param_groups[0]["weight_decay"] == 0.123
    assert scheduler.__class__.__name__ == "ReduceLROnPlateau"


def test_main_worker_and_run_pretraining_accept_explicit_runtime_hyperparameters():
    main_sig = inspect.signature(core.main_worker)
    pre_sig = inspect.signature(run_pretraining)

    for name in [
        "base_lr", "weight_decay", "scheduler_type", "warmup_steps", "t_0",
        "t_mult", "eta_min", "mask_p", "noise_level", "max_class_weight",
        "grad_clip_norm", "random_seed", "metadata_tag",
    ]:
        assert name in main_sig.parameters

    for name in [
        "scheduler_type", "weight_decay", "t_0", "t_mult", "eta_min",
        "grad_clip_norm", "random_seed", "metadata_tag",
    ]:
        assert name in pre_sig.parameters
