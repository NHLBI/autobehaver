from __future__ import annotations

import torch

from autobehaver.cli import build_parser
from autobehaver.model import PosePredTransformer


def test_posepred_cli_no_longer_exposes_weight_initialization() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "--weight_initialization" not in help_text


def test_transformer_standard_weight_init_policy_applied() -> None:
    model = PosePredTransformer(
        num_features=24,
        sequence_length=17,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout_rate=0.1,
        frame_representation="cls+middle",
        num_cls=2,
        prediction_head="mlp",
        pose_pred=True,
        orient_pred=True,
        recon=True,
        num_pose_classes=3,
        num_orient_classes=2,
        positional_encoding_type="learned",
        pretrain_mode=True,
        projection_dim=16,
    )

    enc0 = model.transformer_encoder.layers[0].self_attn.in_proj_weight.detach()
    enc1 = model.transformer_encoder.layers[1].self_attn.in_proj_weight.detach()
    assert not torch.equal(enc0, enc1)

    dec0 = model.transformer_decoder.layers[0].self_attn.in_proj_weight.detach()
    dec1 = model.transformer_decoder.layers[1].self_attn.in_proj_weight.detach()
    assert not torch.equal(dec0, dec1)

    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias).item() == 0
        if isinstance(module, torch.nn.LayerNorm):
            assert torch.allclose(module.weight, torch.ones_like(module.weight))
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert torch.allclose(module.weight, torch.ones_like(module.weight))
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, torch.nn.MultiheadAttention) and module.in_proj_bias is not None:
            assert torch.count_nonzero(module.in_proj_bias).item() == 0

    for param in [model.positional_encoding, model.frame_cls, model.frame_cls_positional_encoding]:
        mean = float(param.detach().mean())
        std = float(param.detach().std(unbiased=False))
        assert abs(mean) < 0.02
        assert 0.005 < std < 0.05


def test_model_constructor_no_weight_initialization_kwarg() -> None:
    try:
        PosePredTransformer(
            num_features=4,
            sequence_length=5,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=16,
            dropout_rate=0.1,
            frame_representation="middle",
            num_cls=1,
            prediction_head="fc",
            pose_pred=True,
            orient_pred=False,
            recon=False,
            num_pose_classes=2,
            num_orient_classes=0,
            positional_encoding_type="sinusoidal",
            pretrain_mode=False,
            projection_dim=8,
            weight_initialization="kaiming",
        )
        raised = False
    except TypeError:
        raised = True
    assert raised
