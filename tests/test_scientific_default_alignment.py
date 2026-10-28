"""Keep shared scientific defaults consistent across native and desktop entry points."""
import pytest
from autobehaver.cli import build_parser, resolve_schedule_mode
from autobehaver.desktop_training import option_arguments
from autobehaver.desktop_settings import advanced_specs
from autobehaver.preinf.posepred_inference.config import PosePredInferenceConfig
from autobehaver.postinf.featureimp.config import build_arg_parser
from autobehaver.postinf.featureimp.runtime_settings import FSEARuntimeSettings


def test_supervised_architecture_and_loader_sized_schedule():
    args = build_parser().parse_args([])
    expected = dict(sequence_length=41, d_model=1024, dim_feedforward=4096,
        num_encoder_layers=1, nhead=32, frame_representation='middle', prediction_head='mlp',
        positional_encoding_type='sinusoidal', batch=1024, dropout_rate=.5,
        weight_decay=.001, noise_level=.05, mask_p=.05, max_class_weight=200,
        pose_weight=1., orient_weight=.1, base_lr=.0001, eta_min=.000001,
        num_epochs=301, cosine_decay_epochs=300, random_seed=7)
    _, desktop = option_arguments({})
    for key, value in expected.items():
        assert getattr(args, key) == value
        assert getattr(desktop, key) == value
    assert not args.noval and args.ablation_preset == 'no_pos'
    assert resolve_schedule_mode(args).cosine_decay_epochs == 300


@pytest.mark.parametrize('flags', [['--swa_mode'], ['--pretrain_mode'], ['--lr_finder_mode'],
                                  ['--cosine_decay_epochs', '0']])
def test_optional_modes_retain_step_based_schedule(flags):
    args = resolve_schedule_mode(build_parser().parse_args(flags))
    assert args.cosine_decay_epochs is None and args.warmup_epochs is None


def test_short_epoch_schedule_and_explicit_overrides():
    _, args = option_arguments(dict(num_epochs=3, cosine_decay_epochs=2, nhead=16, random_seed=21))
    assert resolve_schedule_mode(args).cosine_decay_epochs == 2
    assert args.num_epochs == 3 and args.nhead == 16 and args.random_seed == 21


def test_inference_defaults_are_shared():
    args=build_parser().parse_args([])
    cfg=PosePredInferenceConfig(input_dir='unused')
    specs=advanced_specs()
    for key in ('inf_average','use_tta','tta_noise_level','min_pose_bout','min_orient_bout'):
        assert getattr(args,key) == getattr(cfg,key) == specs['pp_'+key]['default']


def test_featureimp_cli_gui_and_runtime_defaults():
    args=build_arg_parser().parse_args([]); specs=advanced_specs()
    expected=dict(n_splits=5,H_hits_per_feature=100,n_features_per_model=100,
        n_estimators=1000,learning_rate=.01,max_depth=6,model_subsample=.8,
        min_feature_completeness_per_group=.25,MIN_VARIANCE_THRESHOLD=1e-6,
        n_permutations=1,subset_weight_tau_quantile=.95,subset_weight_gamma=2.,
        pca_fixed_k=50,fsea_permutations=100000,fsea_min_size=15,fsea_max_size=500,fsea_fdr_threshold=.15)
    for key,value in expected.items():
        assert getattr(args,key) == value
        assert specs['fi_opt_'+key]['default'] == value
    assert FSEARuntimeSettings().fsea_fdr_threshold == .15
    assert all(getattr(args,key)=='' for key in ('control_group_tag','training_group_a','training_group_b'))
