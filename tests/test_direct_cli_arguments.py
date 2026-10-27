import argparse

import pytest

from autobehaver.argparse_utils import normalize_argv, parse_cli_args
from autobehaver.cli import build_parser


def test_mixed_spelling_and_equals_preserve_values():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--no-postanalysis_strict', action='store_false', dest='strict')
    args = parse_cli_args(parser, ['--output-dir=A folder/with-hyphens', '--no-postanalysis-strict'])
    assert args.output_dir == 'A folder/with-hyphens'
    assert args.strict is False
    assert parse_cli_args(parser, ['--no_postanalysis_strict']).strict is False


def test_end_of_options_preserves_literal_option_names():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('filenames', nargs='*')
    assert parse_cli_args(parser, ['--', '--output-dir']).filenames == ['--output-dir']


def test_exact_names_win_and_ambiguous_aliases_are_rejected():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a_b-c')
    parser.add_argument('--a-b_c')
    assert normalize_argv(['--a_b-c=x'], parser=parser) == ['--a_b-c=x']
    with pytest.raises(SystemExit):
        parse_cli_args(parser, ['--a-b-c=x'])


@pytest.mark.parametrize('arguments', [['--num-w', '2'], ['--config', 'unused.yaml']])
def test_training_rejects_abbreviations_and_retired_inputs(arguments):
    with pytest.raises(SystemExit):
        parse_cli_args(build_parser(), arguments)


def test_training_direct_scientific_options_and_negative_flags():
    args = parse_cli_args(build_parser(), [
        '--directory', 'A folder/with-hyphens', '--random-seed=17', '--inf-average', '3',
        '--no-use-tta', '--no_pose_pred', '--no-orient-pred', '--no_defined_split',
        '--resource_monitor', '--no_resource_monitor_gpu',
    ])
    assert args.directory == 'A folder/with-hyphens'
    assert args.random_seed == 17
    assert args.inf_average == 3
    assert (args.use_tta, args.pose_pred, args.orient_pred, args.defined_split) == (False,) * 4
    assert args.resource_monitor is True
    assert args.resource_monitor_gpu is False
