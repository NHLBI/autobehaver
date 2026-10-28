from pathlib import Path
import importlib
import shlex


def test_direct_command_examples_use_existing_stage_parsers():
    package = Path(__file__).resolve().parents[1]
    command_count = 0
    for filename in ("pipeline_commands.txt",):
        text = (package / "examples" / filename).read_text(encoding="utf-8")
        assert "--config" not in text
        for line in text.splitlines():
            if not line.startswith("python -m "):
                continue
            argv = shlex.split(line)
            stage = argv[2].rsplit(".", 1)[-1]
            if stage == "featureimp":
                parser = importlib.import_module(argv[2] + ".config").build_arg_parser()
            else:
                module = importlib.import_module(argv[2] + ".cli")
                parser = getattr(module, "build_parser" if stage in {"infmovie", "mmgen"} else "build_arg_parser")()
            parser.parse_args(argv[3:])
            command_count += 1
    assert command_count == 9


def test_unused_pipeline_yaml_removed_scientific_assets_retained():
    package = Path(__file__).resolve().parents[1]
    assert not list((package / "configs").glob("*.yaml"))
    assert (package / "ASSET_SHA256SUMS").is_file()


def test_infmovie_removed_noop_flags_are_not_exposed():
    from autobehaver.postinf.infmovie.cli import build_parser
    parser = build_parser()
    help_text = parser.format_help()
    option_strings = [opt for action in parser._actions for opt in action.option_strings]
    assert "--num_movies" not in option_strings
    assert option_strings.count("--skip_first_n_frames") == 1
    assert "--skip_first_n_frames" in help_text
