"""Resolve long-option aliases and paired boolean CLI flags."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _collect_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    opts: set[str] = set()
    for a in getattr(parser, "_actions", []):
        for s in getattr(a, "option_strings", []) or []:
            opts.add(str(s))
    return opts


def normalize_argv(
    argv: Sequence[str],
    *,
    parser: Optional[argparse.ArgumentParser] = None,
) -> list[str]:
    """Resolve hyphen/underscore aliases to options declared by the parser.

    Preserve option values, short flags and tokens after ``--``.
    """

    known: set[str] = set()
    if parser is not None:
        known |= _collect_option_strings(parser)

    aliases: dict[str, list[str]] = {}
    for option in known:
        if option.startswith('--'):
            aliases.setdefault(option[2:].replace('-', '_'), []).append(option)

    out: list[str] = []
    positional_only = False
    for tok in list(argv):
        if tok == '--':
            positional_only = True
        if positional_only:
            out.append(tok)
            continue
        if not tok.startswith("--"):
            out.append(tok)
            continue

        opt, eq, val = tok.partition("=")
        opt2 = opt

        if opt2 not in known:
            candidates = aliases.get(opt2[2:].replace('-', '_'), [])
            if len(candidates) == 1:
                opt2 = candidates[0]

        out.append(opt2 + (eq + val if eq else ""))

    return out


def parse_cli_args(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None):
    """Parse one direct invocation with consistent option spellings."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    parser.allow_abbrev = False
    return parser.parse_args(normalize_argv(arguments, parser=parser))


def add_bool_pair(
    parser: argparse.ArgumentParser,
    flag: str,
    *,
    default: Optional[bool],
    help: str,
    dest: Optional[str] = None,
) -> None:
    """Add mutually exclusive ``--flag`` and ``--no_flag`` switches.

    A default of None preserves an unset value until a switch is supplied.
    """

    base = flag[2:] if flag.startswith("--") else flag
    dest_name = dest or base.replace("-", "_")

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        f"--{base}",
        dest=dest_name,
        action="store_true",
        help=help,
    )
    group.add_argument(
        f"--no_{base}",
        dest=dest_name,
        action="store_false",
        help=f"Disable: {help}" if help else "",
    )

    if default is None:
        parser.set_defaults(**{dest_name: None})
    else:
        parser.set_defaults(**{dest_name: bool(default)})
