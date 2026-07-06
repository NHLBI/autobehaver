"""Small argparse helpers used across Autobehaver CLI entrypoints.

These utilities are intentionally lightweight and dependency-free.

Goals
-----
- Keep one canonical spelling (underscored long options) in --help.
- Accept hyphenated variants (e.g. --file-filter) without duplicating parser code.
- Provide a consistent two-sided boolean style: --flag / --no_flag.
"""

from __future__ import annotations

import argparse
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
    """Normalize long options to the canonical underscored spelling.

    Notes
    -----
    - Only rewrites tokens starting with "--".
    - Supports "--opt=value" style.
    - Does *not* attempt to rewrite short flags ("-i").
    """

    known: set[str] = set()
    if parser is not None:
        known |= _collect_option_strings(parser)

    out: list[str] = []
    for tok in list(argv):
        if not tok.startswith("--"):
            out.append(tok)
            continue

        opt, eq, val = tok.partition("=")
        opt2 = opt

        if "-" in opt2:
            cand = opt2.replace("-", "_")
            if cand in known:
                opt2 = cand

        out.append(opt2 + (eq + val if eq else ""))

    return out


def add_bool_pair(
    parser: argparse.ArgumentParser,
    flag: str,
    *,
    default: Optional[bool],
    help: str,
    dest: Optional[str] = None,
) -> None:
    """Add a two-sided boolean flag: --flag / --no_flag.

    Parameters
    ----------
    flag:
        Either "flag" or "--flag".
    default:
        If True/False, sets a real default. If None, the destination is left as
        tri-state (None unless explicitly set on the CLI).
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
