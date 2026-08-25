"""``qvac-translate`` console script — translate from the command line.

Examples::

    # Bilingual model: no language tag needed
    qvac-translate --models models/en-de "Where is the bus stop?"

    # Multilingual model with a target-language tag prefix (EuroNano / AfriNano)
    qvac-translate --models models/en-xx --tag DE "Good morning, my friend."
    qvac-translate --models models/en-xx --tag SW "Good morning, my friend."

    # Stream stdin (one sentence per line)
    cat lines.txt | qvac-translate --models models/en-xx --tag YO

The ``--tag`` flag is optional and accepts any short alphabetic code; the
CLI does not enforce a whitelist because the model itself decides what
tags it understands. (For multilingual models the convention is the
target-language ISO code, prefixed as ``##TAG`` by Bergamot trainings.)

For low-level batching or beam search, use the Python API directly
(``from qvac_nmt import MarianSSRU, load_npz, Translator``).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

from .translator import load_model


def build_parser(
    prog: str = "qvac-translate",
    description: str | None = None,
    *,
    tag_choices: list[str] | None = None,
    tag_help: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Build the argparse parser.

    ``tag_choices`` is opt-in: if provided, ``--tag`` is restricted to that
    set (used by the AfriNano ``afritranslate`` shim). If ``None`` (default),
    any string consisting of letters/digits is accepted.
    """
    p = argparse.ArgumentParser(
        prog=prog,
        description=description
        or "Translate with a Bergamot-recipe Marian .npz checkpoint, in pure PyTorch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog if epilog is not None else __doc__,
    )
    p.add_argument(
        "--models",
        type=Path,
        required=True,
        help="Directory containing one .npz weight file and one .spm vocab.",
    )
    if tag_choices is not None:
        p.add_argument(
            "--tag",
            choices=tag_choices,
            default=None,
            help=tag_help
            or "Target language tag, prepended as '##TAG'.",
        )
    else:
        p.add_argument(
            "--tag",
            type=_validate_tag,
            default=None,
            help=tag_help
            or "Optional target-language tag for multilingual models; "
            "prepended as '##TAG '. Omit for bilingual or reverse-direction "
            "models. Any short alphanumeric code is accepted; the model "
            "decides which ones are valid.",
        )
    p.add_argument(
        "--max-len",
        type=int,
        default=200,
        help="Maximum target tokens before forced stop (default 200).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print the inferred model variant on load.",
    )
    p.add_argument(
        "text",
        nargs="*",
        help="Sentence(s) to translate. If omitted, reads stdin one line at a time.",
    )
    return p


_TAG_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")


def _validate_tag(value: str) -> str:
    if not _TAG_RE.match(value):
        raise argparse.ArgumentTypeError(
            f"--tag {value!r}: expected 1-8 alphanumeric characters"
        )
    return value.upper()


def run(
    args: argparse.Namespace,
    *,
    pre_render: Callable[[str], str] | None = None,
) -> int:
    """Execute a parsed namespace. Shared by both the generic and AT CLIs.

    ``pre_render`` is applied to each input line before translation; the
    AfriNano ``afritranslate`` shim uses it to enforce that one of its tag
    values is always present.
    """
    if not args.models.is_dir():
        print(f"--models {args.models}: not a directory", file=sys.stderr)
        return 2

    translator = load_model(args.models, max_len=args.max_len, verbose=args.verbose)

    def render(line: str) -> str:
        if pre_render is not None:
            line = pre_render(line)
        if args.tag:
            line = f"##{args.tag} {line}"
        return translator.translate(line)

    if args.text:
        for line in args.text:
            print(render(line))
        return 0

    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if not line:
            print("")
            continue
        print(render(line), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
