"""``afritranslate`` console script — TranslatePsy-AfriNano convenience wrapper.

This is a thin shim around :func:`qvac_nmt.cli_translate.run` that:

1. Restricts ``--tag`` to the eight African target-language codes that
   TranslatePsy-AfriNano was trained with.
2. Sets the program name and help text for the AfriNano product context.

Examples::

    afritranslate --models models/en-xx --tag SW "Good morning, my friend."
    afritranslate --models models/xx-en "Habari yako leo?"

Equivalent without the wrapper::

    qvac-translate --models models/en-xx --tag SW "Good morning, my friend."

The library code is identical; this entry point only whitelists the eight
AfriNano language tags for convenience.
"""
from __future__ import annotations

from . import cli_translate

#: Eight target-language tags TranslatePsy-AfriNano was trained with.
#: Used as the ``--tag`` whitelist for this shim.
AFRITRANSLATE_TAGS: list[str] = ["SW", "HA", "IG", "YO", "ZU", "AM", "LN", "SO"]


def build_parser():
    return cli_translate.build_parser(
        prog="afritranslate",
        description=(
            "Translate with TranslatePsy-AfriNano "
            "(en <-> Swahili / Hausa / Igbo / Yoruba / Zulu / Amharic / "
            "Lingala / Somali) in pure PyTorch."
        ),
        tag_choices=AFRITRANSLATE_TAGS,
        tag_help=(
            "Target-language tag for English -> African direction "
            "(one of: " + ", ".join(AFRITRANSLATE_TAGS) + "). "
            "Will be prepended as '##TAG '. Omit for African -> English."
        ),
        epilog=__doc__,
    )


def main(argv: list[str] | None = None) -> int:
    return cli_translate.run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
