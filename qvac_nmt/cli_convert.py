"""``qvac-convert`` — re-emit a Marian ``.npz`` at a different precision.

Use this once on an fp32 checkpoint to produce smaller fp16 / int8 variants
that ``qvac-translate`` will load directly without any code change.

Examples::

    # fp32 -> fp16 (roughly half the size), recommended for CPU inference
    qvac-convert model.fp32.npz model.fp16.npz --dtype fp16

    # fp32 -> int8 with embedded quantMult scalars (roughly a quarter)
    qvac-convert model.fp32.npz model.int8.npz --dtype int8

The output is a plain Marian-format ``.npz`` and is read back by
:func:`qvac_nmt.load_npz` without configuration. fp16 entries are upcast
on load; int8 entries are dequantized using the embedded
``<name>_quantMult`` scalars. The architecture is auto-detected from the
source's embedded ``special:model.yml`` (which is preserved on output
so the destination remains self-describing).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .loader import convert_dtype

_ALIASES = {
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "int8": "int8",
    "i8": "int8",
}


def build_parser(prog: str = "qvac-convert") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Convert a Marian .npz checkpoint between fp32 / fp16 / int8.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("src", type=Path, help="Input .npz")
    p.add_argument("dst", type=Path, help="Output .npz")
    p.add_argument(
        "--dtype",
        required=True,
        choices=sorted(_ALIASES),
        help="Target dtype.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = _ALIASES[args.dtype]
    if not args.src.is_file():
        print(f"input not found: {args.src}", file=sys.stderr)
        return 2
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    convert_dtype(args.src, args.dst, dtype=target)
    sz = args.dst.stat().st_size / (1024 * 1024)
    print(
        f"wrote {args.dst} ({sz:.1f} MB, {target}) in "
        f"{time.perf_counter() - t0:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
