"""Minimal end-to-end example.

Expected layout (relative to the repo root)::

    models/
    ├── en-xx/
    │   ├── model.npz       (any dtype: fp32, fp16, or int8)
    │   └── vocab.spm
    └── xx-en/
        ├── model.npz
        └── vocab.spm

Works with either TranslatePsy-EuroNano or TranslatePsy-AfriNano
checkpoints (same layout). Adjust the language tags below to match the
model you placed under ``models/``.

Run::

    uv run python examples/basic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from qvac_nmt import load_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    en2xx_dir = ROOT / "models" / "en-xx"
    xx2en_dir = ROOT / "models" / "xx-en"
    for d in (en2xx_dir, xx2en_dir):
        if not d.is_dir() or not list(d.glob("*.npz")) or not list(d.glob("*.spm")):
            print(
                f"missing model: {d}\n"
                f"  expected one .npz and one .spm; see README.md for setup.",
                file=sys.stderr,
            )
            return 2

    en2xx = load_model(en2xx_dir, verbose=True)
    xx2en = load_model(xx2en_dir, verbose=True)

    print("\nEnglish -> target")
    for tag, text in [
        ("DE", "Where is the bus stop?"),
        ("FR", "Good morning, my friend."),
        ("SW", "Where is the nearest hospital?"),
    ]:
        print(f"  [{tag}] {text!r:40s} -> {en2xx.translate(f'##{tag} {text}')}")

    print("\nSource -> English")
    for text in ["Wo ist die Bushaltestelle?", "Habari yako leo?"]:
        print(f"  {text!r:40s} -> {xx2en.translate(text)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
