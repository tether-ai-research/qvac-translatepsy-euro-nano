#!/usr/bin/env python3
"""Populate a directory with real Bergamot-recipe Marian ``.npz`` checkpoints
for ``tests/test_real_checkpoints.py``.

Usage::

    # Default: download all models in DEFAULT_REPOS to ./real_models/
    python scripts/fetch_real_models.py

    # Custom destination + custom repo list
    python scripts/fetch_real_models.py \\
        --dest /tmp/qvac_real \\
        --repos your-org/model-a your-org/model-b

    # Then run the integration tests against it
    QVAC_NMT_REAL_MODELS_DIR=/tmp/qvac_real uv run pytest tests/test_real_checkpoints.py -v

The script fetches only ``*.npz`` and ``*.spm`` files from each repo. Each
repo lands in ``{dest}/{repo_name_with_slashes_replaced}/`` so the
integration test's directory scan picks them up automatically.

Notes on availability of public ``.npz`` checkpoints
----------------------------------------------------

Most distributors of Bergamot-recipe models publish only the int8
``.intgemm.alphas.bin`` runtime format. ``qvac-nmt`` deliberately does not
load that format — it loads the upstream Marian-training ``.npz``. Public
``.npz`` repos are relatively rare; pass the repo IDs you host with
``--repos``. If you have a private Marian ``.npz`` from your own training
run, just drop it in a subdirectory of ``--dest`` (alongside its ``.spm``
vocab) — no download required.

Hugging Face authentication: any of the listed repos may require an
``HF_TOKEN`` environment variable if the repo is gated. The script
forwards whatever is in your environment to ``snapshot_download``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPOS: list[str] = []


def fetch_one(repo_id: str, dest_root: Path) -> Path | None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required. Install it with one of:\n"
            "  uv sync --group dev          # adds it as a dev dep\n"
            "  pip install huggingface-hub  # standalone\n"
            "  pip install qvac-nmt[hf]     # via the optional 'hf' extra",
            file=sys.stderr,
        )
        return None

    local_dir = dest_root / repo_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"-> {repo_id}: downloading to {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=["*.npz", "*.spm"],
        )
    except Exception as exc:
        print(f"   FAILED: {exc}", file=sys.stderr)
        return None

    npzs = list(local_dir.glob("*.npz"))
    spms = list(local_dir.glob("*.spm"))
    if not npzs or not spms:
        print(
            f"   warning: {repo_id} yielded npz={[p.name for p in npzs]} "
            f"spm={[p.name for p in spms]}; the integration test expects "
            f"exactly one of each at the top level.",
            file=sys.stderr,
        )
    else:
        print(f"   ok: {npzs[0].name} ({npzs[0].stat().st_size / 1e6:.1f} MB)")
    return local_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("real_models"),
        help="Destination directory (default: ./real_models).",
    )
    p.add_argument(
        "--repos",
        nargs="+",
        default=DEFAULT_REPOS,
        help="Hugging Face repo IDs to download (no default; pass the repos you host).",
    )
    args = p.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)

    if not args.repos:
        print(
            "no --repos specified; nothing to download.\n"
            "Pass one or more Hugging Face repo IDs, e.g.\n"
            "  python scripts/fetch_real_models.py --repos your-org/your-model",
            file=sys.stderr,
        )
        return 0

    successes = 0
    for repo in args.repos:
        if fetch_one(repo, args.dest) is not None:
            successes += 1

    print(
        f"\nfetched {successes}/{len(args.repos)} repo(s) into {args.dest}\n"
        f"Run integration tests with:\n"
        f"  QVAC_NMT_REAL_MODELS_DIR={args.dest} uv run pytest tests/test_real_checkpoints.py -v"
    )
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
