"""Integration tests against real Bergamot-recipe Marian ``.npz`` checkpoints.

These tests are skipped by default. They run only when one of two env vars
points at a populated directory of model checkpoints:

* ``QVAC_NMT_REAL_MODELS_DIR``  — a path you already have locally.
* ``QVAC_NMT_HF_REPOS``         — comma-separated Hugging Face repo IDs;
  the test downloads each into ``QVAC_NMT_HF_CACHE_DIR`` (default
  ``~/.cache/qvac_nmt/real``) using ``huggingface_hub`` and treats the
  resulting tree as the model directory.

The directory layout is whatever the test sees: every subdirectory that
contains exactly one ``.npz`` and one ``.spm`` is treated as one model
checkpoint. Each is parametrised into the test, so adding a model is
just a matter of dropping it into the directory.

What gets verified for each checkpoint:

1. The Marian YAML embedded in the ``.npz`` matches all
   :data:`qvac_nmt.config_loader._REQUIRED_INVARIANTS` (post-LN, SSRU,
   tied-embeddings-all, sinusoidal positions, ReLU FFN, RNN-style
   autoregressive decoding).
2. ``infer_config`` succeeds and returns a sensible ``Config`` (heads
   divides ``d_model``, vocab > 1024, etc.).
3. ``load_model`` succeeds end-to-end and ``translate("Hello world.")``
   produces a non-empty string.

Note on availability of public test fixtures: most distributors publish
Bergamot-recipe checkpoints in the ``.intgemm.alphas.bin`` runtime format,
not the ``.npz`` Marian-training format that this package consumes.
Public ``.npz`` checkpoints: any Marian export that follows the Bergamot
recipe (TranslatePsy-EuroNano, TranslatePsy-AfriNano, or your own training
run). Pass repo IDs via ``QVAC_NMT_HF_REPOS`` or drop local ``.npz`` /
``.spm`` pairs under ``QVAC_NMT_REAL_MODELS_DIR``.

If you can point the env var at any of these the test runs; otherwise it
skips with a clear message.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from qvac_nmt import infer_config, load_model
from qvac_nmt.config_loader import _REQUIRED_INVARIANTS, read_marian_yaml

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _candidate_dirs(root: Path) -> list[Path]:
    """Return every subdirectory of ``root`` that has one .npz + one .spm."""
    out: list[Path] = []
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        npz = list(sub.glob("*.npz"))
        spm = list(sub.glob("*.spm"))
        if len(npz) == 1 and len(spm) == 1:
            out.append(sub)
    return out


def _populate_from_hf() -> Path | None:
    """If ``QVAC_NMT_HF_REPOS`` is set, snapshot-download each repo into
    ``QVAC_NMT_HF_CACHE_DIR`` and return that root.
    """
    repos = os.environ.get("QVAC_NMT_HF_REPOS", "").strip()
    if not repos:
        return None
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pytest.skip(
            "QVAC_NMT_HF_REPOS is set but huggingface_hub is not installed. "
            "Install with `uv sync --group dev` or `pip install huggingface-hub`."
        )

    cache_root = Path(
        os.environ.get("QVAC_NMT_HF_CACHE_DIR", str(Path.home() / ".cache/qvac_nmt/real"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    for repo_id in (r.strip() for r in repos.split(",") if r.strip()):
        # Slashes are illegal in dir names; encode them.
        local_dir = cache_root / repo_id.replace("/", "__")
        if not local_dir.is_dir() or not list(local_dir.glob("*.npz")):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    allow_patterns=["*.npz", "*.spm"],
                )
            except Exception as exc:
                pytest.skip(f"could not snapshot_download {repo_id}: {exc}")
    return cache_root


def _resolve_models_root() -> Path | None:
    """Return the root dir to scan for real checkpoints, or None to skip."""
    explicit = os.environ.get("QVAC_NMT_REAL_MODELS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    return _populate_from_hf()


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------


def _models_for_pytest() -> list[Path]:
    root = _resolve_models_root()
    if root is None:
        return []
    return _candidate_dirs(root)


_MODELS = _models_for_pytest()
_SKIP_REASON = (
    "set QVAC_NMT_REAL_MODELS_DIR to a directory containing Bergamot-recipe "
    "Marian .npz checkpoints (one subdirectory per model, each holding one "
    ".npz and one .spm) — or set QVAC_NMT_HF_REPOS to a comma-separated list "
    "of Hugging Face repo IDs to auto-download."
)


@pytest.fixture(scope="module")
def model_dirs() -> list[Path]:
    return _MODELS


@pytest.mark.skipif(not _MODELS, reason=_SKIP_REASON)
@pytest.mark.parametrize("model_dir", _MODELS, ids=lambda p: p.name)
class TestRealCheckpoint:
    def test_yaml_satisfies_required_invariants(self, model_dir: Path) -> None:
        npz = next(model_dir.glob("*.npz"))
        y: dict[str, Any] | None = read_marian_yaml(npz)
        assert y is not None, (
            f"{npz} has no embedded special:model.yml and no sibling .yml. "
            f"This package needs the YAML to auto-detect dimensions."
        )
        for key, expected in _REQUIRED_INVARIANTS.items():
            assert y.get(key) == expected, (
                f"{npz}: {key}={y.get(key)!r}, expected {expected!r}. "
                f"This checkpoint is not Bergamot-recipe."
            )

    def test_infer_config_returns_sane_dims(self, model_dir: Path) -> None:
        npz = next(model_dir.glob("*.npz"))
        cfg = infer_config(npz)
        assert cfg.vocab_size > 1024
        assert cfg.d_model >= 64
        assert cfg.d_ffn >= cfg.d_model
        assert cfg.n_heads >= 1
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.enc_layers >= 1
        assert cfg.dec_layers >= 1

    def test_load_and_translate(self, model_dir: Path) -> None:
        translator = load_model(model_dir)
        out = translator.translate("Hello world.")
        assert isinstance(out, str)
        assert len(out) > 0
