"""Pytest fixtures for qvac_nmt.

Tests run without any of the released model checkpoints — they use small
synthetic ``Config``s (vocab=64, d_model=16, ...) so that round-trips and
forward passes complete in milliseconds and CI machines need no external
files.
"""
from __future__ import annotations

import pytest

from qvac_nmt import Config


@pytest.fixture
def tiny_cfg() -> Config:
    """Pocket-sized Config: small enough for unit tests, structurally complete."""
    return Config(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        d_ffn=32,
        enc_layers=2,
        dec_layers=2,
    )


@pytest.fixture
def base_like_cfg() -> Config:
    """A second shape, used to verify auto-detection actually distinguishes."""
    return Config(
        vocab_size=64,
        d_model=24,
        n_heads=4,
        d_ffn=48,
        enc_layers=2,
        dec_layers=4,
    )
