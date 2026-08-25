"""Marian YAML auto-detection tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvac_nmt import (
    Config,
    config_from_marian_yaml,
    describe_variant,
    infer_config,
    read_marian_yaml,
)


def _yaml_for(cfg: Config) -> dict:
    return {
        "type": "transformer",
        "dec-cell": "ssru",
        "transformer-decoder-autoreg": "rnn",
        "transformer-postprocess": "dan",
        "transformer-ffn-activation": "relu",
        "tied-embeddings-all": True,
        "transformer-train-position-embeddings": False,
        "dim-vocabs": [cfg.vocab_size, cfg.vocab_size],
        "dim-emb": cfg.d_model,
        "transformer-heads": cfg.n_heads,
        "transformer-dim-ffn": cfg.d_ffn,
        "enc-depth": cfg.enc_layers,
        "dec-depth": cfg.dec_layers,
    }


# ---------------------------------------------------------------------------
# config_from_marian_yaml
# ---------------------------------------------------------------------------


def test_config_from_yaml_recovers_known_variants() -> None:
    tiny = Config(d_model=256, d_ffn=1536, enc_layers=6, dec_layers=2, n_heads=8)
    base = Config(d_model=512, d_ffn=2048, enc_layers=6, dec_layers=2, n_heads=8)
    base_mem = Config(d_model=384, d_ffn=1536, enc_layers=6, dec_layers=4, n_heads=8)
    for cfg in (tiny, base, base_mem):
        rebuilt = config_from_marian_yaml(_yaml_for(cfg))
        assert rebuilt == cfg


@pytest.mark.parametrize(
    "knob,bad_value",
    [
        ("type", "rnn"),
        ("dec-cell", "lstm"),
        ("transformer-decoder-autoreg", "self-attention"),
        ("transformer-postprocess", "n"),  # pre-LN
        ("transformer-ffn-activation", "gelu"),
        ("tied-embeddings-all", False),
        ("transformer-train-position-embeddings", True),  # learnt, not sinusoidal
    ],
)
def test_unsupported_variants_raise_descriptive_error(knob: str, bad_value: object) -> None:
    y = _yaml_for(Config())
    y[knob] = bad_value
    with pytest.raises(ValueError, match=knob):
        config_from_marian_yaml(y)


def test_yaml_dim_inconsistency_raises() -> None:
    y = _yaml_for(Config())
    y["dim-emb"] = 257  # not divisible by 8
    with pytest.raises(ValueError, match=r"divisible|inconsistency"):
        config_from_marian_yaml(y)


def test_missing_field_raises() -> None:
    y = _yaml_for(Config())
    del y["enc-depth"]
    with pytest.raises(ValueError):
        config_from_marian_yaml(y)


# ---------------------------------------------------------------------------
# describe_variant
# ---------------------------------------------------------------------------


def test_describe_known_variants() -> None:
    assert describe_variant(
        Config(d_model=256, d_ffn=1536, enc_layers=6, dec_layers=2, n_heads=8)
    ) == "Tiny"
    assert describe_variant(
        Config(d_model=512, d_ffn=2048, enc_layers=6, dec_layers=2, n_heads=8)
    ) == "Base"
    assert describe_variant(
        Config(d_model=384, d_ffn=1536, enc_layers=6, dec_layers=4, n_heads=8)
    ) == "Base-memory"


def test_describe_custom_variant_names_dims() -> None:
    label = describe_variant(
        Config(d_model=128, d_ffn=512, enc_layers=4, dec_layers=2, n_heads=4)
    )
    assert "custom" in label
    assert "d_model=128" in label


# ---------------------------------------------------------------------------
# Sibling YAML resolution + strict mode
# ---------------------------------------------------------------------------


def test_read_marian_yaml_falls_back_to_sibling_basename(tmp_path: Path) -> None:
    """When the .npz has no embedded yaml, look for ``<basename>.yml``."""
    p = tmp_path / "model.npz.best-bleu-detok.npz"
    np.savez(str(p), Wemb=np.zeros((4, 4), dtype=np.float32))
    sib = p.with_suffix(p.suffix + ".yml")
    sib.write_text(
        "type: transformer\n"
        "dec-cell: ssru\n"
        "dec-depth: 2\n"
        "dim-emb: 256\n"
        "dim-vocabs: [32000, 32000]\n"
        "enc-depth: 6\n"
        "tied-embeddings-all: true\n"
        "transformer-decoder-autoreg: rnn\n"
        "transformer-dim-ffn: 1536\n"
        "transformer-ffn-activation: relu\n"
        "transformer-heads: 8\n"
        "transformer-postprocess: dan\n"
        "transformer-train-position-embeddings: false\n"
    )
    y = read_marian_yaml(p)
    assert y is not None
    cfg = config_from_marian_yaml(y)
    assert cfg.d_model == 256


def test_infer_config_strict_raises_when_no_yaml(tmp_path: Path) -> None:
    p = tmp_path / "naked.npz"
    np.savez(str(p), Wemb=np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match=r"special:model\.yml|YAML"):
        infer_config(p)


def test_infer_config_non_strict_falls_back_to_default(tmp_path: Path) -> None:
    p = tmp_path / "naked.npz"
    np.savez(str(p), Wemb=np.zeros((4, 4), dtype=np.float32))
    cfg = infer_config(p, strict=False)
    assert cfg == Config()
