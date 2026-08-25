"""``load_npz`` / ``save_npz`` / ``convert_dtype`` round-trip tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qvac_nmt import (
    Config,
    MarianSSRU,
    convert_dtype,
    infer_config,
    load_npz,
    save_npz,
)


def _params_close(a: MarianSSRU, b: MarianSSRU, atol: float, rtol: float = 0.0) -> None:
    pa = dict(a.named_parameters())
    pb = dict(b.named_parameters())
    assert set(pa) == set(pb)
    for name, t_a in pa.items():
        t_b = pb[name]
        assert t_a.shape == t_b.shape, f"{name}: {t_a.shape} vs {t_b.shape}"
        assert torch.allclose(t_a, t_b, atol=atol, rtol=rtol), (
            f"{name}: max diff {(t_a - t_b).abs().max().item():.4g}"
        )


def test_fp32_roundtrip_is_bit_exact(tiny_cfg: Config, tmp_path: Path) -> None:
    """fp32 round-trip must lose nothing."""
    torch.manual_seed(0)
    src = MarianSSRU(tiny_cfg)
    p = tmp_path / "rt.npz"
    save_npz(src, p, dtype="float32")

    dst = MarianSSRU(tiny_cfg)
    load_npz(dst, p)
    _params_close(src, dst, atol=0.0)


def test_fp16_roundtrip_within_half_precision_tolerance(
    tiny_cfg: Config, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    src = MarianSSRU(tiny_cfg)
    p = tmp_path / "rt.fp16.npz"
    save_npz(src, p, dtype="float16")

    dst = MarianSSRU(tiny_cfg)
    load_npz(dst, p)
    _params_close(src, dst, atol=2e-3)


def test_int8_roundtrip_within_per_tensor_quantization_tolerance(
    tiny_cfg: Config, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    src = MarianSSRU(tiny_cfg)
    p = tmp_path / "rt.int8.npz"
    save_npz(src, p, dtype="int8")

    dst = MarianSSRU(tiny_cfg)
    load_npz(dst, p)

    # Per-tensor scalar int8: error proportional to max|W|/127. We don't
    # check parameter-by-parameter equality, only that dequantized weights
    # cluster around the original within ~max|W|/127 + headroom.
    pa = dict(src.named_parameters())
    pb = dict(dst.named_parameters())
    for name, t_a in pa.items():
        t_b = pb[name]
        max_abs = t_a.detach().abs().max().item() or 1e-12
        # Biases / LN scale-bias are kept fp32 -> exact match.
        if "bias" in name or "_ln" in name or "out_bias" in name:
            assert torch.allclose(t_a, t_b, atol=0.0), name
        else:
            assert (t_a - t_b).abs().max().item() <= max_abs / 60.0, name


def test_load_into_wrong_config_raises(tiny_cfg: Config, tmp_path: Path) -> None:
    """Loading a ``d_model=24`` checkpoint into a ``d_model=16`` model must
    raise an actionable ``ValueError``, not corrupt the parameters silently."""
    other_cfg = Config(
        vocab_size=tiny_cfg.vocab_size,
        d_model=24,
        n_heads=4,
        d_ffn=48,
        enc_layers=tiny_cfg.enc_layers,
        dec_layers=tiny_cfg.dec_layers,
    )
    other = MarianSSRU(other_cfg)
    p = tmp_path / "other.npz"
    save_npz(other, p, dtype="float32")

    target = MarianSSRU(tiny_cfg)
    with pytest.raises(ValueError, match=r"infer_config|Wemb shape"):
        load_npz(target, p)


def test_load_with_extra_decoder_layer_raises(tiny_cfg: Config, tmp_path: Path) -> None:
    """If the .npz has more decoder layers than ``cfg.dec_layers``, we must
    detect it via the leftover-keys check and raise."""
    big_cfg = Config(
        vocab_size=tiny_cfg.vocab_size,
        d_model=tiny_cfg.d_model,
        n_heads=tiny_cfg.n_heads,
        d_ffn=tiny_cfg.d_ffn,
        enc_layers=tiny_cfg.enc_layers,
        dec_layers=tiny_cfg.dec_layers + 2,
    )
    big = MarianSSRU(big_cfg)
    p = tmp_path / "big.npz"
    save_npz(big, p, dtype="float32")

    small = MarianSSRU(tiny_cfg)
    with pytest.raises(ValueError, match=r"dec_layers|leftover|expect"):
        load_npz(small, p)


def test_save_synthesizes_yaml_when_missing(tiny_cfg: Config, tmp_path: Path) -> None:
    """A freshly-constructed model has no ``_marian_special``; the saved
    .npz must still embed a ``special:model.yml`` so ``infer_config``
    recovers the exact ``Config``."""
    src = MarianSSRU(tiny_cfg)
    p = tmp_path / "fresh.npz"
    save_npz(src, p, dtype="float32")

    cfg2 = infer_config(p)
    assert cfg2.vocab_size == tiny_cfg.vocab_size
    assert cfg2.d_model == tiny_cfg.d_model
    assert cfg2.n_heads == tiny_cfg.n_heads
    assert cfg2.d_ffn == tiny_cfg.d_ffn
    assert cfg2.enc_layers == tiny_cfg.enc_layers
    assert cfg2.dec_layers == tiny_cfg.dec_layers


def test_convert_dtype_auto_infers_from_npz(
    base_like_cfg: Config, tmp_path: Path
) -> None:
    """Convert a non-Tiny checkpoint without explicit cfg — must pick up
    the dimensions from the embedded YAML, not the package default."""
    src_model = MarianSSRU(base_like_cfg)
    src = tmp_path / "base_like.npz"
    save_npz(src_model, src, dtype="float32")

    dst = tmp_path / "base_like.fp16.npz"
    convert_dtype(src, dst, dtype="float16")  # cfg=None on purpose
    cfg2 = infer_config(dst)
    assert cfg2.d_model == base_like_cfg.d_model
    assert cfg2.dec_layers == base_like_cfg.dec_layers


def test_int8_npz_has_quantmult_companions(tiny_cfg: Config, tmp_path: Path) -> None:
    """Every weight matrix in an int8 ``.npz`` must have a parallel
    ``<name>_quantMult`` scalar; otherwise loading would silently treat the
    int8 bytes as fp32."""
    src = MarianSSRU(tiny_cfg)
    p = tmp_path / "int8.npz"
    save_npz(src, p, dtype="int8")
    z = np.load(str(p))
    matrix_keys = [
        k
        for k, arr in [(k, z[k]) for k in z.files]
        if arr.dtype == np.int8 and not k.startswith("special:")
    ]
    for k in matrix_keys:
        assert f"{k}_quantMult" in z.files, f"missing quantMult for {k}"
