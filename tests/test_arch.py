"""Architecture-level sanity tests.

These exercise the model topology with random weights — no external
checkpoints required. They guard against regressions in shape, dtype,
and the Marian-flavoured quirks (SSRU recurrence, sinusoidal positions).
"""
from __future__ import annotations

import torch

from qvac_nmt import (
    Config,
    MarianSSRU,
    SSRUCell,
    sinusoidal_position_embedding,
)

# ---------------------------------------------------------------------------
# Sinusoidal positions
# ---------------------------------------------------------------------------


def test_sinusoidal_block_layout_and_shape() -> None:
    pe = sinusoidal_position_embedding(n_positions=7, dim=16)
    assert pe.shape == (7, 16)
    # Block layout: dims [0, d/2) are sins, [d/2, d) are cosines. At position 0,
    # the sin half should be 0 and the cos half should be 1.
    assert torch.allclose(pe[0, :8], torch.zeros(8))
    assert torch.allclose(pe[0, 8:], torch.ones(8))


def test_sinusoidal_start_offset_is_consistent() -> None:
    full = sinusoidal_position_embedding(n_positions=5, dim=16)
    tail = sinusoidal_position_embedding(n_positions=2, dim=16, start=3)
    assert torch.allclose(full[3:], tail)


# ---------------------------------------------------------------------------
# SSRU
# ---------------------------------------------------------------------------


def test_ssru_step_and_run_sequence_agree() -> None:
    """``run_sequence`` must equal stepping one token at a time."""
    torch.manual_seed(0)
    cell = SSRUCell(dim=8)
    xs = torch.randn(2, 5, 8)
    seq_h, seq_c = cell.run_sequence(xs)

    c = torch.zeros(2, 8)
    step_h = []
    for t in range(5):
        h, c = cell.step(xs[:, t, :], c)
        step_h.append(h)
    step_h = torch.stack(step_h, dim=1)

    assert torch.allclose(step_h, seq_h, atol=1e-6)
    assert torch.allclose(c, seq_c, atol=1e-6)


def test_ssru_W_has_no_bias() -> None:
    """The Marian SSRU specifies ``W`` without a bias; that's what disk
    layout assumes (no ``decoder_l*_rnn_b`` key)."""
    cell = SSRUCell(dim=8)
    assert cell.W.bias is None
    assert cell.Wf.bias is not None


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


def test_forward_returns_logits_with_correct_shape(tiny_cfg: Config) -> None:
    torch.manual_seed(0)
    model = MarianSSRU(tiny_cfg).eval()
    src = torch.randint(1, tiny_cfg.vocab_size, (2, 7))
    tgt = torch.randint(1, tiny_cfg.vocab_size, (2, 5))
    with torch.no_grad():
        logits = model(src, tgt)
    assert logits.shape == (2, 5, tiny_cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backward_pass_runs(tiny_cfg: Config) -> None:
    torch.manual_seed(0)
    model = MarianSSRU(tiny_cfg)
    src = torch.randint(1, tiny_cfg.vocab_size, (2, 6))
    tgt = torch.randint(1, tiny_cfg.vocab_size, (2, 4))
    logits = model(src, tgt)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, tiny_cfg.vocab_size), tgt.reshape(-1)
    )
    loss.backward()
    # Tied output projection: gradient should flow into the embedding.
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_decoder_zero_token_bootstrap_at_step_0(tiny_cfg: Config) -> None:
    """Marian convention: step 0 of the decoder uses a *zero* token embedding,
    not embed(eos). Verifies that the teacher-forced ``decode_train`` is
    invariant to whatever id we put at position 0 of ``tgt_ids`` (because it
    would be replaced by zero anyway).

    The trick here: with our shift-right convention, ``logits[:, 0]`` only
    depends on the encoder output and the zero bootstrap. Changing
    ``tgt[:, 0]`` doesn't change anything until column 1.
    """
    torch.manual_seed(0)
    model = MarianSSRU(tiny_cfg).eval()
    src = torch.randint(1, tiny_cfg.vocab_size, (1, 6))
    tgt_a = torch.tensor([[3, 7, 9, 1]])
    tgt_b = torch.tensor([[42, 7, 9, 1]])
    with torch.no_grad():
        a = model(src, tgt_a)
        b = model(src, tgt_b)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-6)
    # Whereas later positions may differ (different tgt -> different shift)
    assert not torch.allclose(a[:, 1:], b[:, 1:], atol=1e-6)


def test_tied_output_projection(tiny_cfg: Config) -> None:
    """``model.project`` must use the embedding matrix as the output weight
    (Marian's ``tied-embeddings-all: true``)."""
    model = MarianSSRU(tiny_cfg).eval()
    h = torch.randn(1, 3, tiny_cfg.d_model)
    expected = h @ model.embed.weight.T + model.out_bias
    assert torch.allclose(model.project(h), expected, atol=1e-6)


def test_dtype_propagates_through_encode(tiny_cfg: Config) -> None:
    """After ``model.half()`` the encoder output should be fp16 — i.e. the
    sinusoidal-PE addition didn't silently upcast back to fp32."""
    model = MarianSSRU(tiny_cfg).eval().half()
    src = torch.randint(1, tiny_cfg.vocab_size, (1, 4))
    src_mask = torch.ones_like(src)
    with torch.no_grad():
        enc = model.encode(src, src_mask)
    assert enc.dtype == torch.float16


def test_param_count_is_finite_and_correct_order_of_magnitude(tiny_cfg: Config) -> None:
    n = sum(p.numel() for p in MarianSSRU(tiny_cfg).parameters())
    # 64 vocab * 16 d_model = 1024 just for the embedding.
    assert n > 1024
    assert n < 100_000
