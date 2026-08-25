"""Bergamot-recipe Marian Transformer architecture in pure PyTorch.

This module defines only the model topology — no weight loading, no decoding
logic. See :mod:`qvac_nmt.loader` for ``.npz`` weight I/O and
:mod:`qvac_nmt.translator` for greedy translation.

Architectural recipe (the Bergamot / TranslatePsy-Nano student-model family)::

    type: transformer
    dec-cell: ssru, transformer-decoder-autoreg: rnn
    transformer-postprocess: dan         (post-LayerNorm)
    transformer-ffn-activation: relu
    tied-embeddings-all: true
    transformer-train-position-embeddings: false   (sinusoidal)

Dimensions (`dim-emb`, `transformer-dim-ffn`, `transformer-heads`,
`enc-depth`, `dec-depth`, `dim-vocabs`) are read from each checkpoint's
embedded ``special:model.yml``; the topology defined here adapts to all
released variants automatically.

Marian-specific quirks reproduced here:

1. SSRU recurrence is on the *cell* state ``c_t`` (not the hidden state).
   The output handed to the next sublayer is ``relu(c_t)``. ``W`` has no bias.
   See ``marian-dev/src/rnn/cells.h::SSRU``.
2. Sinusoidal position embeddings use a **block layout**
   (``[sin sin ... sin cos cos ... cos]``, not interleaved) and the frequency
   denominator is ``(d/2 - 1)`` not ``d/2``.
   See ``node_initializers.cpp::sinusoidalPositionEmbeddings``.
3. The decoder is bootstrapped at step 0 with a **zero token embedding**.
   See ``decoder.h::embeddingsFromPrediction``.
4. LayerNorm ``eps = 1e-9`` (Marian default).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """Architecture hyper-parameters for a Bergamot-recipe Marian model.

    The defaults match the smallest variant in the family (``Tiny``, used by
    TranslatePsy-Nano Tiny checkpoints). For real checkpoints prefer
    :func:`qvac_nmt.infer_config`, which derives the right ``Config`` from
    the ``.npz``'s embedded YAML so all variants load without manual tuning.
    """

    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    d_ffn: int = 1536
    enc_layers: int = 6
    dec_layers: int = 2
    pad_id: int = 0
    eos_id: int = 0


# ---------------------------------------------------------------------------
# Sinusoidal position embeddings (Marian flavour)
# ---------------------------------------------------------------------------


def sinusoidal_position_embedding(
    n_positions: int, dim: int, start: int = 0
) -> torch.Tensor:
    """Marian-flavoured sinusoidal position embeddings, ``[n_positions, dim]``.

    Differences vs the textbook ``Attention Is All You Need`` formula:

    * **Block layout**: dims ``[0, d/2)`` are sins, dims ``[d/2, d)`` are cosines
      (no ``0::2`` / ``1::2`` interleaving).
    * Frequency denominator is ``(d/2 - 1)``, not ``d/2``.
    """
    num_timescales = dim // 2
    log_timescale_increment = math.log(10000.0) / (num_timescales - 1)
    inv_freq = torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
    )
    pos = torch.arange(start, start + n_positions, dtype=torch.float32).unsqueeze(1)
    args = pos * inv_freq
    pe = torch.empty(n_positions, dim, dtype=torch.float32)
    pe[:, :num_timescales] = torch.sin(args)
    pe[:, num_timescales:] = torch.cos(args)
    return pe


# ---------------------------------------------------------------------------
# Sublayers
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split(self.q(q_in))
        k = self._split(self.k(k_in))
        v = self._split(self.v(v_in))
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.o(self._merge(out))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.self_attn_ln = nn.LayerNorm(cfg.d_model, eps=1e-9)
        self.ffn = FFN(cfg.d_model, cfg.d_ffn)
        self.ffn_ln = nn.LayerNorm(cfg.d_model, eps=1e-9)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None
    ) -> torch.Tensor:
        # Post-LN: y = LN(x + sublayer(x))
        x = self.self_attn_ln(x + self.self_attn(x, x, x, src_mask))
        x = self.ffn_ln(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(cfg.enc_layers)]
        )

    def forward(
        self, emb: torch.Tensor, src_mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = emb
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ---------------------------------------------------------------------------
# SSRU decoder
# ---------------------------------------------------------------------------


class SSRUCell(nn.Module):
    """Marian's Simpler Simple Recurrent Unit.

    Update equations::

        x_pre = W  @ x_t                       (no bias, no relu)
        f_pre = Wf @ x_t + bf
        c_t   = sigmoid(f_pre) * c_{t-1} + (1 - sigmoid(f_pre)) * x_pre
        h_t   = relu(c_t)                       <- output to next sublayer

    The recurrence is on the *cell* state ``c_t``, not on ``h_t``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.Wf = nn.Linear(dim, dim, bias=True)

    def step(
        self, x_t: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pre = self.W(x_t)
        f = torch.sigmoid(self.Wf(x_t))
        c_t = f * c_prev + (1.0 - f) * x_pre
        return F.relu(c_t), c_t

    def run_sequence(
        self, xs: torch.Tensor, c0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = xs.shape
        if c0 is None:
            c0 = xs.new_zeros(b, d)
        # Pre-compute the affine projections for the whole sequence; only the
        # cell-state recurrence is sequential.
        x_all = self.W(xs)
        f_all = torch.sigmoid(self.Wf(xs))
        outs = []
        c = c0
        for i in range(t):
            c = f_all[:, i, :] * c + (1.0 - f_all[:, i, :]) * x_all[:, i, :]
            outs.append(F.relu(c))
        return torch.stack(outs, dim=1), c


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.rnn = SSRUCell(cfg.d_model)
        self.rnn_ln = nn.LayerNorm(cfg.d_model, eps=1e-9)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.cross_attn_ln = nn.LayerNorm(cfg.d_model, eps=1e-9)
        self.ffn = FFN(cfg.d_model, cfg.d_ffn)
        self.ffn_ln = nn.LayerNorm(cfg.d_model, eps=1e-9)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        src_mask: torch.Tensor | None,
        c_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_seq, c_last = self.rnn.run_sequence(x, c_prev)
        x = self.rnn_ln(x + h_seq)
        x = self.cross_attn_ln(
            x + self.cross_attn(x, encoder_out, encoder_out, src_mask)
        )
        x = self.ffn_ln(x + self.ffn(x))
        return x, c_last

    def step(
        self,
        x_t: torch.Tensor,
        encoder_out: torch.Tensor,
        src_mask: torch.Tensor | None,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_t, c_t = self.rnn.step(x_t, c_prev)
        x_t = self.rnn_ln(x_t + h_t)
        y = self.cross_attn(x_t.unsqueeze(1), encoder_out, encoder_out, src_mask)
        x_t = self.cross_attn_ln(x_t + y.squeeze(1))
        x_t = self.ffn_ln(x_t + self.ffn(x_t))
        return x_t, c_t


class Decoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.dec_layers)]
        )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class MarianSSRU(nn.Module):
    """Marian Transformer encoder + SSRU recurrent decoder, in pure PyTorch.

    Implements the Bergamot / TranslatePsy-Nano student-model recipe: post-LN,
    sinusoidal block-layout positions, ReLU FFN, tied input/output embeddings.
    The same class handles every member of the family — Tiny, Base,
    Base-memory, EuroNano / AfriNano, and custom widths — by parameterising
    width/depth via :class:`Config`.

    The output projection weight is **tied** to the input embedding
    (``tied-embeddings-all: true``); only ``out_bias`` is a separate parameter.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    # ----- embedding helpers -----------------------------------------------

    def src_emb(self, src_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(src_ids) * math.sqrt(self.cfg.d_model)
        T = src_ids.size(1)
        pe = sinusoidal_position_embedding(T, self.cfg.d_model).to(
            device=x.device, dtype=x.dtype
        )
        return x + pe

    def tgt_emb(self, tgt_ids: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        x = self.embed(tgt_ids) * math.sqrt(self.cfg.d_model)
        T = tgt_ids.size(1)
        pe = sinusoidal_position_embedding(T, self.cfg.d_model, start=pos_offset).to(
            device=x.device, dtype=x.dtype
        )
        return x + pe

    # ----- forward halves --------------------------------------------------

    def encode(
        self, src_ids: torch.Tensor, src_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        emb = self.src_emb(src_ids)
        attn_mask = (1.0 - src_pad_mask.to(emb.dtype)).unsqueeze(1).unsqueeze(1) * -1e4
        return self.encoder(emb, attn_mask)

    def project(self, dec_out: torch.Tensor) -> torch.Tensor:
        return F.linear(dec_out, self.embed.weight, self.out_bias)

    def decode_train(
        self,
        encoder_out: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decoder, returns logits ``[B, L, V]``.

        Marian convention (``decoder.h::embeddingsFromPrediction``): step 0 is
        bootstrapped with a zero embedding. From step ``t >= 1`` the input is
        ``embed(tgt[t-1])``. ``logits[:, t]`` predicts ``tgt_ids[:, t]``.
        """
        b = encoder_out.size(0)
        d = self.cfg.d_model
        L = tgt_ids.size(1)

        dt = encoder_out.dtype
        zero_step = encoder_out.new_zeros(b, 1, d)
        if L > 1:
            shifted = self.embed(tgt_ids[:, :-1]) * math.sqrt(d)
            tok_emb = torch.cat([zero_step, shifted], dim=1)
        else:
            tok_emb = zero_step
        pe = sinusoidal_position_embedding(L, d).to(
            device=encoder_out.device, dtype=dt
        )
        emb = tok_emb + pe.unsqueeze(0)

        attn_mask = (1.0 - src_pad_mask.to(dt)).unsqueeze(1).unsqueeze(1) * -1e4
        x = emb
        for layer in self.decoder.layers:
            x, _ = layer(x, encoder_out, attn_mask, c_prev=None)
        return self.project(x)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced encoder + decoder. Use with ``cross_entropy``.

        Example::

            logits = model(src_ids, tgt_ids, src_pad_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, V), tgt_ids.reshape(-1),
                ignore_index=cfg.pad_id,
            )
        """
        if src_pad_mask is None:
            src_pad_mask = (src_ids != self.cfg.pad_id).long()
        encoder_out = self.encode(src_ids, src_pad_mask)
        return self.decode_train(encoder_out, src_pad_mask, tgt_ids)


__all__ = [
    "Config",
    "MultiHeadAttention",
    "FFN",
    "EncoderLayer",
    "Encoder",
    "SSRUCell",
    "DecoderLayer",
    "Decoder",
    "MarianSSRU",
    "sinusoidal_position_embedding",
]
