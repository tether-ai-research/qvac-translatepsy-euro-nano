"""Greedy translator + ``load_model`` convenience.

The translator is intentionally simple: greedy (beam=1) decoding, single
sentence at a time, suitable for CPU-only on-device inference.

For batched / beam-search use cases, build on the ``TinyMarian.encode`` and
``DecoderLayer.step`` primitives directly.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .config_loader import describe_variant, infer_config
from .loader import load_npz
from .model import Config, MarianSSRU, sinusoidal_position_embedding


class Translator:
    """Greedy beam=1 translator.

    Parameters
    ----------
    model : MarianSSRU
        A weight-loaded model. ``Translator`` puts it in ``eval()`` mode.
    sp_path : str | Path
        Path to the SentencePiece ``.spm`` vocab file (joint src/tgt).
    eos_id : int
        End-of-sentence id (Marian convention is 0).
    max_len : int
        Hard upper bound on target tokens before forced stop.
    """

    def __init__(
        self,
        model: MarianSSRU,
        sp_path: str | Path,
        eos_id: int = 0,
        max_len: int = 200,
    ) -> None:
        import sentencepiece as spm

        self.model = model.eval()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(sp_path))
        self.eos_id = eos_id
        self.max_len = max_len

    def encode_input(self, text: str) -> torch.Tensor:
        ids = self.sp.EncodeAsIds(text) + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    @torch.no_grad()
    def translate(self, text: str) -> str:
        """Translate one sentence and return the detokenized string."""
        src_ids = self.encode_input(text)
        src_mask = torch.ones_like(src_ids)
        encoder_out = self.model.encode(src_ids, src_mask)
        dt = encoder_out.dtype

        attn_mask = (1.0 - src_mask.to(dt)).unsqueeze(1).unsqueeze(1) * -1e4
        d = self.model.cfg.d_model
        c_prev_per_layer = [
            encoder_out.new_zeros(1, d) for _ in range(self.model.cfg.dec_layers)
        ]
        out_ids: list[int] = []
        prev_token: int | None = None

        for step in range(self.max_len):
            # Marian convention: zero-token bootstrap at step 0; embed(prev_token) thereafter.
            if prev_token is None:
                tok_emb = encoder_out.new_zeros(1, 1, d)
            else:
                tok_ids = torch.tensor(
                    [[prev_token]], dtype=torch.long, device=encoder_out.device
                )
                tok_emb = self.model.embed(tok_ids) * math.sqrt(d)
            pe = sinusoidal_position_embedding(1, d, start=step).to(
                device=encoder_out.device, dtype=dt
            )
            x_t = (tok_emb + pe).squeeze(1)
            for li, layer in enumerate(self.model.decoder.layers):
                x_t, c_prev_per_layer[li] = layer.step(
                    x_t, encoder_out, attn_mask, c_prev_per_layer[li]
                )
            logits = self.model.project(x_t.unsqueeze(1)).squeeze(1)
            nxt = int(logits.argmax(dim=-1).item())
            if nxt == self.eos_id:
                break
            out_ids.append(nxt)
            prev_token = nxt

        return self.sp.DecodeIds(out_ids)


def load_model(
    model_dir: str | Path,
    *,
    cfg: Config | None = None,
    max_len: int = 200,
    verbose: bool = False,
) -> Translator:
    """Build a :class:`Translator` from a directory containing one ``.npz``
    weight file and one ``.spm`` SentencePiece vocab.

    The directory layout is intentionally minimal::

        my-model-dir/
        ├── model.npz
        └── vocab.spm

    The architecture (Tiny / Base / Base-memory / custom) is auto-detected
    from the Marian YAML embedded in the ``.npz``: pass ``cfg=...`` only if
    you want to override that. The same code path therefore loads any of the
    three released variants without configuration.

    Parameters
    ----------
    model_dir
        Directory with exactly one ``.npz`` and one ``.spm``.
    cfg
        Optional explicit :class:`Config` to bypass YAML auto-detection. Useful
        for checkpoints whose ``special:model.yml`` was stripped.
    max_len
        Maximum target tokens before forced stop.
    verbose
        If ``True``, prints the inferred variant name and key dimensions.
    """
    d = Path(model_dir)
    if d.is_file() and d.suffix == ".npz":
        raise ValueError(
            "load_model expects a directory; for a single .npz, "
            "instantiate MarianSSRU + load_npz + Translator manually."
        )
    npz_files = sorted(d.glob("*.npz"))
    spm_files = sorted(d.glob("*.spm"))
    if len(npz_files) != 1 or len(spm_files) != 1:
        raise FileNotFoundError(
            f"expected exactly one .npz and one .spm under {d}, "
            f"found npz={[p.name for p in npz_files]} spm={[p.name for p in spm_files]}"
        )

    if cfg is None:
        cfg = infer_config(npz_files[0])

    if verbose:
        print(
            f"[qvac-nmt] {npz_files[0].name}: "
            f"variant={describe_variant(cfg)} "
            f"d_model={cfg.d_model} d_ffn={cfg.d_ffn} "
            f"enc={cfg.enc_layers} dec={cfg.dec_layers} "
            f"vocab={cfg.vocab_size}"
        )
    model = MarianSSRU(cfg)
    load_npz(model, npz_files[0])
    return Translator(model, spm_files[0], eos_id=cfg.eos_id, max_len=max_len)


__all__ = ["Translator", "load_model"]
