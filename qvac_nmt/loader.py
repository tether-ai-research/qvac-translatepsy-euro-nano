"""Marian ``.npz`` weight I/O.

Three operations are supported:

* :func:`load_npz` — reads a Marian ``.npz`` checkpoint into a ``MarianSSRU``.
  Accepts fp32, fp16, or int8 (with parallel ``<name>_quantMult`` scalars).
* :func:`save_npz` — writes a ``MarianSSRU`` to ``.npz`` at fp32, fp16, or int8.
* :func:`convert_dtype` — round-trips a ``.npz`` from one precision to another
  without instantiating a model in user code.

Notes on Marian's conventions reproduced here:

* ``Wemb`` is stored ``[vocab, dim]`` (no transpose; same as PyTorch
  ``nn.Embedding.weight``).
* All other matrices are stored in **natural** ``[in, out]`` mathematical
  layout. PyTorch ``nn.Linear.weight`` is ``[out, in]`` so we transpose on
  every load and save.
* Biases / LayerNorm scale & bias are ``[1, dim]`` arrays (we flatten on load,
  reshape on save).
* For int8 storage, each quantized array ``W`` is paired with a scalar
  ``W_quantMult`` such that ``W_fp32 = W.astype(float32) / quantMult``.

The output projection weight is *tied* to ``Wemb``; only ``out_bias`` is
stored separately as ``decoder_ff_logit_out_b``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .model import Config, MarianSSRU

DType = Literal["float32", "float16", "int8"]


# ---------------------------------------------------------------------------
# Minimal special:model.yml synthesis (used by save_npz when the source
# Config did not come from a real Marian checkpoint, e.g. a fresh model).
# Mirrors the architectural invariants enforced by config_from_marian_yaml.
# ---------------------------------------------------------------------------


_MINIMAL_YAML_TEMPLATE = (
    "type: transformer\n"
    "dec-cell: ssru\n"
    "dec-depth: {dec_layers}\n"
    "dim-emb: {d_model}\n"
    "dim-vocabs:\n"
    "  - {vocab_size}\n"
    "  - {vocab_size}\n"
    "enc-depth: {enc_layers}\n"
    "tied-embeddings-all: true\n"
    "transformer-decoder-autoreg: rnn\n"
    "transformer-dim-ffn: {d_ffn}\n"
    "transformer-ffn-activation: relu\n"
    "transformer-heads: {n_heads}\n"
    "transformer-postprocess: dan\n"
    "transformer-postprocess-emb: d\n"
    "transformer-train-position-embeddings: false\n"
)


def _synthesize_special_yaml(cfg: Config) -> np.ndarray:
    """Produce a Marian-compatible ``special:model.yml`` int8 blob from a
    :class:`Config`. Used by :func:`save_npz` to keep round-tripped or
    freshly-constructed checkpoints self-describing.
    """
    text = _MINIMAL_YAML_TEMPLATE.format(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ffn=cfg.d_ffn,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
    )
    return np.frombuffer(text.encode("utf-8"), dtype=np.int8).copy()


# ---------------------------------------------------------------------------
# Key set bookkeeping
# ---------------------------------------------------------------------------


def _expected_keys(model: MarianSSRU) -> set[str]:
    """Return the set of weight key names a checkpoint must provide for
    ``model``. Used by :func:`load_npz` to detect arch / cfg mismatches.

    Excludes ``_quantMult`` companions (they're optional metadata).
    """
    keys = {"Wemb", "decoder_ff_logit_out_b"}
    for i in range(model.cfg.enc_layers):
        L = i + 1
        keys.update({
            f"encoder_l{L}_self_Wq", f"encoder_l{L}_self_bq",
            f"encoder_l{L}_self_Wk", f"encoder_l{L}_self_bk",
            f"encoder_l{L}_self_Wv", f"encoder_l{L}_self_bv",
            f"encoder_l{L}_self_Wo", f"encoder_l{L}_self_bo",
            f"encoder_l{L}_self_Wo_ln_scale", f"encoder_l{L}_self_Wo_ln_bias",
            f"encoder_l{L}_ffn_W1", f"encoder_l{L}_ffn_b1",
            f"encoder_l{L}_ffn_W2", f"encoder_l{L}_ffn_b2",
            f"encoder_l{L}_ffn_ffn_ln_scale", f"encoder_l{L}_ffn_ffn_ln_bias",
        })
    for i in range(model.cfg.dec_layers):
        L = i + 1
        keys.update({
            f"decoder_l{L}_rnn_W", f"decoder_l{L}_rnn_Wf", f"decoder_l{L}_rnn_bf",
            f"decoder_l{L}_rnn_ffn_ln_scale", f"decoder_l{L}_rnn_ffn_ln_bias",
            f"decoder_l{L}_context_Wq", f"decoder_l{L}_context_bq",
            f"decoder_l{L}_context_Wk", f"decoder_l{L}_context_bk",
            f"decoder_l{L}_context_Wv", f"decoder_l{L}_context_bv",
            f"decoder_l{L}_context_Wo", f"decoder_l{L}_context_bo",
            f"decoder_l{L}_context_Wo_ln_scale", f"decoder_l{L}_context_Wo_ln_bias",
            f"decoder_l{L}_ffn_W1", f"decoder_l{L}_ffn_b1",
            f"decoder_l{L}_ffn_W2", f"decoder_l{L}_ffn_b2",
            f"decoder_l{L}_ffn_ffn_ln_scale", f"decoder_l{L}_ffn_ffn_ln_bias",
        })
    return keys


# ---------------------------------------------------------------------------
# load_npz
# ---------------------------------------------------------------------------


def load_npz(model: MarianSSRU, npz_path: str | Path) -> dict[str, str]:
    """Load fp32 / fp16 / int8 weights from a Marian ``.npz`` into ``model``.

    int8 entries are detected automatically when their dtype is ``int8`` and a
    parallel ``<name>_quantMult`` key exists; they are dequantized to fp32 on
    load. fp16 is upcast to fp32. The model dtype is unchanged; if you want
    the model to live in fp16 / bf16 / int8 at runtime, call ``model.half()``
    / ``model.bfloat16()`` / :func:`torch.ao.quantization.quantize_dynamic`
    after loading.

    Side effects:
      * the original ``special:*`` blobs (notably ``special:model.yml``) are
        captured on ``model._marian_special`` so :func:`save_npz` can preserve
        them when re-emitting the checkpoint at a different precision.

    Returns a small notes dict (number of tensors, source dtype mix).

    Raises
    ------
    ValueError
        If the checkpoint's tensor shapes don't match ``model.cfg`` (e.g.
        loading a Base checkpoint into a Tiny model). Use
        :func:`qvac_nmt.infer_config` to build the right ``Config``
        first, or call :func:`qvac_nmt.load_model` which does that for
        you.
    """
    z = np.load(str(npz_path))
    by_name: dict[str, np.ndarray] = {
        k: z[k] for k in z.files if not k.startswith("special:")
    }
    specials: dict[str, np.ndarray] = {
        k: z[k] for k in z.files if k.startswith("special:")
    }

    def _matrix_value(name: str) -> np.ndarray:
        a = by_name[name]
        qm_key = f"{name}_quantMult"
        if a.dtype == np.int8 and qm_key in by_name:
            qm = float(by_name[qm_key].item())
            a = a.astype(np.float32) / qm
        else:
            a = a.astype(np.float32)
        return a

    def matrix(name: str) -> torch.Tensor:
        # natural [in, out] -> [out, in] for nn.Linear.weight
        return torch.from_numpy(np.ascontiguousarray(_matrix_value(name).T).copy())

    def vec(name: str) -> torch.Tensor:
        return torch.from_numpy(by_name[name].astype(np.float32).flatten().copy())

    notes: dict[str, str] = {"source": "npz"}

    with torch.no_grad():
        # ---- embedding (already [vocab, dim], stored as int8 / fp16 / fp32)
        Wemb_raw = by_name["Wemb"]
        if Wemb_raw.dtype == np.int8 and "Wemb_quantMult" in by_name:
            qm = float(by_name["Wemb_quantMult"].item())
            Wemb = torch.from_numpy(Wemb_raw.astype(np.float32) / qm)
        else:
            Wemb = torch.from_numpy(Wemb_raw.astype(np.float32).copy())
        if Wemb.shape != model.embed.weight.shape:
            raise ValueError(
                f"Wemb shape {tuple(Wemb.shape)} != model "
                f"{tuple(model.embed.weight.shape)}. The checkpoint describes "
                f"a different model variant than this Config (likely "
                f"d_model={Wemb.shape[1]} vs d_model={model.embed.weight.shape[1]}). "
                f"Use qvac_nmt.infer_config(npz_path) to auto-detect the "
                f"right Config from the checkpoint's embedded YAML, or call "
                f"qvac_nmt.load_model(model_dir) which does that for you."
            )
        model.embed.weight.copy_(Wemb)
        model.out_bias.copy_(vec("decoder_ff_logit_out_b"))

        # ---- encoder
        for i in range(model.cfg.enc_layers):
            L = i + 1
            layer = model.encoder.layers[i]
            layer.self_attn.q.weight.copy_(matrix(f"encoder_l{L}_self_Wq"))
            layer.self_attn.q.bias.copy_(vec(f"encoder_l{L}_self_bq"))
            layer.self_attn.k.weight.copy_(matrix(f"encoder_l{L}_self_Wk"))
            layer.self_attn.k.bias.copy_(vec(f"encoder_l{L}_self_bk"))
            layer.self_attn.v.weight.copy_(matrix(f"encoder_l{L}_self_Wv"))
            layer.self_attn.v.bias.copy_(vec(f"encoder_l{L}_self_bv"))
            layer.self_attn.o.weight.copy_(matrix(f"encoder_l{L}_self_Wo"))
            layer.self_attn.o.bias.copy_(vec(f"encoder_l{L}_self_bo"))
            layer.self_attn_ln.weight.copy_(vec(f"encoder_l{L}_self_Wo_ln_scale"))
            layer.self_attn_ln.bias.copy_(vec(f"encoder_l{L}_self_Wo_ln_bias"))
            layer.ffn.w1.weight.copy_(matrix(f"encoder_l{L}_ffn_W1"))
            layer.ffn.w1.bias.copy_(vec(f"encoder_l{L}_ffn_b1"))
            layer.ffn.w2.weight.copy_(matrix(f"encoder_l{L}_ffn_W2"))
            layer.ffn.w2.bias.copy_(vec(f"encoder_l{L}_ffn_b2"))
            layer.ffn_ln.weight.copy_(vec(f"encoder_l{L}_ffn_ffn_ln_scale"))
            layer.ffn_ln.bias.copy_(vec(f"encoder_l{L}_ffn_ffn_ln_bias"))

        # ---- decoder
        for i in range(model.cfg.dec_layers):
            L = i + 1
            layer = model.decoder.layers[i]
            layer.rnn.W.weight.copy_(matrix(f"decoder_l{L}_rnn_W"))
            layer.rnn.Wf.weight.copy_(matrix(f"decoder_l{L}_rnn_Wf"))
            layer.rnn.Wf.bias.copy_(vec(f"decoder_l{L}_rnn_bf"))
            layer.rnn_ln.weight.copy_(vec(f"decoder_l{L}_rnn_ffn_ln_scale"))
            layer.rnn_ln.bias.copy_(vec(f"decoder_l{L}_rnn_ffn_ln_bias"))
            layer.cross_attn.q.weight.copy_(matrix(f"decoder_l{L}_context_Wq"))
            layer.cross_attn.q.bias.copy_(vec(f"decoder_l{L}_context_bq"))
            layer.cross_attn.k.weight.copy_(matrix(f"decoder_l{L}_context_Wk"))
            layer.cross_attn.k.bias.copy_(vec(f"decoder_l{L}_context_bk"))
            layer.cross_attn.v.weight.copy_(matrix(f"decoder_l{L}_context_Wv"))
            layer.cross_attn.v.bias.copy_(vec(f"decoder_l{L}_context_bv"))
            layer.cross_attn.o.weight.copy_(matrix(f"decoder_l{L}_context_Wo"))
            layer.cross_attn.o.bias.copy_(vec(f"decoder_l{L}_context_bo"))
            layer.cross_attn_ln.weight.copy_(vec(f"decoder_l{L}_context_Wo_ln_scale"))
            layer.cross_attn_ln.bias.copy_(vec(f"decoder_l{L}_context_Wo_ln_bias"))
            layer.ffn.w1.weight.copy_(matrix(f"decoder_l{L}_ffn_W1"))
            layer.ffn.w1.bias.copy_(vec(f"decoder_l{L}_ffn_b1"))
            layer.ffn.w2.weight.copy_(matrix(f"decoder_l{L}_ffn_W2"))
            layer.ffn.w2.bias.copy_(vec(f"decoder_l{L}_ffn_b2"))
            layer.ffn_ln.weight.copy_(vec(f"decoder_l{L}_ffn_ffn_ln_scale"))
            layer.ffn_ln.bias.copy_(vec(f"decoder_l{L}_ffn_ffn_ln_bias"))

    # Sanity check: every weight/bias key in the file should have been
    # consumed by the loops above (modulo `_quantMult` companions). If
    # something remains, the checkpoint very likely has more layers than
    # `model.cfg` declares.
    consumed = _expected_keys(model)
    leftover = [
        k
        for k in by_name
        if k not in consumed and not k.endswith("_quantMult")
    ]
    if leftover:
        # Show only the first few to keep the message readable.
        sample = ", ".join(sorted(leftover)[:5])
        more = f" (+{len(leftover) - 5} more)" if len(leftover) > 5 else ""
        raise ValueError(
            f"checkpoint has {len(leftover)} weight key(s) that "
            f"model.cfg doesn't expect, e.g. {sample}{more}. "
            f"This usually means the model was constructed with the wrong "
            f"Config (e.g. dec_layers mismatch). Use "
            f"qvac_nmt.infer_config(npz_path) to derive the right Config."
        )

    # Stash the original special:* blobs so save_npz can round-trip them.
    if specials:
        model._marian_special = {k: np.array(v) for k, v in specials.items()}  # type: ignore[attr-defined]

    src_dtype = str(by_name["Wemb"].dtype)
    has_int8 = any(v.dtype == np.int8 for v in by_name.values())
    notes["loaded_tensors"] = str(len(by_name))
    notes["wemb_dtype"] = src_dtype
    notes["int8_present"] = "yes" if has_int8 else "no"
    notes["specials"] = ",".join(sorted(specials)) if specials else ""
    return notes


# ---------------------------------------------------------------------------
# save_npz
# ---------------------------------------------------------------------------


def save_npz(
    model: MarianSSRU,
    npz_path: str | Path,
    dtype: DType = "float32",
    *,
    preserve_specials: bool = True,
) -> None:
    """Dump ``model`` to a Marian-compatible ``.npz`` at the given precision.

    Disk sizes for the public Tiny model:

    ============  =========
    dtype           size
    ============  =========
    float32       ~65 MB
    float16       ~33 MB
    int8          ~17 MB
    ============  =========

    For ``int8`` we apply per-tensor scalar quantization to weight matrices
    and embeddings (``q = clip(round(W * 127/max|W|), -127, 127)``) and store
    the inverse scale in a parallel ``<name>_quantMult`` array. Biases and
    LayerNorm scale/bias are kept in fp32 (matches Marian).

    Parameters
    ----------
    preserve_specials
        If ``True`` (default) and ``model`` was loaded by :func:`load_npz`,
        the original ``special:*`` blobs (notably ``special:model.yml``) are
        copied through so the new checkpoint remains self-describing for
        :func:`infer_config`.
    """
    if dtype not in {"float32", "float16", "int8"}:
        raise ValueError(f"unsupported dtype {dtype!r}")

    out: dict[str, np.ndarray] = {}

    if preserve_specials:
        marian_special = getattr(model, "_marian_special", None)
        if marian_special:
            for k, v in marian_special.items():
                out[k] = v
        else:
            # Model was constructed fresh (no source .npz). Synthesize a
            # minimal special:model.yml so the resulting checkpoint is still
            # self-describing for infer_config.
            out["special:model.yml"] = _synthesize_special_yaml(model.cfg)

    Wemb = model.embed.weight.detach().cpu().numpy()
    out_bias = (
        model.out_bias.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
    )

    if dtype == "float32":
        out["Wemb"] = Wemb.astype(np.float32)
    elif dtype == "float16":
        out["Wemb"] = Wemb.astype(np.float16)
    else:
        max_abs = float(np.abs(Wemb).max())
        qm = 127.0 / max(max_abs, 1e-12)
        out["Wemb"] = np.clip(np.rint(Wemb * qm), -127, 127).astype(np.int8)
        out["Wemb_quantMult"] = np.array([qm], dtype=np.float32)
    out["decoder_ff_logit_out_b"] = out_bias

    def _store_mat(name: str, w: torch.Tensor) -> None:
        # PyTorch [out, in] -> Marian [in, out]
        a = w.detach().cpu().numpy().T
        if dtype == "float32":
            out[name] = a.astype(np.float32)
        elif dtype == "float16":
            out[name] = a.astype(np.float16)
        else:
            max_abs = float(np.abs(a).max())
            qm = 127.0 / max(max_abs, 1e-12)
            out[name] = np.clip(np.rint(a * qm), -127, 127).astype(np.int8)
            out[f"{name}_quantMult"] = np.array([qm], dtype=np.float32)

    def _store_vec(name: str, v: torch.Tensor) -> None:
        out[name] = v.detach().cpu().numpy().astype(np.float32).reshape(1, -1)

    for i in range(model.cfg.enc_layers):
        L = i + 1
        layer = model.encoder.layers[i]
        _store_mat(f"encoder_l{L}_self_Wq", layer.self_attn.q.weight)
        _store_vec(f"encoder_l{L}_self_bq", layer.self_attn.q.bias)
        _store_mat(f"encoder_l{L}_self_Wk", layer.self_attn.k.weight)
        _store_vec(f"encoder_l{L}_self_bk", layer.self_attn.k.bias)
        _store_mat(f"encoder_l{L}_self_Wv", layer.self_attn.v.weight)
        _store_vec(f"encoder_l{L}_self_bv", layer.self_attn.v.bias)
        _store_mat(f"encoder_l{L}_self_Wo", layer.self_attn.o.weight)
        _store_vec(f"encoder_l{L}_self_bo", layer.self_attn.o.bias)
        _store_vec(f"encoder_l{L}_self_Wo_ln_scale", layer.self_attn_ln.weight)
        _store_vec(f"encoder_l{L}_self_Wo_ln_bias", layer.self_attn_ln.bias)
        _store_mat(f"encoder_l{L}_ffn_W1", layer.ffn.w1.weight)
        _store_vec(f"encoder_l{L}_ffn_b1", layer.ffn.w1.bias)
        _store_mat(f"encoder_l{L}_ffn_W2", layer.ffn.w2.weight)
        _store_vec(f"encoder_l{L}_ffn_b2", layer.ffn.w2.bias)
        _store_vec(f"encoder_l{L}_ffn_ffn_ln_scale", layer.ffn_ln.weight)
        _store_vec(f"encoder_l{L}_ffn_ffn_ln_bias", layer.ffn_ln.bias)

    for i in range(model.cfg.dec_layers):
        L = i + 1
        layer = model.decoder.layers[i]
        _store_mat(f"decoder_l{L}_rnn_W", layer.rnn.W.weight)
        _store_mat(f"decoder_l{L}_rnn_Wf", layer.rnn.Wf.weight)
        _store_vec(f"decoder_l{L}_rnn_bf", layer.rnn.Wf.bias)
        _store_vec(f"decoder_l{L}_rnn_ffn_ln_scale", layer.rnn_ln.weight)
        _store_vec(f"decoder_l{L}_rnn_ffn_ln_bias", layer.rnn_ln.bias)
        _store_mat(f"decoder_l{L}_context_Wq", layer.cross_attn.q.weight)
        _store_vec(f"decoder_l{L}_context_bq", layer.cross_attn.q.bias)
        _store_mat(f"decoder_l{L}_context_Wk", layer.cross_attn.k.weight)
        _store_vec(f"decoder_l{L}_context_bk", layer.cross_attn.k.bias)
        _store_mat(f"decoder_l{L}_context_Wv", layer.cross_attn.v.weight)
        _store_vec(f"decoder_l{L}_context_bv", layer.cross_attn.v.bias)
        _store_mat(f"decoder_l{L}_context_Wo", layer.cross_attn.o.weight)
        _store_vec(f"decoder_l{L}_context_bo", layer.cross_attn.o.bias)
        _store_vec(f"decoder_l{L}_context_Wo_ln_scale", layer.cross_attn_ln.weight)
        _store_vec(f"decoder_l{L}_context_Wo_ln_bias", layer.cross_attn_ln.bias)
        _store_mat(f"decoder_l{L}_ffn_W1", layer.ffn.w1.weight)
        _store_vec(f"decoder_l{L}_ffn_b1", layer.ffn.w1.bias)
        _store_mat(f"decoder_l{L}_ffn_W2", layer.ffn.w2.weight)
        _store_vec(f"decoder_l{L}_ffn_b2", layer.ffn.w2.bias)
        _store_vec(f"decoder_l{L}_ffn_ffn_ln_scale", layer.ffn_ln.weight)
        _store_vec(f"decoder_l{L}_ffn_ffn_ln_bias", layer.ffn_ln.bias)

    np.savez(str(npz_path), **out)


# ---------------------------------------------------------------------------
# convert_dtype
# ---------------------------------------------------------------------------


def convert_dtype(
    src_npz: str | Path,
    dst_npz: str | Path,
    dtype: DType,
    cfg: Config | None = None,
) -> None:
    """One-shot ``.npz`` -> ``.npz`` precision conversion.

    Internally loads ``src_npz`` into a fresh ``MarianSSRU``, then re-saves at
    ``dtype``. The architecture (Tiny / Base / Base-memory / custom) is
    auto-detected from the source checkpoint's embedded
    ``special:model.yml``; pass ``cfg=Config(...)`` only to override that
    (e.g. for checkpoints whose YAML metadata was stripped).

    The destination ``.npz`` keeps the (possibly rewritten) Marian YAML so it
    remains self-describing for further conversions.
    """
    if cfg is None:
        # Late import: avoids a circular dep at module import time.
        from .config_loader import infer_config

        cfg = infer_config(src_npz)
    model = MarianSSRU(cfg)
    load_npz(model, src_npz)
    save_npz(model, dst_npz, dtype=dtype)


__all__ = ["DType", "load_npz", "save_npz", "convert_dtype"]
