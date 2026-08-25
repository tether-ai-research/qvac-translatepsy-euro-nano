"""Architecture auto-detection from Marian's training YAML.

Marian writes its complete training config into every checkpoint as a UTF-8
blob under the special key ``special:model.yml``. We read that blob (or, as
fallback, a sibling ``*.yml`` file) and translate it into a :class:`Config`
so the package transparently supports the Tiny / Base / Base-memory variants
of TranslatePsy-Nano (EuroNano / AfriNano) — and any other Marian model that
fits the same *shape* of architecture (transformer encoder + SSRU decoder +
post-LN + sinusoidal positions + tied embeddings).

This module also enforces a clear error if a checkpoint's training config
describes a variant that this package doesn't implement (e.g. a
self-attention decoder, learnt position embeddings, or pre-LN). A loud
ValueError is much friendlier than silent garbage output.

Public API:

* :func:`read_marian_yaml` — find and parse the Marian YAML for a ``.npz``.
* :func:`config_from_marian_yaml` — build a :class:`Config` from that YAML
  (or raise on unsupported variants).
* :func:`infer_config` — convenience wrapper used by :func:`load_model`.
* :func:`describe_variant` — friendly variant name for a :class:`Config`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .model import Config

# ---------------------------------------------------------------------------
# YAML reading
# ---------------------------------------------------------------------------


def _decode_special_blob(arr: np.ndarray) -> dict[str, Any]:
    """Decode Marian's ``special:model.yml`` int8 array to a YAML dict."""
    raw = arr.tobytes().rstrip(b"\x00").decode("utf-8")
    return yaml.safe_load(raw) or {}


def read_marian_yaml(npz_path: str | Path) -> dict[str, Any] | None:
    """Locate and parse the Marian training YAML for a ``.npz`` checkpoint.

    Resolution order (most authoritative first):

    1. The ``special:model.yml`` array embedded inside the ``.npz`` itself.
       This is what Marian writes during training and is guaranteed to match
       the exact weight set in the file.
    2. A sibling ``<basename>.yml`` (e.g. ``model.npz.best-chrf.npz.yml``).
    3. ``model.npz.yml`` in the same directory (the un-versioned training
       config dump, identical for all checkpoints from one training run).

    Returns the parsed YAML dict, or ``None`` if no source is available.
    """
    p = Path(npz_path)

    # 1. embedded
    try:
        z = np.load(str(p), allow_pickle=False)
        if "special:model.yml" in z.files:
            return _decode_special_blob(z["special:model.yml"])
    except Exception:
        pass

    # 2. sibling <basename>.yml — handles e.g. .../model.npz.best-chrf.npz.yml
    sib_basename = p.with_suffix(p.suffix + ".yml")
    if sib_basename.exists():
        return yaml.safe_load(sib_basename.read_text()) or {}

    # 3. fallback: <dir>/model.npz.yml
    fallback = p.parent / "model.npz.yml"
    if fallback.exists():
        return yaml.safe_load(fallback.read_text()) or {}

    return None


# ---------------------------------------------------------------------------
# YAML -> Config
# ---------------------------------------------------------------------------


# Architecture knobs the package *fixes* (we don't have code for variants).
# A Marian checkpoint that disagrees with any of these would silently produce
# garbage if we loaded it, so we raise instead.
_REQUIRED_INVARIANTS: dict[str, Any] = {
    "type": "transformer",
    "dec-cell": "ssru",
    "transformer-decoder-autoreg": "rnn",
    "transformer-postprocess": "dan",            # post-LayerNorm with dropout-add-norm
    "transformer-ffn-activation": "relu",
    "tied-embeddings-all": True,
    "transformer-train-position-embeddings": False,  # i.e. sinusoidal
}


def config_from_marian_yaml(y: dict[str, Any]) -> Config:
    """Translate a Marian training YAML dict into a :class:`Config`.

    Architectural dims that vary across the released variants
    (``dim-emb``, ``transformer-dim-ffn``, ``transformer-heads``, ``enc-depth``,
    ``dec-depth``, ``dim-vocabs``) are read out directly. All other knobs are
    asserted against :data:`_REQUIRED_INVARIANTS`; if any disagrees, a
    descriptive :class:`ValueError` is raised pointing at the offending key.
    """
    for key, expected in _REQUIRED_INVARIANTS.items():
        actual = y.get(key, "<missing>")
        if actual != expected:
            raise ValueError(
                f"Marian YAML has {key}={actual!r}; qvac-nmt only "
                f"supports {key}={expected!r}. The checkpoint describes a "
                "model variant that this package does not implement."
            )

    try:
        vocab_size = int(y["dim-vocabs"][0])
        d_model = int(y["dim-emb"])
        n_heads = int(y["transformer-heads"])
        d_ffn = int(y["transformer-dim-ffn"])
        enc_layers = int(y["enc-depth"])
        dec_layers = int(y["dec-depth"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Marian YAML is missing a required architecture field: {exc}"
        ) from exc

    if d_model % n_heads != 0:
        raise ValueError(
            f"Marian YAML inconsistency: dim-emb={d_model} not divisible by "
            f"transformer-heads={n_heads}"
        )

    return Config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ffn=d_ffn,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
    )


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------


def infer_config(
    npz_path: str | Path,
    *,
    strict: bool = True,
) -> Config:
    """Infer a :class:`Config` from a Marian ``.npz`` checkpoint.

    Parameters
    ----------
    npz_path
        Path to a Marian-format ``.npz`` weight file.
    strict
        If ``True`` (default) and no Marian YAML is reachable for this file,
        raise :class:`FileNotFoundError`. If ``False``, fall back silently to
        the package default :class:`Config` (Tiny dimensions). Use
        ``strict=False`` only when you know the checkpoint is Tiny.

    Returns
    -------
    Config
        Architecture spec inferred from the checkpoint's metadata.
    """
    y = read_marian_yaml(npz_path)
    if y is None:
        if strict:
            raise FileNotFoundError(
                f"No Marian YAML found for {npz_path}. The checkpoint has no "
                f"embedded special:model.yml, no sibling <basename>.yml, and "
                f"no model.npz.yml in {Path(npz_path).parent}. Pass an explicit "
                f"`cfg=Config(...)` (or call infer_config(..., strict=False) to "
                f"assume the Tiny variant)."
            )
        return Config()
    return config_from_marian_yaml(y)


# ---------------------------------------------------------------------------
# Friendly variant naming
# ---------------------------------------------------------------------------


_KNOWN_VARIANTS: list[tuple[str, Config]] = [
    ("Tiny",        Config(d_model=256, d_ffn=1536, enc_layers=6, dec_layers=2, n_heads=8)),
    ("Base-memory", Config(d_model=384, d_ffn=1536, enc_layers=6, dec_layers=4, n_heads=8)),
    ("Base",        Config(d_model=512, d_ffn=2048, enc_layers=6, dec_layers=2, n_heads=8)),
]


def describe_variant(cfg: Config) -> str:
    """Return a human-readable label for a :class:`Config`.

    Returns one of ``"Tiny"``, ``"Base-memory"``, ``"Base"``, or
    ``"custom"`` followed by the dims if no canonical match is found.
    """
    for name, ref in _KNOWN_VARIANTS:
        if (
            cfg.d_model == ref.d_model
            and cfg.d_ffn == ref.d_ffn
            and cfg.enc_layers == ref.enc_layers
            and cfg.dec_layers == ref.dec_layers
            and cfg.n_heads == ref.n_heads
        ):
            return name
    return (
        f"custom(d_model={cfg.d_model}, d_ffn={cfg.d_ffn}, "
        f"enc={cfg.enc_layers}, dec={cfg.dec_layers}, heads={cfg.n_heads})"
    )


__all__ = [
    "read_marian_yaml",
    "config_from_marian_yaml",
    "infer_config",
    "describe_variant",
]
