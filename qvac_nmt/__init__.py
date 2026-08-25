"""Pure-PyTorch inference for the Bergamot-recipe Marian NMT family.

``qvac_nmt`` loads any Marian ``.npz`` checkpoint that uses the SSRU-decoder
recipe standardised for fast on-device CPU translation: transformer encoder,
SSRU recurrent decoder, post-LayerNorm, sinusoidal positions, tied embeddings,
ReLU FFN. The primary shipped models are TranslatePsy-EuroNano and
TranslatePsy-AfriNano; any Marian checkpoint matching the same recipe works.

Quick start::

    from qvac_nmt import load_model

    m = load_model("models/en-de")
    print(m.translate("Hello, how are you?"))

Lower-level building blocks::

    from qvac_nmt import (
        Config, MarianSSRU, Translator,
        load_npz, save_npz, convert_dtype,
        infer_config, describe_variant,
    )

The package has *no* dependency on bergamot, marian, or any C++ runtime
at import time or inference time. Checkpoints are read directly from the
Marian ``.npz`` format (fp32, fp16, or int8 with parallel ``_quantMult``
scalars).
"""
from __future__ import annotations

from .config_loader import (
    config_from_marian_yaml,
    describe_variant,
    infer_config,
    read_marian_yaml,
)
from .loader import DType, convert_dtype, load_npz, save_npz
from .model import (
    FFN,
    Config,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MarianSSRU,
    MultiHeadAttention,
    SSRUCell,
    sinusoidal_position_embedding,
)
from .translator import Translator, load_model

__all__ = [
    "Config",
    "MarianSSRU",
    "Translator",
    "load_model",
    "load_npz",
    "save_npz",
    "convert_dtype",
    "DType",
    "infer_config",
    "read_marian_yaml",
    "config_from_marian_yaml",
    "describe_variant",
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "MultiHeadAttention",
    "FFN",
    "SSRUCell",
    "sinusoidal_position_embedding",
]

__version__ = "0.1.0"
