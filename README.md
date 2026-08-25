# qvac-nmt

Pure-PyTorch inference for **Bergamot-recipe Marian NMT models** — the SSRU-decoder student-model architecture used for fast on-device CPU translation. This repo is the inference and conversion toolkit for **TranslatePsy-Nano**, which includes two model families:

- **TranslatePsy-EuroNano** — English ↔ European languages
- **TranslatePsy-AfriNano** — English ↔ African languages

```python
from qvac_nmt import load_model

m = load_model("models/en-de")
print(m.translate("Where is the bus stop?"))
# -> "Wo ist die Bushaltestelle?"
```

No `bergamot-translator`, no `marian-decoder`, no C++ runtime — just `torch`, `numpy`, `sentencepiece`, and `pyyaml`. Marian `.npz` checkpoints are loaded directly; once the weights are PyTorch tensors, fine-tuning, distillation, mixed-precision conversion, and deployment to MPS / CUDA are all standard PyTorch.

## Supported model family

`qvac-nmt` loads any Marian `.npz` checkpoint that uses the SSRU-decoder student-model recipe. The loader auto-detects which model it has by reading the `special:model.yml` blob that Marian embeds in every `.npz`, and asserts these architectural invariants:


| Marian YAML key                         | Required value       |
| --------------------------------------- | -------------------- |
| `type`                                  | `transformer`        |
| `dec-cell`                              | `ssru`               |
| `transformer-decoder-autoreg`           | `rnn`                |
| `transformer-postprocess`               | `dan` (post-LN)      |
| `transformer-ffn-activation`            | `relu`               |
| `tied-embeddings-all`                   | `true`               |
| `transformer-train-position-embeddings` | `false` (sinusoidal) |


Per-variant dimensions (`dim-emb`, `transformer-dim-ffn`, `transformer-heads`, `enc-depth`, `dec-depth`, `dim-vocabs`) are read out of the YAML — the same code path loads every variant.

If a checkpoint disagrees with any invariant the loader raises a `ValueError` naming the offending key, instead of silently producing garbage.

**Known compatible checkpoints:**

- TranslatePsy-EuroNano and TranslatePsy-AfriNano multilingual en↔xx / xx↔en models (Tiny / Base / Base-memory variants).
- Anything you train yourself with Marian under the recipe above.

**Not compatible:**

- `Helsinki-NLP/opus-mt-`* (vanilla Marian — self-attention decoder, learnt positions, untied embeddings).
- Bergamot int8 binaries (`*.intgemm.alphas.bin`). Use the Bergamot runtime for those.

## Highlights

- **Auto-detect every variant.** Tiny, Base, Base-memory, custom widths/depths — same `load_model(...)` call.
- **fp32 / fp16 / int8** — all three precisions are read transparently. A `qvac-convert` CLI re-emits a checkpoint at any of them, preserving the embedded YAML so converted files remain self-describing.
- **Strict on architecture mismatches.** Loud errors, not silent corruption.
- **Self-contained tests.** The unit suite synthesises tiny random models in-process, so no model downloads are required to develop or run CI.
- **Optional AfriNano CLI.** `afritranslate` is a thin shim around `qvac-translate` that whitelists the eight AfriNano target-language tags for convenience.

## Install

```bash
git clone https://github.com/tetherto/qvac-research-translations-nmt.git
cd qvac-research-translations-nmt
uv sync                           # or: pip install -e .
```

Tested on Python 3.10 – 3.12, PyTorch ≥ 2.1, macOS arm64 and Linux x86_64. CPU only; no GPU required.

## Model files

Place one `.npz` weight file and one `.spm` SentencePiece vocab per model:

```
models/
├── en-de/
│   ├── model.npz       # any dtype: fp32, fp16, or int8
│   └── vocab.spm
└── en-xx/              # multilingual model
    ├── model.npz
    └── vocab.spm
```

Any of Marian's standard checkpoint names works (`model.npz`, `model.npz.best-bleu-detok.npz`, `model.npz.best-chrf.npz`, …). To shrink an fp32 checkpoint:

```bash
qvac-convert models/en-de/model.npz models/en-de/model.fp16.npz --dtype fp16
qvac-convert models/en-de/model.npz models/en-de/model.int8.npz --dtype int8
```

The output is a plain Marian-format `.npz` and the loader handles it without any code change.

## Library API

```python
from qvac_nmt import load_model

# Bilingual model: no language tag needed
en2de = load_model("models/en-de")
print(en2de.translate("Hello, how are you?"))
# -> "Hallo, wie geht es dir?"

# Multilingual model with a target-language tag (EuroNano / AfriNano)
en2xx = load_model("models/en-xx", verbose=True)
# [qvac-nmt] model.npz: variant=Tiny d_model=256 d_ffn=1536 enc=6 dec=2 vocab=32000
print(en2xx.translate("##DE Where is the bus stop?"))
# -> "Wo ist die Bushaltestelle?"
print(en2xx.translate("##SW Where is the nearest hospital?"))
# -> "Hospitali iliyo karibu nayo iko wapi?"
```

`Translator` is greedy (beam=1), one sentence at a time. For batched or beam-search workflows, build on the lower-level primitives:

```python
from qvac_nmt import Config, MarianSSRU, infer_config, load_npz

cfg = infer_config("models/en-de/model.npz")     # auto-detect dims
model = MarianSSRU(cfg)
load_npz(model, "models/en-de/model.npz")
model.half()                                     # or .bfloat16(), .to("mps"), .to("cuda")

# Teacher-forced forward pass returns logits ready for cross_entropy:
logits = model(src_ids, tgt_ids)                 # [B, L, V]
```

`MarianSSRU.encode(...)` returns encoder hidden states, `DecoderLayer.step(...)` is the per-token decoder step, and `MarianSSRU.project(...)` produces logits over the joint vocabulary.

To inspect a checkpoint without loading it:

```python
from qvac_nmt import infer_config, describe_variant

cfg = infer_config("models/en-de/model.npz")
print(describe_variant(cfg), cfg)
# Base Config(vocab_size=32000, d_model=512, n_heads=8, d_ffn=2048, enc_layers=6, dec_layers=2, ...)
```

## Command line

After installation:

```bash
# Bilingual model
qvac-translate --models models/en-de "Where is the bus stop?"

# Multilingual model with a tag (EuroNano example)
qvac-translate --models models/en-xx --tag DE "Good morning, my friend."

# AfriNano convenience wrapper (whitelists the 8 Afri tags)
afritranslate --models models/en-xx --tag SW "Good morning, my friend."

# Stream stdin (one sentence per line)
cat lines.txt | qvac-translate --models models/en-de

# Convert checkpoint precision
qvac-convert models/en-de/model.npz \
             models/en-de/model.fp16.npz \
             --dtype fp16
```

Without installation: `python -m qvac_nmt --models models/en-de "..."`.

The `--tag` flag is optional; the validator only enforces "1-8 alphanumeric characters" and uppercases the value. The model decides which tags it actually understands. (For the AfriNano-flavoured `afritranslate` CLI, `--tag` is restricted to `SW HA IG YO ZU AM LN SO`.)

## Picking a precision

Indicative numbers on Apple M-series, single thread, single sentence at a time, Tiny-sized model (d_model=256, ~17 M params):


| `.npz` dtype     | RAM (loaded fp32) | latency (ms/sent) | throughput (sent/s) | disk  |
| ---------------- | ----------------- | ----------------- | ------------------- | ----- |
| `fp32`           | 65 MB             | 4.8               | 210                 | 65 MB |
| `fp16`           | 65 MB             | 4.6               | 215                 | 33 MB |
| `int8` (dequant) | 65 MB             | 4.9               | 205                 | 17 MB |


The activation path runs in fp32 by default; weights are upcast on load. Calling `model.half()` keeps weights and activations in fp16 and halves the working-set RAM with negligible quality loss. PyTorch dynamic int8 (`torch.ao.quantization.quantize_dynamic`) is supported but does not produce a speed-up at these matrix sizes (≤ 256 × 1536) on Apple Silicon — the kernel is dispatch-bound, not arithmetic-bound.

**Recommendation:** ship the `int8` `.npz` (smallest on disk) and let the loader dequantize to fp32 in RAM.

## Architecture notes

The smallest member of the family — the recipe `Config()` defaults to — is laid out as:


| Component             | Setting                                            |
| --------------------- | -------------------------------------------------- |
| Encoder               | 6-layer Transformer, post-LayerNorm                |
| Decoder               | 2-layer **SSRU** (recurrent — not self-attention)  |
| `d_model` / `d_ffn`   | 256 / 1536                                         |
| Heads                 | 8                                                  |
| Positional embeddings | Sinusoidal (Marian block layout)                   |
| Tied embeddings       | input == output projection (`tied-embeddings-all`) |
| Vocab                 | 32 000 SentencePiece pieces (joint src/tgt)        |
| Activation            | ReLU                                               |
| LayerNorm `eps`       | `1e-9` (Marian default)                            |


The defaults exist so that `MarianSSRU(Config())` builds a small model suitable for unit tests or training from scratch. For loading any released checkpoint, prefer auto-detection:

```python
from qvac_nmt import MarianSSRU, infer_config, load_npz

cfg = infer_config("models/some-base-model/model.npz")
model = MarianSSRU(cfg)
load_npz(model, "models/some-base-model/model.npz")
```

### Marian compatibility details

A handful of Marian conventions are deliberately reproduced in `qvac_nmt/model.py` and `qvac_nmt/loader.py`. They differ from the textbook Transformer and matter for bit-level fidelity:

1. **SSRU recurrence is on the cell state**, not the hidden state:
  `c_t = sigmoid(Wf x + bf) * c_{t-1} + (1 − sigmoid(Wf x + bf)) * (W x)`,
   with `h_t = relu(c_t)` handed to the next sublayer. `W` has no bias.
2. **Block-layout sinusoidal positions**: dims `[0, d/2)` are sins, dims `[d/2, d)` are cosines (no interleaving). Frequency denominator is `(d/2 − 1)`, not `d/2`.
3. **Decoder bootstrap at step 0** uses a *zero* token embedding, not `embed(eos)`. From step 1 onward the input is `embed(prev_token)`. The position embedding is added in both cases.
4. **Weight layout on disk**: matrices are stored in natural `[in, out]` mathematical layout in `.npz` (transposed compared to PyTorch's `nn.Linear.weight` `[out, in]`). The loader handles this automatically.
5. **LayerNorm `eps = 1e-9`** (Marian default), not torch's `1e-5`.

## Scope

`qvac-nmt` is a focused PyTorch implementation of the Marian forward and backward pass for the Bergamot-recipe student-model family — sized for fine-tuning, distillation, mixed-precision experiments, and CPU/GPU/MPS inference. The same toolkit covers TranslatePsy-EuroNano and TranslatePsy-AfriNano.

It is **not** a re-implementation of `bergamot-translator`. There is no batched int8 GEMM kernel, no streaming HTML translator, no service abstraction, and no beam search. If you need any of those, use Bergamot directly. If you need a PyTorch-native fp32 / fp16 / int8 inference path with full backward-pass support, this is what that looks like.

It is also **not** a generic Marian loader. Vanilla Marian / OPUS-MT / Helsinki-NLP checkpoints use a self-attention decoder and other architectural choices; loading them here would raise a clear `ValueError` rather than producing garbage. For those, use `transformers`' `MarianMTModel`.

## Development

```bash
uv sync                       # installs dev deps (pytest, ruff, huggingface_hub)
uv run pytest                 # ~0.15 s, no model files required
uv run ruff check .
uv build                      # sdist + wheel
```

The default `pytest` run uses synthetic random models and never touches disk-served checkpoints. To exercise real Bergamot `.npz` files (e.g. before publishing a release), populate a directory and point an env var at it:

```bash
# Option A: fetch checkpoints you host (pass your Hugging Face repo IDs)
HF_TOKEN=... uv run python scripts/fetch_real_models.py --dest real_models --repos your-org/your-model

# Option B: drop your own .npz/.spm pairs into subdirectories of real_models/

QVAC_NMT_REAL_MODELS_DIR=real_models uv run pytest tests/test_real_checkpoints.py -v
```

CI runs the synthetic suite on Python 3.10 / 3.11 / 3.12 plus a wheel-import smoke test. The integration suite runs on push to `main` if the repo has an `HF_TOKEN` secret configured. See `.github/workflows/ci.yml` and `[CONTRIBUTING.md](CONTRIBUTING.md)`.

## License

Code: Apache-2.0 (see `[LICENSE](LICENSE)`).
Model weights and tokenizers: see the license bundled with each checkpoint you use.
