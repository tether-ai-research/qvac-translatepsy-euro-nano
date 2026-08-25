# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — Initial release

### Added

- **Pure-PyTorch implementation** of the Bergamot-recipe Marian student
model: 6-layer transformer encoder + N-layer SSRU recurrent decoder,
post-LayerNorm, sinusoidal block-layout positions, ReLU FFN, tied
input/output embeddings. Loads any Marian `.npz` whose embedded
`special:model.yml` matches that recipe — including TranslatePsy-EuroNano,
TranslatePsy-AfriNano, or your own Marian training output under the same recipe.
- `load_model(...)` convenience: drop a directory containing one `.npz`
and one `.spm` and get back a greedy `Translator`.
- `load_npz` / `save_npz` / `convert_dtype` for fp32 / fp16 / int8 Marian
`.npz` checkpoints, with parallel `<name>_quantMult` companion arrays
for int8.
- **Architecture auto-detection** from the embedded `special:model.yml`.
Tiny / Base / Base-memory / custom widths and depths share one code
path; mismatched recipes raise a descriptive `ValueError` instead of
silently producing garbage.
- CLIs:
  - `qvac-translate` — generic translator. `--tag` accepts any 1-8
  alphanumeric code; the model decides what's valid.
  - `qvac-convert` — change `.npz` precision (fp32 ↔ fp16 ↔ int8). The
  embedded `special:model.yml` is rewritten so the destination remains
  self-describing.
  - `afritranslate` — thin shim around `qvac-translate` that whitelists
  the eight target-language tags
  (`SW HA IG YO ZU AM LN SO`) used by TranslatePsy-AfriNano.
  - `python -m qvac_nmt` entry point.
- Public class is `MarianSSRU` (renamed from `TinyMarian` — the same
class handles every variant, not just Tiny). The class name reflects
the architectural feature: a Marian transformer with the SSRU
recurrent decoder cell.
- Unit tests covering: model topology, SSRU recurrence equivalence,
sinusoidal block layout, fp32 / fp16 / int8 round-trips, YAML
auto-detection, both CLIs (generic and AfriNano shim), and
argument-parser edge cases.
- Integration test (`tests/test_real_checkpoints.py`) gated by
`QVAC_NMT_REAL_MODELS_DIR` or `QVAC_NMT_HF_REPOS`. Verifies that real
Bergamot `.npz` checkpoints satisfy the architectural invariants and
produce non-empty translations.
- Helper `scripts/fetch_real_models.py` that snapshots a list of
Hugging Face repos into a local directory ready for the integration
test.
- GitHub Actions CI on Python 3.10 / 3.11 / 3.12 (lint + unit tests +
wheel build + smoke import); plus an opt-in integration job that
runs against real Bergamot `.npz` checkpoints when `HF_TOKEN` is set.
