# Contributing

Thanks for taking the time to look. Bug reports and small focused PRs are
very welcome.

## Local development

```bash
git clone https://github.com/tetherto/qvac-research-translations-nmt.git
cd qvac-research-translations-nmt
uv sync                 # installs runtime + dev dependencies (pytest, ruff, huggingface_hub)
```

The project uses `[uv](https://docs.astral.sh/uv/)` to manage the
virtualenv and lockfile. If you prefer plain `pip`, `pip install -e '.[hf]'` plus `pip install pytest ruff` works too — but the CI runs
against the `uv.lock`.

## Running checks

```bash
uv run ruff check .              # lint (autofixes available with --fix)
uv run pytest                    # ~0.15 s, no model files required
uv build                         # smoke-build sdist + wheel
```

The default test suite uses synthetic random weight tensors — no
external model files required. The same checks run in
`.github/workflows/ci.yml` against Python 3.10, 3.11, and 3.12.

### Integration tests

`tests/test_real_checkpoints.py` validates the package against real
Bergamot-recipe `.npz` checkpoints. It is skipped by default. To run it
locally:

```bash
# Option A: drop your own .npz/.spm pairs (one per subdirectory) into a folder
QVAC_NMT_REAL_MODELS_DIR=/path/to/your/models uv run pytest tests/test_real_checkpoints.py -v

# Option B: fetch from Hugging Face (pass the repo IDs you host)
HF_TOKEN=... uv run python scripts/fetch_real_models.py --dest real_models --repos your-org/your-model
QVAC_NMT_REAL_MODELS_DIR=real_models uv run pytest tests/test_real_checkpoints.py -v

# Option C: let the test download for you (comma-separated repo IDs via env)
HF_TOKEN=... QVAC_NMT_HF_REPOS=your-org/model-a,your-org/model-b \
  uv run pytest tests/test_real_checkpoints.py -v
```

The CI workflow runs the integration job on push to `main` only, and
only when the `HF_TOKEN` repo secret is configured. PR runs skip the
integration job to avoid leaking secrets to forks.

## Code style

- Public APIs live in `qvac_nmt/__init__.py`; everything exposed there
should have a docstring describing what callers can rely on.
- `qvac_nmt/model.py` should remain free of weight-loading and decoding
logic — keep it pure topology.
- `qvac_nmt/_afritranslate_cli.py` is the only place that should
reference AfriNano-specific assumptions (the eight target-language
tags). Everything else stays generic over the Bergamot recipe.
- The Marian compatibility quirks (SSRU recurrence, sinusoidal block
layout, decoder zero-bootstrap, post-LN, weight transpose) are
load-bearing. If you change them, please add or update the
corresponding test in `tests/test_arch.py`.

## Reporting bugs

When opening an issue, please include:

1. The output of `python -c "import qvac_nmt, torch; print(qvac_nmt.__version__, torch.__version__)"`.
2. A minimal reproduction (≤ 30 lines) — synthesise weights with
  `MarianSSRU(Config(...))` if you can, so the issue can be reproduced
   without any released checkpoints.
3. The full traceback. If it is a "this checkpoint refuses to load"
  issue, please also paste the embedded YAML —
   `python -c "from qvac_nmt import read_marian_yaml; import json; print(json.dumps(read_marian_yaml('your.npz'), indent=2))"`.

If the bug is "translation quality is bad on language X", that's
upstream to the model weights rather than this package; please file it
with whoever published the checkpoint you are running.
