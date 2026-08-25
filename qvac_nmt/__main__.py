"""Entry point so ``python -m qvac_nmt ...`` invokes the translate CLI."""
from __future__ import annotations

from .cli_translate import main

if __name__ == "__main__":
    raise SystemExit(main())
