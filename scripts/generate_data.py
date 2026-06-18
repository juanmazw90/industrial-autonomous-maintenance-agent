#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.0",
#   "pyarrow>=15.0",
# ]
# ///
"""
Standalone data generator — runs without installing anything:
    uv run scripts/generate_data.py
    uv run scripts/generate_data.py --days 180 --seed 99 --output data/synthetic

For the full ml-package version (after `uv sync`):
    uv run --package amia-ml generate-data
"""

import sys
from pathlib import Path

# When running standalone (PEP 723), import the generator directly from source.
# When running inside the workspace (after uv sync), the installed package takes over.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "ml"))

from amia_ml.synthetic.generator import main  # noqa: E402

if __name__ == "__main__":
    main()
