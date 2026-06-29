"""Centralized filesystem locations.

Everything is resolved relative to the repository root (this package's
parent), so every entry point — ``python -m engine.cc_backtest``, a pytest
run, a notebook cell — finds the data and figure directories regardless of
the current working directory.

``DATA_DIR`` is the single source of truth for where bulk data lives — the
``data/`` directory. Every CSV/JSONL/checksum reference resolves through it (via
``data_path``), so relocating the data tree is a one-line change here.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# All bulk data (price CSVs, option-chain dailies, rolls, ledgers, the checksum
# manifest) lives under data/. This is the single switch that locates it.
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def data_path(name: str) -> str:
    """Absolute path (as ``str``, for the CSV/gzip/JSON readers) to ``name`` under ``DATA_DIR``."""
    return str(DATA_DIR / name)
