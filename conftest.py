"""Shared pytest fixtures — a session-scoped chain-store cache.

The real-chain suites parse the same option-daily stores many times over.
Within test_real_cc_backtest.py alone the class-scoped `market` fixtures
reload the MSFT canonical store ~6x, SPY 2x, and QQQ 2x; then
test_explorations.py and test_edge_search.py reload them again through
load_naked_run. Every load re-parses the full CSV — and in CI, where only the
.gz ships, also re-decompresses it (up to 488MB for SPY). That redundant
parsing, not the test logic, dominates the suite's wall time.

`load_chain_store` returns a fully pre-cleaned, READ-ONLY store: the mark
clamp happens at load time, `candidates` and `marks` rows are tuples, and the
overlay never mutates a row (verified). So the parsed result is safe to share
across tests. This module memoizes it for the test session, keyed on
(path, extra_paths, start), and swaps the memoized version into every module
that calls `load_chain_store` so both direct callers and load_naked_run hit
one cache.

This is a TEST-ONLY optimization. It changes no pinned number — a cached
store is byte-identical to a fresh load — only how many times the CSVs are
parsed. The cost it trades for that: all distinct stores stay resident for the
session (~6GB peak across MSFT/SPY/QQQ and their backfills), versus the
class-scoped status quo that freed each store between classes. That fits a
16GB CI runner with headroom.
"""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import pytest

import real_cc_backtest

# The genuine loader, captured before any patching. Cache keyed on the full
# argument tuple, so a clipped store (start=...) and an unclipped one are
# distinct entries — never conflated.
_REAL_LOAD = real_cc_backtest.load_chain_store
_CACHE: dict[tuple[str, tuple[str, ...], str | None], dict[str, dict[str, Any]]] = {}

# Every module that holds a `load_chain_store` binding worth redirecting. The
# `is _REAL_LOAD` guard below means listing a module that didn't import it (or
# imported a different symbol) is a harmless no-op, so this list can be
# generous without risk.
_PATCH_TARGETS = (
    'real_cc_backtest',
    'explorations',
    'edge_search',
    'walk_forward_real',
    'trend_gate',
    'test_real_cc_backtest',
)


def _cached_load_chain_store(
    path: str, extra_paths: Sequence[str] = (), start: str | None = None
) -> dict[str, dict[str, Any]]:
    key = (path, tuple(extra_paths), start)
    store = _CACHE.get(key)
    if store is None:
        store = _REAL_LOAD(path, extra_paths, start)
        _CACHE[key] = store
    return store


@pytest.fixture(scope='session', autouse=True)
def _shared_chain_store_cache():
    """Redirect `load_chain_store` to the session-memoized version in every
    module that calls it, for the whole test session, then restore it.

    Session-scoped and autouse, so it is set up before any class-scoped
    `market` fixture parses a store — the first load of each distinct store
    populates the cache and every later identical load is a dict lookup."""
    patched = []
    for name in _PATCH_TARGETS:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if getattr(mod, 'load_chain_store', None) is _REAL_LOAD:
            mod.load_chain_store = _cached_load_chain_store  # type: ignore[attr-defined]
            patched.append(mod)
    yield
    for mod in patched:
        mod.load_chain_store = _REAL_LOAD  # type: ignore[attr-defined]
    _CACHE.clear()
