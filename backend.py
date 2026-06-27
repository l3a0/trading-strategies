"""backend.py — the Backend protocol (F1 of docs/integration_plan.md).

The honest core (edge_search / evalue_fdr / read_gate_wire) is hypothesis-BLIND: it keys on coordinates
and reads a result row, never engine internals. This module makes that seam EXPLICIT — the `Backend`
protocol any hypothesis domain must satisfy — and ships `OptionBackend`, the option domain's
implementation, proving the existing generative option path already satisfies it.

It is ADDITIVE and BEHAVIOR-PRESERVING. `OptionBackend` FORWARDS to the existing generative functions
(generative_grammar.validate_composition / canonical_key / enumerate_compositions, generative_engine.
derive_family / _entry_signature / score_composition, edge_search._data_lineage_hash) without changing
any of them — no existing code is edited — so every pinned number is untouched. F1 is just the seam: it
is the precondition for a second backend (the Qlib-backed factor backend, F2) to plug into the SAME
honest core.

The protocol's six methods are exactly the contract `run_composition_round` already calls implicitly:

  enumerate()        -> the bounded, pre-specified candidate space (the grammar)
  validate(c)        -> production-rule gate; RAISES off-grammar
  canonical_key(c)   -> content-addressed, order-invariant, sign-excluded identity (the dedup key)
  mechanism(c)       -> the mechanism gate; None == fail-closed (mechanism-incoherent)
  lineage(c)         -> the data-lineage hash the comparison's result ran against
  score(c)           -> the honest-core-facing row {ticker, predicted_sign, t_stat_newey_west, p_value,
                        n_days, sign_ok, measurement_invalid, family, ...}

A backend instance binds to ONE data context (one ticker's loaded chains + run config for options; one
panel for factors): the campaign loads a ticker, then scores a batch. enumerate / validate / canonical_key
are data-independent (the grammar); mechanism / lineage / score read the bound data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from edge_search import STRUCTURE_CAPITAL, STRUCTURE_END, _data_lineage_hash
from generative_engine import _entry_signature, derive_family, score_composition
from generative_grammar import (Composition, canonical_key, enumerate_compositions,
                                 validate_composition)


@runtime_checkable
class Backend(Protocol):
    """The contract a hypothesis domain provides to the honest core. The core (e-LOND / the ledger /
    the seal) consumes only `score`'s row; the other five methods are the domain's grammar, identity,
    mechanism, and lineage. An implementation binds to one data context (one ticker for options, one
    panel for factors). `@runtime_checkable` so a conformance test can assert structural satisfaction
    (method presence — Protocol isinstance does not check signatures)."""

    def enumerate(self) -> list[Any]:
        """The bounded, pre-specified candidate space (the grammar)."""
        ...

    def validate(self, candidate: Any) -> Any:
        """Production-rule gate: returns the candidate unchanged, RAISES on an off-grammar one."""
        ...

    def canonical_key(self, candidate: Any) -> str:
        """A content-addressed, order-invariant, sign-excluded identity — the dedup key."""
        ...

    def mechanism(self, candidate: Any) -> str | None:
        """The mechanism gate against the bound data — the claimed-family analog. `None` is the
        fail-closed verdict for a mechanism-incoherent candidate (the foil-paper defense)."""
        ...

    def lineage(self, candidate: Any) -> str:
        """The data-lineage hash the comparison's result ran against (data + engine version)."""
        ...

    def score(self, candidate: Any) -> dict[str, Any]:
        """The honest-core-facing result row — the only thing the core reads."""
        ...


@dataclass
class OptionBackend:
    """The option domain as a `Backend`, bound to ONE ticker's pre-loaded chains + run config. Every
    method FORWARDS to the existing generative functions unchanged — a formalizing ADAPTER, not a
    reimplementation — so `score` is byte-identical to `run_composition_round`'s per-composition row and
    no pinned number moves. The candidate type is `Composition`."""

    ticker: str
    dates: list[str]
    prices: list[float]
    store: dict
    capital: float = STRUCTURE_CAPITAL
    end: str = STRUCTURE_END
    checksums: dict[str, str] | None = None
    max_legs: int = 2
    hedge_mode: str = 'combined'
    entry_guard: str = 'each_short_positive'
    management: str = 'hold'
    params: dict | None = None

    def _params(self) -> dict:
        return {**(self.params or {}), 'capital': self.capital}

    def enumerate(self) -> list[Composition]:
        """The bounded generative slice (every single-leg + same-expiration two-leg structure)."""
        return enumerate_compositions(self.max_legs)

    def validate(self, candidate: Composition) -> Composition:
        return validate_composition(candidate)

    def canonical_key(self, candidate: Composition) -> str:
        return canonical_key(candidate)

    def mechanism(self, candidate: Composition) -> str | None:
        """The composition's engine-derived family at its first invertible entry, or `None`
        (mechanism-incoherent). Mirrors `score`'s inline gate, exposed as the queryable form; the two
        agree by construction (same `_entry_signature` -> `derive_family`)."""
        sig = _entry_signature(candidate, self.dates, self.prices, self.store, self._params())
        family = derive_family(sig) if sig is not None else None
        return family.value if family else None

    def lineage(self, candidate: Composition) -> str:
        """The (data + engine) lineage for scoring against this ticker. Candidate-independent for the
        single-expiration slice (NOT grammar-dependent — a TERM widening would fold the far checksum,
        matching `run_composition_round`)."""
        return _data_lineage_hash(self.ticker, self.end, self.capital, self.checksums)

    def score(self, candidate: Composition) -> dict[str, Any]:
        """The honest-core-facing row — `score_composition` enriched with `end` + `data_lineage_hash`,
        byte-identical to `run_composition_round`'s per-composition row."""
        row = score_composition(candidate, self.ticker, self.dates, self.prices, self.store,
                                capital=self.capital, hedge_mode=self.hedge_mode,
                                entry_guard=self.entry_guard, management=self.management,
                                params=self.params)
        return {**row, 'end': self.end, 'data_lineage_hash': self.lineage(candidate)}
