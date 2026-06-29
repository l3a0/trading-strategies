"""factor_backend.py — the factor domain as a `Backend` (F2 of docs/integration_plan.md).

The honest core is hypothesis-BLIND: it keys on coordinates and reads a `score` row, never engine
internals (proven by F1, backend.py). F2 is the second backend — alpha factors — that plugs into the
SAME core, demonstrating the seam is domain-general: a factor's score row feeds `online_fdr_survivors`
(the e-LOND FDR control) exactly as an option structure's does.

DEPENDENCY-LIGHT BY DESIGN. F2 is the SCORER, not the grammar. The Information Coefficient (rank
correlation of a factor's cross-sectional values with forward returns) and its ICIR t-stat are a few
lines of pandas/numpy — no Qlib, no alphalens. The one-sided p reuses the repo's shared `_asymptotic_p`
null (in `evalue_fdr`), so factor and structure p-values sit on the same footing and feed the same e-LOND
control. Qlib's expression engine is F3's grammar and alphalens/Qlib remain OPTIONAL accelerators
for scale; the minimal-dependency core (evalue_fdr's design) is unchanged. The factor primitives here (momentum / reversal / lowvol over a few windows) are a small
fixed slice — F3 generalizes them to a bounded formula grammar.

STATUS (docs/integration_plan.md phasing):
  * THE MECHANISM GATE is LIVE (H1b). A factor has no greek signature to read, so its mechanism is the
    LOADING REGRESSION (factor_mechanism.py): `mechanism` types a factor by the registered premium it
    loads on, or `None` for a mechanism-incoherent one. `score` wires it in (factor_engine.py shares the
    path), so an incoherent row gets `family=None`, `p_value=None`, and fails closed (e=0 under e-LOND,
    never flags) — `measurement_invalid` now fires on DATA-INSUFFICIENCY *or* mechanism-incoherence.
  * PROMOTION stays CLOSED and survivors EXPLORATORY until the Phase-C time-axis holdout exists. The
    factor menu-walker proposer (factor_search.py, F4) runs the search loop, but a flagged cell escalates
    to manual pre-registration + the holdout — it is never auto-promoted.

A FactorBackend binds to ONE equity panel (a universe of prices); a factor is scored cross-sectionally
over that universe, so the honest core's (canonical_key, ticker) cell becomes (factor_key, universe) —
the `ticker` slot carries the universe id, no core change.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evalue_fdr import _asymptotic_p   # the shared asymptotic-p convention (one definition; option-independent)
from factor_mechanism import loading_family

# the F2 primitive slice (a small fixed menu; F3 generalizes to a bounded formula grammar)
FACTOR_NAMES: tuple[str, ...] = ('momentum', 'reversal', 'lowvol')
WINDOWS: tuple[int, ...] = (5, 20, 60)          # lookback buckets (days)
MIN_IC_PERIODS = 30                              # below this, the IC t-stat is data-insufficient
FACTOR_ENGINE_VERSION = 'factor-v2'             # bump when IC/score mechanics change (re-lineages); v2 = H1b gate
# The factor panel's as-of date — the factor analog of the option `STRUCTURE_END`. A SEPARATE constant on
# purpose: the equity-panel as-of is independent of the option-chain as-of (they coincide today, but
# coupling them would be wrong), and keeping it here is what frees the factor modules from importing
# edge_search. Same value, so every pinned factor row's `end` is unchanged.
FACTOR_END = '2026-06-06'


class FactorPrimitiveError(ValueError):
    """A factor off the F2 primitive grammar — raised at validation, never a scored cell."""


@dataclass(frozen=True)
class Factor:
    """One F2 factor primitive: a named cross-sectional signal over a lookback `window`, plus the
    a-priori bet on its IC sign. `predicted_sign` is EXCLUDED from `factor_key` (the sign-shopping guard,
    matching the composition grammar)."""
    name: str                    # 'momentum' | 'reversal' | 'lowvol'
    window: int                  # lookback in trading days, a WINDOWS bucket
    predicted_sign: int = 1      # -1 | +1


def validate_factor(f: Factor) -> Factor:
    """Production-rule gate: name in the menu, window a committed bucket, sign in {-1,+1}. RAISES
    `FactorPrimitiveError` off-grammar; returns the factor unchanged on success."""
    if f.name not in FACTOR_NAMES:
        raise FactorPrimitiveError(f'factor name {f.name!r} not in {FACTOR_NAMES}')
    if f.window not in WINDOWS or type(f.window) is not int:
        raise FactorPrimitiveError(f'window {f.window!r} not a committed WINDOWS bucket')
    if f.predicted_sign not in (-1, 1) or type(f.predicted_sign) is not int:
        raise FactorPrimitiveError(f'predicted_sign must be int -1 or +1, got {f.predicted_sign!r}')
    return f


def factor_key(f: Factor) -> str:
    """A content-addressed, sign-excluded identity — sha256 of (name, window). A factor and its
    sign-flipped twin share one key and cannot re-spend the FDR budget (the composition pattern)."""
    return hashlib.sha256(f'{f.name}/w{f.window}'.encode('utf-8')).hexdigest()[:16]


def evaluate_factor(f: Factor, prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the factor's cross-sectional values from a price panel (dates x tickers). Pure pandas —
    the F2 primitive evaluator (F3's grammar replaces this with a Qlib expression). Higher value = the
    factor's stronger signal; `predicted_sign` carries the directional bet, applied at scoring."""
    rets = prices.pct_change()
    if f.name == 'momentum':
        return prices / prices.shift(f.window) - 1.0           # trailing return
    if f.name == 'reversal':
        return -(prices / prices.shift(f.window) - 1.0)        # short-term reversal
    if f.name == 'lowvol':
        return -rets.rolling(f.window).std()                   # low realized vol (negated)
    raise FactorPrimitiveError(f'unknown factor {f.name!r}')      # unreachable post-validate


def information_coefficient(values: pd.DataFrame, prices: pd.DataFrame, fwd: int = 1) -> np.ndarray:
    """The per-period rank Information Coefficient: the Spearman correlation, each date, between the
    factor's cross-sectional values and the `fwd`-period forward return — the factor analog of the
    structure's daily vol-P&L series. LOOK-AHEAD-FREE: the signal at date t is correlated with the return
    from t to t+fwd (`shift(-fwd)`), which is known only after t — a positive shift would leak the future.
    Periods with fewer than 3 ranked names are dropped (a rank correlation on 1-2 names is spurious)."""
    forward = prices.shift(-fwd) / prices - 1.0
    ics: list[float] = []
    for date in values.index:
        fv, fr = values.loc[date], forward.loc[date]
        pair = pd.concat([fv, fr], axis=1).dropna()
        if len(pair) >= 3:
            ic = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())   # Spearman == Pearson on ranks (no scipy dep)
            if pd.notna(ic):
                ics.append(float(ic))
    return np.asarray(ics, dtype=float)


def ic_to_row(ic: np.ndarray, family: str | None, predicted_sign: int, key: str, universe: str, end: str,
              lineage: str, min_periods: int = MIN_IC_PERIODS) -> dict[str, Any]:
    """Build the honest-core-facing row from a factor's IC series + its mechanism `family` (H1b) — the
    SINGLE row source shared by the primitive scorer (`FactorBackend`) and the grammar scorer
    (`factor_engine.GrammarFactorBackend`), so both emit the IDENTICAL contract: {phase, key, ticker,
    predicted_sign, family, mechanism_ok, measurement_invalid, n_days, t_stat_newey_west, sign_ok,
    p_value, end, data_lineage_hash}.

    `measurement_invalid` (never flags) fires on EITHER axis: DATA-INSUFFICIENCY (too few IC periods / a
    zero-variance IC) OR MECHANISM-INCOHERENCE (`family is None` — the loading regression found no
    registered premium). A mechanism-incoherent factor that HAS data keeps its t-stat for transparency but
    its `p_value` is None (e=0, never flags) — the foil-paper defense, mirroring the option path's
    family-None branch. A coherent factor (family set) scores normally."""
    n = int(ic.size)
    row = {'phase': 'factor', 'key': key, 'ticker': universe, 'predicted_sign': predicted_sign,
           'family': family, 'mechanism_ok': family is not None, 'end': end, 'data_lineage_hash': lineage}
    if n < min_periods or ic.std(ddof=1) == 0:
        return {**row, 'measurement_invalid': True, 'n_days': n,                    # data-insufficient
                't_stat_newey_west': None, 'sign_ok': False, 'p_value': None}
    t_ic = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(n)))        # the ICIR t-stat
    t_sign = (t_ic > 0) - (t_ic < 0)
    incoherent = family is None                                   # mechanism-incoherent: keep t, never flag
    return {**row, 'measurement_invalid': incoherent, 'n_days': n, 't_stat_newey_west': t_ic,
            'sign_ok': bool(t_sign == predicted_sign),
            'p_value': None if incoherent else round(_asymptotic_p(t_ic, predicted_sign), 4)}


@dataclass
class FactorBackend:
    """The factor domain as a `Backend`, bound to ONE equity panel. `score` emits the honest-core-facing
    row — the ICIR t-stat in `t_stat_newey_west`, its one-sided asymptotic p in `p_value` (the SAME
    `_asymptotic_p` null the structure path uses), so the row feeds `online_fdr_survivors` unchanged. The
    candidate type is `Factor`."""

    universe: str                          # the panel id — fills the honest core's `ticker` slot
    prices: pd.DataFrame                   # dates x tickers
    checksum: str = ''                     # a panel content hash (lineage input)
    end: str = FACTOR_END                  # as-of date the panel is loaded through
    fwd: int = 1                           # forward-return horizon for the IC (periods)
    min_periods: int = MIN_IC_PERIODS

    def enumerate(self) -> list[Factor]:
        """The F2 primitive slice: every (name, window), harvesting convention (predicted_sign +1)."""
        return [Factor(name, w, 1) for name in FACTOR_NAMES for w in WINDOWS]

    def validate(self, candidate: Factor) -> Factor:
        return validate_factor(candidate)

    def canonical_key(self, candidate: Factor) -> str:
        return factor_key(candidate)

    def mechanism(self, candidate: Factor) -> str | None:
        """The factor's family by the LOADING REGRESSION (H1b, live): type the named factor's signal by the
        registered premium it loads on, or None for a mechanism-incoherent factor that loads on no known
        premium. A MEASUREMENT, not a label — the foil-paper defense (declaring a family without checking
        it is exactly what the regression prevents)."""
        return loading_family(evaluate_factor(candidate, self.prices), self.prices)

    def lineage(self, candidate: Factor) -> str:
        """The (data + engine) lineage — sha over the universe id + panel checksum + engine version.
        Candidate-independent (the panel is the result-moving input), mirroring the option path."""
        payload = f'{self.universe}|{self.checksum}|{self.end}|{FACTOR_ENGINE_VERSION}'
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    def score(self, candidate: Factor) -> dict[str, Any]:
        """The honest-core-facing row: evaluate the factor's signal ONCE, compute its IC series AND its
        mechanism `family` (the loading regression) from it, and hand both to `ic_to_row`. A coherent
        factor scores normally; a mechanism-incoherent one fails closed (the gate is now live, H1b)."""
        signal = evaluate_factor(candidate, self.prices)
        ic = information_coefficient(signal, self.prices, self.fwd)
        family = loading_family(signal, self.prices)
        return ic_to_row(ic, family, candidate.predicted_sign, factor_key(candidate), self.universe,
                         self.end, self.lineage(candidate), self.min_periods)
