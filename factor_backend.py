"""factor_backend.py — the factor domain as a `Backend` (F2 of docs/integration_plan.md).

The honest core is hypothesis-BLIND: it keys on coordinates and reads a `score` row, never engine
internals (proven by F1, backend.py). F2 is the second backend — alpha factors — that plugs into the
SAME core, demonstrating the seam is domain-general: a factor's score row feeds `online_fdr_survivors`
(the e-LOND FDR control) exactly as an option structure's does.

DEPENDENCY-LIGHT BY DESIGN. F2 is the SCORER, not the grammar. The Information Coefficient (rank
correlation of a factor's cross-sectional values with forward returns) and its ICIR t-stat are a few
lines of pandas/numpy — no Qlib, no alphalens. The one-sided p reuses the option path's `_asymptotic_p`
null (from edge_search), so factor and structure p-values sit on the same footing and feed the same
e-LOND control. Qlib's expression engine is F3's grammar and alphalens/Qlib remain OPTIONAL accelerators
for scale; the minimal-dependency core (evalue_fdr's design) is unchanged. The factor primitives here (momentum / reversal / lowvol over a few windows) are a small
fixed slice — F3 generalizes them to a bounded formula grammar.

TWO THINGS DEFERRED, on purpose (docs/integration_plan.md phasing):
  * THE MECHANISM GATE is H1. A factor has no greek signature to read; its mechanism is the
    LOADING REGRESSION (regress the factor's returns on known premia, require a correctly-signed loading
    on the claimed family). Until H1 builds it, `mechanism` returns None and rows type `family=None` —
    the alignment gate is a no-op for factors, so they lean ENTIRELY on e-LOND + the Phase-C holdout
    (the doc's caveat 1). `measurement_invalid` here means DATA-INSUFFICIENCY (too few IC periods), NOT
    mechanism-incoherence — the two are decoupled for factors.
  * PROMOTION stays CLOSED and survivors EXPLORATORY until the Phase-C time-axis holdout exists. F2
    builds no search loop and records nothing; it is the scorer + a wiring proof only.

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

from edge_search import STRUCTURE_END, _asymptotic_p

# the F2 primitive slice (a small fixed menu; F3 generalizes to a bounded formula grammar)
FACTOR_NAMES: tuple[str, ...] = ('momentum', 'reversal', 'lowvol')
WINDOWS: tuple[int, ...] = (5, 20, 60)          # lookback buckets (days)
MIN_IC_PERIODS = 30                              # below this, the IC t-stat is data-insufficient
FACTOR_ENGINE_VERSION = 'factor-v1'             # bump when IC/score mechanics change (re-lineages)


class FactorGrammarError(ValueError):
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
    `FactorGrammarError` off-grammar; returns the factor unchanged on success."""
    if f.name not in FACTOR_NAMES:
        raise FactorGrammarError(f'factor name {f.name!r} not in {FACTOR_NAMES}')
    if f.window not in WINDOWS or type(f.window) is not int:
        raise FactorGrammarError(f'window {f.window!r} not a committed WINDOWS bucket')
    if f.predicted_sign not in (-1, 1) or type(f.predicted_sign) is not int:
        raise FactorGrammarError(f'predicted_sign must be int -1 or +1, got {f.predicted_sign!r}')
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
    raise FactorGrammarError(f'unknown factor {f.name!r}')      # unreachable post-validate


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
            ic = pair.iloc[:, 0].corr(pair.iloc[:, 1], method='spearman')
            if pd.notna(ic):
                ics.append(float(ic))
    return np.asarray(ics, dtype=float)


@dataclass
class FactorBackend:
    """The factor domain as a `Backend`, bound to ONE equity panel. `score` emits the honest-core-facing
    row — the ICIR t-stat in `t_stat_newey_west`, its one-sided asymptotic p in `p_value` (the SAME
    `_asymptotic_p` null the structure path uses), so the row feeds `online_fdr_survivors` unchanged. The
    candidate type is `Factor`."""

    universe: str                          # the panel id — fills the honest core's `ticker` slot
    prices: pd.DataFrame                   # dates x tickers
    checksum: str = ''                     # a panel content hash (lineage input)
    end: str = STRUCTURE_END               # as-of date the panel is loaded through
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
        """No greek to read — the factor mechanism gate is the LOADING REGRESSION, built in H1. Until
        then this returns None (the alignment gate is a no-op for factors; they lean on e-LOND + the
        holdout). NOT a stand-in label: declaring a family without checking it is the foil-paper failure
        mode the regression exists to prevent."""
        return None

    def lineage(self, candidate: Factor) -> str:
        """The (data + engine) lineage — sha over the universe id + panel checksum + engine version.
        Candidate-independent (the panel is the result-moving input), mirroring the option path."""
        payload = f'{self.universe}|{self.checksum}|{self.end}|{FACTOR_ENGINE_VERSION}'
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    def score(self, candidate: Factor) -> dict[str, Any]:
        """The honest-core-facing row: evaluate the factor, compute its ICIR t-stat, and emit
        {phase, key, ticker, predicted_sign, family, mechanism_ok, measurement_invalid, n_days,
        t_stat_newey_west, sign_ok, p_value, end, data_lineage_hash} — the same shape the option path
        emits, byte-compatible with what `online_fdr_survivors` reads. `measurement_invalid` here is
        DATA-INSUFFICIENCY (too few IC periods); `family`/`mechanism_ok` are None/False until H1."""
        ic = information_coefficient(evaluate_factor(candidate, self.prices), self.prices, self.fwd)
        sign = candidate.predicted_sign
        n = int(ic.size)
        row = {'phase': 'factor', 'key': factor_key(candidate), 'ticker': self.universe,
               'predicted_sign': sign, 'family': None, 'mechanism_ok': False,
               'end': self.end, 'data_lineage_hash': self.lineage(candidate)}
        if n < self.min_periods or ic.std(ddof=1) == 0:
            return {**row, 'measurement_invalid': True, 'n_days': n,
                    't_stat_newey_west': None, 'sign_ok': False, 'p_value': None}
        t_ic = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(n)))    # the ICIR t-stat
        t_sign = (t_ic > 0) - (t_ic < 0)
        return {**row, 'measurement_invalid': False, 'n_days': n, 't_stat_newey_west': t_ic,
                'sign_ok': bool(t_sign == sign), 'p_value': round(_asymptotic_p(t_ic, sign), 4)}
