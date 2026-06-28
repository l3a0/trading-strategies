"""factor_engine.py — the engine layer of the factor grammar (F3b of docs/integration_plan.md).

Bridges the pure factor grammar (factor_grammar.py) to scoring, exactly as `generative_engine.py`
bridges `generative_grammar.py` to the options engine. F3 enumerated `Expr` trees but could not score
them; F3b evaluates an `Expr` on a price panel to a factor-value signal, wraps it as a scoreable
candidate (`ExprFactor` = an `Expr` + a falsifiable `predicted_sign`), and scores it through the SAME
Information-Coefficient path the named primitives use — so a grammar-built factor feeds the honest core
(`online_fdr_survivors`) identically to an F2 primitive.

MAXIMAL REUSE: scoring delegates to `factor_backend.information_coefficient` (the rank-IC) and
`factor_backend.ic_to_row` (the single honest-core row source), so the primitive and grammar scorers emit
the byte-identical contract — there is one IC computation and one row builder, not two. DEPENDENCY-LIGHT:
the evaluator is pure pandas (rolling, cross-sectional rank/zscore, elementwise arithmetic); no Qlib, no
scipy. The mechanism gate (the loading regression) is still H1, so `mechanism` returns None / `family` is
None, and promotion stays CLOSED until the Phase-C holdout.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from edge_search import STRUCTURE_END
from factor_backend import FACTOR_ENGINE_VERSION, MIN_IC_PERIODS, ic_to_row, information_coefficient
from factor_grammar import (Expr, ExprGrammarError, canonical_expr_key, enumerate_exprs, validate_expr)
from factor_mechanism import loading_family


def evaluate_expr(expr: Expr, prices: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a validated `Expr` tree on a price panel (dates x tickers) -> a factor-value DataFrame.
    Pure pandas: time-series ops roll along the date axis; cross-sectional ops (`rank`/`zscore`) act
    ACROSS tickers per date (axis=1); arithmetic is elementwise. Division by zero -> NaN (dropped at IC
    time). Assumes `expr` is on-grammar (raises `ExprGrammarError` on the unreachable bad node)."""
    op = expr.op
    if op == 'field':
        if expr.operand == 'close':
            return prices
        if expr.operand == 'ret':
            return prices.pct_change()
        raise ExprGrammarError(f'unknown operand {expr.operand!r}')      # unreachable post-validate
    a = evaluate_expr(expr.args[0], prices)
    if op == 'ts_mean':
        return a.rolling(expr.window).mean()
    if op == 'ts_std':
        return a.rolling(expr.window).std()
    if op == 'ts_delta':
        return a - a.shift(expr.window)
    if op == 'rank':
        return a.rank(axis=1)                                           # cross-sectional rank per date
    if op == 'zscore':
        return a.sub(a.mean(axis=1), axis=0).div(a.std(axis=1), axis=0)  # cross-sectional z per date
    if op == 'neg':
        return -a
    b = evaluate_expr(expr.args[1], prices)
    if op == 'add':
        return a + b
    if op == 'sub':
        return a - b
    if op == 'mul':
        return a * b
    if op == 'div':
        return (a / b).replace([np.inf, -np.inf], np.nan)
    raise ExprGrammarError(f'unknown op {op!r}')                         # unreachable post-validate


@dataclass(frozen=True)
class ExprFactor:
    """A scoreable grammar candidate: an `Expr` paired with a falsifiable `predicted_sign` (the a-priori
    bet on the IC's sign). The sign is EXCLUDED from `canonical_key` (it keys on the Expr alone), the
    sign-shopping guard, exactly as `Composition`/`Factor` do."""
    expr: Expr
    predicted_sign: int = 1


@dataclass
class GrammarFactorBackend:
    """The factor EXPRESSION grammar as a `Backend`, bound to ONE equity panel — the F3 grammar made
    scoreable. Same protocol and same honest-core row as `FactorBackend` (the named primitives); the only
    difference is the candidate (`ExprFactor`) and its evaluation (`evaluate_expr`). Reuses the shared IC
    + row source, so the grammar and primitive scorers are interchangeable to the honest core."""

    universe: str                          # the panel id — fills the honest core's `ticker` slot
    prices: pd.DataFrame                   # dates x tickers
    checksum: str = ''                     # a panel content hash (lineage input)
    end: str = STRUCTURE_END
    fwd: int = 1
    min_periods: int = MIN_IC_PERIODS

    def enumerate(self) -> list[ExprFactor]:
        """The bounded grammar slice (factor_grammar.enumerate_exprs), each at the +1 harvesting bet."""
        return [ExprFactor(e, 1) for e in enumerate_exprs()]

    def validate(self, candidate: ExprFactor) -> ExprFactor:
        validate_expr(candidate.expr)
        if candidate.predicted_sign not in (-1, 1) or type(candidate.predicted_sign) is not int:
            raise ExprGrammarError(f'predicted_sign must be int -1 or +1, got {candidate.predicted_sign!r}')
        return candidate

    def canonical_key(self, candidate: ExprFactor) -> str:
        return canonical_expr_key(candidate.expr)              # sign-excluded (keys on the Expr alone)

    def mechanism(self, candidate: ExprFactor) -> str | None:
        """The factor's family by the LOADING REGRESSION (H1b, live): type the Expr's signal by the
        registered premium it loads on, or None for a mechanism-incoherent Expr that loads on no known
        premium — the factor's derive_family, a MEASUREMENT not a label (the foil-paper defense)."""
        return loading_family(evaluate_expr(candidate.expr, self.prices), self.prices)

    def lineage(self, candidate: ExprFactor) -> str:
        """The (data + engine) lineage — the panel + engine version, shared with `FactorBackend` (the IC
        scoring is identical), so a primitive and an Expr that produce the same signal share lineage."""
        payload = f'{self.universe}|{self.checksum}|{self.end}|{FACTOR_ENGINE_VERSION}'
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    def score(self, candidate: ExprFactor) -> dict[str, Any]:
        """The honest-core-facing row: evaluate the Expr to a signal ONCE, compute its rank-IC AND its
        mechanism `family` (the loading regression) from it, and hand both to the SHARED `ic_to_row` —
        byte-identical in shape to a primitive's row, so it feeds e-LOND the same. A coherent Expr scores
        normally; a mechanism-incoherent one fails closed (H1b)."""
        signal = evaluate_expr(candidate.expr, self.prices)
        ic = information_coefficient(signal, self.prices, self.fwd)
        family = loading_family(signal, self.prices)
        return ic_to_row(ic, family, candidate.predicted_sign, canonical_expr_key(candidate.expr),
                         self.universe, self.end, self.lineage(candidate), self.min_periods)
