"""Pins for the factor engine (factor_engine.py, F3b of docs/integration_plan.md).

The grammar (F3) made scoreable: the `Expr` evaluator, `GrammarFactorBackend` (the grammar as a
`Backend`), and the F3b deliverable — a grammar-built factor scores through the SAME rank-IC + row source
as the F2 primitives and feeds the SAME honest core. All always-run on the synthetic deterministic panel.
"""
from __future__ import annotations

import pytest

from backend import Backend
from factor_engine import ExprFactor, GrammarFactorBackend, evaluate_expr
from factor_grammar import Expr, ExprGrammarError, canonical_expr_key, leaf
from test_factor_backend import CONTRACT, _panel


def _backend(**kw) -> GrammarFactorBackend:
    return GrammarFactorBackend('SYNTH', _panel(), checksum='cafe', **kw)


class TestExprEvaluator:
    """The evaluator maps an Expr tree to a factor-value panel — pure pandas, no scipy."""

    def test_field_leaves(self) -> None:
        p = _panel()
        assert evaluate_expr(leaf('close'), p).equals(p)
        assert evaluate_expr(leaf('ret'), p).equals(p.pct_change())

    def test_time_series_op(self) -> None:
        p = _panel()
        got = evaluate_expr(Expr('ts_mean', args=(leaf('close'),), window=5), p)
        assert got.equals(p.rolling(5).mean())

    def test_cross_sectional_rank_is_per_date(self) -> None:
        ranked = evaluate_expr(Expr('rank', args=(leaf('close'),)), _panel())
        last = ranked.iloc[-1].dropna()                              # a full cross-section of ranks
        assert last.min() == 1 and last.max() == len(last) and last.nunique() == len(last)

    def test_composite_evaluates_to_panel_shape(self) -> None:
        p = _panel()
        e = Expr('sub', args=(leaf('close'), Expr('ts_mean', args=(leaf('close'),), window=20)))
        assert evaluate_expr(e, p).shape == p.shape


class TestGrammarBackendProtocol:
    """GrammarFactorBackend satisfies the same Backend protocol as the option + primitive backends."""

    def test_is_a_backend(self) -> None:
        assert isinstance(_backend(), Backend)

    def test_enumerate_and_validate(self) -> None:
        fb = _backend()
        cands = fb.enumerate()
        assert cands and all(isinstance(c, ExprFactor) and c.predicted_sign == 1 for c in cands)
        assert fb.validate(cands[0]) is cands[0]

    def test_validate_raises_off_grammar(self) -> None:
        with pytest.raises(ExprGrammarError):
            _backend().validate(ExprFactor(Expr('ts_mean', args=(leaf('close'),), window=7)))   # bad window

    def test_canonical_key_excludes_sign(self) -> None:
        fb, e = _backend(), Expr('ts_mean', args=(leaf('ret'),), window=20)
        assert fb.canonical_key(ExprFactor(e, 1)) == fb.canonical_key(ExprFactor(e, -1)) == canonical_expr_key(e)


class TestGrammarScoring:
    """A grammar Expr scores through rank-IC into the honest-core row; mechanism/family deferred to H1."""

    def test_predictive_expr_scores_valid(self) -> None:
        # ts_mean(ret, 20) is a momentum signal -> positive IC on the persistent-drift panel
        row = _backend().score(ExprFactor(Expr('ts_mean', args=(leaf('ret'),), window=20), 1))
        assert row['measurement_invalid'] is False
        assert row['t_stat_newey_west'] > 1.0 and row['sign_ok'] is True
        assert CONTRACT <= set(row)

    def test_mechanism_and_family_none_until_h1(self) -> None:
        fb = _backend()
        c = ExprFactor(Expr('ts_mean', args=(leaf('ret'),), window=20), 1)
        assert fb.mechanism(c) is None and fb.score(c)['family'] is None

    def test_row_matches_the_shared_row_source(self) -> None:
        # the grammar scorer emits the SAME row the shared ic_to_row builds — one contract source
        from factor_backend import ic_to_row, information_coefficient
        fb = _backend()
        c = ExprFactor(Expr('ts_std', args=(leaf('ret'),), window=20), 1)
        ic = information_coefficient(evaluate_expr(c.expr, fb.prices), fb.prices, fb.fwd)
        expected = ic_to_row(ic, 1, canonical_expr_key(c.expr), 'SYNTH', fb.end, fb.lineage(c), fb.min_periods)
        assert fb.score(c) == expected


class TestGrammarFeedsHonestCore:
    """THE F3b DELIVERABLE: grammar-built factor rows feed the SAME e-LOND control as primitives + options,
    unchanged. (A 10-expr slice — enough to prove the wiring without scoring all 63.)"""

    def test_grammar_rows_get_elond_verdicts(self) -> None:
        from evalue_fdr import online_fdr_survivors
        fb = _backend()
        rows = [fb.score(c) for c in fb.enumerate()[:10]]
        judged = online_fdr_survivors(rows)
        assert len(judged) == len(rows) == 10
        for r in judged:
            assert 'e_value' in r and 'elond_level' in r and isinstance(r['elond_survivor'], bool)
        assert any(r['e_value'] > 0 for r in judged)                # calibration actually ran
