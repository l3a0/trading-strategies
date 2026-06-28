"""Pins for the bounded factor expression grammar (factor_grammar.py, F3 of docs/integration_plan.md).

All ALWAYS-RUN, pure-structural (no data, no scoring — this is the grammar layer): the production-rule
validator, the PARTIAL normalized-form canonical key (commutativity + double-neg collapse), and the
bounded deterministic enumerator. The evaluator + scoring bridge are F3b.
"""
from __future__ import annotations

import pytest

from factor_grammar import (MAX_DEPTH, Expr, ExprGrammarError, canonical_expr_key, enumerate_exprs,
                            leaf, validate_expr)


class TestFactorGrammarValidate:
    """The production-rule gate: well-formed expressions pass, off-grammar ones raise."""

    def test_valid_forms(self) -> None:
        for e in (leaf('close'),
                  Expr('ts_mean', args=(leaf('close'),), window=20),
                  Expr('rank', args=(Expr('ts_std', args=(leaf('ret'),), window=5),)),
                  Expr('sub', args=(leaf('close'), Expr('ts_mean', args=(leaf('close'),), window=20)))):
            assert validate_expr(e) is e

    @pytest.mark.parametrize('bad', [
        Expr('field', operand='volume'),                              # operand off-menu
        Expr('ts_mean', args=(leaf('close'),), window=7),             # window off-bucket
        Expr('bogus', args=(leaf('close'),)),                        # unknown op
        Expr('add', args=(leaf('close'),)),                          # binary with 1 arg
        Expr('rank', args=(leaf('close'),), window=20),              # window on a unary op
        Expr('ts_mean', operand='close', args=(leaf('close'),), window=20),   # op carrying an operand
        Expr('add', args=(leaf('close'), leaf('ret')), window=5),    # window on a binary op
        Expr('ts_mean', args=(leaf('close'),)),                      # TS op missing its window
        Expr('ts_mean', args=(leaf('close'), leaf('ret')), window=5),   # TS op with 2 args
        Expr('field', operand='close', args=(leaf('ret'),)),         # a field leaf carrying args
    ])
    def test_off_grammar_raises(self, bad: Expr) -> None:
        with pytest.raises(ExprGrammarError):
            validate_expr(bad)

    def test_depth_boundary(self) -> None:
        assert MAX_DEPTH == 3
        ok = Expr('rank', args=(Expr('ts_mean', args=(leaf('close'),), window=5),))   # leaf=1, ts=2, rank=3
        assert validate_expr(ok) is ok                              # depth 3 PASSES (the boundary itself)
        deep = Expr('add', args=(ok, leaf('ret')))                  # ...wrapping it -> depth 4
        with pytest.raises(ExprGrammarError):
            validate_expr(deep)                                     # depth 4 raises


class TestFactorCanonicalKey:
    """The PARTIAL normal form: commutative ops are order-invariant, double-neg folds, deeper identities
    are (deliberately) NOT collapsed."""

    def test_commutative_ops_are_order_invariant(self) -> None:
        a, b = leaf('close'), Expr('ts_mean', args=(leaf('ret'),), window=20)
        for op in ('add', 'mul'):
            assert canonical_expr_key(Expr(op, args=(a, b))) == canonical_expr_key(Expr(op, args=(b, a)))

    def test_non_commutative_ops_keep_order(self) -> None:
        a, b = leaf('close'), Expr('ts_mean', args=(leaf('ret'),), window=20)
        assert canonical_expr_key(Expr('sub', args=(a, b))) != canonical_expr_key(Expr('sub', args=(b, a)))

    def test_double_negation_folds(self) -> None:
        x = leaf('close')
        nn = Expr('neg', args=(Expr('neg', args=(x,)),))
        assert validate_expr(nn) is nn                               # neg(neg(x)) is well-formed
        assert canonical_expr_key(nn) == canonical_expr_key(x)       # ...and collapses to x

    def test_distinct_expressions_distinct_keys(self) -> None:
        assert (canonical_expr_key(Expr('ts_mean', args=(leaf('close'),), window=5))
                != canonical_expr_key(Expr('ts_mean', args=(leaf('close'),), window=20)))

    @pytest.mark.parametrize('a,b', [
        (leaf('close'), leaf('ret')),                                                         # operand
        (Expr('ts_mean', args=(leaf('ret'),), window=5), Expr('ts_mean', args=(leaf('ret'),), window=20)),  # window
        (Expr('ts_mean', args=(leaf('ret'),), window=5), Expr('ts_std', args=(leaf('ret'),), window=5)),    # ts op
        (Expr('rank', args=(leaf('close'),)), Expr('zscore', args=(leaf('close'),))),         # unary op
        (Expr('sub', args=(leaf('close'), leaf('ret'))), Expr('sub', args=(leaf('ret'), leaf('close')))),   # non-comm order
    ])
    def test_no_false_collapse(self, a: Expr, b: Expr) -> None:
        # the safety property: genuinely-different expressions NEVER share a key (a false collapse would
        # silently skip a real factor). The normal form only collapses CORRECT equivalences.
        assert canonical_expr_key(a) != canonical_expr_key(b)


class TestFactorEnumerate:
    """The bounded deterministic slice: every expression valid, distinct canonical keys, reproducible."""

    def test_enumerate_is_valid_deduped_and_deterministic(self) -> None:
        es = enumerate_exprs()
        # 18 base features (3 TS_OPS x 2 OPERANDS x 3 WINDOWS) + 36 rank/zscore wraps + 6 (feature - its MA)
        # + 3 cross products = 63, none collapsed by the partial normal form (all genuinely distinct).
        assert len(es) == 63
        for e in es:
            assert validate_expr(e) is e                            # every enumerated expr is on-grammar
        keys = [canonical_expr_key(e) for e in es]
        assert len(set(keys)) == len(keys)                          # deduped by canonical key
        assert keys == sorted(keys)                                 # canonical-key order
        assert [canonical_expr_key(e) for e in enumerate_exprs()] == keys   # deterministic
