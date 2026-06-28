"""factor_grammar.py — a bounded alpha-factor expression grammar (F3 of docs/integration_plan.md).

F2 (factor_backend.py) scored three NAMED primitives (momentum / reversal / lowvol). F3 generalizes
them to a bounded EXPRESSION grammar: a factor is an `Expr` tree over base series (`close`, `ret`),
time-series and cross-sectional operators, and a small window menu. This is the PURE grammar layer —
a production-rule validator, a content-addressed canonical key, and a bounded enumerator — with NO
evaluator and NO scoring, exactly as `generative_grammar.py` was pure before `generative_engine.py`
wired the engine. The evaluator + the FactorBackend scoring bridge are the F3b follow-on.

DEPENDENCY-LIGHT + HAND-ROLLED, matching the repo's grammar style (no Qlib, no SymPy, no egg). Qlib's
expression engine is the optional accelerator for scale; this small bounded grammar is enough to take the
factor backend past three primitives and to exercise the canonicalization problem honestly.

CANONICALIZATION IS PARTIAL, ON PURPOSE (docs/integration_plan.md, caveat 2: "exact canonicalization is
unsolved at scale"). `canonical_expr_key` is a NORMALIZED-FORM hash: it folds the cheap, exact
equivalences — argument order for commutative ops (`add`/`mul`), double negation (`neg(neg(x))==x`) — so
those collapse to one key, but it does NOT chase deeper algebraic identities (distributivity, `x-x==0`,
log/scale rules). A leaked duplicate (two spellings of one factor that the normal form misses) is
CONSERVATIVE for the e-LOND control — it over-counts the comparison budget, raising the bar, never
lowering it — so partial canonicalization is safe, just lossy; the cost is a weaker pre-specification
denominator, not an invalid FDR.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

# --- the frozen grammar space (human-signed; widening any of these is a pinned governance act) --------
OPERANDS: tuple[str, ...] = ('close', 'ret')          # base series; ret == daily pct-change of close
TS_OPS: tuple[str, ...] = ('ts_mean', 'ts_std', 'ts_delta')   # time-series, need a window
UNARY_OPS: tuple[str, ...] = ('rank', 'zscore', 'neg')        # cross-sectional / sign, no window
BINARY_OPS: tuple[str, ...] = ('add', 'sub', 'mul', 'div')    # two sub-expressions
COMMUTATIVE: frozenset[str] = frozenset({'add', 'mul'})       # arg order does not matter (canonicalized)
WINDOWS: tuple[int, ...] = (5, 20, 60)
MAX_DEPTH = 3                # space cap: the deepest operator nesting (a base operand is depth 1)


class ExprGrammarError(ValueError):
    """An expression off the grammar — raised at validation, never a scored cell."""


@dataclass(frozen=True)
class Expr:
    """One node of a bounded factor expression. A LEAF is `op='field'` with `operand` a base series; an
    operator node carries `args` (sub-`Expr`s) and, for a time-series op, a `window`. Identity is
    `canonical_expr_key` (order-invariant for commutative ops), NOT dataclass equality."""
    op: str                      # 'field' | a TS_OPS / UNARY_OPS / BINARY_OPS operator
    operand: str = ''            # the base series, iff op == 'field'
    args: tuple = ()             # sub-Exprs, iff op != 'field'
    window: int = 0              # the lookback, iff op in TS_OPS


# --- constructors (sugar; every path goes through validate) -------------------------------------------
def leaf(name: str) -> Expr:
    return Expr('field', operand=name)


# --- the production-rule validator --------------------------------------------------------------------
def _depth(e: Expr) -> int:
    return 1 if e.op == 'field' else 1 + max((_depth(a) for a in e.args), default=0)


def validate_expr(e: Expr) -> Expr:
    """Type-strict production-rule gate. RAISES `ExprGrammarError` off-grammar; returns `e` unchanged
    on success. Checks the operator alphabet, per-op arity, windows present iff a time-series op (a
    committed bucket), base operands, and the depth cap — recursively."""
    if not isinstance(e, Expr):
        raise ExprGrammarError(f'not an Expr: {e!r}')
    if e.op == 'field':
        if e.operand not in OPERANDS:
            raise ExprGrammarError(f'operand {e.operand!r} not in {OPERANDS}')
        if e.args or e.window:
            raise ExprGrammarError('a field leaf takes no args or window')
        return e
    if e.operand:
        raise ExprGrammarError(f'op {e.op!r} must not carry an operand')
    if e.op in TS_OPS:
        if len(e.args) != 1:
            raise ExprGrammarError(f'{e.op} takes 1 arg, got {len(e.args)}')
        if e.window not in WINDOWS or type(e.window) is not int:
            raise ExprGrammarError(f'{e.op} window {e.window!r} not a committed WINDOWS bucket')
    elif e.op in UNARY_OPS:
        if len(e.args) != 1:
            raise ExprGrammarError(f'{e.op} takes 1 arg, got {len(e.args)}')
        if e.window:
            raise ExprGrammarError(f'{e.op} takes no window')
    elif e.op in BINARY_OPS:
        if len(e.args) != 2:
            raise ExprGrammarError(f'{e.op} takes 2 args, got {len(e.args)}')
        if e.window:
            raise ExprGrammarError(f'{e.op} takes no window')
    else:
        raise ExprGrammarError(f'unknown op {e.op!r}')
    for a in e.args:
        validate_expr(a)
    if _depth(e) > MAX_DEPTH:
        raise ExprGrammarError(f'depth {_depth(e)} exceeds MAX_DEPTH={MAX_DEPTH}')
    return e


# --- the canonical normal form (PARTIAL canonicalization; content-addressed identity) -----------------
def _normal_form(e: Expr) -> str:
    """A normalized string form: commutative args sorted, double-neg folded — recursively. Two spellings
    that differ only by those collapse to one string (one key); deeper identities are NOT chased (partial,
    see the module docstring)."""
    if e.op == 'field':
        return f'f:{e.operand}'
    if e.op == 'neg' and e.args[0].op == 'neg':            # neg(neg(x)) == x
        return _normal_form(e.args[0].args[0])
    parts = [_normal_form(a) for a in e.args]
    if e.op in COMMUTATIVE:
        parts.sort()
    win = f':{e.window}' if e.op in TS_OPS else ''
    return f'{e.op}{win}({",".join(parts)})'


def canonical_expr_key(e: Expr) -> str:
    """A content-addressed identity: sha256 of the partial normal form. Order-invariant for commutative
    ops and double-neg-folded, so those equivalent spellings share one key and cannot re-spend the e-LOND
    budget. Partial — it does not collapse deeper algebraic identities (conservative; see the docstring)."""
    return hashlib.sha256(_normal_form(e).encode('utf-8')).hexdigest()[:16]


# --- the bounded enumerator (a deterministic slice of the grammar) ------------------------------------
def enumerate_exprs() -> list[Expr]:
    """A bounded, deterministic slice of the grammar — the factor analog of `enumerate_compositions`.
    Yields valid `Expr`s in canonical-key order, deduped: every base time-series feature (`ts_op(field,
    w)`), each wrapped in a cross-sectional `rank`/`zscore`, plus a few two-arm combinations (a feature
    minus its own moving average; a feature times another). A SLICE, not the whole space — deeper trees
    are reachable widenings, and the run-time stop is the saturation readout (as in the option grammar)."""
    seen: set[str] = set()
    out: list[tuple[str, Expr]] = []

    def _add(e: Expr) -> None:
        try:
            validate_expr(e)
        except ExprGrammarError:
            return
        k = canonical_expr_key(e)
        if k not in seen:
            seen.add(k)
            out.append((k, e))

    base = [Expr(op, args=(leaf(fld),), window=w)             # ts_op(field, w) — the base features
            for op in TS_OPS for fld in OPERANDS for w in WINDOWS]
    for f in base:
        _add(f)
        for un in ('rank', 'zscore'):                         # cross-sectional wrap of each base feature
            _add(Expr(un, args=(f,)))
    for fld in OPERANDS:                                      # feature minus its own moving average
        for w in WINDOWS:
            _add(Expr('sub', args=(leaf(fld), Expr('ts_mean', args=(leaf(fld),), window=w))))
    for w in WINDOWS:                                         # one cross-feature product (commutative)
        _add(Expr('mul', args=(Expr('ts_mean', args=(leaf('ret'),), window=w),
                               Expr('ts_std', args=(leaf('ret'),), window=w))))
    return [e for _, e in sorted(out)]                        # canonical-key order — deterministic
