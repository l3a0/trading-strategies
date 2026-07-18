"""generative_grammar.py — Phase 1 of the generative grammar (docs/generative_grammar_plan.md).

The grammar CORE: leg primitives composed into a bounded, typed `Composition`, a production-rule
validator, a canonical content-addressed identity, and the reachable-count bound. NO ENGINE, NO LLM,
NO SEAL SURFACE — this module is pure and deterministic. It is ADDITIVE: it does not touch the live
closed-grammar code (STRUCTURE_GRAMMAR / _validate_grammar / grid_universe_size in edge_search.py);
later phases wire the engine and the proposer onto it.

The honest-search precondition the closed lattice provided (a finite, pre-specified, human-signed
hypothesis space) is replaced here by a bounded production grammar: leg coordinates are LATTICE
BUCKETS (never free numbers — that is the numberless-value boundary, owned by `validate_composition`,
not `assert_numberless`), and the reachable space is bounded by human-signed caps. The 8 committed
overlays are a VERIFIED SUB-GRAMMAR (`composition_of` maps each named template to a Composition; the
70 named grid points get 70 distinct canonical keys — pinned by test_generative_grammar.py).

Governance knobs (owner-signed; the pinned artifact, per the doc's four-part split):
  * two SPACE CAPS — MAX_LEGS, MAX_EXPIRATIONS;
  * one CORRECTNESS INVARIANT — MAX_NET_DELTA = 1.0, fixed by the engine's [-1,+1] hedge clamp,
    runtime-enforced in Phase 2 (delta is entry-day dependent), NOT a dial;
  * the LIFETIME e-LOND BUDGET — the real power governor, enforced when recording (Phase 3+);
  * the BUCKET SETS — frozen here to exactly the values the 7 overlays use, so the named sub-grammar
    stays tight; widening a bucket is a later, separately-pinned governance step.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

# --- the frozen grammar space (human-signed; widening any of these is a pinned governance act) ----
# Bucket sets, frozen to EXACTLY the values the 7 named overlays use (per the Phase-1 decision):
#   target_delta (0.15/0.25/0.50) + short_delta (0.20/0.25/0.30) + wing_delta (0.05/0.10)
DELTAS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)   # |delta| buckets; 0.50 == ATM
#   dte (21/30/45) + near_dte (21/30) + far_dte (60/90)
DTES: tuple[int, ...] = (21, 30, 45, 60, 90)
RIGHTS: tuple[str, ...] = ('call', 'put')
SIDES: tuple[str, ...] = ('short', 'long')              # sign -1 / +1
HEDGES: tuple[str, ...] = ('combined',)                 # the only hedge rule (general -Σ sign·delta)

MAX_LEGS = 4                 # space cap (the iron condor's 4 legs is the committed max)
MAX_EXPIRATIONS = 2         # space cap (the calendar's 2 expiries is the committed max)
MAX_NET_DELTA = 1.0         # correctness INVARIANT (the [-1,+1] hedge clamp), runtime-enforced Phase 2
LIFETIME_ELOND_BUDGET = 256  # owner-signed power governor (total recorded comparisons), enforced Phase 3+


# --- the composition types -------------------------------------------------------------------------
@dataclass(frozen=True)
class Leg:
    """One leg of a composition. The strike is a TARGET (resolved to a real contract at engine time,
    Phase 2): `('delta', d)` selects the |delta|=d strike (d in DELTAS; 0.50 is ATM), `('same',)`
    shares the composition's single delta-anchored strike (the calendar's far leg). `dte` is an
    absolute days-to-expiry target in DTES."""
    side: str                    # 'short' | 'long'
    right: str                   # 'call' | 'put'
    strike: tuple                # ('delta', d) | ('same',)
    dte: int


@dataclass(frozen=True)
class Composition:
    """A bounded leg combination + a falsifiable direction. `predicted_sign` is the a-priori bet
    (the tail `_asymptotic_p` scores); it is DELIBERATELY EXCLUDED from `canonical_key`, the same
    sign-shopping guard the closed grammar's `_ledger_key` enforces — a structure and its
    sign-flipped twin share one identity and cannot re-spend the e-LOND budget."""
    legs: tuple[Leg, ...]
    predicted_sign: int          # -1 | +1
    hedge: str = 'combined'
    # `_` makes the dataclass eq/hash stable regardless of leg input order is NOT relied on — identity
    # is `canonical_key`, not dataclass equality (two leg orders are the same composition there).
    _: tuple = field(default=(), repr=False, compare=False)


# --- the production-rule validator (replaces _validate_grammar; owns the numberless-value boundary) -
class GrammarError(ValueError):
    """A composition that is off-grammar — raised at validation, never a scored cell."""


def validate_composition(comp: Composition) -> Composition:
    """Type-strict production-rule gate. RAISES `GrammarError` on any off-grammar composition; this
    is where the numberless-VALUE boundary lives — every coordinate must be a committed lattice
    bucket, so a result-derived strike/dte cannot enter (assert_numberless, a key-name guard, can't
    see it). Returns the composition unchanged on success (for chaining)."""
    legs = comp.legs
    if not isinstance(legs, tuple) or not (1 <= len(legs) <= MAX_LEGS):
        raise GrammarError(f'legs must be a tuple of 1..{MAX_LEGS}, got {len(legs) if isinstance(legs, tuple) else type(legs).__name__}')
    if comp.predicted_sign not in (-1, 1) or type(comp.predicted_sign) is not int:
        raise GrammarError(f'predicted_sign must be int -1 or +1, got {comp.predicted_sign!r}')
    if comp.hedge not in HEDGES:
        raise GrammarError(f'hedge {comp.hedge!r} not in {HEDGES}')
    for i, leg in enumerate(legs):
        if not isinstance(leg, Leg):
            raise GrammarError(f'leg {i} is not a Leg: {leg!r}')
        if leg.side not in SIDES:
            raise GrammarError(f'leg {i} side {leg.side!r} not in {SIDES}')
        if leg.right not in RIGHTS:
            raise GrammarError(f'leg {i} right {leg.right!r} not in {RIGHTS}')
        # dte must be a committed bucket member, type-strict (no 30.0 for 30) — the value boundary
        if leg.dte not in DTES or type(leg.dte) is not int:
            raise GrammarError(f'leg {i} dte {leg.dte!r} not a committed DTES bucket')
        strike = leg.strike
        if not (isinstance(strike, tuple) and strike):
            raise GrammarError(f'leg {i} strike must be a non-empty tuple, got {strike!r}')
        if strike[0] == 'delta':
            # EXACT shape ('delta', d) — a 3rd element would smuggle a free number past the boundary
            if len(strike) != 2:
                raise GrammarError(f"leg {i} delta strike must be ('delta', d), got {strike!r}")
            d = strike[1]
            # TYPE-STRICT float (like dte's int): Fraction(1,4) / Decimal('0.25') / np.float64 == 0.25
            # but spell to a DIFFERENT canonical token, so a non-float that == a bucket would re-spend.
            if type(d) is not float or d not in DELTAS:
                raise GrammarError(f'leg {i} strike delta {d!r} not a committed DELTAS float bucket')
        elif strike[0] == 'same':
            if len(strike) != 1:
                raise GrammarError(f"leg {i} same strike must be ('same',), got {strike!r}")
        else:
            raise GrammarError(f"leg {i} strike {strike!r} must be ('delta', d) or ('same',)")
    # expiry cap
    n_exp = len({leg.dte for leg in legs})
    if n_exp > MAX_EXPIRATIONS:
        raise GrammarError(f'{n_exp} distinct expiries exceeds MAX_EXPIRATIONS={MAX_EXPIRATIONS}')
    # legs must be DISTINCT. A repeated identical leg is N× one leg — a pure SCALE-MULTIPLE, not a new
    # economic structure: its leg P&L (Σ sign·mid·shares) and its −Σ sign·delta hedge are both LINEAR in
    # size, so the Newey-West t-stat / Sharpe / family are byte-identical to the single leg — yet (L,) and
    # (L, L) carry DISTINCT canonical keys and would each spend e-LOND budget on the SAME idea, defeating
    # the totality guarantee below. Size is not a grammar axis, so a multiset of legs is off-grammar.
    # (Surfaced by the Phase-4 adversarial seal verification; the menu-walker's self-pair fed it too.)
    if len(set(legs)) != len(legs):
        raise GrammarError('duplicate legs — a composition is a SET of distinct legs (size is not a '
                           'grammar axis; a repeated leg is a scale-multiple, not a new structure)')
    # a `('same',)` leg shares ONE unambiguous delta-anchored strike (one delta value at one tenor),
    # at a DIFFERENT tenor than that anchor. A same-tenor 'same' is a redundant spelling of a
    # `('delta', d)` leg (by parity the |delta| strike coincides), so forbidding it keeps the canonical
    # form TOTAL — one structure, one key, no e-LOND re-spend.
    same_legs = [leg for leg in legs if leg.strike[0] == 'same']
    if same_legs:
        delta_legs = [leg for leg in legs if leg.strike[0] == 'delta']
        anchor_deltas = {leg.strike[1] for leg in delta_legs}
        anchor_dtes = {leg.dte for leg in delta_legs}
        if len(anchor_deltas) != 1 or len(anchor_dtes) != 1:
            raise GrammarError(
                "a ('same',) leg needs exactly one delta-anchored strike at one tenor, found "
                f'{len(anchor_deltas)} delta value(s) at {len(anchor_dtes)} tenor(s)')
        anchor_dte = next(iter(anchor_dtes))
        if any(leg.dte == anchor_dte for leg in same_legs):
            raise GrammarError("a ('same',) leg must be at a DIFFERENT tenor than its delta anchor "
                               '(a same-tenor share is a redundant spelling — use the delta strike)')
    return comp


# --- the canonical normal form (replaces _overlay_params_key; content-addressed identity) ----------
def _leg_token(leg: Leg) -> str:
    strike = f'd{leg.strike[1]}' if leg.strike[0] == 'delta' else 'same'
    return f'{leg.side}/{leg.right}/{strike}/dte{leg.dte}'


def canonical_key(comp: Composition) -> str:
    """A content-addressed identity: the sha256 (truncated) of the SORTED leg tokens + the hedge rule.
    TOTAL (every legal composition has exactly one key) and ORDER-INVARIANT (leg input order does not
    matter), so two spellings of one structure collapse to one key — they cannot re-spend the lifetime
    e-LOND budget. `predicted_sign` is EXCLUDED (the sign-shopping guard, matching `_ledger_key`)."""
    payload = '|'.join(sorted(_leg_token(leg) for leg in comp.legs)) + f'#hedge:{comp.hedge}'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


# --- the reachable-count bound (replaces grid_universe_size; a governance/review number) -----------
def leg_type_count() -> int:
    """Distinct delta-targeted leg primitives = |SIDES| × |RIGHTS| × |DELTAS| × |DTES| (the `('same',)`
    relational leg is not a free primitive — it only exists anchored to a delta strike)."""
    return len(SIDES) * len(RIGHTS) * len(DELTAS) * len(DTES)


def reachable_upper_bound() -> int:
    """An UPPER BOUND on distinct compositions of 1..MAX_LEGS delta-targeted legs — the count of
    multisets, `Σ_{k=1..MAX_LEGS} C(L+k-1, k)` with `L = leg_type_count()`. It IGNORES the
    ≤MAX_EXPIRATIONS and `('same',)`/coherence constraints (which only SHRINK it), so it is a loose
    ceiling, not the live count. Its point is to make the doc's claim concrete: the raw reachable
    space is astronomical, so the LIFETIME_ELOND_BUDGET — not this count — is the power bound; this is
    a governance/review number only."""
    L = leg_type_count()
    return sum(math.comb(L + k - 1, k) for k in range(1, MAX_LEGS + 1))


# --- the named grammar as a verified sub-grammar (composition_of) ----------------------------------
def composition_of(overlay: str, params: dict, predicted_sign: int = 1) -> Composition:
    """Map a named closed-grammar template `(overlay, params)` to its Composition — the proof that the
    7 named overlays are a SUB-GRAMMAR of the generative one. Leg structures are exactly what the
    vol_premium selectors build (every strike delta-targeted; ATM == delta 0.50; the calendar's far
    leg same-strike). The resulting `canonical_key` is the named template's identity in the generative
    grammar, so the published ledger dedups against it unchanged."""
    p = params
    if overlay == 'short_vol':                                   # 1 short call, delta-targeted
        legs = (Leg('short', 'call', ('delta', p['target_delta']), p['dte']),)
    elif overlay == 'straddle':                                  # short call + short put, ATM (Δ0.50)
        legs = (Leg('short', 'call', ('delta', 0.50), p['dte']),
                Leg('short', 'put', ('delta', 0.50), p['dte']))
    elif overlay == 'strangle':                                  # short call + short put, OTM symmetric
        s = p['short_delta']
        legs = (Leg('short', 'call', ('delta', s), p['dte']),
                Leg('short', 'put', ('delta', s), p['dte']))
    elif overlay == 'iron_condor':                               # short body + long wings, both sides
        s, w = p['short_delta'], p['wing_delta']
        legs = (Leg('short', 'call', ('delta', s), p['dte']),
                Leg('short', 'put', ('delta', s), p['dte']),
                Leg('long', 'call', ('delta', w), p['dte']),
                Leg('long', 'put', ('delta', w), p['dte']))
    elif overlay == 'risk_reversal':                            # short rich put + long cheap call
        s = p['short_delta']
        legs = (Leg('short', 'put', ('delta', s), p['dte']),
                Leg('long', 'call', ('delta', s), p['dte']))
    elif overlay == 'credit_spread':                            # short put + long further-OTM put wing
        s, w = p['short_delta'], p['wing_delta']
        legs = (Leg('short', 'put', ('delta', s), p['dte']),
                Leg('long', 'put', ('delta', w), p['dte']))
    elif overlay == 'call_credit_spread':                       # short call + long higher-strike call wing
        s, w = p['short_delta'], p['wing_delta']                # (widening 5 — the CARRY call side)
        legs = (Leg('short', 'call', ('delta', s), p['dte']),
                Leg('long', 'call', ('delta', w), p['dte']))
    elif overlay == 'calendar':                                 # short near call + long far call, same K
        legs = (Leg('short', 'call', ('delta', 0.50), p['near_dte']),
                Leg('long', 'call', ('same',), p['far_dte']))
    else:
        raise GrammarError(f'unknown overlay {overlay!r}')
    return validate_composition(Composition(legs=legs, predicted_sign=predicted_sign))


# --- the deterministic menu-walker over the production grammar (Phase 3) ----------------------------
def enumerate_compositions(max_legs: int = 2) -> list[Composition]:
    """Deterministic menu-walk over a BOUNDED SLICE of the production grammar — the generative analog of
    `enumerate_grammar_templates`. Yields valid `Composition`s (predicted_sign +1, the harvesting
    convention) in canonical-key order, deduped, with NO randomness.

    The slice (Phase 3): every single-leg structure, plus every same-expiration two-leg structure — the
    straddle / strangle / risk-reversal / spread families. It is a SLICE, not the whole space: the
    `('same',)` calendar family and 3-4-leg structures (the iron condor) are reachable widenings of the
    walk, not yet enumerated; and the run-time stop is the SATURATION READOUT, not this enumerator (the
    full reachable space is astronomical — `reachable_upper_bound()` — so the menu-walker scores a
    sampled prefix and stops when the e-LOND bar overtakes the data ceiling, per
    docs/generative_grammar_plan.md). Mechanism-incoherent and high-net-delta cells in the slice are
    filtered at SCORE time (`derive_family` / the `|net_delta| <= 1.0` runtime gate), not here."""
    seen: set[str] = set()
    out: list[tuple[str, Composition]] = []

    def _add(legs: tuple[Leg, ...]) -> None:
        try:
            comp = validate_composition(Composition(legs=legs, predicted_sign=1))
        except GrammarError:
            return
        key = canonical_key(comp)
        if key not in seen:
            seen.add(key)
            out.append((key, comp))

    leg_types = [(side, right, d) for side in SIDES for right in RIGHTS for d in DELTAS]
    for side, right, d in leg_types:                              # every single-leg structure
        for dte in DTES:
            _add((Leg(side, right, ('delta', d), dte),))
    if max_legs >= 2:                                             # every same-expiration two-leg structure
        for dte in DTES:
            for i, (s1, r1, d1) in enumerate(leg_types):
                for s2, r2, d2 in leg_types[i + 1:]:             # DISTINCT unordered pairs — never the
                    # i==i self-pair (two identical legs = a scale-multiple, off-grammar; would otherwise
                    # be enumerated then rejected by validate_composition, double-charging nothing but
                    # wasting the slot — so skip it at the source).
                    _add((Leg(s1, r1, ('delta', d1), dte), Leg(s2, r2, ('delta', d2), dte)))
    return [comp for _, comp in sorted(out)]                     # canonical-key order — deterministic
