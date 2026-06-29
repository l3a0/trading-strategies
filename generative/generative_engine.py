"""generative_engine.py — the engine-touching layer of the generative grammar (Phase 2).

Bridges the pure grammar core (generative_grammar.py) to the options engine (vol_premium.py). This
first increment is the rule-based FAMILY classifier `derive_family`: it replaces the closed grammar's
hand-authored per-overlay family table (`structure_family` / `STRUCTURE_GRAMMAR[o].family`) with a
RULE over the engine's robust signature axes, so an ARBITRARY composition's mechanism can be typed
without a human declaring it — the auto-typing an open grammar needs (docs/generative_grammar_plan.md,
"the mechanism check goes inline").

The COMPOSER (`compose_legs` / `run_composition`) resolves a `Composition` to the engine's entry legs
and runs it through the single generic overlay engine, replicating each named selector's inter-leg
constraints (the anchor leg sets the expiration; in-expiration shorts are band-filtered; long wings are
struck strictly further OTM than their same-right short; a `('same',)` leg matches the anchor's strike
at a later expiry). A composed NAMED overlay is byte-identical to its hand-written form — the
dataset-gated equivalence proof (docs/generative_grammar_plan.md, Phase 2).
"""
from __future__ import annotations
from common.paths import data_path

from typing import Any, Callable

import pandas as pd

from search.edge_search import STRUCTURE_CAPITAL, PremiumFamily
from generative.generative_grammar import Composition, Leg, canonical_key, composition_of
from realchains.real_cc_backtest import COMMISSION_PER_SHARE, select_entry
from realchains.vol_premium import run_real_structure_overlay, select_put_entry

# candidate tuple = (dte, delta, bid, ask, mid, expiration, strike, contractID)
_BAND = {'call': (0.05, 0.60), 'put': (-0.60, -0.05)}      # the entry bands select_entry/put use


def derive_family(signature: dict[str, Any]) -> PremiumFamily | None:
    """Classify a structure's robust greek signature into a `PremiumFamily` by RULE — the generative
    replacement for the per-overlay family table. `signature` is `structure_greek_signature`'s output
    `{legs, expirations, net_vega, net_delta, net_skew}` (vega/delta in {short, long, neutral}; skew in
    {short_rich, long_rich, flat}).

    Returns `None` for a MECHANISM-INCOHERENT signature that maps to no registered family — e.g. a
    long-vega SINGLE-expiration structure that harvests no committed premium. That `None` is the
    fail-closed verdict the inline gate turns into a rejected / `measurement_invalid` cell (the
    foil-paper defense, per composition rather than per overlay name).

    The rule reproduces the 7 committed overlays' DECLARED families exactly (pinned by
    test_generative_engine.py; and via `TestGrammarSignatureMatchesEngine` pinning declared ==
    engine-derived, it is engine-consistent too), in priority order:

      * TWO expirations           -> TERM     (opposite-sign vega across tenors — the calendar);
      * net_skew + NEUTRAL vega   -> SKEW     (the risk reversal's wing asymmetry on a vega-flat book);
      * LONG delta + SHORT vega   -> CARRY    (the credit spread — theta-positive defined-risk);
      * (else) SHORT vega         -> VARIANCE (the four short-vol overlays);
      * (else)                    -> None     (no term, no skew-on-neutral-vega, no short premium).
    """
    if signature['expirations'] >= 2:
        return PremiumFamily.TERM
    if signature['net_skew'] != 'flat' and signature['net_vega'] == 'neutral':
        return PremiumFamily.SKEW
    if signature['net_delta'] == 'long' and signature['net_vega'] == 'short':
        return PremiumFamily.CARRY
    if signature['net_vega'] == 'short':
        return PremiumFamily.VARIANCE
    return None


# --- the composer: resolve a Composition to the engine's entry legs (byte-identical to the selectors) -
def _signed_target(leg: Leg) -> float:
    """The signed vendor-delta target for a delta-targeted leg: +d for a call, -d for a put."""
    return leg.strike[1] if leg.right == 'call' else -leg.strike[1]


def _in_band(c: tuple, right: str) -> bool:
    lo, hi = _BAND[right]
    return lo < c[1] < hi


def _leg_dict(c: tuple, leg: Leg, fill: str) -> dict[str, Any]:
    """Build the engine leg dict from a resolved candidate — byte-identical to the named `_legs_*`
    builders: shorts fill at bid (c[2]), longs at ask (c[3]), or the mid (c[4]) under fill='mid', with
    COMMISSION_PER_SHARE baked into entry_net (subtracted for a short sale, added for a buy)."""
    sign = -1 if leg.side == 'short' else 1
    px = c[4] if fill != 'bid_ask' else (c[2] if sign == -1 else c[3])
    entry_net = px - COMMISSION_PER_SHARE if sign == -1 else px + COMMISSION_PER_SHARE
    return {'sign': sign, 'right': leg.right, 'strike': c[6], 'contract': c[7],
            'entry_net': entry_net, 'mid': c[4], 'delta': c[1], 'expiration': c[5]}


def compose_legs(composition: Composition,
                 min_gap_dte: int = 30) -> Callable[[dict, dict], list[dict] | None]:
    """Turn a `Composition` into a `select(day, params)` callable for `run_real_structure_overlay`,
    resolving legs exactly as the named selectors do (so a composed named overlay is byte-identical):

      * group legs by dte target — each group is one expiration;
      * the group's ANCHOR is its first CALL leg (else its first put) — resolved via `select_entry` /
        `select_put_entry`, fixing the expiration E (and the same-strike reference K);
      * other SHORT legs -> the band-filtered candidate of their right in E nearest the signed target;
      * LONG wings -> the candidate of their right in E struck strictly further OTM than the same-right
        short (buyable, NOT band-filtered) nearest the signed target;
      * a `('same',)` leg -> the anchor strike K at a LATER expiration (>= min_gap_dte beyond the anchor)
        nearest its far-dte target — the calendar far leg.

    Returns `None` (no trade that day) if any leg is unavailable, exactly like the named selectors."""
    legs = composition.legs
    delta_legs = [leg for leg in legs if leg.strike[0] == 'delta']
    same_legs = [leg for leg in legs if leg.strike[0] == 'same']
    strike_anchor = delta_legs[0] if same_legs else None      # the validator guarantees one delta tenor

    def _select(day: dict, params: dict) -> list[dict] | None:
        fill = str(params.get('fill', 'bid_ask'))
        cands = day['candidates']
        picked: dict[Leg, tuple] = {}
        dtes: list[int] = []
        for leg in delta_legs:
            if leg.dte not in dtes:
                dtes.append(leg.dte)
        for dte in dtes:
            group = [leg for leg in delta_legs if leg.dte == dte]
            anchor = next((leg for leg in group if leg.right == 'call'), group[0])
            picker = select_entry if anchor.right == 'call' else select_put_entry
            a = picker(day, dte, _signed_target(anchor))
            if a is None:
                return None
            picked[anchor], expiry = a, a[5]
            shorts: dict[str, tuple] = {anchor.right: a} if anchor.side == 'short' else {}
            # shorts before longs — a long wing references its same-right short's strike
            for leg in sorted((x for x in group if x is not anchor), key=lambda x: x.side == 'long'):
                tgt = _signed_target(leg)
                if leg.side == 'short':
                    pool = [c for c in cands if c[5] == expiry and c[2] > 0 and _in_band(c, leg.right)]
                    if not pool:
                        return None
                    c = min(pool, key=lambda c: abs(c[1] - tgt))
                    shorts[leg.right] = c
                else:
                    s = shorts.get(leg.right)
                    if s is None:                          # a non-anchor long with no same-right short
                        return None                         # is outside the named grammar (Phase 2b+)
                    pool = [c for c in cands if c[5] == expiry and c[3] > 0
                            and (c[1] > 0 and c[6] > s[6] if leg.right == 'call'
                                 else c[1] < 0 and c[6] < s[6])]
                    if not pool:
                        return None
                    c = min(pool, key=lambda c: abs(c[1] - tgt))
                picked[leg] = c
        if same_legs:
            a = picked[strike_anchor]
            k, near_exp, near_dte = a[6], a[5], a[0]
            for leg in same_legs:
                far = [c for c in cands if c[6] == k and c[5] > near_exp and c[3] > 0
                       and (c[1] > 0 if leg.right == 'call' else c[1] < 0)
                       and c[0] - near_dte >= min_gap_dte]
                if not far:
                    return None
                picked[leg] = min(far, key=lambda c: abs(c[0] - leg.dte))
        return [_leg_dict(picked[leg], leg, fill) for leg in legs]

    return _select


def run_composition(composition: Composition, dates: list[str], prices: list[float],
                    store: dict, params: dict | None = None, *,
                    hedge_mode: str = 'combined', entry_guard: str = 'each_short_positive',
                    management: str = 'hold',
                    ) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Run a `Composition` through the single generic overlay engine — the ADDITIVE generic runner (it
    does not touch the named-overlay dispatch). The defaults are the generative grammar's own config
    (delta-hedged 'combined', held to expiry); the engine config is exposed so the equivalence test can
    hold it equal to a named overlay's STRUCTURE_SPEC and isolate the COMPOSER's leg selection (which is
    what the sub-grammar proof is about — the run config is the overlay's spec, not the composer's)."""
    return run_real_structure_overlay(
        dates, prices, store, params or {}, select=compose_legs(composition),
        entry_guard=entry_guard, hedge_mode=hedge_mode, management=management)


def _entry_signature(composition: Composition, dates: list[str], prices: list[float], store: dict,
                     params: dict, rf: float = 0.045) -> dict[str, Any] | None:
    """The engine's ACTUAL greek signature at the composition's first invertible entry — the inline
    mechanism check, computed exactly as `test_vol_premium.TestGrammarSignatureMatchesEngine` computes it
    for a named overlay: replay the composer's leg selection day by day until one resolves, back each
    leg's IV out of its mid at its OWN tenor, and reduce to `structure_greek_signature`'s robust axes
    `{legs, expirations, net_vega, net_delta, net_skew}`. Returns `None` if no day yields an invertible
    entry, so `derive_family` fails closed. It is a SEPARATE scan from `run_composition` (which does not
    expose entry legs), costing one selection pass up to the first entry."""
    from realchains.vol_premium import structure_greek_signature
    select = compose_legs(composition)
    for i, d in enumerate(dates):
        day = store.get(d)
        if day is None:
            continue
        legs = select(day, params)
        if not legs:
            continue
        years = (pd.Timestamp(legs[0]['expiration']) - pd.Timestamp(d)).days / 365.0
        try:
            return structure_greek_signature(legs, prices[i], years, rf=rf, entry_date=d)
        except ValueError:
            continue
    return None


def score_composition(composition: Composition, ticker: str, dates: list[str], prices: list[float],
                      store: dict, *, capital: float = STRUCTURE_CAPITAL,
                      hedge_mode: str = 'combined', entry_guard: str = 'each_short_positive',
                      management: str = 'hold', params: dict | None = None) -> dict[str, Any]:
    """Score ONE `Composition` on a ticker's pre-loaded chains — the generative KILL-GATE, the analog of
    `edge_search.structure_kill_gate` for a composition (keyed by `canonical_key`, not a template name).
    Runs the composition through the engine and scores its daily delta-hedged vol-P&L by the HAC t-stat's
    closed-form asymptotic null (no RNG), the same statistic the named gate uses, AND applies the INLINE
    MECHANISM GATE: it types the composition by the engine's actual entry signature (`_entry_signature` ->
    `derive_family`) and FAILS CLOSED (`p_value=None`, never flags) for a mechanism-incoherent composition
    that harvests no registered premium — the foil-paper defense applied per composition, not per overlay
    name (a structure can't survive on a lucky t-stat alone if its mechanism is incoherent). A non-trading
    composition is `measurement_invalid` (p=None -> e=0), the campaign analog of the equivalence test's
    must-trade guard. Phase 3b/3c — the per-cell scorer the menu-walker and the Phase-4 author feed into."""
    from search.edge_search import _asymptotic_p
    from realchains.vol_premium import short_vol_statistics
    p = {**(params or {}), 'capital': capital}
    summary, trades, eq = run_composition(composition, dates, prices, store, p,
                                          hedge_mode=hedge_mode, entry_guard=entry_guard,
                                          management=management)
    sign = composition.predicted_sign
    row = {'phase': 'structure', 'key': canonical_key(composition), 'ticker': ticker,
           'predicted_sign': sign}
    if not trades:
        return {**row, 'family': None, 'mechanism_ok': False, 'measurement_invalid': True,
                'no_trades': True, 't_stat_newey_west': None, 'sign_ok': False, 'p_value': None}
    sig = _entry_signature(composition, dates, prices, store, p)
    family = derive_family(sig) if sig is not None else None  # the inline mechanism gate
    st = short_vol_statistics(eq, summary['capital'], rf=summary['risk_free_rate'])
    t_nw = float(st['t_stat_newey_west'])
    t_sign = (t_nw > 0) - (t_nw < 0)                          # +1 / -1 / 0, matching np.sign
    # fail-closed: a mechanism-incoherent composition (family None) keeps its t-stat for transparency but
    # NEVER flags (p=None -> e=0), so a structure harvesting no registered premium can't survive a lucky t.
    return {**row, 'family': family.value if family else None, 'mechanism_ok': family is not None,
            'measurement_invalid': family is None,
            'n_days': st['n_days'], 't_stat_newey_west': t_nw, 'nw_lag': st['nw_lag'],
            'sharpe': st['sharpe'], 'ann_excess_return_pct': st['ann_excess_return_pct'],
            'sign_ok': bool(t_sign == sign),
            'p_value': None if family is None else round(_asymptotic_p(t_nw, sign), 4)}


# --- Phase 3c: the recording loop (design A) -------------------------------------------------------------
# Generative compositions are judged over the lifetime e-LOND stream with the PUBLISHED idea_ledger.jsonl
# as the READ-ONLY stream head: the 56+ published cells fix the discount sequence (so a generative cell's
# gamma_t continues from t=|published|+1, never restarting at 1), and the fresh verdicts are recorded to a
# SEPARATE gen_ledger.jsonl — the published, pinned ledger is never mutated. A "cell" is a (canonical_key,
# ticker) pair: canonical_key is the STRUCTURE identity (ticker-independent), so a structure on two tickers
# is two cells that each spend budget, exactly like the named campaign's 8x7.
GEN_LEDGER_PATH = data_path('gen_ledger.jsonl')
IDEA_LEDGER_PATH = data_path('idea_ledger.jsonl')
# The two committed short-vol cells carry template names that are NOT their overlay; every other published
# template is either a bare overlay name or a systematic 'overlay__params' menu-walk name (split on '__').
_OVERLAY_ALIAS = {'short_call_25': 'short_vol', 'short_call_atm': 'short_vol'}


def _row_overlay(template: str) -> str:
    """Recover the overlay name from a published ledger row's `template`: a systematic menu-walk name is
    `overlay__params...` (split on '__'); the two committed short-vol cells carry aliases; every other
    committed template name IS its overlay. Tied to the committed naming convention (pinned by the test)."""
    if '__' in template:
        return template.split('__', 1)[0]
    return _OVERLAY_ALIAS.get(template, template)


def _published_cell_keys(rows: list[dict]) -> set[tuple[str, str]]:
    """The set of (canonical_key, ticker) CELLS already in the published lifetime ledger — the generative
    identity of every published cell, so a coincident composition dedups against it rather than re-spending
    e-LOND budget. RAISES on an unmappable row (fail-loud: a naming-convention drift must not silently skip
    a row and let a duplicate double-count)."""
    return {(canonical_key(composition_of(_row_overlay(r['template']), r['params'])), r['ticker'])
            for r in rows}


def judge_compositions_against_published(new_rows: list[dict], *, idea_path: str = IDEA_LEDGER_PATH,
                                         prior_rows: list[dict] | None = None) -> list[dict]:
    """Judge fresh generative rows over the lifetime e-LOND stream with the published ledger as the
    read-only stream HEAD (design A). The published cells fix the discount sequence; a fresh cell coincident
    with a published one (same (canonical_key, ticker)) is dropped (it already spent its budget). Returns
    ONLY the fresh rows, each carrying its lifetime `elond_survivor` verdict. The published ledger is read,
    never written. Mirrors `edge_search.judge_against_lifetime_stream`, keyed on the generative cell."""
    from search.edge_search import load_idea_ledger
    from search.evalue_fdr import online_fdr_survivors
    prior = list(prior_rows) if prior_rows is not None else load_idea_ledger(idea_path)
    seen = _published_cell_keys(prior)
    fresh = []
    for r in sorted(new_rows, key=lambda r: (r['key'], r['ticker'])):   # deterministic stream order
        cell = (r['key'], r['ticker'])
        if cell in seen:
            continue
        seen.add(cell)
        fresh.append(r)
    judged = online_fdr_survivors(prior + fresh)                        # prior fixes the discount head
    return judged[len(prior):]


def record_compositions(judged_rows: list[dict], gen_path: str = GEN_LEDGER_PATH) -> int:
    """Append fresh generative verdicts to the SEPARATE generative ledger (design A), deduped by
    (canonical_key, ticker) against what it already holds. Returns the count newly appended; the published
    idea_ledger.jsonl is never touched. e-LOND is online, so an appended verdict is permanent."""
    import json
    import os
    existing: set[tuple[str, str]] = set()
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            existing = {(r['key'], r['ticker']) for r in (json.loads(line) for line in f if line.strip())}
    added = 0
    with open(gen_path, 'a') as f:
        for r in judged_rows:
            cell = (r['key'], r['ticker'])
            if cell in existing:
                continue
            existing.add(cell)
            f.write(json.dumps(r, sort_keys=True) + '\n')
            added += 1
    return added


def run_composition_round(compositions: list[Composition], ticker: str, dates: list[str],
                          prices: list[float], store: dict, *, capital: float = STRUCTURE_CAPITAL,
                          end: str = '2026-06-06', checksums: dict | None = None, record: bool = False,
                          idea_path: str = IDEA_LEDGER_PATH, gen_path: str = GEN_LEDGER_PATH,
                          hedge_mode: str = 'combined', entry_guard: str = 'each_short_positive',
                          management: str = 'hold', params: dict | None = None) -> dict:
    """Phase 3c: score a batch of compositions on ONE ticker, judge them over the lifetime e-LOND stream
    (published ledger as the read-only head, design A), and OPTIONALLY record the fresh verdicts to the
    separate generative ledger. `record=False` (default) is the dry/preview path — scores + judges, mutates
    NOTHING; `record=True` is the owner-gated consequential path (spends e-LOND budget, grows
    gen_ledger.jsonl). The single-expiration slice carries no far-chain lineage (overlay=None); a TERM
    widening must fold the far checksum here. The loop the menu-walker and the Phase-4 author both drive."""
    from search.edge_search import _data_lineage_hash
    lineage = _data_lineage_hash(ticker, end, capital, checksums)        # NOT grammar-dependent
    scored = [{**score_composition(c, ticker, dates, prices, store, capital=capital, hedge_mode=hedge_mode,
                                   entry_guard=entry_guard, management=management, params=params),
               'end': end, 'data_lineage_hash': lineage} for c in compositions]
    fresh = judge_compositions_against_published(scored, idea_path=idea_path, prior_rows=None)
    recorded = record_compositions(fresh, gen_path) if record else 0
    return {'scored': len(scored), 'fresh': len(fresh), 'recorded': recorded, 'rows': fresh}
