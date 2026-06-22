"""edge_search.py — MVP harness for an automated, FDR-controlled edge search.

WHAT THIS IS. A thin harness that sweeps a committed batch of cheap
entry-conditioning hypotheses — each one RE-TAGS the cycles the pinned naked
runs already produced (no new engine runs) — through one shared kill-gate,
logs every result to an append-only ledger, and judges the whole batch with a
single Benjamini-Yekutieli pass. It answers one question: across the cheap
entry-conditioning template class, does ANY candidate survive campaign-wide
false-discovery-rate control?

WHY IT LOOKS LIKE THIS. explorations.py already proved that the cooldown,
IV-richness, and trend scouts are the SAME statistic wearing different masks:
tag each cycle with a binary treatment rule, compute D_A = mean(treated P&L)
- mean(other P&L), and calibrate against a same-count permutation null. This
module lifts that shared gate out and wraps it in the two things automation
needs to stay honest:

  1. A multiple-testing ledger. Test nine (or nine thousand) hypotheses at
     p < 0.05 and noise alone hands you false positives; significance is
     judged across the WHOLE batch with Benjamini-Yekutieli (the
     dependence-robust FDR procedure — these candidates are correlated:
     shared tickers, overlapping option cycles, nested windows), not
     per-hypothesis.
  2. A sealed vault. A machine that generates and tests in a loop can never
     "commit before seeing the number," so we commit the DATA it never sees
     instead: the search loads only SEARCH_TICKERS; SEALED_TICKERS are held
     out for a later, manual confirmation step (NOT automated here). QQQ is a
     weak vault on purpose (~0.8 correlated with the search set); the strong
     vault is a structurally different underlying the search never saw, which
     is a premium-data fetch and out of MVP scope.

EXPLORATORY, exactly like explorations.py — a kill-gate, never a registered
verdict. A survivor earns a pre-registration (docs/prereg_trend_gate.md is the
template), not a headline. The ledger pins the campaign so a swept dead end
stays dead instead of being re-derived next session.

TWO PHASES, one ledger. (1) The cheap RE-TAG class — templates that re-tag the
existing naked cycles (no engine re-runs), scored by the D_A split against a
permutation null. (2) The ENGINE-RE-RUN class (lower in this file) — structure
strategies (short-vol / straddle / iron-condor) that CHANGE the trades, so each
(template, ticker) candidate runs a full run_real_*_overlay and is scored by
short_vol_statistics' Newey-West HAC t-stat against its asymptotic normal null
(a closed-form p, no per-candidate permutation). Both phases share the
Benjamini-Yekutieli ledger and the sealed-vault discipline; they are parallel
kill-gates and neither bends the other.

The re-tag null is per-template: the default is the uniform same-count shuffle
(what iv_richness_scout uses), and a template whose treatment has temporal
structure supplies its own — cooldown uses the structure-preserving
trigger-placement permutation cooldown_scout uses (redraw each ticker's rips
from its own terminals). BY (not BH) is the FDR procedure because the candidates
are dependent.

Usage:
    python edge_search.py            # the re-tag campaign (MSFT+SPY, QQQ sealed)
    python edge_search.py structure  # the engine-re-run campaign (TLT sealed)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy as np

from explorations import (
    IV_FLOOR,
    RV_WINDOW,
    _ann_vol,
    _d_a,
    _ord,
    load_entry_ivs,
    load_naked_run,
    post_rip_mask,
)

# --- campaign configuration (committed before the numbers are read) ---------

# The search runs on these; SEALED_TICKERS are held out and never loaded here.
# A survivor is confirmed on the sealed set in a separate, manual step — the
# automation-compatible substitute for pre-registration (commit the data the
# machine can't see, since it can't commit a hypothesis before seeing it).
SEARCH_TICKERS: tuple[str, ...] = ('MSFT', 'SPY')
SEALED_TICKERS: tuple[str, ...] = ('QQQ',)

CAMPAIGN_SEED = 20260613   # one campaign seed → per-candidate seed = SEED + i
N_PERM = 1000              # permutation draws per candidate
FDR_Q = 0.10               # target false-discovery rate for the BY pass

# Template parameter grids. Each (template, setting) pair is one candidate.
COOLDOWN_NS: tuple[int, ...] = (7, 30, 60, 90)        # calendar days
TREND_WINDOWS: tuple[int, ...] = (21, 63, 126, 252)   # trailing-return, days


@dataclass(frozen=True)
class Campaign:
    """A ticker batch for one campaign run: the names the search SPENDS sample
    on (`search`), and the names held SEALED (`sealed`) — never loaded, the
    manual-confirmation / pre-registration substitute. The two sets must be
    disjoint (a sealed ticker can never be searched), so the seal is enforced
    here, in config. Point `run_batch` at a different Campaign to sweep the same
    templates on the next batch of tickers; roll a fresh underlying into
    `sealed` each round so the held-out vault stays genuinely unseen as the
    search expands across names."""
    search: tuple[str, ...]
    sealed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # coerce to tuples so the frozen dataclass stays hashable even if a
        # caller passes lists, then enforce the seal: no ticker in both sets.
        object.__setattr__(self, 'search', tuple(self.search))
        object.__setattr__(self, 'sealed', tuple(self.sealed))
        overlap = sorted(set(self.search) & set(self.sealed))
        if overlap:
            raise ValueError(
                f'campaign seal violated: {overlap} are both searched and '
                f'sealed — a sealed ticker must never enter the search set')


# The published MVP batch: MSFT + SPY searched, QQQ sealed. `main()` runs this.
DEFAULT_CAMPAIGN = Campaign(search=SEARCH_TICKERS, sealed=SEALED_TICKERS)


# --- the per-cycle data every template tags against (built once) ------------

@dataclass
class CycleData:
    """Pooled, per-cycle arrays the templates re-tag. Built once from the
    naked runs so every candidate is a cheap re-tag, not a re-run."""
    pnls: np.ndarray                       # per-cycle P&L
    entry_ords: list[int]                  # entry-date ordinals
    ticker_ids: list[str]                  # per-cycle ticker
    rip_ords_by_ticker: dict[str, list[int]]  # sorted rip terminal ordinals
    trailing_rv: np.ndarray                # trailing realized vol at entry (nan if short)
    trailing_ret: dict[int, np.ndarray]    # window -> trailing return at entry (nan if short)
    richness: np.ndarray                   # entry IV - trailing RV (nan if IV missing/<floor)
    tickers: list[str]                     # the search set actually loaded
    # all terminal ordinals per ticker — the pool the cooldown trigger-placement
    # null redraws fake rips from. Defaulted so ad-hoc CycleData(...) in tests
    # need not supply it (only build_cycle_data and the cooldown null use it).
    term_ords_by_ticker: dict[str, list[int]] = field(default_factory=dict)


def build_cycle_data(
    runs: Sequence[dict[str, Any]],
    iv_loader: Callable[..., dict[tuple[str, str], float]] = load_entry_ivs,
) -> CycleData:
    """Pool the naked cycles and precompute every per-cycle quantity the
    templates need: rip ordinals (cooldown), trailing realized vol (the
    generic vol-confound probe), trailing returns over each window (the
    momentum/trend templates), and entry-IV richness (the VRP template).

    `iv_loader` is injectable so the synthetic test layer can run without the
    multi-hundred-MB option-daily CSVs."""
    pnls: list[float] = []
    entry_ords: list[int] = []
    ticker_ids: list[str] = []
    trailing_rv: list[float] = []
    trailing_ret: dict[int, list[float]] = {k: [] for k in TREND_WINDOWS}
    richness: list[float] = []
    rip_ords_by_ticker: dict[str, list[int]] = {}
    term_ords_by_ticker: dict[str, list[int]] = {}

    for r in runs:
        ticker = r['ticker']
        prices = np.asarray(r['prices'], dtype=float)
        logret = np.diff(np.log(prices))
        idx = {d: i for i, d in enumerate(r['dates'])}
        cycles = r['cycles']
        rip_ords_by_ticker[ticker] = sorted(
            _ord(c['terminal_date']) for c in cycles if c['rip'])
        term_ords_by_ticker[ticker] = sorted(_ord(c['terminal_date']) for c in cycles)
        # one streaming pass over the dailies for this ticker's entry IVs
        wanted = {(c['entry_date'], c['entry_contract']) for c in cycles
                  if c.get('entry_contract')}
        ivs = iv_loader(ticker, wanted)
        for c in cycles:
            pnls.append(float(c['pnl']))
            entry_ords.append(_ord(c['entry_date']))
            ticker_ids.append(ticker)
            i = idx.get(c['entry_date'])
            # trailing realized vol (known at entry); nan if history too short
            if i is not None and i >= RV_WINDOW:
                trv = _ann_vol(logret[i - RV_WINDOW:i])
            else:
                trv = float('nan')
            trailing_rv.append(trv)
            # trailing return over each window; nan if history too short
            for k in TREND_WINDOWS:
                if i is not None and i >= k:
                    trailing_ret[k].append(float(prices[i] / prices[i - k] - 1.0))
                else:
                    trailing_ret[k].append(float('nan'))
            # entry-IV richness vs trailing RV; nan if IV missing / below floor
            iv = ivs.get((c['entry_date'], c.get('entry_contract')))
            if iv is not None and iv >= IV_FLOOR and not np.isnan(trv):
                richness.append(iv - trv)
            else:
                richness.append(float('nan'))

    return CycleData(
        pnls=np.asarray(pnls, dtype=float),
        entry_ords=entry_ords,
        ticker_ids=ticker_ids,
        rip_ords_by_ticker=rip_ords_by_ticker,
        term_ords_by_ticker=term_ords_by_ticker,
        trailing_rv=np.asarray(trailing_rv, dtype=float),
        trailing_ret={k: np.asarray(v, dtype=float) for k, v in trailing_ret.items()},
        richness=np.asarray(richness, dtype=float),
        tickers=[r['ticker'] for r in runs],
    )


# --- the hypothesis-template enumerator -------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One fully-specified, individually-testable hypothesis. `tag` returns
    (treated, valid) boolean arrays over the pooled cycles: `treated` is the
    entry-conditioning mask, `valid` flags cycles where the rule is defined
    (e.g. enough history for the trailing window)."""
    template: str
    params: tuple[tuple[str, Any], ...]   # hashable; dict(params) to read
    predicted_sign: int                   # -1 ⇒ predict D_A < 0 (treated worse)
    tag: Callable[[], tuple[np.ndarray, np.ndarray]]
    # optional structure-preserving permutation null; None = the uniform
    # same-count shuffle (the default). A template whose treatment has temporal
    # structure (cooldown's rip clustering) supplies its own faithful null here.
    null_fn: Callable[..., np.ndarray] | None = None

    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)


def enumerate_candidates(cd: CycleData) -> list[Candidate]:
    """Expand the mechanism templates into the committed batch. Refuses to
    emit a candidate without a sign prediction — the constraint that keeps the
    batch a structured family of falsifiable bets, not a blind grid."""
    n = len(cd.pnls)
    all_valid = np.ones(n, dtype=bool)
    out: list[Candidate] = []

    # Template 1: post-rip cooldown. Hypothesis: a cycle entered within N days
    # of a same-ticker rip does WORSE (the stock is "running"). Predict D_A<0.
    for N in COOLDOWN_NS:
        def tag_cooldown(N: int = N) -> tuple[np.ndarray, np.ndarray]:
            mask = post_rip_mask(cd.entry_ords, cd.ticker_ids,
                                 cd.rip_ords_by_ticker, N)
            return mask, all_valid
        out.append(Candidate('cooldown', (('N', N),), -1, tag_cooldown,
                             null_fn=_cooldown_null))

    # Template 2: trailing up-move. Hypothesis: a cycle entered after a
    # positive trailing-k-day return does WORSE (momentum forfeits the right
    # tail). Predict D_A<0 — this is the repo's recurring lesson under test.
    for k in TREND_WINDOWS:
        def tag_trend(k: int = k) -> tuple[np.ndarray, np.ndarray]:
            ret = cd.trailing_ret[k]
            valid = ~np.isnan(ret)
            treated = valid & (ret > 0)
            return treated, valid
        out.append(Candidate('up_trend', (('window', k),), -1, tag_trend))

    # Template 3: IV richness (the VRP gate). Hypothesis: a cycle whose entry
    # IV exceeds trailing realized vol does BETTER (richer premium). Predict
    # D_A>0. Carries a known low-vol confound the vol_confound column exposes.
    def tag_iv() -> tuple[np.ndarray, np.ndarray]:
        rich = cd.richness
        valid = ~np.isnan(rich)
        treated = valid & (rich > 0)
        return treated, valid
    out.append(Candidate('iv_rich', (), +1, tag_iv))

    return out


# --- the shared kill-gate ----------------------------------------------------

def _add_one_p(perm: np.ndarray, observed: float, predicted_sign: int) -> float:
    """One-sided add-one Monte Carlo p-value (Davison & Hinkley 1997), in the
    predicted direction. Counts permutation statistics at least as extreme as
    the observed one toward the prediction. Matches the prereg §5.2 convention
    and keeps the test exact (never reports p = 0)."""
    if predicted_sign < 0:
        extreme = int(np.sum(perm <= observed))   # predicted treated worse
    else:
        extreme = int(np.sum(perm >= observed))    # predicted treated better
    return (1 + extreme) / (1 + len(perm))


def _uniform_null(pnls: np.ndarray, n_treated: int,
                  rng: np.random.Generator, n_perm: int) -> np.ndarray:
    """The default null: a uniform same-count label shuffle over the valid
    cycles. Works for any binary tag, but ignores any temporal structure in the
    treatment — the lowest-common-denominator null."""
    perm = np.empty(n_perm, dtype=float)
    size = len(pnls)
    for j in range(n_perm):
        fake = np.zeros(size, dtype=bool)
        fake[rng.choice(size, size=n_treated, replace=False)] = True
        perm[j] = pnls[fake].mean() - pnls[~fake].mean()
    return perm


def _cooldown_null(cd: CycleData, cand: Candidate,
                   rng: np.random.Generator, n_perm: int) -> np.ndarray:
    """Structure-preserving trigger-placement null for the cooldown template:
    redraw each ticker's rip dates from its OWN terminals (same count), recompute
    the post-rip mask, recompute D_A. Preserves the per-ticker rip count and the
    treatment's temporal clustering — the faithful null cooldown_scout uses,
    where the generic uniform shuffle would break it. (Cooldown's `valid` is all
    cycles, so D_A is over the full pooled P&L, exactly like the scout.)"""
    horizon = int(cand.params_dict()['N'])
    perm = np.empty(n_perm, dtype=float)
    for j in range(n_perm):
        fake: dict[str, list[int]] = {}
        for ticker, rips in cd.rip_ords_by_ticker.items():
            terms = cd.term_ords_by_ticker[ticker]
            picks = rng.choice(len(terms), size=len(rips), replace=False)
            fake[ticker] = sorted(terms[i] for i in picks)
        mask = post_rip_mask(cd.entry_ords, cd.ticker_ids, fake, horizon)
        d = _d_a(cd.pnls, mask)
        perm[j] = d if d is not None else np.nan
    return perm


def kill_gate(cd: CycleData, cand: Candidate, rng: np.random.Generator,
              n_perm: int = N_PERM) -> dict[str, Any]:
    """Run one candidate through the shared D_A split + permutation null and
    return its ledger row (without the campaign-level BY verdict, added
    later). Restricts every computation to the candidate's `valid` cycles."""
    treated, valid = cand.tag()
    pnls = cd.pnls[valid]
    mask = treated[valid]
    n_treated = int(mask.sum())
    n_other = int((~mask).sum())
    row: dict[str, Any] = {
        'template': cand.template,
        'params': cand.params_dict(),
        'predicted_sign': cand.predicted_sign,
        'n_valid': int(valid.sum()),
        'n_treated': n_treated,
        'n_other': n_other,
        'n_perm': n_perm,
        'seed': None,   # the per-candidate seed is stamped by run_campaign
        'search_tickers': list(cd.tickers),
    }
    d_a = _d_a(pnls, mask)
    if d_a is None or n_treated == 0 or n_other == 0:
        # degenerate (an empty cell) — recorded, never a survivor
        row.update({'D_A': None, 'sign_ok': False, 'p_value': None,
                    'vol_confound': None})
        return row

    # the null: the template's own structure-preserving permutation if it has
    # one (cooldown's trigger placement), else the uniform same-count shuffle.
    if cand.null_fn is None:
        perm = _uniform_null(pnls, n_treated, rng, n_perm)
    else:
        perm = cand.null_fn(cd, cand, rng, n_perm)
    p_value = _add_one_p(perm, d_a, cand.predicted_sign)
    sign_ok = bool(np.sign(d_a) == cand.predicted_sign)

    # generic vol-level confound probe: does this tag mostly sort cycles by
    # trailing volatility? (the trap that made iv_richness's split look real).
    vc = _vol_confound(cd.trailing_rv[valid], mask)

    row.update({
        'D_A': round(float(d_a), 2),
        'sign_ok': sign_ok,
        'p_value': round(p_value, 4),
        'vol_confound': round(vc, 4) if vc is not None else None,
    })
    return row


def _vol_confound(trailing_rv: np.ndarray, mask: np.ndarray) -> float | None:
    """mean(trailing RV | treated) - mean(trailing RV | other), over cycles
    with a defined trailing RV. A large magnitude flags a tag that is really a
    volatility-level sort rather than the claimed signal."""
    defined = ~np.isnan(trailing_rv)
    a = trailing_rv[defined & mask]
    b = trailing_rv[defined & ~mask]
    if len(a) == 0 or len(b) == 0:
        return None
    return float(a.mean() - b.mean())


# --- the FDR correction (judged across the whole batch) ---------------------

def benjamini_yekutieli(pvals: Sequence[float | None], q: float = FDR_Q) -> list[bool]:
    """Benjamini-Yekutieli step-up FDR control, valid under arbitrary
    dependence. Equivalent to Benjamini-Hochberg but with the threshold
    divided by the harmonic factor c(n) = Σ 1/i — the price for not assuming
    the tests are independent (they are not: shared tickers, overlapping
    cycles, nested windows).

    Sort the p-values ascending; find the largest rank k with
    p(k) ≤ (k / (n·c)) · q; reject every hypothesis with p ≤ p(k). `None`
    p-values (degenerate candidates) never survive but still count toward n —
    they were tests you ran. Returns a survivor flag per input position."""
    n = len(pvals)
    if n == 0:
        return []
    c = float(np.sum(1.0 / np.arange(1, n + 1)))
    # None → treated as p = 1.0 (cannot be rejected) but counted in n
    eff = [1.0 if p is None else float(p) for p in pvals]
    order = sorted(range(n), key=lambda i: eff[i])
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if eff[i] <= (rank / (n * c)) * q:
            k_max = rank
    if k_max == 0:
        return [False] * n
    threshold = eff[order[k_max - 1]]
    return [p is not None and float(p) <= threshold for p in pvals]


# --- the ledger wrapper: run the batch, judge it, record it -----------------

def run_campaign(cd: CycleData, seed: int = CAMPAIGN_SEED,
                 n_perm: int = N_PERM, q: float = FDR_Q) -> list[dict[str, Any]]:
    """The wrapper: enumerate the batch, run each candidate through the shared
    kill-gate with a per-candidate seed, then add the campaign-wide BY verdict
    to every row. Returns the ledger rows (deterministic in `seed`)."""
    candidates = enumerate_candidates(cd)
    rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        rng = np.random.default_rng(seed + i)
        row = kill_gate(cd, cand, rng, n_perm=n_perm)
        row['seed'] = seed + i   # record the actual per-candidate seed
        rows.append(row)

    survivors = benjamini_yekutieli([r['p_value'] for r in rows], q=q)
    for r, surv in zip(rows, survivors):
        r['fdr_q'] = q
        r['by_survivor'] = bool(surv)
        # a CLEAN survivor also has the predicted sign and no dominating vol
        # confound; reported separately so a confounded "win" can't masquerade.
        r['clean_survivor'] = bool(surv and r.get('sign_ok'))
    return rows


def write_ledger(rows: Sequence[dict[str, Any]], path: str = 'edge_ledger.jsonl') -> None:
    """Append-only ledger: one immutable JSON row per candidate, full
    provenance (seed, search tickers, statistic, p-value, verdict)."""
    stamp = datetime.now().isoformat(timespec='seconds')
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps({**r, 'run_at': stamp}) + '\n')


def load_search_runs(
    search_tickers: Sequence[str] = SEARCH_TICKERS,
) -> list[dict[str, Any]]:
    """Load ONLY the given search tickers' naked runs. The seal is enforced by
    OMISSION — a sealed ticker is simply never passed here, so no candidate can
    train on it. Defaults to SEARCH_TICKERS (the MSFT + SPY, QQQ-sealed batch)."""
    return [load_naked_run(t) for t in search_tickers]


def run_batch(
    campaign: Campaign = DEFAULT_CAMPAIGN,
    seed: int = CAMPAIGN_SEED,
    n_perm: int = N_PERM,
    q: float = FDR_Q,
    iv_loader: Callable[..., dict[tuple[str, str], float]] = load_entry_ivs,
) -> list[dict[str, Any]]:
    """Run one campaign against a ticker batch: load ONLY the batch's search
    tickers (the sealed ones are never touched), build the cycle data, and run
    the templates through the kill-gate + BY. This is the entry point for
    sweeping the existing templates on the next batch of tickers — pass a
    Campaign whose `search` is the new names and whose `sealed` holds out a
    fresh underlying for confirmation. `iv_loader` is injectable for tests."""
    cd = build_cycle_data(load_search_runs(campaign.search), iv_loader=iv_loader)
    return run_campaign(cd, seed=seed, n_perm=n_perm, q=q)


def _format_summary(rows: Sequence[dict[str, Any]],
                    campaign: Campaign = DEFAULT_CAMPAIGN) -> str:
    lines = [
        f'Campaign: search={list(campaign.search)} sealed={list(campaign.sealed)} '
        f'q={FDR_Q} n_perm={N_PERM}',
        f'{"template":<10} {"params":<14} {"D_A":>9} {"sign":>4} '
        f'{"p":>7} {"vol_conf":>9} {"BY":>3} {"clean":>5}',
    ]
    for r in rows:
        params = ','.join(f'{k}={v}' for k, v in r['params'].items()) or '-'
        d_a = '-' if r['D_A'] is None else f'{r["D_A"]:.0f}'
        p = '-' if r['p_value'] is None else f'{r["p_value"]:.3f}'
        vc = '-' if r['vol_confound'] is None else f'{r["vol_confound"]:.3f}'
        lines.append(
            f'{r["template"]:<10} {params:<14} {d_a:>9} '
            f'{"ok" if r["sign_ok"] else "x":>4} {p:>7} {vc:>9} '
            f'{"Y" if r["by_survivor"] else ".":>3} '
            f'{"Y" if r["clean_survivor"] else ".":>5}')
    n_clean = sum(r['clean_survivor'] for r in rows)
    lines.append(f'\nclean survivors after BY: {n_clean} / {len(rows)}')
    return '\n'.join(lines)


# ============================================================================
# Engine-re-run phase — the structure-side template class.
#
# The re-tag class above is cheap because it never changes the trades. The
# structure-side ideas — the delta-neutral short-vol / straddle / iron-condor
# strategies — DO change the trades, so each candidate is (template, ticker)
# and runs a full run_real_*_overlay engine pass rather than re-tagging fixed
# cycles. It is scored by short_vol_statistics' Newey-West HAC t-stat against
# its ASYMPTOTIC normal null: no per-candidate permutation — the closed-form p
# is the only mechanical difference from the re-tag gate. The batch is the
# template x ticker cross-section, judged whole by the same Benjamini-Yekutieli
# pass, with a non-equity name (TLT) sealed by omission. A PARALLEL phase: it
# imports the engine lazily and never touches the re-tag gate above.
#
# Same epistemic object: EXPLORATORY, sample-spending, kill-or-justify. A
# survivor earns a pre-registration and a manual sealed-vault confirmation,
# never an automated verdict.
# ============================================================================

# Seal a structurally-different underlying the structure work never used. TLT
# (long bonds) is the strong vault here; QQQ — the re-tag seal — appears in the
# structure cross-section, so it cannot seal this phase.
STRUCTURE_SEARCH: tuple[str, ...] = ('MSFT', 'SPY', 'QQQ', 'GLD', 'XLE', 'EEM', 'NVDA')
STRUCTURE_SEALED: tuple[str, ...] = ('TLT',)
STRUCTURE_CAMPAIGN = Campaign(search=STRUCTURE_SEARCH, sealed=STRUCTURE_SEALED)

STRUCTURE_CAPITAL = 100_000
STRUCTURE_END = '2026-06-06'   # as-of date the chains are loaded through (single source)
# Engine-version tag folded into the data-lineage hash (#3a). The recorded
# statistic is pinned to (data + capital + this version), NOT inferred live from
# engine code — so a change to the overlay / short_vol_statistics mechanics or the
# frozen engine config (rf=0.045, hedge_cost_bps=1.0 in vol_premium) that re-computes
# a different t-stat for the SAME data must BUMP this, which re-lineages and
# re-records rather than silently keeping a stale answer-key row.
STRUCTURE_ENGINE_VERSION = 'v1'


# --- the closed grammar: the menu the search builds templates from -----------
# Interlock #1 of the LLM-ideation protocol. The same grammar is enforced in
# BOTH StructureTemplate.__post_init__ (the authoring object) AND
# StructureCandidate.__post_init__ (the object that actually reaches the
# kill-gate and the BY pool), via _validate_grammar below — so an off-menu value
# cannot sneak in one layer down. A closed grammar makes the template menu FINITE
# and on the record: every comparison the FDR ledger counts is provably one of
# grid_universe_size() templates, and widening the menu is a deliberate, reviewed
# edit that bumps the pinned size. That is the countability the FDR accounting
# rests on — a continuous knob like target_delta=0.241 is a hard error at
# construction, not an infinite fishing ground. The grid is a SUPERSET of the
# committed STRUCTURE_TEMPLATES below (standard option deltas / DTEs), so it
# leaves room to enumerate without re-running the published campaign.
#
# The grammar is ECONOMICALLY TYPED (the scaffold a generative widening builds on, no widening
# here — still grid_universe_size()==30): each overlay carries a PREMIUM FAMILY and a net-greek
# SIGNATURE. The committed three are all VARIANCE (short gamma/vega at one expiry). Enforcement is
# two-layer: (1) _assert_grammar_well_typed gates at IMPORT that every overlay carries a registered
# family + a complete signature (PRESENCE only — it can't run the engine without data); (2) the
# signature is CROSS-CHECKED against the engine's ACTUAL greeks by the dataset-gated
# TestGrammarSignatureMatchesEngine, which runs each overlay on real chains, backs the IV out of
# each entry leg's mid, computes BS net gamma/vega (vol_premium.structure_greek_signature), and
# asserts the engine-derived {legs, expirations, net_gamma, net_vega} matches the declared
# signature. So for the committed overlays a composition that DECLARES short gamma/vega while the
# engine runs something long-vega FAILS the test — mechanism is CHECKED against the engine, not a
# post-hoc label (the contrast paper's failure mode). The guarantee is per-verified-overlay (a test
# that must run with data, not a constructor invariant): a widening must
# ADD its structure to that test (on a ticker that actually trades it) to inherit it. STRUCTURE_GRAMMAR
# is the typed source of truth; ALLOWED_GRID is its flat lattice view (same dict objects), so
# grid_universe_size / _validate_grammar / enumerate_grammar_templates are byte-unchanged.


class PremiumFamily(Enum):
    """The economic mechanism a structure claims to harvest — the typing that keeps the grammar
    mechanism-coherent as it grows. VARIANCE (short gamma/vega, one expiration) covers the four
    short-vol overlays; SKEW (the risk reversal, widening 2) harvests the put-call skew; TERM /
    CARRY are registered for a future widening."""
    VARIANCE = 'variance'   # short realized-vs-implied variance (short gamma/vega, one expiry)
    SKEW = 'skew'           # delta-offset wing asymmetry (risk reversal — sell rich put, buy cheap call)
    TERM = 'term'           # opposite-sign vega across two expirations (calendar / diagonal)
    CARRY = 'carry'         # theta-positive defined-risk


@dataclass
class OverlayGrammar:
    """One overlay's slot in the structure grammar: its parameter lattices (the knob menu), its
    premium `family`, and a declared net-greek `signature`. The reachable templates for the
    overlay are the Cartesian product of its lattices."""
    lattices: dict[str, tuple[Any, ...]]
    family: PremiumFamily
    signature: dict[str, Any]


# The declared signature is three ROBUST economic axes — net_vega (VARIANCE), net_delta (DIRECTION),
# net_skew (the SKEW edge: do the SHORT legs sit at higher IV than the LONG legs?). net_GAMMA is
# deliberately absent: for offset-leg structures the iron-condor's short gamma and the risk-reversal's
# long gamma overlap in magnitude, so no tolerance pins both — vol_premium.structure_greek_signature
# carries the full rationale. Each axis is cross-checked against the engine by the dataset-gated
# TestGrammarSignatureMatchesEngine.
STRUCTURE_GRAMMAR: dict[str, OverlayGrammar] = {
    'short_vol':   OverlayGrammar({'target_delta': (0.15, 0.25, 0.50), 'dte': (21, 30, 45)},
                                  PremiumFamily.VARIANCE,
                                  {'expirations': 1, 'legs': 1, 'net_vega': 'short',
                                   'net_delta': 'short', 'net_skew': 'flat'}),
    'straddle':    OverlayGrammar({'dte': (21, 30, 45)},
                                  PremiumFamily.VARIANCE,
                                  {'expirations': 1, 'legs': 2, 'net_vega': 'short',
                                   'net_delta': 'neutral', 'net_skew': 'flat'}),
    'iron_condor': OverlayGrammar({'dte': (21, 30, 45), 'short_delta': (0.20, 0.25, 0.30),
                                   'wing_delta': (0.05, 0.10)},
                                  PremiumFamily.VARIANCE,
                                  {'expirations': 1, 'legs': 4, 'net_vega': 'short',
                                   'net_delta': 'neutral', 'net_skew': 'long_rich'}),
    'strangle':    OverlayGrammar({'dte': (21, 30, 45), 'short_delta': (0.20, 0.25, 0.30)},
                                  PremiumFamily.VARIANCE,   # widening 1: the straddle's OTM cousin
                                  {'expirations': 1, 'legs': 2, 'net_vega': 'short',
                                   'net_delta': 'neutral', 'net_skew': 'flat'}),
    'risk_reversal': OverlayGrammar({'dte': (21, 30, 45), 'short_delta': (0.20, 0.25, 0.30)},
                                  PremiumFamily.SKEW,       # widening 2: the first NEW family
                                  {'expirations': 1, 'legs': 2, 'net_vega': 'neutral',
                                   'net_delta': 'long', 'net_skew': 'short_rich'}),
}

# Flat lattice view of the grammar — byte-identical to the prior ALLOWED_GRID literal (SAME dict
# objects as STRUCTURE_GRAMMAR[...].lattices), so every consumer of ALLOWED_GRID is unchanged.
ALLOWED_GRID: dict[str, dict[str, tuple[Any, ...]]] = {
    name: og.lattices for name, og in STRUCTURE_GRAMMAR.items()
}


def structure_family(overlay: str) -> PremiumFamily:
    """The premium family an overlay is typed to (the economic mechanism it claims)."""
    return STRUCTURE_GRAMMAR[overlay].family


def _assert_grammar_well_typed() -> None:
    """Economic-typing scaffold, layer 1 of 2 (runs at IMPORT): gate that every overlay carries a
    REGISTERED PremiumFamily and a net-greek signature with all expected keys PRESENT, and that
    ALLOWED_GRID matches the grammar's lattices (a structural/key-level check — value drift is
    impossible since ALLOWED_GRID shares the same dict objects, so this only catches a future
    hand-written grid whose top-level structure diverges). This is PRESENCE only: it can't run the
    engine without market data, so a MIS-declared signature (net_vega='short' on an actually
    long-vega engine) is NOT caught here. Layer 2 — the signature-vs-engine cross-check that DOES
    catch a mis-declaration — is the dataset-gated TestGrammarSignatureMatchesEngine (it backs the
    IV out of each entry leg and compares BS net greeks to the declared signature). Pinned by the
    always-run TestClosedGrammar."""
    for name, og in STRUCTURE_GRAMMAR.items():
        if not isinstance(og.family, PremiumFamily):
            raise ValueError(f'{name}: {og.family!r} is not a registered PremiumFamily')
        missing = {'expirations', 'legs', 'net_vega', 'net_delta', 'net_skew'} - set(og.signature)
        if missing:
            raise ValueError(f'{name}: net-greek signature missing {sorted(missing)}')
    if ALLOWED_GRID != {n: og.lattices for n, og in STRUCTURE_GRAMMAR.items()}:
        raise ValueError('ALLOWED_GRID drifted from STRUCTURE_GRAMMAR lattices')


_assert_grammar_well_typed()


def grid_universe_size() -> int:
    """The count of distinct templates ALLOWED_GRID can express — the size of
    the reachable hypothesis universe (sum over overlays of the product of each
    knob's DISTINCT option count). Pinned by test_edge_search: bump the grid,
    bump the pin, on the record."""
    return sum(math.prod(len(set(v)) for v in grid.values())
               for grid in ALLOWED_GRID.values())


def _validate_grammar(label: str, overlay: str,
                      params: tuple[tuple[str, Any], ...], predicted_sign: int) -> None:
    """Enforce the closed grammar on an (overlay, params, sign) triple. Raises
    ValueError on an off-menu overlay/param value, a missing/extra/duplicate knob,
    or a predicted_sign that is not exactly the int -1 or +1. Membership is
    TYPE-STRICT (30.0 does not match int 30; True does not match +1): a proposer
    must pass the exact grid literal, which is the countability contract. Shared
    by StructureTemplate and StructureCandidate so the constraint sits on both the
    authoring object and the object that enters the BY pool."""
    if overlay not in ALLOWED_GRID:
        raise ValueError(f'{label}: overlay {overlay!r} off-menu; '
                         f'known overlays = {sorted(ALLOWED_GRID)}')
    grid = ALLOWED_GRID[overlay]
    keys = dict(params)
    if len(keys) != len(params):
        raise ValueError(f'{label}: duplicate param key in {params}')
    if set(keys) != set(grid):
        raise ValueError(f'{label}: params {sorted(keys)} must match the {overlay!r} '
                         f'knobs {sorted(grid)} exactly (none missing, none extra)')
    for k, v in keys.items():
        if not any(v == g and type(v) is type(g) for g in grid[k]):
            raise ValueError(f'{label}: {k}={v!r} off-menu; allowed {grid[k]}')
    if type(predicted_sign) is not int or predicted_sign not in (-1, +1):
        raise ValueError(f'{label}: predicted_sign must be the int -1 or +1, '
                         f'got {predicted_sign!r}')


@dataclass(frozen=True)
class StructureTemplate:
    """One structure strategy + its parameter setting, drawn from ALLOWED_GRID.
    `overlay` names the vol_premium engine to run (resolved lazily). Every
    committed template predicts a POSITIVE delta-hedged premium (+1): the
    short-vol seller is paid for bearing variance risk. `__post_init__` REFUSES
    to build a template whose overlay/params/sign are off-menu — so an off-grid
    value (a continuous-knob fish like target_delta=0.241) is a hard error at
    construction, never a silent extra comparison."""
    name: str
    overlay: str            # 'short_vol' | 'straddle' | 'iron_condor'
    params: tuple[tuple[str, Any], ...]
    predicted_sign: int     # mandatory: -1 or +1, a falsifiable direction (no default)

    def __post_init__(self) -> None:
        _validate_grammar(self.name, self.overlay, self.params, self.predicted_sign)


# The committed structure batch: the short call at two deltas (0.25 = the
# variance-premium wing of the +2.54 headline, 0.50 = ATM, max gamma/vega), the
# two-leg ATM straddle, and the defined-risk iron condor — every existing
# vol_premium overlay, one row each. Each states its +1 sign explicitly (no
# default), and every value is a member of ALLOWED_GRID above.
STRUCTURE_TEMPLATES: tuple[StructureTemplate, ...] = (
    StructureTemplate('short_call_25', 'short_vol', (('target_delta', 0.25), ('dte', 30)), +1),
    StructureTemplate('short_call_atm', 'short_vol', (('target_delta', 0.50), ('dte', 30)), +1),
    StructureTemplate('straddle', 'straddle', (('dte', 30),), +1),
    StructureTemplate('iron_condor', 'iron_condor',
                      (('dte', 30), ('short_delta', 0.25), ('wing_delta', 0.10)), +1),
    StructureTemplate('strangle', 'strangle',         # widening 1 (the OTM straddle)
                      (('dte', 30), ('short_delta', 0.25)), +1),
    StructureTemplate('risk_reversal', 'risk_reversal',   # widening 2 (the first NEW family: SKEW)
                      (('dte', 30), ('short_delta', 0.25)), +1),
)


@dataclass(frozen=True)
class StructureCandidate:
    """One (template, ticker) cell — a single engine overlay to run and score.
    Grammar-validated at construction (the same `_validate_grammar` as
    StructureTemplate), because THIS is the object that reaches the kill-gate and
    the BY pool — so no off-grid params can sneak in below the template layer."""
    template: str
    ticker: str
    overlay: str
    params: tuple[tuple[str, Any], ...]
    predicted_sign: int

    def __post_init__(self) -> None:
        _validate_grammar(f'{self.template}@{self.ticker}', self.overlay,
                          self.params, self.predicted_sign)

    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)


def enumerate_structure_candidates(
    campaign: Campaign = STRUCTURE_CAMPAIGN,
) -> list[StructureCandidate]:
    """The template x ticker cross-section. The seal is enforced by OMISSION: a
    sealed ticker (TLT) is never enumerated, so no candidate can run on it."""
    return [
        StructureCandidate(t.name, tk, t.overlay, t.params, t.predicted_sign)
        for tk in campaign.search
        for t in STRUCTURE_TEMPLATES
    ]


def _overlay_params_key(overlay: str, params: dict[str, Any]) -> tuple[str, str]:
    """Order-free identity of a (overlay, params) point: params canonicalized with
    json sort_keys, exactly as _ledger_key canonicalizes them — so a grid point and a
    committed template match regardless of param tuple order."""
    return (overlay, json.dumps(params, sort_keys=True))


def enumerate_grammar_templates() -> list[StructureTemplate]:
    """Expand ALLOWED_GRID into EVERY grammar-valid template — the full menu the
    deterministic menu-walker proposes from (grid_universe_size() of them). The four
    committed STRUCTURE_TEMPLATES keep their hand-chosen names (so a menu-walker cell that
    coincides with a committed one shares its _ledger_key and dedups against the published
    ledger instead of re-counting the same hypothesis under a new name); every other grid
    point gets a deterministic systematic name. Every template predicts +1 — the committed
    convention that the short-vol seller is paid for bearing variance risk.

    The systematic naming is part of the grammar's on-the-record identity: once a
    menu-walked cell is recorded (via `propose --record`), its name is frozen into the
    lifetime ledger's _ledger_key, so changing this scheme later would re-count those cells.
    Treat a naming change like a grammar widening — a human-signed, pinned edit."""
    import itertools
    committed = {_overlay_params_key(t.overlay, dict(t.params)): t.name
                 for t in STRUCTURE_TEMPLATES}
    out: list[StructureTemplate] = []
    for overlay, grid in ALLOWED_GRID.items():
        knobs = sorted(grid)   # canonical knob order (param tuple order is identity-free)
        for combo in itertools.product(*(grid[k] for k in knobs)):
            params = tuple((k, v) for k, v in zip(knobs, combo))
            name = committed.get(_overlay_params_key(overlay, dict(params))) \
                or f'{overlay}__' + '_'.join(f'{k}{v}' for k, v in params)
            out.append(StructureTemplate(name, overlay, params, +1))
    return out


def _asymptotic_p(t_nw: float, predicted_sign: int) -> float:
    """One-sided p-value from the HAC t-stat's asymptotic N(0,1) null. The
    structure phase's whole point: short_vol_statistics' Newey-West t is
    asymptotically standard normal under H0 (zero premium), so the p is
    CLOSED-FORM — no per-candidate permutation. predicted_sign=+1 tests the
    upper tail: p = P(Z >= t) = erfc(t / sqrt 2) / 2."""
    z = t_nw if predicted_sign >= 0 else -t_nw
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _put_chain_paths(ticker: str) -> list[str]:
    """The separate put-chain file to merge for a ticker whose CANONICAL store is calls-only.
    SPY/MSFT/QQQ keep puts in `{ticker}_option_dailies_puts.csv` (a separate published asset);
    GLD/XLE/EEM/NVDA carry puts in the canonical file already. Merging the puts at load is what
    lets the PUT-LEG structures (straddle, iron condor) actually enter — without it those campaign
    cells never trade and record a vacuous ~0 t-stat (the calls-only defect). Returns [] when no
    separate file exists (the bare name resolves to its .gz twin the same way the canonical does)."""
    import os
    base = f'{ticker.lower()}_option_dailies_puts.csv'
    return [base] if (os.path.exists(base) or os.path.exists(base + '.gz')) else []


def _load_ticker_data(ticker: str, end: str = STRUCTURE_END) -> tuple[Any, list[str], list[float]]:
    """Load one ticker's era-clipped chain store + matching unadjusted prices ONCE,
    reused across all that ticker's templates in a campaign — the store parse, not
    the overlay, is the per-cell cost, so caching it cuts the campaign from one load
    per (template, ticker) cell to one per ticker. LIVE CHAIN_CLEAN_START (exploratory
    sees the corrected boundary). For SPY/MSFT/QQQ the separate puts file is MERGED
    (_put_chain_paths) so the put-leg structures trade. Engine deps imported lazily so
    re-tag-only use of this module stays light."""
    from real_cc_backtest import (CHAIN_CLEAN_START, load_chain_store,
                                   load_unadjusted_prices)
    extra = _put_chain_paths(ticker)
    store = load_chain_store(f'{ticker.lower()}_option_dailies.csv',
                             extra_paths=extra,
                             start=CHAIN_CLEAN_START.get(ticker))
    days = sorted(store)
    if extra:
        # A merged puts file can PREDATE the calls (QQQ puts go back to 2011, its calls to 2016)
        # and the ticker may have no era clip — so without this the window would stretch into a
        # calls-free span where no structure can enter (every template needs a CALL leg), diluting
        # the t-stat with idle rf days and re-measuring even the call cells. Restrict the window to
        # CALL days (a positive-delta candidate — puts are negative-delta), which is exactly the
        # calls-file day set, so merging puts is purely ADDITIVE: it gives the put-leg structures a
        # put to trade against WITHOUT moving the call-cell measurement window.
        call_days = [d for d in days if any(c[1] > 0 for c in store[d]['candidates'])]
        if call_days:
            days = call_days
    dates, prices = load_unadjusted_prices(ticker, days[0], end)
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    return store, [d for d, _ in pairs], [p for _, p in pairs]


def structure_kill_gate(cand: StructureCandidate,
                        loaded: tuple[Any, list[str], list[float]],
                        capital: float = STRUCTURE_CAPITAL) -> dict[str, Any]:
    """One structure candidate → its ledger row, given the ticker's pre-loaded
    (store, dates, prices). Runs the overlay and scores the daily vol-P&L by the
    HAC t-stat's asymptotic null — no RNG, closed-form p (the only mechanical
    difference from the re-tag gate)."""
    from vol_premium import (run_real_iron_condor_overlay,
                             run_real_risk_reversal_overlay,
                             run_real_short_vol_overlay,
                             run_real_straddle_overlay, run_real_strangle_overlay,
                             short_vol_statistics)
    overlays = {'short_vol': run_real_short_vol_overlay,
                'straddle': run_real_straddle_overlay,
                'iron_condor': run_real_iron_condor_overlay,
                'strangle': run_real_strangle_overlay,
                'risk_reversal': run_real_risk_reversal_overlay}
    store, dates, prices = loaded
    summary, trades, eq = overlays[cand.overlay](dates, prices, store,
                                                 {**cand.params_dict(), 'capital': capital})
    if not trades:
        # The structure never ENTERED (e.g. a put-leg overlay on a calls-only store). No
        # measurement happened, so flag measurement_invalid (p=None -> e=0: counts toward the
        # stream, never flags) rather than scoring the idle flat rf-credit curve as a real ~0
        # t-stat — the campaign analog of the equivalence test's must_trade guard.
        return {'phase': 'structure', 'template': cand.template, 'ticker': cand.ticker,
                'params': cand.params_dict(), 'predicted_sign': cand.predicted_sign,
                'measurement_invalid': True, 'no_trades': True,
                't_stat_newey_west': None, 'sign_ok': False, 'p_value': None}
    st = short_vol_statistics(eq, summary['capital'], rf=summary['risk_free_rate'])
    t_nw = float(st['t_stat_newey_west'])
    return {
        'phase': 'structure',
        'template': cand.template,
        'ticker': cand.ticker,
        'params': cand.params_dict(),
        'predicted_sign': cand.predicted_sign,
        'n_days': st['n_days'],
        't_stat_newey_west': t_nw,
        'nw_lag': st['nw_lag'],
        'sharpe': st['sharpe'],
        'ann_excess_return_pct': st['ann_excess_return_pct'],
        'sign_ok': bool(np.sign(t_nw) == cand.predicted_sign),
        'p_value': round(_asymptotic_p(t_nw, cand.predicted_sign), 4),
    }


def _ticker_scale_ratio(loaded: tuple[Any, list[str], list[float]]) -> float | None:
    """The price-vs-chain scale guard (validate_dailies.scale_ratio) on a loaded ticker:
    median ATM-strike / price. ~1.0 is healthy; far from 1.0 means the price file is off
    the chain's as-traded scale (a split mismatch like XLE pre-fix), so the overlay's
    delta-hedge would run on the wrong price scale and the measurement is invalid."""
    from validate_dailies import scale_ratio
    store, dates, prices = loaded
    pxd = dict(zip(dates, prices))
    atm: dict[str, float] = {}
    for day in store:
        calls = [c for c in store[day]['candidates'] if 0.0 < c[1] < 1.0]  # c = (dte,delta,...,strike,cid)
        if calls:
            atm[day] = min(calls, key=lambda c: abs(c[1] - 0.5))[6]
    return scale_ratio(atm, pxd)


def run_structure_campaign(campaign: Campaign = STRUCTURE_CAMPAIGN,
                           q: float = FDR_Q,
                           capital: float = STRUCTURE_CAPITAL,
                           scorer: Callable[[StructureCandidate], dict[str, Any]] | None = None,
                           candidates: Sequence[StructureCandidate] | None = None,
                           ) -> list[dict[str, Any]]:
    """Enumerate the template x ticker structure batch, run each engine overlay,
    score by the HAC t-stat's asymptotic p, and judge the whole batch by BY.
    DETERMINISTIC — overlays + closed-form p, no RNG, so it reproduces without a seed.
    Each ticker's store loads ONCE (cached across its templates). A price-vs-chain SCALE
    GUARD runs first: a ticker whose price file is off the chain's as-traded scale (a
    split mismatch like XLE pre-fix) is flagged measurement_invalid and scored p=None — it
    still COUNTS toward BY's n (a comparison you ran) but can never be rejected, so it
    never masquerades as a survivor and never shrinks the denominator to loosen the bar for
    the other cells. `scorer` is injectable so the synthetic test layer can exercise the
    FDR/flagging path without the engine. `candidates` overrides the enumerated batch (the
    menu-walker proposer passes its proposed cells); None = the committed cross-section.
    NOTE: the e-LOND pass here is PER-BATCH (the head-of-stream view); the proposer re-judges
    over the lifetime stream via judge_against_lifetime_stream before recording (#3b)."""
    from validate_dailies import SCALE_TOL
    cands = enumerate_structure_candidates(campaign) if candidates is None else list(candidates)
    if scorer is None:
        cache: dict[str, tuple[Any, list[str], list[float]]] = {}
        invalid: dict[str, float] = {}

        def _default_score(cand: StructureCandidate) -> dict[str, Any]:
            if cand.ticker not in cache:
                cache[cand.ticker] = _load_ticker_data(cand.ticker)
                ratio = _ticker_scale_ratio(cache[cand.ticker])
                if ratio is not None and abs(ratio - 1.0) > SCALE_TOL:
                    invalid[cand.ticker] = ratio
            if cand.ticker in invalid:
                return {'phase': 'structure', 'template': cand.template, 'ticker': cand.ticker,
                        'params': cand.params_dict(), 'predicted_sign': cand.predicted_sign,
                        'measurement_invalid': True, 'scale_ratio': round(invalid[cand.ticker], 3),
                        't_stat_newey_west': None, 'sign_ok': False, 'p_value': None}
            return structure_kill_gate(cand, cache[cand.ticker], capital)

        scorer = _default_score

    rows = [scorer(c) for c in cands]
    # e-LOND is the FDR CONTROL OF RECORD (#3b, docs/prereg_fdr_budget.md): the
    # campaign's cells form the stream in enumeration order; a cell is FLAGGED (a
    # survivor) iff its calibrated e-value clears the e-LOND level. A
    # measurement_invalid cell calibrates to e=0 (p=None) and can never be flagged.
    # This SUPERSEDES the per-batch BY gate, retained below only as a diagnostic.
    # (Note the sign is already baked in: p_value is one-sided in the predicted
    # direction, so a wrong-signed cell gets a large p -> small e -> never flagged.)
    from evalue_fdr import online_fdr_survivors
    rows = online_fdr_survivors(rows)   # adds e_value, elond_level, elond_survivor
    # BY, RETIRED from the gate, kept as a within-campaign DIAGNOSTIC. The #46
    # N-shrink defense still holds for it: measurement_invalid cells carry
    # p_value=None, go INTO the BY call, count toward n, and can never be rejected
    # (dropping them would shrink n and loosen the bar for the other cells).
    by = benjamini_yekutieli([r['p_value'] for r in rows], q=q)
    for r, surv in zip(rows, by):
        r['fdr_q'] = q
        r['by_survivor'] = bool(surv)                          # diagnostic
        r['clean_survivor'] = bool(surv and r.get('sign_ok'))  # BY diagnostic
    return rows


def _format_structure_summary(rows: Sequence[dict[str, Any]],
                              campaign: Campaign = STRUCTURE_CAMPAIGN) -> str:
    from evalue_fdr import ONLINE_FDR_ALPHA
    lines = [
        f'Structure campaign: search={list(campaign.search)} '
        f'sealed={list(campaign.sealed)} alpha={ONLINE_FDR_ALPHA} '
        f'(e-LOND control; BY q={FDR_Q} diagnostic)',
        f'{"template":<15} {"ticker":<6} {"t_NW":>6} {"p":>7} '
        f'{"exc%":>6} {"shrp":>6} {"eL":>3} {"BY":>3}',
    ]
    for r in rows:
        if r.get('measurement_invalid'):
            lines.append(f'{r["template"]:<15} {r["ticker"]:<6} '
                         f'{"INVALID":>6} {"":>7} {"scale " + str(r.get("scale_ratio")):>13}'
                         f'   .   .   (e=0 / p=None: counts toward the stream, never flagged)')
            continue
        lines.append(
            f'{r["template"]:<15} {r["ticker"]:<6} '
            f'{r["t_stat_newey_west"]:>+6.2f} {r["p_value"]:>7.4f} '
            f'{r["ann_excess_return_pct"]:>6.1f} {r["sharpe"]:>6.2f} '
            f'{"Y" if r["elond_survivor"] else ".":>3} '
            f'{"Y" if r["by_survivor"] else ".":>3}')
    n_elond = sum(r['elond_survivor'] for r in rows)
    n_by = sum(r['by_survivor'] for r in rows)
    n_scored = sum(not r.get('measurement_invalid') for r in rows)
    lines.append(f'\ne-LOND survivors (control): {n_elond} / {len(rows)} cells  '
                 f'(BY diagnostic: {n_by}; {len(rows) - n_scored} measurement-invalid '
                 f'-> e=0, count toward the stream but never flagged)')
    return '\n'.join(lines)


# --- the lifetime idea ledger (interlock #3a: the guess-counter) -------------
# A COMMITTED, append-only record of every distinct structure comparison ever run
# against a data lineage — distinct from the .gitignore'd edge_ledger.jsonl (the
# regenerable per-run results). It is the foundation the cumulative-n BY threshold
# (#3b) and the scrubbed proposer scoreboard (#2) will read: it makes "how many
# comparisons has the program ever spent" a countable, on-the-record number rather
# than a per-session reset. Deduped + timestamp-free, so it is DETERMINISTIC —
# re-running a campaign on the same data lineage adds no rows (it is the same
# comparison), and the git history is the timeline. Structure-phase for now; the
# re-tag phase records into the same file once it carries a per-row lineage.
#
# NOTE: this ledger CARRIES the result statistics (p-value, t-stat) — it is the
# answer key, committed deliberately. An LLM proposer must NEVER read it; #2's
# scrubbed scoreboard is the redacted view it is allowed to see.
IDEA_LEDGER_PATH = 'idea_ledger.jsonl'


def _read_data_checksums(path: str = 'data_checksums.sha256') -> dict[str, str]:
    """filename -> sha256, parsed from the committed checksum manifest. Missing
    file -> empty map (a fresh checkout without published data still records a
    well-formed lineage, just with 'MISSING' store checksums)."""
    out: dict[str, str] = {}
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    out[parts[1]] = parts[0]
    except FileNotFoundError:
        pass
    return out


def _data_lineage_hash(ticker: str, end: str, capital: float = STRUCTURE_CAPITAL,
                       checksums: dict[str, str] | None = None) -> str:
    """A short, deterministic id for the (data + engine) a comparison's RESULT ran
    against: this ticker's chain-store checksum + its era-clip boundary + the end
    date + the deployed capital + the engine-version tag. The rule is exactly the
    inputs that move the t-stat, nothing more, nothing less — so two comparisons
    share a lineage iff they would produce the SAME result. That is what lets #3b
    count cumulative-n WITHIN a lineage and refuse to mix comparisons run against
    different data.

    Deliberately NOT folded in: the closed grammar (ALLOWED_GRID). The menu never
    enters the engine, so the same (template, params) gives a byte-identical t-stat
    no matter what else the grid can express — folding it in would re-lineage every
    comparison on a menu edit and silently RESET the lifetime counter (a fresh
    false-discovery budget on every grid widening). The grammar's countability role
    lives where it belongs: grid_universe_size and the pinned 28-cell batch (the
    denominator the BY diagnostic of a single batch needs), not the per-comparison
    identity. `checksums` is
    injectable for tests."""
    from real_cc_backtest import CHAIN_CLEAN_START
    checks = _read_data_checksums() if checksums is None else checksums
    sha = checks.get(f'{ticker.lower()}_option_dailies.csv.gz', 'MISSING')
    payload = {'ticker': ticker, 'store_sha': sha,
               'clean_start': CHAIN_CLEAN_START.get(ticker, ''),
               'end': end, 'capital': capital,
               'engine': STRUCTURE_ENGINE_VERSION}
    # For a calls-only-canonical ticker (SPY/MSFT/QQQ) the loaded store is calls+puts
    # MERGED (_load_ticker_data), so the store identity is BOTH files — fold the puts
    # checksum too. Without this a put-leg result would change while the lineage stayed
    # fixed (the stale-answer-key failure this hash exists to prevent). Only added when a
    # separate puts file exists, so a no-puts ticker's lineage is byte-unchanged.
    if _put_chain_paths(ticker):
        payload['puts_sha'] = checks.get(f'{ticker.lower()}_option_dailies_puts.csv.gz', 'MISSING')
    canon = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _ledger_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Identity of a comparison: (lineage, phase, template, ticker, params). Two
    rows with the same key ARE the same comparison — re-running it is not a new
    look, so record_trials dedupes on this."""
    return (row['data_lineage_hash'], row['phase'], row['template'],
            row['ticker'], json.dumps(row['params'], sort_keys=True))


def structure_ledger_rows(campaign_rows: Sequence[dict[str, Any]],
                          end: str = STRUCTURE_END,
                          capital: float = STRUCTURE_CAPITAL) -> list[dict[str, Any]]:
    """Project run_structure_campaign rows into lean lifetime-ledger rows: the
    hypothesis (template/ticker/params/sign), the decisive statistic, the verdict of
    record (`elond_survivor` — the e-LOND FDR control, #3b), the retained BY
    diagnostic (`by_survivor`), and a per-ticker data-lineage hash. Carries the
    result — it is the answer key, not the scrubbed view (#2). `end` / `capital`
    default to the phase constants the campaign runs with, so the recorded lineage is
    provably the span + size the comparison ran on (pass the same values you ran
    run_structure_campaign with)."""
    return [{
        'phase': 'structure',
        'template': r['template'],
        'ticker': r['ticker'],
        'params': r['params'],
        'predicted_sign': r['predicted_sign'],
        'statistic_kind': 't_nw',
        'statistic': r.get('t_stat_newey_west'),
        'p_value': r.get('p_value'),
        'elond_survivor': bool(r.get('elond_survivor', False)),   # FDR control of record (#3b)
        'by_survivor': bool(r.get('by_survivor', False)),         # retained BY diagnostic
        'measurement_invalid': bool(r.get('measurement_invalid', False)),
        'fdr_q': r.get('fdr_q'),
        'end': end,
        'data_lineage_hash': _data_lineage_hash(r['ticker'], end, capital),
    } for r in campaign_rows]


def load_idea_ledger(path: str = IDEA_LEDGER_PATH) -> list[dict[str, Any]]:
    """Read the committed lifetime ledger (missing file -> empty). A malformed line
    is intentionally FATAL (json.loads raises): for a machine-written, never-hand-
    edited, append-only record, refusing to extend on top of corruption beats
    silently losing comparisons from the lifetime count. Do not add a skip-bad-lines
    clause."""
    try:
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def record_trials(ledger_rows: Sequence[dict[str, Any]],
                  path: str = IDEA_LEDGER_PATH) -> int:
    """Append only the comparisons NOT already in the ledger (dedup by
    _ledger_key), preserving existing lines. Returns the count newly added.
    Append-only + deduped = deterministic: recording the same campaign twice is a
    no-op, so the committed file changes only when a genuinely new comparison is
    run. THIS is the counter that never silently resets."""
    seen = {_ledger_key(r) for r in load_idea_ledger(path)}
    fresh: list[dict[str, Any]] = []
    for r in ledger_rows:
        k = _ledger_key(r)
        if k not in seen:
            seen.add(k)
            fresh.append(r)
    fresh.sort(key=_ledger_key)   # canonical file order, independent of caller's row order
    if fresh:
        with open(path, 'a', encoding='utf-8') as f:
            for r in fresh:
                f.write(json.dumps(r, sort_keys=True) + '\n')
    return len(fresh)


def judge_against_lifetime_stream(
    new_ledger_rows: Sequence[dict[str, Any]],
    path: str = IDEA_LEDGER_PATH,
    prior_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Judge new ledger-format rows as the TAIL of the committed lifetime e-LOND stream,
    not as a fresh batch restarting at t=1.

    e-LOND is the cumulative-n FDR control of record (#3b, docs/prereg_fdr_budget.md): a
    hypothesis at stream position t faces level alpha_t = alpha*gamma_t*(R_{t-1}+1), so its bar
    depends on EVERYTHING before it. run_structure_campaign runs e-LOND over one batch in
    ISOLATION — correct for the published one-shot (the batch IS the head of the stream), but
    judging each appended batch alone restarts t at 1 and re-faces the loosest head-of-stream bar
    1/(alpha*gamma_1): a silent per-session budget reset, the multiple-looks leak the registration
    exists to prevent (prereg §0). This closes it for the recording path — the lifetime-stream
    judging the registration describes but run_structure_campaign alone does not deliver.

    Places the committed prior ledger AHEAD of the new rows, runs ONE e-LOND pass over the whole
    concatenation, and returns the new rows with `elond_survivor` corrected to the lifetime-stream
    verdict — schema otherwise unchanged (no e_value/elond_level leaks into the committed ledger).
    The new rows are ordered by `_ledger_key` for the pass to MATCH the order record_trials commits
    them in, so the recorded verdict is exactly what a future re-judge of the file reproduces.
    Because e-LOND is ONLINE (a row's decision depends only on rows before it), each verdict is
    fixed on arrival and never moves under later appends, so recording it is permanent. Rows
    already in the prior ledger OR repeated within the batch (same `_ledger_key`) are NOT a fresh
    look: they are not re-appended (no double-count, exactly as record_trials dedups) and return
    their committed-position verdict. `prior_rows` is injectable for tests (else loaded from
    `path`)."""
    from evalue_fdr import online_fdr_survivors
    prior = list(load_idea_ledger(path)) if prior_rows is None else list(prior_rows)
    seen = {_ledger_key(r) for r in prior}
    fresh: list[dict[str, Any]] = []
    for r in new_ledger_rows:        # dedup against prior AND within the batch — EXACTLY as
        k = _ledger_key(r)           # record_trials does — so the judged stream is byte-for-byte the
        if k not in seen:            # sequence record_trials commits, and the recorded verdict is
            seen.add(k)              # reproducible by a future re-judge of the file.
            fresh.append(r)
    fresh.sort(key=_ledger_key)
    judged = online_fdr_survivors(prior + fresh)
    survivor = {_ledger_key(r): bool(r['elond_survivor']) for r in judged}
    return [{**r, 'elond_survivor': survivor[_ledger_key(r)]} for r in new_ledger_rows]


# --- the number-free scoreboard (interlock #2: what the proposer may read) ---
# An ALLOW-LIST projection of the lifetime ledger: the hypothesis coordinates a
# proposer needs to avoid re-suggesting duds (template / ticker / params /
# predicted_sign) plus a ONE-BIT verdict — and nothing else. Every result
# statistic (p-value, t-stat, fdr_q, the lineage hash) is dropped BY CONSTRUCTION,
# not redacted after the fact: scrub_ledger_row copies only SAFE_FIELDS, so a
# result column added to the ledger later cannot leak (it is simply never copied).
# The magnitude is the dangerous channel — a near-miss t-stat tells a proposer
# WHERE to fish; the one-bit KILLED/SURVIVED does not. This is the redacted view an
# LLM proposer is allowed to read; it must NEVER read idea_ledger.jsonl (the answer
# key). Allow-list beats deny-list/regex-redaction precisely because template names
# carry digits (short_call_25) and grid values collide with results (fdr_q 0.10 ==
# wing_delta 0.10) — only structural field-selection is airtight.
SAFE_FIELDS: tuple[str, ...] = ('phase', 'template', 'ticker', 'params', 'predicted_sign')


def scrub_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one lifetime-ledger row to the proposer-safe fields + a one-bit
    verdict. Allow-list: only SAFE_FIELDS survive, so no result statistic can leak
    through a forgotten field. `params` is defensively copied so a consumer mutating
    the corpus cannot reach back into the source row. The verdict keys off
    `elond_survivor` — the e-LOND FDR control of record (#3b), NOT the retained
    `by_survivor` diagnostic: a SURVIVED cell is exactly one e-LOND flags (the one the
    prereg escalates to manual pre-registration), so the corpus exclusion tracks the
    control, not the diagnostic. The two are not guaranteed to coincide — e-LOND's
    (R+1) reward can flag a cell BY does not — so keying off the diagnostic would
    mislabel an e-LOND survivor KILLED and leak it back into automated proposal.
    measurement_invalid surfaces as INVALID — a per-TICKER data-quality state (the
    scale mismatch), not a per-hypothesis result; it tells a proposer which ticker's
    price file needs a human fix, nothing about where an edge lives."""
    out = {f: (dict(row[f]) if f == 'params' else row[f]) for f in SAFE_FIELDS}
    out['verdict'] = ('INVALID' if row.get('measurement_invalid')
                      else 'SURVIVED' if row.get('elond_survivor') else 'KILLED')
    return out


def build_proposer_corpus(ledger_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The scrubbed view of the lifetime ledger an automated proposer may read —
    names + verdict, no numbers beyond the hypothesis coordinates. SURVIVED rows are
    EXCLUDED: a survivor is the one genuine "fish here" coordinate (an e-LOND-flagged
    cell — the FDR control of record, #3b — announcing the comparison that cleared the
    bar), and per the manual-graduation discipline it escalates to human pre-
    registration out-of-band — it must never feed back into automated proposal. So
    the corpus is the duds to avoid (KILLED) plus unmeasurable tickers (INVALID).
    Re-proposing an excluded survivor cell is harmless: record_trials dedupes it."""
    return [s for r in ledger_rows
            if (s := scrub_ledger_row(r))['verdict'] != 'SURVIVED']


def load_proposer_corpus(path: str = IDEA_LEDGER_PATH) -> list[dict[str, Any]]:
    """Load the committed lifetime ledger and return ONLY its scrubbed projection —
    the single function a proposer-facing surface should call (never load_idea_ledger,
    which carries the answer key).

    CONTINGENCY (not yet an interlock): this projection is leak-proof only for a
    proposer that reads THROUGH it. idea_ledger.jsonl is committed to git and carries
    the full answer key, and nothing yet DENIES a repo-aware agent from reading it
    directly — "the proposer must never read the ledger" is an honor-system
    convention today, not a mechanized control. The access boundary (a vault dir + a
    scoped Read-deny, or committing only the scrubbed projection to the proposer-
    visible path) is the unbuilt interlock that makes this scoreboard meaningful."""
    return build_proposer_corpus(load_idea_ledger(path))


def render_proposer_corpus(scrubbed: Sequence[dict[str, Any]]) -> str:
    """A markdown table of the scrubbed corpus — what's been tried and whether it
    survived, with every result statistic absent. Safe to hand to a proposer."""
    if not scrubbed:
        return '(no comparisons recorded yet)'
    lines = ['| template | ticker | params | predicted | verdict |',
             '| --- | --- | --- | --- | --- |']
    for r in scrubbed:
        params = ', '.join(f'{k}={v}' for k, v in r['params'].items()) or '-'
        sign = '+1' if r['predicted_sign'] > 0 else '-1'
        lines.append(f'| {r["template"]} | {r["ticker"]} | {params} | {sign} | {r["verdict"]} |')
    return '\n'.join(lines)


# --- Phase 1: the deterministic menu-walker proposer (no LLM) ----------------
# The smallest end-to-end slice of the proposer loop, with a DUMB ENUMERATOR standing in
# for the future LLM author: read the scrubbed corpus -> propose grammar-valid cells not yet
# tried -> grammar-gate -> run the engine -> judge over the lifetime e-LOND stream (#3b) ->
# record -> next round re-reads the corpus and skips them. Zero model nondeterminism, zero
# read-gate exposure (there is no model to deny yet) — it proves the plumbing the LLM later
# plugs into, swapping its JSON output for the enumerator while the gate/judge/record stay
# identical. The proposer reads ONLY load_proposer_corpus, never the answer-key ledger.

def _is_onboarded(ticker: str) -> bool:
    """True iff the ticker's option-daily store is present, so a proposed cell on it can
    actually run. Keys on the CANONICAL `{ticker}_option_dailies.csv[.gz]` only — the primary
    store run_structure_campaign loads and the pinned spans clip to; a backfill-only
    (`{ticker}_option_dailies_<era>...`) ticker is correctly treated as un-onboarded (a backfill
    is an extended-span MERGE, not a primary store). An un-onboarded ticker is NOT auto-fetched
    (premium data costs money) — the proposer routes it to the human-gated onboard pipeline."""
    import os
    base = os.path.join(os.path.dirname(__file__), f'{ticker.lower()}_option_dailies.csv')
    return os.path.exists(base) or os.path.exists(base + '.gz')


def _proposer_key(template: str, ticker: str, params: dict[str, Any]) -> tuple[str, str, str]:
    """The proposer's LINEAGE-FREE dedup identity: (template, ticker, canonical params),
    matching the scrubbed corpus's coordinates exactly (scrub_ledger_row drops the lineage
    hash — it is not in SAFE_FIELDS). ONE canonicalizer, shared by _cand_key (candidate side)
    and run_proposer_round (corpus side), so the proposer's skip and the corpus cannot desync.

    Lineage-free is deliberate: within the published data lineage it skips what's already
    tried. It does NOT auto-re-open a cell after a data refresh — the lineage-free corpus keeps
    a refreshed cell skipped — so picking up a refresh is a SEPARATE path (the corpus carrying
    lineage, or a forced re-run), not something record_trials' write-time _ledger_key delivers
    through the proposer. (record_trials still dedups on the full lineage-aware _ledger_key at
    write, so a refreshed cell that DID reach it would be a new row; the proposer just never
    reaches it for a skipped cell.)"""
    return (template, ticker, json.dumps(params, sort_keys=True))


def _cand_key(c: StructureCandidate) -> tuple[str, str, str]:
    """A candidate's proposer dedup key (see _proposer_key)."""
    return _proposer_key(c.template, c.ticker, c.params_dict())


def propose_structure_candidates(
    campaign: Campaign = STRUCTURE_CAMPAIGN,
    tried_keys: set[tuple[str, str, str]] | None = None,
    templates: Sequence[StructureTemplate] | None = None,
) -> tuple[list[StructureCandidate], list[str]]:
    """The deterministic MENU-WALKER — the stand-in for the future LLM author. Cross every
    grammar template (enumerate_grammar_templates) with every ONBOARDED search ticker, drop
    the cells already tried (`tried_keys`, the scrubbed corpus's coordinates). Returns
    (candidates, needs_onboard): an un-onboarded search ticker is never run, only flagged for
    the human-gated onboard pipeline. The grammar-gate is enforced at StructureCandidate
    construction — exactly the gate the LLM's output will hit. The seal holds by omission
    (campaign.search never contains a sealed ticker)."""
    tried = tried_keys or set()
    templates = enumerate_grammar_templates() if templates is None else list(templates)
    cands: list[StructureCandidate] = []
    needs_onboard: list[str] = []
    for tk in campaign.search:
        if not _is_onboarded(tk):
            needs_onboard.append(tk)
            continue
        for t in templates:
            c = StructureCandidate(t.name, tk, t.overlay, t.params, t.predicted_sign)
            if _cand_key(c) not in tried:
                cands.append(c)
    return cands, needs_onboard


def run_proposer_round(
    campaign: Campaign = STRUCTURE_CAMPAIGN,
    path: str = IDEA_LEDGER_PATH,
    capital: float = STRUCTURE_CAPITAL,
    scorer: Callable[[StructureCandidate], dict[str, Any]] | None = None,
    run: bool = True,
    record: bool = False,
) -> dict[str, Any]:
    """One menu-walker proposer round (Phase 1 — deterministic, NO LLM): the loop the LLM
    will later plug into, with the enumerator standing in for the author.

      READ scrubbed corpus -> PROPOSE (menu-walk grammar x onboarded tickers, minus tried)
      -> GRAMMAR-GATE (StructureCandidate) -> RUN (engine, scored per-batch) -> JUDGE over the
      lifetime e-LOND stream (judge_against_lifetime_stream, #3b) -> RECORD -> next round
      re-reads the corpus and skips them.

    `run=False` is a cheap PREVIEW — it proposes the untried cells but runs no engine and
    writes nothing. `run=True, record=False` runs + lifetime-judges without writing (a dry
    run). `record=True` appends the judged rows to the lifetime ledger (and implies run). The
    proposer reads only the scrubbed corpus, never the answer-key ledger. `scorer` is
    injectable for the synthetic test layer."""
    corpus = load_proposer_corpus(path)
    tried = {_proposer_key(r['template'], r['ticker'], r['params']) for r in corpus}
    cands, needs_onboard = propose_structure_candidates(campaign, tried)
    result: dict[str, Any] = {'proposed': len(cands), 'recorded': 0,
                              'needs_onboard': needs_onboard, 'candidates': cands,
                              'rows': [], 'ledger_rows': []}
    if not cands or not run:
        return result
    rows = run_structure_campaign(campaign, capital=capital, scorer=scorer, candidates=cands)
    ledger_rows = judge_against_lifetime_stream(
        structure_ledger_rows(rows, capital=capital), path=path)
    result['rows'] = rows
    result['ledger_rows'] = ledger_rows
    if record:
        result['recorded'] = record_trials(ledger_rows, path)
    return result


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'structure':
        print('Running the structure (engine-re-run) campaign — TLT sealed '
              '(a few minutes cold) ...', flush=True)
        rows = run_structure_campaign()
        write_ledger(rows)
        print(_format_structure_summary(rows))
        if '--record' in sys.argv:
            # judge the batch as the TAIL of the committed lifetime stream (cumulative-n
            # e-LOND), not in isolation — so an appended batch never resets the budget.
            ledger_rows = judge_against_lifetime_stream(structure_ledger_rows(rows))
            n = record_trials(ledger_rows)
            print(f'\nidea_ledger: +{n} new comparison(s) recorded to '
                  f'{IDEA_LEDGER_PATH} (deduped; e-LOND judged over the lifetime stream)')
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'propose':
        # Phase 1: the deterministic menu-walker proposer (no LLM). Default is a cheap
        # PREVIEW (propose untried cells, run nothing); --run scores them; --record records.
        record = '--record' in sys.argv
        run = record or '--run' in sys.argv
        print(f'Menu-walker proposer (deterministic, no LLM); grammar = '
              f'{grid_universe_size()} templates'
              f'{" — running engine, a while cold ..." if run else " (preview)"}', flush=True)
        res = run_proposer_round(run=run, record=record)
        print(f'proposed {res["proposed"]} untried cell(s) across onboarded search tickers')
        if res['needs_onboard']:
            print(f'  needs onboard (NOT run — human-gated fetch): {res["needs_onboard"]}')
        if run:
            print(f'  ran + lifetime-judged {len(res["rows"])} cell(s); '
                  f'recorded {res["recorded"]} to {IDEA_LEDGER_PATH}'
                  if record else
                  f'  ran + lifetime-judged {len(res["rows"])} cell(s) (dry run — nothing recorded)')
        else:
            print('  preview only — pass --run to score, --record to record')
        return
    print('Loading search runs (sealed set excluded; a few minutes cold) ...',
          flush=True)
    rows = run_batch(DEFAULT_CAMPAIGN)
    write_ledger(rows)
    print(_format_summary(rows, DEFAULT_CAMPAIGN))


if __name__ == '__main__':
    main()
