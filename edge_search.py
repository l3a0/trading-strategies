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

import json
import math
from dataclasses import dataclass, field
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
STRUCTURE_SEARCH: tuple[str, ...] = ('MSFT', 'SPY', 'QQQ', 'GLD', 'XLE', 'EEM')
STRUCTURE_SEALED: tuple[str, ...] = ('TLT',)
STRUCTURE_CAMPAIGN = Campaign(search=STRUCTURE_SEARCH, sealed=STRUCTURE_SEALED)

STRUCTURE_CAPITAL = 100_000


@dataclass(frozen=True)
class StructureTemplate:
    """One structure strategy + its parameter setting. `overlay` names the
    vol_premium engine to run (resolved lazily). Every template predicts a
    POSITIVE delta-hedged premium (+1): the short-vol seller is paid for
    bearing variance risk."""
    name: str
    overlay: str            # 'short_vol' | 'straddle' | 'iron_condor'
    params: tuple[tuple[str, Any], ...]
    predicted_sign: int = +1


# The committed structure batch: the short call at two deltas (0.25 = the
# variance-premium wing of the +2.54 headline, 0.50 = ATM, max gamma/vega), the
# two-leg ATM straddle, and the defined-risk iron condor — every existing
# vol_premium overlay, one row each.
STRUCTURE_TEMPLATES: tuple[StructureTemplate, ...] = (
    StructureTemplate('short_call_25', 'short_vol', (('target_delta', 0.25), ('dte', 30))),
    StructureTemplate('short_call_atm', 'short_vol', (('target_delta', 0.50), ('dte', 30))),
    StructureTemplate('straddle', 'straddle', (('dte', 30),)),
    StructureTemplate('iron_condor', 'iron_condor',
                      (('dte', 30), ('short_delta', 0.25), ('wing_delta', 0.10))),
)


@dataclass(frozen=True)
class StructureCandidate:
    """One (template, ticker) cell — a single engine overlay to run and score."""
    template: str
    ticker: str
    overlay: str
    params: tuple[tuple[str, Any], ...]
    predicted_sign: int

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


def _asymptotic_p(t_nw: float, predicted_sign: int) -> float:
    """One-sided p-value from the HAC t-stat's asymptotic N(0,1) null. The
    structure phase's whole point: short_vol_statistics' Newey-West t is
    asymptotically standard normal under H0 (zero premium), so the p is
    CLOSED-FORM — no per-candidate permutation. predicted_sign=+1 tests the
    upper tail: p = P(Z >= t) = erfc(t / sqrt 2) / 2."""
    z = t_nw if predicted_sign >= 0 else -t_nw
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _load_ticker_data(ticker: str, end: str = '2026-06-06') -> tuple[Any, list[str], list[float]]:
    """Load one ticker's era-clipped chain store + matching unadjusted prices ONCE,
    reused across all that ticker's templates in a campaign — the store parse, not
    the overlay, is the per-cell cost, so caching it cuts the campaign from one load
    per (template, ticker) cell to one per ticker. LIVE CHAIN_CLEAN_START (exploratory
    sees the corrected boundary). Engine deps imported lazily so re-tag-only use of
    this module stays light."""
    from real_cc_backtest import (CHAIN_CLEAN_START, load_chain_store,
                                   load_unadjusted_prices)
    store = load_chain_store(f'{ticker.lower()}_option_dailies.csv',
                             start=CHAIN_CLEAN_START.get(ticker))
    days = sorted(store)
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
                             run_real_short_vol_overlay,
                             run_real_straddle_overlay, short_vol_statistics)
    overlays = {'short_vol': run_real_short_vol_overlay,
                'straddle': run_real_straddle_overlay,
                'iron_condor': run_real_iron_condor_overlay}
    store, dates, prices = loaded
    summary, _, eq = overlays[cand.overlay](dates, prices, store,
                                            {**cand.params_dict(), 'capital': capital})
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
                           ) -> list[dict[str, Any]]:
    """Enumerate the template x ticker structure batch, run each engine overlay,
    score by the HAC t-stat's asymptotic p, and judge the whole batch by BY.
    DETERMINISTIC — overlays + closed-form p, no RNG, so it reproduces without a seed.
    Each ticker's store loads ONCE (cached across its templates). A price-vs-chain SCALE
    GUARD runs first: a ticker whose price file is off the chain's as-traded scale (a
    split mismatch like XLE pre-fix) is flagged measurement_invalid and EXCLUDED from the
    BY batch — it never inflates n or masquerades as a survivor. `scorer` is injectable so
    the synthetic test layer can exercise the FDR/flagging path without the engine."""
    from validate_dailies import SCALE_TOL
    cands = enumerate_structure_candidates(campaign)
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
    scored = [r for r in rows if not r.get('measurement_invalid')]
    survivors = benjamini_yekutieli([r['p_value'] for r in scored], q=q)
    for r, surv in zip(scored, survivors):
        r['fdr_q'] = q
        r['by_survivor'] = bool(surv)
        r['clean_survivor'] = bool(surv and r['sign_ok'])
    for r in rows:
        if r.get('measurement_invalid'):
            r['fdr_q'] = q
            r['by_survivor'] = False
            r['clean_survivor'] = False
    return rows


def _format_structure_summary(rows: Sequence[dict[str, Any]],
                              campaign: Campaign = STRUCTURE_CAMPAIGN) -> str:
    lines = [
        f'Structure campaign: search={list(campaign.search)} '
        f'sealed={list(campaign.sealed)} q={FDR_Q} (HAC-t asymptotic null)',
        f'{"template":<15} {"ticker":<6} {"t_NW":>6} {"p":>7} '
        f'{"exc%":>6} {"shrp":>6} {"BY":>3} {"clean":>5}',
    ]
    for r in rows:
        if r.get('measurement_invalid'):
            lines.append(f'{r["template"]:<15} {r["ticker"]:<6} '
                         f'{"INVALID":>6} {"":>7} {"scale " + str(r.get("scale_ratio")):>13}'
                         f'   .     .   (excluded from BY)')
            continue
        lines.append(
            f'{r["template"]:<15} {r["ticker"]:<6} '
            f'{r["t_stat_newey_west"]:>+6.2f} {r["p_value"]:>7.4f} '
            f'{r["ann_excess_return_pct"]:>6.1f} {r["sharpe"]:>6.2f} '
            f'{"Y" if r["by_survivor"] else ".":>3} '
            f'{"Y" if r["clean_survivor"] else ".":>5}')
    n_clean = sum(r['clean_survivor'] for r in rows)
    n_scored = sum(not r.get('measurement_invalid') for r in rows)
    lines.append(f'\nclean survivors after BY: {n_clean} / {n_scored} scored '
                 f'({len(rows) - n_scored} measurement-invalid, excluded)')
    return '\n'.join(lines)


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'structure':
        print('Running the structure (engine-re-run) campaign — TLT sealed '
              '(a few minutes cold) ...', flush=True)
        rows = run_structure_campaign()
        write_ledger(rows)
        print(_format_structure_summary(rows))
        return
    print('Loading search runs (sealed set excluded; a few minutes cold) ...',
          flush=True)
    rows = run_batch(DEFAULT_CAMPAIGN)
    write_ledger(rows)
    print(_format_summary(rows, DEFAULT_CAMPAIGN))


if __name__ == '__main__':
    main()
