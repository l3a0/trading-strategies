"""Tests for edge_search.py — the MVP FDR-controlled edge-search harness.

Two layers, like the other real-chain suites:

- Always-run unit tests: the Benjamini-Yekutieli FDR step (including its
  harmonic penalty over plain Benjamini-Hochberg and its None-handling), the
  hypothesis-template enumerator (counts, signs, params), the per-cycle
  CycleData builder (trailing-vol / trailing-return / richness validity), the
  shared kill-gate (one-sided add-one p direction, determinism, the
  degenerate empty-cell guard), the vol-confound probe, and the seal
  (SEALED_TICKERS never overlap the search set).
- TestEdgeSearchCampaign (dataset-gated): pins the MVP campaign on the real
  MSFT + SPY chains (QQQ sealed) — the decisive output is that NO candidate in
  the cheap entry-conditioning class survives campaign-wide BY.

All pins are EXPLORATORY numbers, not registered verdicts — pinned so the
swept class is not re-derived. See docs/edge_search.md.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from edge_search import (
    ALLOWED_GRID,
    COOLDOWN_NS,
    DEFAULT_CAMPAIGN,
    PremiumFamily,
    STRUCTURE_GRAMMAR,
    SEALED_TICKERS,
    SEARCH_TICKERS,
    STRUCTURE_CAMPAIGN,
    STRUCTURE_SEALED,
    STRUCTURE_SEARCH,
    STRUCTURE_TEMPLATES,
    TREND_WINDOWS,
    WIRE_VERSION,
    Campaign,
    Candidate,
    CycleData,
    OverlayGrammar,
    ProposalBatch,
    StructureCandidate,
    StructureTemplate,
    grid_universe_size,
    main,
    structure_family,
    _add_one_p,
    _assert_grammar_well_typed,
    _assert_llm_boundary,
    _asymptotic_p,
    _cooldown_null,
    _cand_key,
    _data_lineage_hash,
    _engine_importable_from_cwd,
    _ledger_key,
    _resolve_llm_author,
    _vol_confound,
    benjamini_yekutieli,
    build_cycle_data,
    build_proposer_corpus,
    enumerate_candidates,
    enumerate_grammar_templates,
    enumerate_structure_candidates,
    judge_against_lifetime_stream,
    kill_gate,
    propose_structure_candidates,
    run_proposer_round,
    score_and_record,
    assert_numberless,
    _load_ticker_data,
    load_idea_ledger,
    load_proposer_corpus,
    load_search_runs,
    llm_propose_candidates,
    record_trials,
    render_proposer_corpus,
    run_batch,
    run_campaign,
    run_structure_campaign,
    scrub_ledger_row,
    structure_ledger_rows,
)
from evalue_fdr import online_fdr_survivors
from read_gate_wire import BANNED_RESULT_FIELDS
from test_real_cc_backtest import _HAVE_MSFT_DAILIES, _HAVE_SPY_DAILIES

_HAVE_SEARCH = _HAVE_MSFT_DAILIES and _HAVE_SPY_DAILIES


def _have_dailies(ticker: str) -> bool:
    base = os.path.join(os.path.dirname(__file__), f'{ticker.lower()}_option_dailies.csv')
    return os.path.exists(base) or os.path.exists(base + '.gz')


# the structure campaign needs all of its search tickers' chains (the sealed TLT is
# never loaded); the unadjusted price files are committed, so they are always present.
_HAVE_STRUCTURE = all(_have_dailies(t) for t in STRUCTURE_SEARCH)


def _synthetic_runs(seed: int = 0):
    """Two small fabricated naked runs (MSFT + SPY) with hand-controllable
    cycles, for the always-run layer — no option-daily CSVs required."""
    def one(ticker: str, s: int):
        rng = np.random.default_rng(s)
        dates = [f'2020-{m:02d}-{d:02d}' for m in range(1, 13) for d in (1, 15)]
        prices = list(100 + np.cumsum(rng.normal(0, 1, len(dates))))
        cycles = []
        for i in range(0, len(dates) - 2, 2):
            cycles.append({
                'entry_date': dates[i], 'entry_contract': f'{ticker}{i}',
                'terminal_date': dates[i + 1], 'action': 'close',
                'pnl': float(rng.normal(0, 500)), 'rip': i % 4 == 0,
            })
        return {'ticker': ticker, 'dates': dates, 'prices': prices, 'cycles': cycles}
    return [one('MSFT', seed + 1), one('SPY', seed + 2)]


def _fake_iv(constant: float = 0.25):
    return lambda ticker, wanted: {k: constant for k in wanted}


# ---- always-run: Benjamini-Yekutieli FDR ----

class TestBenjaminiYekutieli:
    def test_known_vector(self) -> None:
        """n=5, q=0.10, c=Σ(1/i)=2.2833. Thresholds (k/(n·c))·q rise from
        0.00876 to 0.0438; the step-up rejects p ≤ p(k_max). Here k_max=2
        (0.012 ≤ 0.01752, but 0.03 > 0.02628), so only the first two survive."""
        pv = [0.001, 0.012, 0.03, 0.21, 0.5]
        assert benjamini_yekutieli(pv, 0.10) == [True, True, False, False, False]

    def test_more_conservative_than_bh(self) -> None:
        """The harmonic penalty is the whole point of BY: at q=0.10 the p=0.03
        hypothesis would pass plain Benjamini-Hochberg (rank-3 threshold
        3/5·0.10 = 0.06) but fails BY (0.03 > 0.02628). Encodes the dependence
        robustness the correlated candidates need."""
        pv = [0.001, 0.012, 0.03, 0.21, 0.5]
        # BH (c=1) would keep rank-3; BY (c=2.28) does not.
        assert benjamini_yekutieli(pv, 0.10)[2] is False

    def test_none_counts_toward_n_but_never_survives(self) -> None:
        """A degenerate candidate (p = None) cannot be rejected, yet still
        inflates n — it was a test you ran."""
        assert benjamini_yekutieli([0.001, None, 0.5], 0.10) == [True, False, False]

    def test_all_fail_and_empty(self) -> None:
        assert benjamini_yekutieli([0.4, 0.6, 0.9], 0.10) == [False, False, False]
        assert benjamini_yekutieli([], 0.10) == []


# ---- always-run: enumerator ----

class TestEnumerator:
    def test_batch_shape(self) -> None:
        """One candidate per (template, setting): |cooldown| + |up_trend| + 1
        IV-richness. Each carries a sign prediction (the falsifiability the
        enumerator enforces)."""
        cd = build_cycle_data(_synthetic_runs(), iv_loader=_fake_iv())
        cands = enumerate_candidates(cd)
        assert len(cands) == len(COOLDOWN_NS) + len(TREND_WINDOWS) + 1
        by_tmpl = {}
        for c in cands:
            by_tmpl.setdefault(c.template, []).append(c)
        assert len(by_tmpl['cooldown']) == len(COOLDOWN_NS)
        assert len(by_tmpl['up_trend']) == len(TREND_WINDOWS)
        assert len(by_tmpl['iv_rich']) == 1
        # sign predictions: entry-after-an-up-move templates predict D_A<0,
        # the VRP template predicts D_A>0.
        assert all(c.predicted_sign == -1 for c in by_tmpl['cooldown'])
        assert all(c.predicted_sign == -1 for c in by_tmpl['up_trend'])
        assert by_tmpl['iv_rich'][0].predicted_sign == +1

    def test_params_are_readable(self) -> None:
        cd = build_cycle_data(_synthetic_runs(), iv_loader=_fake_iv())
        cd_by = {(c.template, c.params_dict().get('N') or c.params_dict().get('window')): c
                 for c in enumerate_candidates(cd)}
        assert ('cooldown', COOLDOWN_NS[0]) in cd_by
        assert ('up_trend', TREND_WINDOWS[-1]) in cd_by


# ---- always-run: CycleData builder ----

class TestBuildCycleData:
    def test_validity_and_richness(self) -> None:
        """Trailing vol/return are nan until enough history exists; richness is
        nan when the injected IV is missing or below the floor."""
        runs = _synthetic_runs()
        # one cycle's IV missing, one below floor, rest fine
        present = {}
        for r in runs:
            for c in r['cycles']:
                present[(c['entry_date'], c['entry_contract'])] = 0.30
        # drop MSFT's first cycle IV, set SPY's first below floor
        first_msft = (runs[0]['cycles'][0]['entry_date'], runs[0]['cycles'][0]['entry_contract'])
        first_spy = (runs[1]['cycles'][0]['entry_date'], runs[1]['cycles'][0]['entry_contract'])
        del present[first_msft]
        present[first_spy] = 0.01  # below IV_FLOOR
        cd = build_cycle_data(runs, iv_loader=lambda t, w: {k: present[k] for k in w if k in present})
        # earliest cycles have no trailing window → nan
        assert np.isnan(cd.trailing_ret[TREND_WINDOWS[-1]][0])
        # rip ordinals recorded per ticker, sorted
        assert set(cd.rip_ords_by_ticker) == {'MSFT', 'SPY'}
        assert cd.rip_ords_by_ticker['MSFT'] == sorted(cd.rip_ords_by_ticker['MSFT'])
        # at least one richness is nan (the dropped + below-floor IVs)
        assert np.isnan(cd.richness).any()


# ---- always-run: kill-gate + p-value direction + confound ----

def _const_cycledata(pnls, treated, trailing_rv=None):
    """A minimal CycleData plus a Candidate whose tag returns a fixed mask."""
    n = len(pnls)
    pnls = np.asarray(pnls, float)
    treated = np.asarray(treated, bool)
    rv = np.asarray(trailing_rv if trailing_rv is not None else [0.2] * n, float)
    cd = CycleData(pnls=pnls, entry_ords=list(range(n)),
                   ticker_ids=['X'] * n, rip_ords_by_ticker={'X': []},
                   trailing_rv=rv, trailing_ret={}, richness=np.full(n, np.nan),
                   tickers=['X'])
    return cd, treated


class TestKillGate:
    def test_add_one_p_direction(self) -> None:
        perm = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # predict D_A<0: count perms ≤ observed; observed=-1 → {-2,-1}=2 → 3/6
        assert _add_one_p(perm, -1.0, -1) == pytest.approx(3 / 6)
        # predict D_A>0: count perms ≥ observed; observed=1 → {1,2}=2 → 3/6
        assert _add_one_p(perm, 1.0, +1) == pytest.approx(3 / 6)

    def test_treated_clearly_worse_is_significant(self) -> None:
        """Treated cycles all lose, others all win; predicted_sign -1. D_A is
        strongly negative and lands in the low tail → small one-sided p."""
        cd, treated = _const_cycledata(
            pnls=[-1000] * 6 + [1000] * 6, treated=[True] * 6 + [False] * 6)
        cand = Candidate('t', (), -1, lambda: (treated, np.ones(12, bool)))
        row = kill_gate(cd, cand, np.random.default_rng(0), n_perm=500)
        assert row['D_A'] < 0 and row['sign_ok'] is True
        assert row['p_value'] < 0.05

    def test_degenerate_empty_cell(self) -> None:
        """No treated cycles → empty cell → recorded with None, never a win."""
        cd, treated = _const_cycledata(pnls=[1, 2, 3], treated=[False, False, False])
        cand = Candidate('t', (), -1, lambda: (treated, np.ones(3, bool)))
        row = kill_gate(cd, cand, np.random.default_rng(0), n_perm=50)
        assert row['D_A'] is None and row['p_value'] is None
        assert row['sign_ok'] is False

    def test_determinism(self) -> None:
        cd, treated = _const_cycledata(
            pnls=[-500, 400, -300, 600, -100, 200], treated=[True, False] * 3)
        cand = Candidate('t', (), -1, lambda: (treated, np.ones(6, bool)))
        a = kill_gate(cd, cand, np.random.default_rng(7), n_perm=300)
        b = kill_gate(cd, cand, np.random.default_rng(7), n_perm=300)
        assert a['p_value'] == b['p_value'] and a['D_A'] == b['D_A']

    def test_vol_confound_sign(self) -> None:
        """The probe is mean(RV|treated) − mean(RV|other)."""
        rv = np.array([0.30, 0.30, 0.10, 0.10])
        assert _vol_confound(rv, np.array([True, True, False, False])) == pytest.approx(0.20)
        # undefined when a cell is empty / all-nan
        assert _vol_confound(np.array([np.nan, np.nan]), np.array([True, False])) is None


# ---- always-run: per-template permutation null ----

class TestPerTemplateNull:
    """A Candidate can carry its own structure-preserving null (`null_fn`);
    the default (None) routes to the uniform same-count shuffle."""

    def test_kill_gate_dispatches_to_custom_null_fn(self) -> None:
        """When a Candidate supplies null_fn, kill_gate uses it for the
        permutation distribution instead of the uniform shuffle."""
        cd, treated = _const_cycledata(
            pnls=[-1000] * 6 + [1000] * 6, treated=[True] * 6 + [False] * 6)
        called = []

        def sentinel(cd_, cand_, rng_, n_perm_):
            called.append(n_perm_)
            return np.full(n_perm_, 999.0)   # a null far above the observed D_A

        cand = Candidate('t', (), -1, lambda: (treated, np.ones(12, bool)),
                         null_fn=sentinel)
        row = kill_gate(cd, cand, np.random.default_rng(0), n_perm=50)
        assert called == [50]                # the custom null was invoked
        # observed D_A = -2000 (predicted_sign -1); no perm (all 999) ≤ it → 1/51
        # (kill_gate rounds the p-value to 4 dp)
        assert row['p_value'] == pytest.approx(1 / 51, abs=1e-4)

    def test_cooldown_null_deterministic_and_count_preserving(self) -> None:
        """The cooldown trigger-placement null redraws each ticker's rips from
        its own terminals (count preserved), recomputes D_A; seeded → it is
        deterministic and (on a healthy pool) never degenerate."""
        terms = list(range(1, 13, 2))         # 6 terminal ordinals
        cd = CycleData(
            pnls=np.array([100, -200, 150, -50, 80, -120], float),
            entry_ords=list(range(0, 12, 2)), ticker_ids=['X'] * 6,
            rip_ords_by_ticker={'X': sorted([terms[0], terms[2]])},  # 2 rips
            trailing_rv=np.full(6, np.nan), trailing_ret={},
            richness=np.full(6, np.nan), tickers=['X'],
            term_ords_by_ticker={'X': sorted(terms)})                # 6 terminals
        cand = Candidate('cooldown', (('N', 3),), -1,
                         lambda: (np.zeros(6, bool), np.ones(6, bool)),
                         null_fn=_cooldown_null)
        a = _cooldown_null(cd, cand, np.random.default_rng(1), 100)
        b = _cooldown_null(cd, cand, np.random.default_rng(1), 100)
        assert len(a) == 100 and np.array_equal(a, b)   # deterministic
        assert np.isfinite(a).all()                      # no degenerate draw


# ---- always-run: campaign invariants + the seal ----

class TestCampaignInvariants:
    def test_clean_survivor_requires_sign_ok(self) -> None:
        """clean_survivor ⟺ (by_survivor AND sign_ok), so a wrong-signed BY
        survivor can never be reported as a clean win."""
        cd = build_cycle_data(_synthetic_runs(), iv_loader=_fake_iv())
        rows = run_campaign(cd, seed=42, n_perm=200)
        for r in rows:
            assert r['clean_survivor'] == bool(r['by_survivor'] and r['sign_ok'])
            assert r['fdr_q'] == pytest.approx(0.10)
            assert r['search_tickers'] == ['MSFT', 'SPY']

    def test_campaign_is_deterministic(self) -> None:
        cd = build_cycle_data(_synthetic_runs(), iv_loader=_fake_iv())
        a = run_campaign(cd, seed=42, n_perm=200)
        b = run_campaign(cd, seed=42, n_perm=200)
        assert [r['p_value'] for r in a] == [r['p_value'] for r in b]

    def test_seal_holds_qqq_out(self) -> None:
        """The vault is enforced in code: the sealed set never overlaps the
        search set, so no candidate can train on it."""
        assert set(SEARCH_TICKERS).isdisjoint(SEALED_TICKERS)
        assert 'QQQ' in SEALED_TICKERS and 'QQQ' not in SEARCH_TICKERS


# ---- always-run: the ticker batch is a parameter (Campaign), not a constant --

class TestCampaignConfig:
    """The search/sealed ticker sets are a Campaign config, not hardcoded
    constants — so the same templates sweep the next batch of tickers, with the
    seal enforced in config and by omission at load time."""

    def test_default_campaign_matches_module_constants(self) -> None:
        """Backward-compat: the default batch is exactly the published one, so
        the pinned campaign is unchanged."""
        assert DEFAULT_CAMPAIGN.search == SEARCH_TICKERS
        assert DEFAULT_CAMPAIGN.sealed == SEALED_TICKERS
        assert set(DEFAULT_CAMPAIGN.search).isdisjoint(DEFAULT_CAMPAIGN.sealed)

    def test_campaign_enforces_disjoint_seal(self) -> None:
        """A ticker can't be both searched and sealed — the seal is a config
        invariant. Lists are coerced to tuples so the frozen config is hashable."""
        c = Campaign(search=['AAA', 'BBB'], sealed=['CCC'])
        assert c.search == ('AAA', 'BBB') and c.sealed == ('CCC',)
        with pytest.raises(ValueError):
            Campaign(search=('AAA', 'BBB'), sealed=('BBB',))

    def test_load_search_runs_is_parameterized(self, monkeypatch) -> None:
        """load_search_runs loads exactly the tickers it is given (default =
        SEARCH_TICKERS), and never a sealed name."""
        seen: list[str] = []
        monkeypatch.setattr('edge_search.load_naked_run',
                            lambda t: (seen.append(t), {'ticker': t})[1])
        runs = load_search_runs(['AAA', 'BBB'])
        assert seen == ['AAA', 'BBB']
        assert [r['ticker'] for r in runs] == ['AAA', 'BBB']
        seen.clear()
        assert load_search_runs([]) == [] and seen == []   # empty → loads nothing
        seen.clear()
        load_search_runs()                                  # default
        assert seen == list(SEARCH_TICKERS) and 'QQQ' not in seen

    def test_run_batch_loads_only_search_never_sealed(self, monkeypatch) -> None:
        """run_batch sweeps a Campaign end-to-end: it requests exactly the
        search tickers, never the sealed ones, and the rows carry that search
        set — so the templates run on whatever batch the Campaign names."""
        seen: list[str] = []
        synth = {r['ticker']: r for r in _synthetic_runs()}
        monkeypatch.setattr('edge_search.load_naked_run',
                            lambda t: (seen.append(t), synth[t])[1])
        camp = Campaign(search=('MSFT', 'SPY'), sealed=('QQQ',))
        rows = run_batch(camp, n_perm=100, iv_loader=_fake_iv())
        assert seen == ['MSFT', 'SPY'] and 'QQQ' not in seen
        assert rows and all(r['search_tickers'] == ['MSFT', 'SPY'] for r in rows)


# ---- dataset-gated: the pinned real-chain campaign ----

@pytest.fixture(scope='module')
def campaign():
    if not _HAVE_SEARCH:
        pytest.skip('needs MSFT + SPY option dailies (or their committed .gz twins)')
    cd = build_cycle_data(load_search_runs())
    return cd, run_campaign(cd)


@pytest.mark.skipif(
    not _HAVE_SEARCH,
    reason='needs MSFT + SPY option dailies (or their committed .gz twins)',
)
class TestEdgeSearchCampaign:
    """Pin the MVP campaign on the real MSFT + SPY chains, QQQ sealed
    (docs/edge_search.md). EXPLORATORY, not a registered verdict — pinned so
    the swept cheap entry-conditioning class is not re-derived. Deterministic:
    naked runs on the clean canonical chains (CHAIN_CLEAN_START applied),
    campaign seed 20260613, same-count permutation null, BY at q=0.10.
    """

    def test_batch_size(self, campaign) -> None:
        _, rows = campaign
        assert len(rows) == len(COOLDOWN_NS) + len(TREND_WINDOWS) + 1  # 9

    def test_no_clean_survivor(self, campaign) -> None:
        """The decisive output: NO candidate in the cheap entry-conditioning
        class survives campaign-wide BY at q=0.10."""
        _, rows = campaign
        assert sum(r['clean_survivor'] for r in rows) == 0
        assert sum(r['by_survivor'] for r in rows) == 0

    @staticmethod
    def _by_key(rows) -> dict:
        out = {}
        for r in rows:
            p = r['params']
            out[(r['template'], p.get('N') or p.get('window'))] = r
        return out

    def test_cooldown_wrong_signed_every_horizon(self, campaign) -> None:
        """The cooldown template predicts post-rip cycles do WORSE (D_A<0). On
        MSFT+SPY it is wrong-signed at every horizon (D_A>0 — post-rip cycles
        lose less), the same sign the pooled three-ticker scout found. Its null
        is the structure-preserving trigger-placement permutation (not the
        uniform shuffle), so the wrong-signed arrangement sits deep in the HIGH
        tail with p high across N — the pattern cooldown_scout reports."""
        _, rows = campaign
        cool = [r for r in rows if r['template'] == 'cooldown']
        assert all(r['D_A'] > 0 for r in cool)
        assert all(r['sign_ok'] is False for r in cool)
        k = self._by_key(rows)
        # D_A is the observed split — unchanged by the choice of null
        assert k[('cooldown', 30)]['D_A'] == pytest.approx(581.06, abs=1.0)
        assert k[('cooldown', 90)]['D_A'] == pytest.approx(1217.04, abs=1.0)
        # p-values under the trigger-placement null: all deep in the high tail
        assert k[('cooldown', 30)]['p_value'] == pytest.approx(0.908, abs=0.01)
        assert k[('cooldown', 60)]['p_value'] == pytest.approx(0.895, abs=0.01)
        assert k[('cooldown', 90)]['p_value'] == pytest.approx(0.969, abs=0.01)
        assert all(r['p_value'] > 0.5 for r in cool)   # all in the high tail

    def test_up_trend_mostly_wrong_signed_and_insignificant(self, campaign) -> None:
        """The up-move template predicts entries after a trailing gain do WORSE
        (D_A<0). At most one window is even sign-correct, and that one is
        statistical noise (|D_A| ~ 0, p well above 0.10) — the recurring lesson
        that conditioning entry on recent up-moves has the sign backwards."""
        _, rows = campaign
        up = [r for r in rows if r['template'] == 'up_trend']
        sign_ok = [r for r in up if r['sign_ok']]
        assert len(sign_ok) <= 1
        for r in sign_ok:
            assert r['p_value'] > 0.10
            assert abs(r['D_A']) < 100

    def test_iv_rich_suggestive_but_confounded_and_not_fdr_significant(self, campaign) -> None:
        """The one sign-correct, individually-suggestive candidate (the VRP
        gate, D_A>0 at p~0.09) is exactly the documented trap: it is the
        low-vol confound (rich-IV entries cluster in calm markets, so
        vol_confound<0), and it does NOT survive campaign-wide BY."""
        _, rows = campaign
        iv = next(r for r in rows if r['template'] == 'iv_rich')
        assert iv['sign_ok'] is True
        assert iv['D_A'] == pytest.approx(733.15, abs=1.0)
        assert iv['p_value'] == pytest.approx(0.095, abs=0.01)
        assert iv['by_survivor'] is False
        assert iv['vol_confound'] < 0

    def test_smallest_p_misses_the_by_threshold(self, campaign) -> None:
        """Even the best candidate's p clears the BY rank-1 bar by a wide
        margin — the multiple-testing math, not a single p-value, is what
        empties the class."""
        _, rows = campaign
        ps = [r['p_value'] for r in rows if r['p_value'] is not None]
        n = len(rows)
        c = float(sum(1.0 / i for i in range(1, n + 1)))
        assert min(ps) > (1 / (n * c)) * 0.10


# --------------------------------------------------------------------------- #
# Engine-re-run (structure) phase
# --------------------------------------------------------------------------- #
class TestClosedGrammar:
    """Interlock #1: the closed grammar (ALLOWED_GRID + _validate_grammar). The same
    check runs on StructureTemplate (authoring) AND StructureCandidate (the object
    that enters the BY pool), so the hypothesis universe is finite and countable —
    the precondition the FDR accounting rests on."""

    def test_grid_universe_size_pinned(self) -> None:
        # short_vol 3x3=9, straddle 3, iron_condor 3x3x2=18, strangle 3x3=9, risk_reversal 3x3=9,
        # credit_spread 3x3x2=18, calendar 2x2=4 -> 70. Bump the grid, bump this pin — the universe
        # size is on the record by design.
        assert grid_universe_size() == 70
        by_hand = sum(
            len(grid['target_delta']) * len(grid['dte']) if ov == 'short_vol'
            else len(grid['dte']) if ov == 'straddle'
            else len(grid['dte']) * len(grid['short_delta']) if ov in ('strangle', 'risk_reversal')
            else len(grid['near_dte']) * len(grid['far_dte']) if ov == 'calendar'
            else len(grid['dte']) * len(grid['short_delta']) * len(grid['wing_delta'])
            for ov, grid in ALLOWED_GRID.items())
        assert by_hand == grid_universe_size()

    def test_committed_templates_are_on_menu(self) -> None:
        # Importing edge_search already constructed STRUCTURE_TEMPLATES; if any cell
        # were off-menu, __post_init__ would have raised at import. Re-affirm here:
        # every committed template's params match its overlay grid exactly, value-wise.
        assert len(STRUCTURE_TEMPLATES) == 8   # + strangle (1), risk-reversal (2), credit-spread (3), calendar (4)
        for t in STRUCTURE_TEMPLATES:
            grid = ALLOWED_GRID[t.overlay]
            params = dict(t.params)
            assert set(params) == set(grid)
            for k, v in params.items():
                assert v in grid[k]
            assert t.predicted_sign in (-1, +1)

    def test_offmenu_value_raises(self) -> None:
        # the continuous-knob fish: 0.241 is not on the delta menu
        with pytest.raises(ValueError, match='off-menu'):
            StructureTemplate('x', 'short_vol', (('target_delta', 0.241), ('dte', 30)), +1)

    def test_unknown_overlay_raises(self) -> None:
        with pytest.raises(ValueError, match='overlay'):
            StructureTemplate('x', 'butterfly', (('dte', 30),), +1)

    def test_param_keys_must_match_grid_exactly(self) -> None:
        # an extra knob the overlay doesn't define
        with pytest.raises(ValueError, match='must match'):
            StructureTemplate('x', 'straddle', (('dte', 30), ('target_delta', 0.25)), +1)
        # a missing required knob (short_vol needs both target_delta and dte)
        with pytest.raises(ValueError, match='must match'):
            StructureTemplate('x', 'short_vol', (('dte', 30),), +1)

    def test_duplicate_param_key_raises(self) -> None:
        with pytest.raises(ValueError, match='duplicate'):
            StructureTemplate('x', 'straddle', (('dte', 30), ('dte', 45)), +1)

    def test_predicted_sign_mandatory_and_validated(self) -> None:
        # no default: omitting predicted_sign is a TypeError (missing positional arg)
        with pytest.raises(TypeError):
            StructureTemplate('x', 'straddle', (('dte', 30),))  # type: ignore[call-arg]
        # only -1 / +1 are valid directions
        for bad in (0, 2, -2):
            with pytest.raises(ValueError, match='predicted_sign'):
                StructureTemplate('x', 'straddle', (('dte', 30),), bad)

    def test_candidate_is_also_grammar_validated(self) -> None:
        # The honesty-relevant object is StructureCandidate — it reaches the kill-gate
        # and the BY pool. An off-grid candidate must be a hard error too, or the
        # continuous-knob fish just swims around the template gate one layer down.
        with pytest.raises(ValueError, match='off-menu'):
            StructureCandidate('fish', 'MSFT', 'short_vol',
                               (('target_delta', 0.241), ('dte', 30)), +1)
        # the on-menu candidate the enumerator actually builds constructs fine
        StructureCandidate('short_call_25', 'MSFT', 'short_vol',
                           (('target_delta', 0.25), ('dte', 30)), +1)

    def test_committed_batch_is_the_by_denominator(self) -> None:
        # The cross-section size — the e-LOND stream length and the BY diagnostic's
        # denominator — is the run count, not the grammar universe, so pin it.
        # 8 committed templates x 7 search tickers = 56 cells (the credit-spread widening took it
        # 6->7 / 42->49, then the calendar widening 7->8 / 49->56).
        cands = enumerate_structure_candidates(STRUCTURE_CAMPAIGN)
        assert len(cands) == 56
        assert len(cands) == len(STRUCTURE_TEMPLATES) * len(STRUCTURE_SEARCH)
        # and the committed menu is a subset of the reachable universe (8 <= 70),
        # so widening either the templates or the grid is a deliberate, pinned edit.
        assert len(STRUCTURE_TEMPLATES) <= grid_universe_size()

    def test_grid_menus_have_no_duplicate_values(self) -> None:
        # a fat-fingered duplicate (e.g. dte:(21,30,30,45)) would silently inflate
        # the universe count; the menu must be a true set of options per knob.
        for overlay, grid in ALLOWED_GRID.items():
            for knob, values in grid.items():
                assert len(set(values)) == len(values), f'{overlay}.{knob} has a duplicate'

    def test_membership_is_type_strict(self) -> None:
        # bool is an int subclass (True == 1), and 30.0 == 30 — a guard whose job is
        # rejecting off-spec input must not let either through where an int is meant.
        with pytest.raises(ValueError, match='predicted_sign'):
            StructureTemplate('x', 'straddle', (('dte', 30),), True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match='off-menu'):
            StructureTemplate('x', 'straddle', (('dte', 30.0),), +1)
        with pytest.raises(ValueError, match='off-menu'):
            StructureCandidate('x', 'MSFT', 'straddle', (('dte', 30.0),), +1)

    def test_grammar_is_economically_typed(self) -> None:
        # every overlay carries a registered PremiumFamily + a complete signature. The four short-vol
        # overlays are VARIANCE; the risk reversal (widening 2) is the first SKEW; the credit spread
        # (widening 3) is the first CARRY; the calendar (widening 4) is the first TERM (two
        # expirations). net_GAMMA is intentionally NOT a signature axis (it is un-pinnable for
        # offset-leg structures — see structure_greek_signature); the three robust axes are
        # net_vega / net_delta / net_skew.
        assert {f.name for f in PremiumFamily} == {'VARIANCE', 'SKEW', 'TERM', 'CARRY'}
        assert set(STRUCTURE_GRAMMAR) == {'short_vol', 'straddle', 'iron_condor', 'strangle',
                                          'risk_reversal', 'credit_spread', 'calendar'}
        families = {name: og.family for name, og in STRUCTURE_GRAMMAR.items()}
        assert families['risk_reversal'] is PremiumFamily.SKEW
        assert families['credit_spread'] is PremiumFamily.CARRY
        assert families['calendar'] is PremiumFamily.TERM
        assert all(families[n] is PremiumFamily.VARIANCE
                   for n in ('short_vol', 'straddle', 'iron_condor', 'strangle'))
        for name, og in STRUCTURE_GRAMMAR.items():
            assert structure_family(name) is og.family
            assert {'expirations', 'legs', 'net_vega', 'net_delta', 'net_skew'} <= set(og.signature)
        # the calendar is the first TWO-expiration structure; every other overlay is single-expiration.
        assert STRUCTURE_GRAMMAR['calendar'].signature['expirations'] == 2
        assert all(og.signature['expirations'] == 1 for name, og in STRUCTURE_GRAMMAR.items()
                   if name != 'calendar')

    def test_allowed_grid_is_the_grammar_flat_view(self) -> None:
        # ALLOWED_GRID is the grammar's lattices, SAME dict objects (so grid_universe_size /
        # _validate_grammar / enumerate_grammar_templates are byte-unchanged, and the
        # lineage-no-reset-loophole test's mutate/restore still operates on the live grid).
        assert ALLOWED_GRID == {n: og.lattices for n, og in STRUCTURE_GRAMMAR.items()}
        assert ALLOWED_GRID['short_vol'] is STRUCTURE_GRAMMAR['short_vol'].lattices

    def test_well_typed_assertion_rejects_untyped_overlay(self, monkeypatch) -> None:
        # the import-time scaffold catches an overlay added without a registered family or a
        # complete signature (PRESENCE gate — it does NOT yet check the signature against the
        # engine's greeks). monkeypatch.setitem auto-restores STRUCTURE_GRAMMAR at teardown.
        monkeypatch.setitem(STRUCTURE_GRAMMAR, '_probe',
                            OverlayGrammar({'dte': (30,)}, 'not_a_family', {}))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match='not a registered PremiumFamily'):
            _assert_grammar_well_typed()
        monkeypatch.setitem(STRUCTURE_GRAMMAR, '_probe',
                            OverlayGrammar({'dte': (30,)}, PremiumFamily.VARIANCE, {'expirations': 1}))
        with pytest.raises(ValueError, match='signature missing'):
            _assert_grammar_well_typed()


class TestIdeaLedger:
    """Interlock #3a: the committed, append-only lifetime trial ledger. Deterministic
    (deduped, timestamp-free), so it is the guess-counter that never silently resets.
    Always-run — uses synthetic campaign rows + the committed data_checksums."""

    @staticmethod
    def _row(template='short_call_25', ticker='MSFT', params=None,
             p=0.5, t=1.0, surv=False, elond=False):
        # a run_structure_campaign row: carries BOTH the e-LOND control flag and the
        # retained BY diagnostic (online_fdr_survivors + benjamini_yekutieli).
        return {'phase': 'structure', 'template': template, 'ticker': ticker,
                'params': params or {'target_delta': 0.25, 'dte': 30},
                'predicted_sign': 1, 't_stat_newey_west': t, 'p_value': p,
                'elond_survivor': elond, 'by_survivor': surv, 'fdr_q': 0.10}

    def test_lineage_hash_deterministic_and_sensitive(self) -> None:
        h1 = _data_lineage_hash('MSFT', '2026-06-06')
        assert h1 == _data_lineage_hash('MSFT', '2026-06-06')   # deterministic
        assert h1 != _data_lineage_hash('MSFT', '2025-01-01')   # end date matters
        assert h1 != _data_lineage_hash('SPY', '2026-06-06')    # ticker matters
        assert h1 != _data_lineage_hash('MSFT', '2026-06-06', 200_000)  # capital matters
        assert len(h1) == 16
        # a different store checksum is a different lineage (the data changed)
        assert h1 != _data_lineage_hash('MSFT', '2026-06-06',
                                        checksums={'msft_option_dailies.csv.gz': 'deadbeef'})

    def test_lineage_ignores_grammar_no_reset_loophole(self) -> None:
        # Widening the menu must NOT change a comparison's lineage. The engine result
        # for a fixed (template, params, ticker, data) is invariant to what else the
        # grid can express, so folding ALLOWED_GRID into the lineage would re-record
        # every prior look as "new" on a grid edit and hand #3b a fresh false-discovery
        # budget — the exact reset the lifetime counter exists to prevent. The menu's
        # countability lives in grid_universe_size + the pinned 49-cell batch, not here.
        before = _data_lineage_hash('MSFT', '2026-06-06')
        orig = ALLOWED_GRID['short_vol']['target_delta']
        ALLOWED_GRID['short_vol']['target_delta'] = orig + (0.70,)   # widen the menu
        try:
            after = _data_lineage_hash('MSFT', '2026-06-06')
        finally:
            ALLOWED_GRID['short_vol']['target_delta'] = orig          # restore
        assert before == after

    def test_structure_ledger_rows_schema(self) -> None:
        rows = structure_ledger_rows([self._row(p=0.83, t=-0.96)])
        r = rows[0]
        assert r['phase'] == 'structure' and r['template'] == 'short_call_25'
        assert r['statistic_kind'] == 't_nw' and r['statistic'] == -0.96
        assert r['p_value'] == 0.83
        # the answer key carries the verdict of record (elond_survivor, #3b) AND the
        # retained BY diagnostic — both projected from the campaign row.
        assert r['elond_survivor'] is False and r['by_survivor'] is False
        assert len(r['data_lineage_hash']) == 16

    def test_ledger_records_elond_verdict_of_record(self) -> None:
        """The committed answer key records elond_survivor — the FDR control of record
        (#3b) — not just the BY diagnostic. A campaign cell e-LOND flagged round-trips
        into the ledger as elond_survivor=True, independent of its by_survivor bit."""
        flagged = structure_ledger_rows([self._row(elond=True, surv=False)])[0]
        assert flagged['elond_survivor'] is True and flagged['by_survivor'] is False

    def test_record_dedupes_reruns(self, tmp_path) -> None:
        p = str(tmp_path / 'idea_ledger.jsonl')
        rows = structure_ledger_rows([self._row(), self._row(ticker='SPY')])
        assert record_trials(rows, p) == 2
        # re-recording THE SAME comparisons adds nothing — same lineage + candidate
        assert record_trials(rows, p) == 0
        assert len(load_idea_ledger(p)) == 2

    def test_record_appends_new_comparison(self, tmp_path) -> None:
        p = str(tmp_path / 'idea_ledger.jsonl')
        record_trials(structure_ledger_rows([self._row()]), p)
        added = record_trials(structure_ledger_rows(
            [self._row(template='straddle', params={'dte': 30})]), p)
        assert added == 1                      # a different template is a new comparison
        assert len(load_idea_ledger(p)) == 2

    def test_load_missing_ledger_is_empty(self, tmp_path) -> None:
        assert load_idea_ledger(str(tmp_path / 'nope.jsonl')) == []


class TestLifetimeStreamJudge:
    """Interlock #3b correctness: judge_against_lifetime_stream judges a new batch as the TAIL of
    the committed lifetime e-LOND stream, not in isolation. run_structure_campaign runs e-LOND
    per-batch (correct only for the published head-of-stream one-shot); judging each APPENDED batch
    alone restarts the discount sequence at t=1 — a silent per-session budget reset, the
    multiple-looks leak docs/prereg_fdr_budget.md exists to prevent. Always-run, synthetic (no
    engine, no datasets)."""

    @staticmethod
    def _lrow(template='t', ticker='AAA', p=0.5, params=None, lineage='lin0'):
        # a ledger-format row (post structure_ledger_rows): carries p_value + a distinct _ledger_key.
        return {'phase': 'structure', 'template': template, 'ticker': ticker,
                'params': params or {'dte': 30}, 'predicted_sign': 1,
                'statistic_kind': 't_nw', 'statistic': 0.0, 'p_value': p,
                'elond_survivor': False, 'by_survivor': False, 'measurement_invalid': False,
                'fdr_q': 0.10, 'end': '2026-06-06', 'data_lineage_hash': lineage}

    def test_empty_prior_equals_per_batch(self) -> None:
        """With no prior, the lifetime judge IS the per-batch judge — so the published first batch
        (empty ledger before it) is unaffected and TestStructureCampaign's 0/35 head-of-stream pin
        stands."""
        rows = [self._lrow(template=f't{i}', p=p) for i, p in enumerate([0.0001, 0.5, 0.5])]
        per_batch = [r['elond_survivor'] for r in online_fdr_survivors(rows)]
        lifetime = [r['elond_survivor'] for r in judge_against_lifetime_stream(rows, prior_rows=[])]
        assert per_batch == lifetime

    def test_closes_the_per_session_reset(self) -> None:
        """The decisive pin: a strong cell (tiny p -> big e) IS flagged as the head of the stream,
        but the SAME cell placed deep in the lifetime stream faces a far tighter gamma_t bar and is
        NOT flagged. Judging each appended batch in isolation (the bug) would flag it every time."""
        strong = self._lrow(template='strong', p=1e-6)
        head = online_fdr_survivors([strong])[0]
        assert head['elond_survivor'] is True            # head of stream: loosest 1/(alpha*gamma_1) bar
        prior = [self._lrow(template=f'dud{i}', p=0.9) for i in range(12)]   # 12 non-survivors ahead
        tail = judge_against_lifetime_stream([strong], prior_rows=prior)[0]
        assert tail['elond_survivor'] is False           # deep tail: gamma_t shrank the bar past its e

    def test_already_recorded_rows_not_double_counted(self) -> None:
        """A row already in the prior ledger is not a fresh look: it is not re-appended to the
        stream, and it returns its existing-position verdict (record_trials would dedup it anyway)."""
        a, b = self._lrow(template='a', p=0.5), self._lrow(template='b', p=0.5)
        prior = [a, b]
        out = judge_against_lifetime_stream([a, b], prior_rows=prior)
        base = online_fdr_survivors(prior)
        assert [r['elond_survivor'] for r in out] == [r['elond_survivor'] for r in base]

    def test_online_verdict_stable_under_later_appends(self) -> None:
        """e-LOND is online: a row's decision depends only on rows BEFORE it, so a verdict is fixed
        on arrival and never moves when more cells are appended later — which is why recording it
        here is permanent and correct."""
        early = self._lrow(template='early', p=1e-6)
        first = judge_against_lifetime_stream([early], prior_rows=[])[0]['elond_survivor']
        later = [self._lrow(template=f'after{i}', p=0.5) for i in range(5)]
        again = online_fdr_survivors([early] + later)[0]['elond_survivor']
        assert first is True and again == first          # appends after `early` don't move its verdict

    def test_judged_rows_keep_ledger_schema(self) -> None:
        """The corrected rows keep the ledger schema exactly (only elond_survivor may change) — no
        e_value / elond_level leaks into the committed answer key."""
        r = self._lrow(template='x', p=0.2)
        out = judge_against_lifetime_stream([r], prior_rows=[])[0]
        assert set(out) == set(r)
        assert 'e_value' not in out and 'elond_level' not in out

    def test_within_batch_duplicate_matches_record_trials_stream(self) -> None:
        """The judge must dedup WITHIN the batch exactly as record_trials does, so the stream it
        scores equals the deduped sequence record_trials commits — otherwise an intra-batch
        duplicate would inflate the stream length and shift downstream cells' gamma_t, and the
        recorded verdict would not reproduce on a re-judge of the file. (Latent: the real --record
        path enumerates unique cells; pinned so a future proposer batch can't silently diverge.)"""
        dup, x = self._lrow(template='dup', p=0.5), self._lrow(template='x', p=0.5)
        out = judge_against_lifetime_stream([dup, x, dup], prior_rows=[])   # internal duplicate
        # record_trials commits the deduped+sorted batch; judging THAT must reproduce the verdicts
        committed = sorted({_ledger_key(r): r for r in [dup, x, dup]}.values(), key=_ledger_key)
        ref = {_ledger_key(r): r['elond_survivor'] for r in online_fdr_survivors(committed)}
        assert all(o['elond_survivor'] == ref[_ledger_key(o)] for o in out)
        assert out[0]['elond_survivor'] == out[2]['elond_survivor']   # both dups, same verdict

    def test_measurement_invalid_occupies_a_position_and_never_flags(self) -> None:
        """A measurement_invalid row (p_value=None -> e=0) can never flag, yet it OCCUPIES a stream
        position — so a borderline cell after it faces the tighter t=2 bar, not the head bar."""
        invalid = {**self._lrow(template='broken'), 'p_value': None, 'measurement_invalid': True}
        assert judge_against_lifetime_stream([invalid], prior_rows=[])[0]['elond_survivor'] is False
        borderline = self._lrow(template='borderline', p=2.78e-4)   # flags at the head, not at t=2
        at_head = judge_against_lifetime_stream([borderline], prior_rows=[])[0]
        after_invalid = judge_against_lifetime_stream([borderline], prior_rows=[invalid])[0]
        assert at_head['elond_survivor'] is True and after_invalid['elond_survivor'] is False

    def test_prior_survivor_raises_R_for_a_later_cell(self) -> None:
        """e-LOND's (R+1) reward flows ACROSS the prior/new boundary: a survivor already in the
        lifetime stream raises R for an appended cell, loosening its bar. The same new cell flags
        when a prior survivor precedes it but not when the prior is a non-survivor at that position."""
        probe = self._lrow(template='probe', p=1e-4)
        with_surv = judge_against_lifetime_stream(
            [probe], prior_rows=[self._lrow(template='winner', p=1e-9)])[0]
        without = judge_against_lifetime_stream(
            [probe], prior_rows=[self._lrow(template='dud', p=0.9)])[0]
        assert with_surv['elond_survivor'] is True and without['elond_survivor'] is False


class TestProposerCorpus:
    """Interlock #2: the number-free scoreboard — an allow-list projection of the
    lifetime ledger. A proposer may read this; it must never read the ledger itself.
    The airtight guarantee is structural (scrub copies only SAFE_FIELDS), not a
    regex over the rendered text — template names carry digits and grid values
    collide with result values, so only field-selection is leak-proof."""

    @staticmethod
    def _ledger_row(template='short_call_25', ticker='MSFT',
                    elond=False, by=False, invalid=False):
        # a row as it lives in idea_ledger.jsonl — carries the answer key (both the
        # e-LOND control flag and the retained BY diagnostic).
        return {'phase': 'structure', 'template': template, 'ticker': ticker,
                'params': {'target_delta': 0.25, 'dte': 30}, 'predicted_sign': 1,
                'statistic_kind': 't_nw', 'statistic': 7.654321, 'p_value': 0.0123456,
                'elond_survivor': elond, 'by_survivor': by,
                'measurement_invalid': invalid, 'fdr_q': 0.10,
                'end': '2026-06-06', 'data_lineage_hash': 'abcd1234'}

    def test_scrub_is_allow_list_only_safe_keys_survive(self) -> None:
        s = scrub_ledger_row(self._ledger_row())
        assert set(s) == {'phase', 'template', 'ticker', 'params', 'predicted_sign', 'verdict'}
        # every result-bearing field is dropped by construction, not redaction —
        # including the raw FDR flags (only the one-bit verdict survives, never the
        # control/diagnostic bits that would tell a proposer which gate fired).
        for forbidden in ('statistic', 'statistic_kind', 'p_value', 'fdr_q',
                          'data_lineage_hash', 'elond_survivor', 'by_survivor'):
            assert forbidden not in s

    def test_verdict_is_one_bit(self) -> None:
        assert scrub_ledger_row(self._ledger_row())['verdict'] == 'KILLED'
        assert scrub_ledger_row(self._ledger_row(elond=True))['verdict'] == 'SURVIVED'
        assert scrub_ledger_row(self._ledger_row(invalid=True))['verdict'] == 'INVALID'

    def test_verdict_keys_off_elond_control_not_by_diagnostic(self) -> None:
        """The SURVIVED verdict (hence the corpus exclusion) tracks elond_survivor —
        the FDR control of record (#3b) — NOT by_survivor, the retained diagnostic.
        e-LOND and BY are not guaranteed to coincide (e-LOND's (R+1) reward can flag a
        cell BY does not), so the two cross cases are the decisive regression:"""
        # control flags, diagnostic does not -> SURVIVED (the leak the fix prevents:
        # an e-LOND survivor must never be mislabeled KILLED and re-proposed).
        assert scrub_ledger_row(self._ledger_row(elond=True, by=False))['verdict'] == 'SURVIVED'
        # diagnostic flags, control does not -> KILLED (BY does not promote anything;
        # the prereg is explicit that "only e-LOND flags").
        assert scrub_ledger_row(self._ledger_row(elond=False, by=True))['verdict'] == 'KILLED'

    def test_render_omits_the_answer_key(self) -> None:
        corpus = render_proposer_corpus(build_proposer_corpus([self._ledger_row()]))
        # the distinctive result magnitudes must NOT appear in what the proposer reads
        assert '7.654321' not in corpus and '0.0123456' not in corpus
        # but the hypothesis coordinates + the one-bit verdict do
        assert 'short_call_25' in corpus and 'target_delta=0.25' in corpus and 'KILLED' in corpus

    def test_empty_corpus_renders_safely(self) -> None:
        assert render_proposer_corpus([]) == '(no comparisons recorded yet)'

    def test_survivor_is_excluded_from_corpus(self) -> None:
        # SURVIVED — an e-LOND-flagged cell (the control of record) — is the one "fish
        # here" coordinate; it must not feed the automated proposer (it escalates to
        # manual pre-registration out-of-band). KILLED and INVALID stay (duds to avoid
        # / broken tickers); only the e-LOND winner is dropped. A BY-only flag is a
        # diagnostic, not the control, so that cell is KILLED and STAYS in the corpus.
        rows = [self._ledger_row(template='killed_one'),
                self._ledger_row(template='winner', elond=True),
                self._ledger_row(template='by_only', by=True),
                self._ledger_row(template='broken', invalid=True)]
        corpus = build_proposer_corpus(rows)
        verdicts = {r['template']: r['verdict'] for r in corpus}
        # no 'winner'; 'by_only' is retained as KILLED (the diagnostic does not exclude)
        assert verdicts == {'killed_one': 'KILLED', 'by_only': 'KILLED', 'broken': 'INVALID'}

    def test_params_defensively_copied(self) -> None:
        # the corpus is a safe boundary — mutating it must not reach the source row
        row = self._ledger_row()
        s = scrub_ledger_row(row)
        s['params']['target_delta'] = 0.99
        assert row['params']['target_delta'] == 0.25   # source untouched

    def test_load_proposer_corpus_is_scrubbed(self, tmp_path) -> None:
        # full path: a real KILLED ledger row (with p_value/statistic) -> record -> view
        p = str(tmp_path / 'idea_ledger.jsonl')
        row = {'phase': 'structure', 'template': 'straddle', 'ticker': 'SPY',
               'params': {'dte': 30}, 'predicted_sign': 1, 't_stat_newey_west': 0.4,
               'p_value': 0.34, 'elond_survivor': False, 'by_survivor': False,
               'fdr_q': 0.10}
        record_trials(structure_ledger_rows([row]), p)
        corpus = load_proposer_corpus(p)
        assert len(corpus) == 1
        assert 'p_value' not in corpus[0] and 'statistic' not in corpus[0]
        assert corpus[0]['verdict'] == 'KILLED' and corpus[0]['template'] == 'straddle'


class TestMenuWalkerProposer:
    """Phase 1: the deterministic menu-walker proposer (no LLM) — the loop the LLM later
    plugs into. read scrubbed corpus -> propose grammar-valid untried cells -> grammar-gate
    -> run -> lifetime-judge (#3b) -> record -> re-read. Always-run, synthetic (injected
    scorer + monkeypatched onboarding + temp ledger; no engine, no datasets)."""

    @staticmethod
    def _scorer(cand):
        # a run_structure_campaign-shaped row from an injected scorer (no engine)
        return {'phase': 'structure', 'template': cand.template, 'ticker': cand.ticker,
                'params': cand.params_dict(), 'predicted_sign': cand.predicted_sign,
                't_stat_newey_west': 0.5, 'sign_ok': True, 'p_value': 0.3}

    def test_grammar_menu_is_the_full_universe(self) -> None:
        tmpls = enumerate_grammar_templates()
        assert len(tmpls) == grid_universe_size()                 # the whole reachable menu (66)
        assert len({t.name for t in tmpls}) == len(tmpls)         # names unique
        assert all(t.predicted_sign == +1 for t in tmpls)         # committed sign convention
        # the committed cells keep their hand-chosen names so a menu-walked cell that coincides
        # with one dedups against the published ledger instead of re-counting it under a new name
        assert {t.name for t in STRUCTURE_TEMPLATES} <= {t.name for t in tmpls}

    def test_menu_walker_dedups_and_routes_unonboarded(self, monkeypatch) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: tk != 'NEW')
        camp = Campaign(search=('AAA', 'NEW'))
        cands, need = propose_structure_candidates(camp, set())
        assert need == ['NEW']                                    # un-onboarded -> flagged, not run
        assert len(cands) == grid_universe_size()                 # only AAA's full menu runs
        assert all(c.ticker == 'AAA' for c in cands)
        # dedup: a tried cell is dropped (keyed on the corpus coordinates)
        tried = {_cand_key(cands[0])}
        cands2, _ = propose_structure_candidates(camp, tried)
        assert len(cands2) == len(cands) - 1

    def test_round_records_via_lifetime_judge_and_re_reads(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        p = str(tmp_path / 'idea_ledger.jsonl')
        camp = Campaign(search=('AAA',))
        res = run_proposer_round(camp, path=p, scorer=self._scorer, run=True, record=True)
        assert res['proposed'] == grid_universe_size()            # the full untried menu
        assert res['recorded'] == grid_universe_size()            # all recorded (fresh ledger)
        assert len(load_idea_ledger(p)) == grid_universe_size()
        # re-read: the corpus now carries every cell, so a second round proposes/records nothing
        res2 = run_proposer_round(camp, path=p, scorer=self._scorer, run=True, record=True)
        assert res2['proposed'] == 0 and res2['recorded'] == 0

    def test_preview_runs_no_engine_and_writes_nothing(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        p = str(tmp_path / 'idea_ledger.jsonl')
        called = []
        res = run_proposer_round(Campaign(search=('AAA',)), path=p,
                                 scorer=lambda c: called.append(c) or {}, run=False)
        assert res['proposed'] == grid_universe_size() and res['recorded'] == 0
        assert called == []                                       # run=False: no engine/scorer call
        assert load_idea_ledger(p) == []                          # and nothing written

    def test_dry_run_judges_without_recording(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        p = str(tmp_path / 'idea_ledger.jsonl')
        res = run_proposer_round(Campaign(search=('AAA',)), path=p, scorer=self._scorer,
                                 run=True, record=False)
        assert res['proposed'] == grid_universe_size()            # ran + judged ...
        assert len(res['ledger_rows']) == grid_universe_size()
        assert res['recorded'] == 0 and load_idea_ledger(p) == []  # ... but wrote nothing


class TestLLMProposer:
    """Phase 2: the LLM AUTHOR contract — the drop-in for the menu-walker. Always-run,
    synthetic (a stub author, injected scorer, monkeypatched onboarding, temp ledger; no
    model, no engine). Pins the gate/dedup/seal/cap AND the provenance invariant: the
    exact model id is recorded to a SEPARATE audit log and never re-keys or re-spends the
    model-blind comparison ledger. The sandbox/oracle process boundary (docs/read_gate.md)
    is what makes a REAL author safe to activate; this is the contract it plugs into."""

    @staticmethod
    def _scorer(cand):
        return {'phase': 'structure', 'template': cand.template, 'ticker': cand.ticker,
                'params': cand.params_dict(), 'predicted_sign': cand.predicted_sign,
                't_stat_newey_west': 0.5, 'sign_ok': True, 'p_value': 0.3}

    @staticmethod
    def _author(proposals, model='claude-test'):
        def author(menu, corpus, onboarded):
            return ProposalBatch(tuple(proposals), model_requested=model,
                                 model_served=f'{model}-snap', temperature=0.0,
                                 prompt_sha=f'sha-{model}')
        return author

    # a valid committed grid point (short_call_25); reused across tests
    _CELL = {'overlay': 'short_vol', 'ticker': 'AAA',
             'params': {'target_delta': 0.25, 'dte': 30}, 'predicted_sign': 1}

    def test_gate_rejects_off_menu_sealed_offcampaign_and_sign(self, monkeypatch) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: tk == 'AAA')
        camp = Campaign(search=('AAA', 'BBB'), sealed=('TLT',))
        proposals = [
            self._CELL,                                                                  # valid
            {**self._CELL, 'params': {'target_delta': 0.241, 'dte': 30}},                # off-grammar
            {**self._CELL, 'ticker': 'TLT'},                                             # sealed
            {**self._CELL, 'ticker': 'ZZZ'},                                             # off-campaign
            {'overlay': 'straddle', 'ticker': 'AAA', 'params': {'dte': 30},
             'predicted_sign': -1},                                                      # sign mismatch
            {**self._CELL, 'params': 'target_delta=0.25,dte=30'},                        # malformed (stringified -> dict() ValueError)
            {**self._CELL, 'predicted_sign': True},                                      # bool sign (not int) -> rejected
            self._CELL,                                                                  # dup of #1
        ]
        cands, need, rejected, batch = llm_propose_candidates(
            self._author(proposals), camp, corpus=[], tried_keys=set())
        assert [(c.template, c.ticker) for c in cands] == [('short_call_25', 'AAA')]     # only the valid one
        assert need == []
        reasons = ' '.join(r['reason'] for r in rejected)
        assert 'off-grammar' in reasons and 'sealed' in reasons
        assert 'off-campaign' in reasons and 'predicted_sign' in reasons
        assert 'malformed' in reasons                                                    # stringified params didn't crash the round
        assert batch.model_served == 'claude-test-snap'                                  # carried, not gated

    def test_unonboarded_search_ticker_routes_to_needs_onboard(self, monkeypatch) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: tk == 'AAA')
        camp = Campaign(search=('AAA', 'BBB'), sealed=('TLT',))
        cands, need, rejected, _ = llm_propose_candidates(
            self._author([{**self._CELL, 'ticker': 'BBB'}]), camp, corpus=[], tried_keys=set())
        assert need == ['BBB'] and cands == []                                           # flagged, not run

    def test_dedup_against_corpus_coordinates(self, monkeypatch) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        camp = Campaign(search=('AAA',))
        tried = {('short_call_25', 'AAA', json.dumps({'target_delta': 0.25, 'dte': 30}, sort_keys=True))}
        cands, _, _, _ = llm_propose_candidates(
            self._author([self._CELL]), camp, corpus=[], tried_keys=tried)
        assert cands == []                                                               # already tried -> dropped

    def test_max_batch_caps_accepted(self, monkeypatch) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        camp = Campaign(search=('AAA',))
        menu = enumerate_grammar_templates()
        many = [{'overlay': t.overlay, 'ticker': 'AAA', 'params': dict(t.params),
                 'predicted_sign': t.predicted_sign} for t in menu]                      # the whole menu
        cands, _, _, _ = llm_propose_candidates(
            self._author(many), camp, corpus=[], tried_keys=set(), max_batch=3)
        assert len(cands) == 3                                                           # capped

    def test_round_records_comparison_and_provenance(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        led = str(tmp_path / 'idea_ledger.jsonl')
        prov = str(tmp_path / 'proposal_provenance.jsonl')
        res = run_proposer_round(Campaign(search=('AAA',)), path=led, scorer=self._scorer,
                                 run=True, record=True, author=self._author([self._CELL], 'A'),
                                 round_id='r1', provenance_path=prov)
        assert res['proposed'] == 1 and res['recorded'] == 1
        ledger = load_idea_ledger(led)
        assert len(ledger) == 1
        # the comparison row is MODEL-AGNOSTIC: no model/prompt field leaked in
        assert not ({'model_served', 'model_requested', 'prompt_sha'} & set(ledger[0]))
        prov_rows = [json.loads(ln) for ln in open(prov)]
        assert len(prov_rows) == 1 and prov_rows[0]['model_served'] == 'A-snap'
        assert prov_rows[0]['round_id'] == 'r1'

    def test_model_id_does_not_rekey_or_respend(self, monkeypatch, tmp_path) -> None:
        """The invariant: the model is provenance, not lineage. The same cell scored under
        two different models yields a BYTE-IDENTICAL comparison row, and a second model
        re-proposing a tried cell re-spends nothing."""
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        camp = Campaign(search=('AAA',))
        # same cell, two different models, two fresh ledgers
        a, b = (str(tmp_path / 'a.jsonl'), str(tmp_path / 'b.jsonl'))
        pa, pb = (str(tmp_path / 'pa.jsonl'), str(tmp_path / 'pb.jsonl'))
        run_proposer_round(camp, path=a, scorer=self._scorer, run=True, record=True,
                           author=self._author([self._CELL], 'A'), round_id='r', provenance_path=pa)
        run_proposer_round(camp, path=b, scorer=self._scorer, run=True, record=True,
                           author=self._author([self._CELL], 'B'), round_id='r', provenance_path=pb)
        # comparison rows identical regardless of model (model never touches _ledger_key /
        # _data_lineage_hash); provenance differs
        assert load_idea_ledger(a) == load_idea_ledger(b)
        assert json.loads(open(pa).read())['model_served'] == 'A-snap'
        assert json.loads(open(pb).read())['model_served'] == 'B-snap'
        # re-spend guard: model B proposing the SAME cell into A's ledger adds nothing
        res = run_proposer_round(camp, path=a, scorer=self._scorer, run=True, record=True,
                                 author=self._author([self._CELL], 'B'), round_id='r2', provenance_path=pa)
        assert res['proposed'] == 0 and res['recorded'] == 0                             # tried -> no new comparison
        assert len(load_idea_ledger(a)) == 1                                             # COMPARISON ledger unchanged (no re-spend)
        prov_after = [json.loads(ln) for ln in open(pa)]
        assert len(prov_after) == 2                                                      # the re-proposal IS audited ...
        assert prov_after[-1]['model_served'] == 'B-snap' and prov_after[-1]['accepted'] == []  # ... but accepted nothing


class TestReadGateOracleSeam:
    """PR 1 of the read-gate: `score_and_record` — the oracle's one-bit entry point, the
    ONLY way across the boundary to the engine. It records BEFORE replying and hands back
    only the scrubbed scoreboard (no t-stats). Always-run, synthetic (injected scorer; no
    engine/datasets). The wall (separate processes) is a later PR; this pins the contract."""

    @staticmethod
    def _scorer(cand):
        return {'phase': 'structure', 'template': cand.template, 'ticker': cand.ticker,
                'params': cand.params_dict(), 'predicted_sign': cand.predicted_sign,
                't_stat_newey_west': 0.5, 'sign_ok': True, 'p_value': 0.3}

    _MODEL = {'model_requested': 'claude-x', 'model_served': 'claude-x-snap',
              'temperature': 0.0, 'prompt_sha': 'abc'}
    _CELL = {'overlay': 'short_vol', 'ticker': 'AAA',
             'params': {'target_delta': 0.25, 'dte': 30}, 'predicted_sign': 1}

    def test_records_and_returns_one_bit_view(self, monkeypatch, tmp_path) -> None:
        # NOTE: record-before-reply is STRUCTURAL (run_proposer_round records, then
        # score_and_record composes the reply from its return) — this asserts the
        # recording is present by return time + the reply is the scrubbed one-bit view.
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        led = str(tmp_path / 'idea_ledger.jsonl')
        reply = score_and_record([self._CELL], round_id='r1', model=self._MODEL,
                                 campaign=Campaign(search=('AAA',)), path=led,
                                 provenance_path=str(tmp_path / 'prov.jsonl'), scorer=self._scorer)
        assert len(load_idea_ledger(led)) == 1                                    # the comparison row is on disk
        assert reply['recorded'] == 1 and reply['wire_version'] == WIRE_VERSION
        # the reply is the SCRUBBED scoreboard — a one-bit verdict, never the raw rows
        assert 'rows' not in reply and 'ledger_rows' not in reply and 'candidates' not in reply
        assert [r['verdict'] for r in reply['corpus']] == ['KILLED']
        assert all('p_value' not in r and 't_stat_newey_west' not in r for r in reply['corpus'])

    def test_assert_numberless_catches_a_leak(self) -> None:
        assert_numberless({'corpus': [{'template': 'x', 'verdict': 'KILLED'}]})   # clean: no raise
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'corpus': [{'template': 'x', 't_stat_newey_west': 2.1}]})
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'a': {'b': [{'p_value': 0.01}]}})                  # nested leak caught too

    def test_banned_set_covers_engine_result_fields(self) -> None:
        # the guard's promise ("a future leak fails loudly") only holds if every result key
        # the engine produces is banned — incl. the easy-to-miss n_days / nw_lag / sign_ok
        assert {'t_stat_newey_west', 'p_value', 'e_value', 'elond_survivor', 'by_survivor',
                'n_days', 'nw_lag', 'sign_ok', 'data_lineage_hash'} <= BANNED_RESULT_FIELDS

    def test_proposer_cannot_crash_oracle_with_a_banned_named_key(self, monkeypatch, tmp_path) -> None:
        # S1 regression: an untrusted proposal carrying a banned-named key gets rejected,
        # and the echoed rejected[].proposal is re-scrubbed to coordinates — so it neither
        # rides the reply nor trips assert_numberless (which would crash the oracle).
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        led = str(tmp_path / 'l.jsonl')
        sneaky = {**self._CELL, 'ticker': 'ZZZ', 't_stat_newey_west': 9.9}        # off-campaign + banned key
        reply = score_and_record([sneaky], round_id='r', model=self._MODEL,
                                 campaign=Campaign(search=('AAA',)), path=led,
                                 provenance_path=str(tmp_path / 'p.jsonl'), scorer=self._scorer)
        assert_numberless(reply)                                                  # did not raise
        echoed = reply['rejected'][0]['proposal']
        assert set(echoed) == {'overlay', 'ticker', 'params', 'predicted_sign'}   # scrubbed to coords
        assert 't_stat_newey_west' not in echoed

    def test_missing_model_field_raises_before_recording(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        led = str(tmp_path / 'l.jsonl')
        bad_model = {k: v for k, v in self._MODEL.items() if k != 'prompt_sha'}
        with pytest.raises(ValueError, match='prompt_sha'):
            score_and_record([self._CELL], round_id='r', model=bad_model,
                             campaign=Campaign(search=('AAA',)), path=led,
                             provenance_path=str(tmp_path / 'p.jsonl'), scorer=self._scorer)
        assert load_idea_ledger(led) == []                                        # nothing recorded on a bad request


class TestStructurePhase:
    """Always-run synthetic layer for the structure phase — the HAC-t asymptotic p,
    the template x ticker enumerator + seal, and the FDR / scale-guard / flagging
    logic via an injected scorer (no engine, no data)."""

    def test_asymptotic_p_one_sided_upper_tail(self) -> None:
        # predicted_sign=+1 tests P(Z >= t): t=0 -> 0.5, t=+2.326 -> ~0.01, t<0 -> >0.5
        assert _asymptotic_p(0.0, +1) == pytest.approx(0.5, abs=1e-9)
        assert _asymptotic_p(2.326, +1) == pytest.approx(0.01, abs=1e-3)
        assert _asymptotic_p(-2.0, +1) == pytest.approx(0.9772, abs=1e-3)
        # a negative-sign prediction flips the tail
        assert _asymptotic_p(-2.326, -1) == pytest.approx(0.01, abs=1e-3)

    def test_enumerate_cross_product_and_seal(self) -> None:
        cands = enumerate_structure_candidates(STRUCTURE_CAMPAIGN)
        assert len(cands) == len(STRUCTURE_TEMPLATES) * len(STRUCTURE_SEARCH)
        assert {c.ticker for c in cands} == set(STRUCTURE_SEARCH)
        # the seal is by OMISSION — no sealed ticker (TLT) is ever a candidate
        assert STRUCTURE_SEALED == ('TLT',)
        assert all(c.ticker not in STRUCTURE_SEALED for c in cands)
        for tk in STRUCTURE_SEARCH:
            assert {c.template for c in cands if c.ticker == tk} == {t.name for t in STRUCTURE_TEMPLATES}

    def test_every_template_predicts_positive_premium(self) -> None:
        assert all(t.predicted_sign == +1 for t in STRUCTURE_TEMPLATES)

    def test_campaign_fdr_and_scale_exclusion_via_injected_scorer(self) -> None:
        """run_structure_campaign over an injected scorer: a scale-INVALID ticker is
        carried into BY as p=None (counts toward n, never a survivor), and a genuinely
        tiny p among the scored cells survives BY."""
        def scorer(cand: StructureCandidate) -> dict:
            base = dict(phase='structure', template=cand.template, ticker=cand.ticker,
                        params=cand.params_dict(), predicted_sign=1)
            if cand.ticker == 'BBB':   # a scale-broken ticker → measurement-invalid
                return {**base, 'measurement_invalid': True, 'scale_ratio': 2.0,
                        't_stat_newey_west': None, 'sign_ok': False, 'p_value': None}
            p = 0.0001 if cand.template == 'short_call_25' else 0.5
            return {**base, 't_stat_newey_west': 3.0, 'sign_ok': True, 'p_value': p}

        rows = run_structure_campaign(Campaign(search=('AAA', 'BBB')), scorer=scorer)
        invalid = [r for r in rows if r.get('measurement_invalid')]
        scored = [r for r in rows if not r.get('measurement_invalid')]
        assert len(invalid) == len(STRUCTURE_TEMPLATES)   # every BBB cell flagged invalid
        assert len(scored) == len(STRUCTURE_TEMPLATES)    # every AAA cell scored
        # invalid cells go INTO BY (p=None) but can never be rejected
        assert all(r['p_value'] is None for r in invalid)
        assert all(not r['by_survivor'] for r in invalid)
        # BY over [0.0001, 0.5, 0.5, 0.5, None, None, None, None] at q=0.10 ->
        # only the tiny p survives (the four Nones still count toward n=8)
        survivors = [r for r in scored if r['clean_survivor']]
        assert len(survivors) == 1
        assert survivors[0]['ticker'] == 'AAA' and survivors[0]['template'] == 'short_call_25'
        # e-LOND is the CONTROL of record: the same tiny-p cell clears the e-LOND bar at
        # the head of the stream (e=1/(2*sqrt 0.0001)=50 >= 1/(alpha*gamma_1)); invalid
        # cells calibrate to e=0 and are never flagged.
        elond = [r for r in rows if r['elond_survivor']]
        assert len(elond) == 1
        assert elond[0]['ticker'] == 'AAA' and elond[0]['template'] == 'short_call_25'
        assert all(r['e_value'] == 0.0 for r in invalid)

    def test_measurement_invalid_does_not_shrink_by_denominator(self) -> None:
        """A cell flagged measurement_invalid must still consume a BY comparison
        (p=None counts toward n), so flagging it can never loosen the rejection bar
        for the other cells — a data-dependent N-shrink lever the pre-fix code had.

        Construct a borderline cell (p=0.005) that survives BY's rank-1 threshold at
        n=7 but NOT at n=8, then toggle one OTHER cell between scored-valid and
        measurement-invalid. With n preserved (the fix), the borderline cell's verdict
        is identical either way. On the pre-fix code, flagging dropped the cell before
        BY (n: 8 -> 7), loosening the bar enough to rescue the borderline cell."""
        BORDERLINE, LARGE = 0.005, 0.9     # 0.005 survives BY at n=7, fails at n=8
        FLAGGED = ('BBB', 'iron_condor')   # the single cell toggled valid <-> invalid
        BORDER_CELL = ('AAA', 'short_call_25')

        def make_scorer(flag_invalid: bool):
            def scorer(cand: StructureCandidate) -> dict:
                base = dict(phase='structure', template=cand.template, ticker=cand.ticker,
                            params=cand.params_dict(), predicted_sign=1)
                if flag_invalid and (cand.ticker, cand.template) == FLAGGED:
                    return {**base, 'measurement_invalid': True, 'scale_ratio': 2.0,
                            't_stat_newey_west': None, 'sign_ok': False, 'p_value': None}
                p = BORDERLINE if (cand.ticker, cand.template) == BORDER_CELL else LARGE
                return {**base, 't_stat_newey_west': 3.0, 'sign_ok': True, 'p_value': p}
            return scorer

        camp = Campaign(search=('AAA', 'BBB'))
        valid_rows = run_structure_campaign(camp, scorer=make_scorer(False))
        flagged_rows = run_structure_campaign(camp, scorer=make_scorer(True))

        # both runs enumerate the same 8 cells; BY's effective n is 8 in BOTH
        assert len(valid_rows) == len(flagged_rows) == 2 * len(STRUCTURE_TEMPLATES)

        def remaining_survivors(rows):
            return {(r['ticker'], r['template']) for r in rows
                    if r['by_survivor'] and (r['ticker'], r['template']) != FLAGGED}

        # the remaining cells' BY verdicts are identical whether or not FLAGGED is
        # invalid — flagging preserved n, so the threshold did not move.
        assert remaining_survivors(valid_rows) == remaining_survivors(flagged_rows)
        # concretely: the borderline cell does NOT survive in either run (it would,
        # had flagging shrunk n to 7).
        def border(rows):
            return next(r for r in rows
                        if (r['ticker'], r['template']) == BORDER_CELL)
        assert border(valid_rows)['by_survivor'] is False
        assert border(flagged_rows)['by_survivor'] is False


@pytest.fixture(scope='module')
def structure_campaign():
    if not _HAVE_STRUCTURE:
        pytest.skip("needs the structure search tickers' option dailies (or .gz twins)")
    return run_structure_campaign()


@pytest.mark.skipif(
    not _HAVE_STRUCTURE,
    reason='needs MSFT/SPY/QQQ/GLD/XLE/EEM option dailies (or their committed .gz twins)',
)
class TestStructureCampaign:
    """Pin the engine-re-run campaign on the real chains — short-vol / straddle /
    iron-condor / strangle / risk-reversal / credit-spread across
    MSFT/SPY/QQQ/GLD/XLE/EEM/NVDA, TLT sealed, scored by the HAC-t asymptotic null
    and judged by e-LOND (BY a diagnostic). EXPLORATORY, not a registered verdict.
    Deterministic (overlays + closed-form p, no RNG); cells use the LIVE
    CHAIN_CLEAN_START (exploratory sees the corrected boundary)."""

    @staticmethod
    def _cell(rows, template, ticker):
        return next(r for r in rows if r['template'] == template and r['ticker'] == ticker)

    def test_batch_all_scored_one_invalid(self, structure_campaign) -> None:
        rows = structure_campaign
        assert len(rows) == len(STRUCTURE_TEMPLATES) * len(STRUCTURE_SEARCH)  # 56 (credit-spread + calendar)
        # Exactly ONE cell is measurement-invalid: the MSFT calendar. MSFT's listed chains carry no
        # far-dated call at the near leg's exact strike (a same-strike calendar needs the strike
        # quoted >=30 DTE beyond the near, which MSFT's grid doesn't list), so the structure never
        # enters and the no_trades guard flags it. It still COUNTS toward the e-LOND/BY denominator
        # (n stays 56) but can never be flagged — the campaign analog of must_trade. Every other
        # search-ticker cell is scale-valid and traded.
        invalid = [r for r in rows if r.get('measurement_invalid')]
        assert len(invalid) == 1
        assert invalid[0]['template'] == 'calendar' and invalid[0]['ticker'] == 'MSFT'
        assert invalid[0].get('no_trades') is True

    def test_no_survivor(self, structure_campaign) -> None:
        """The decisive output: no structure candidate is flagged by e-LOND (the FDR
        control of record, #3b), and none survives the BY diagnostic either."""
        rows = structure_campaign
        assert sum(r['elond_survivor'] for r in rows) == 0   # e-LOND: the control flag
        assert all('e_value' in r for r in rows)             # every cell calibrated p->e
        assert sum(r['clean_survivor'] for r in rows) == 0   # BY diagnostic
        assert sum(r['by_survivor'] for r in rows) == 0

    def test_spy_short_call_strongest_but_misses_by(self, structure_campaign) -> None:
        """SPY short-call (the exploratory cousin of the frozen +2.54 headline) is the
        strongest cell, yet its p clears the BY rank-1 bar by a wide margin."""
        rows = structure_campaign
        spy = self._cell(rows, 'short_call_25', 'SPY')
        assert spy['t_stat_newey_west'] == pytest.approx(2.17, abs=0.06)
        ps = [r['p_value'] for r in rows if r['p_value'] is not None]
        assert spy['p_value'] == pytest.approx(min(ps), abs=1e-6)
        n = len(ps)
        c = float(sum(1.0 / i for i in range(1, n + 1)))
        assert min(ps) > (1 / (n * c)) * 0.10

    def test_xle_repaired_not_a_survivor(self, structure_campaign) -> None:
        """Regression on the split fix: XLE short-call is t~-1.7 (no edge), NOT the
        t=+4.16 the halved split-adjusted price file fabricated, and the scale guard
        leaves it SCORED (ratio ~1.0 after the fix), not excluded."""
        rows = structure_campaign
        xle = self._cell(rows, 'short_call_25', 'XLE')
        assert xle['t_stat_newey_west'] == pytest.approx(-1.72, abs=0.10)
        assert xle['t_stat_newey_west'] < 0           # the fabricated +4.16 is gone
        assert not xle.get('measurement_invalid')     # scale-valid after the fix
        assert xle['by_survivor'] is False

    def test_put_leg_cells_trade_on_calls_only_tickers(self, structure_campaign) -> None:
        """Regression on the calls-only defect: the canonical SPY/MSFT/QQQ stores carry no puts,
        so the put-leg structures (straddle, iron condor, credit spread) USED to never enter —
        recording a vacuous ~0 t-stat (straddle == iron_condor, the tell). With the separate puts
        file now merged at load (_put_chain_paths), they trade: each cell is a real measurement
        (scored, not measurement_invalid) and the put-leg structures' t-stats differ from one
        another (distinct structures, distinct t-stats — the credit spread, a 2-leg put structure,
        is not the 4-leg iron condor)."""
        rows = structure_campaign
        for tk in ('SPY', 'MSFT', 'QQQ'):
            strad = self._cell(rows, 'straddle', tk)
            ic = self._cell(rows, 'iron_condor', tk)
            cs = self._cell(rows, 'credit_spread', tk)
            for cell in (strad, ic, cs):
                assert not cell.get('measurement_invalid')
                assert cell['t_stat_newey_west'] is not None
            # the calls-only bug made these identical (both flat rf curves); now they differ
            assert strad['t_stat_newey_west'] != ic['t_stat_newey_west']
            assert cs['t_stat_newey_west'] != ic['t_stat_newey_west']  # credit spread != iron condor

    def test_risk_reversal_all_wrong_signed(self, structure_campaign) -> None:
        """Widening 2 (the first NEW family, SKEW): every risk-reversal cell is wrong-signed
        (negative alpha over cash) on all 7 search tickers — no harvestable put-call skew premium
        at these names/era. The structure's large RAW P&L is rf-interest; the alpha the campaign
        scores is negative, so the SKEW family enters the lifetime stream as 7 more nulls (0/56)."""
        rows = structure_campaign
        rr = [r for r in rows if r['template'] == 'risk_reversal']
        assert len(rr) == len(STRUCTURE_SEARCH)        # 7 tickers
        assert all(r['t_stat_newey_west'] < 0 for r in rr)
        assert all(r['sign_ok'] is False for r in rr)
        assert all(not r['elond_survivor'] and not r['by_survivor'] for r in rr)

    def test_credit_spread_all_wrong_signed(self, structure_campaign) -> None:
        """Widening 3 (the first CARRY structure, the bull put credit spread): every credit-spread
        cell is wrong-signed (negative alpha over cash) on all 7 search tickers — the defined-risk
        carry collects a credit but the delta-hedged vol-P&L is negative at these names/era. So the
        CARRY family enters the lifetime stream as 7 more nulls and the verdict holds at 0/56.
        Per-ticker HAC-t: MSFT -2.08 / SPY -0.91 / QQQ -0.72 / GLD -3.24 / XLE -2.74 / EEM -2.21 /
        NVDA -0.06 (all short of significance)."""
        rows = structure_campaign
        cs = [r for r in rows if r['template'] == 'credit_spread']
        assert len(cs) == len(STRUCTURE_SEARCH)        # 7 tickers
        assert all(r['t_stat_newey_west'] < 0 for r in cs)
        assert all(r['sign_ok'] is False for r in cs)
        assert all(not r['elond_survivor'] and not r['by_survivor'] for r in cs)
        # not measurement_invalid: the put legs trade on calls-only SPY/MSFT/QQQ (merged puts)
        assert all(not r.get('measurement_invalid') for r in cs)

    def test_calendar_all_wrong_signed_or_invalid(self, structure_campaign) -> None:
        """Widening 4 (the first TERM family, the long calendar): six of seven calendar cells trade
        and are wrong-signed (negative alpha — a long-vega calendar pays for term-structure exposure
        these names/era don't reward), and the seventh (MSFT) is measurement-invalid because MSFT's
        chains don't list a far call at the near's strike. So the TERM family enters the lifetime
        stream as six more nulls plus one invalid — none flagged, and 0/56 holds."""
        rows = structure_campaign
        cal = [r for r in rows if r['template'] == 'calendar']
        assert len(cal) == len(STRUCTURE_SEARCH)        # 7 tickers
        traded = [r for r in cal if not r.get('measurement_invalid')]
        invalid = [r for r in cal if r.get('measurement_invalid')]
        assert len(traded) == 6 and len(invalid) == 1
        assert invalid[0]['ticker'] == 'MSFT'           # the listed-strike gap
        assert all(r['t_stat_newey_west'] < 0 for r in traded)   # all six traded cells wrong-signed
        assert all(r['sign_ok'] is False for r in traded)
        assert all(not r['elond_survivor'] and not r['by_survivor'] for r in cal)
        # the SPY calendar is the strongest-wrong-signed (a clean two-expiration measurement)
        spy = self._cell(rows, 'calendar', 'SPY')
        assert spy['t_stat_newey_west'] == pytest.approx(-2.44, abs=0.10)

    @pytest.mark.skipif(not _have_dailies('QQQ'), reason='needs qqq_option_dailies.csv (+ _puts)')
    def test_puts_merge_keeps_window_on_calls_span(self) -> None:
        """The merge is purely ADDITIVE on the measurement window. QQQ's puts file starts 2011 but
        its calls start 2016 (and QQQ has no era clip), so without the call-day clip the merged
        window would stretch back to 2011 — a calls-free span that dilutes the t-stat with idle rf
        days and re-measures even the call cells. The clip restricts the window to call days, so a
        merged ticker's window still begins at its calls-file start (2016 for QQQ), not the puts'."""
        _store, dates, _prices = _load_ticker_data('QQQ')
        assert dates[0] >= '2016-01-01'   # the QQQ calls-file start, NOT the 2011 puts start


# --------------------------------------------------------------------------- #
# Per-onboarded-ticker single-ticker structure campaigns
# --------------------------------------------------------------------------- #
# As each new ticker is onboarded and its store published, its single-ticker
# structure campaign (the onboarding smoke test) gets a CI-reproducible pin here
# — same engine-re-run phase, run on its own chain alone (BY over its 6 template
# cells). EXPLORATORY, not a registered verdict: pinned so the onboarding sweep
# is not re-derived. Every onboarded ticker so far reads 0/6 survivors with
# every cell wrong-signed (t_NW < 0), the recurring lesson that a bare short-vol,
# spread, or skew structure carries no edge on these chains.


@pytest.fixture(scope='module')
def nvda_structure_campaign():
    if not _have_dailies('NVDA'):
        pytest.skip('needs NVDA option dailies (or its committed .gz twin)')
    return run_structure_campaign(Campaign(search=('NVDA',)))


@pytest.mark.skipif(
    not _have_dailies('NVDA'),
    reason='needs NVDA option dailies (or its committed .gz twin)',
)
class TestNvdaStructureCampaign:
    """Pin NVDA's structure cells as a focused per-ticker check — NVDA is also one
    of the seven search tickers in the main 56-cell campaign, so this isolates it
    on the published chain: all eight structures (short-vol x2 / straddle / iron-condor /
    strangle / risk-reversal / credit-spread / calendar) run on NVDA alone, scored by the HAC-t
    asymptotic null; e-LOND is the control (BY a diagnostic) over the 8 template cells.
    EXPLORATORY, not a registered verdict. Deterministic (overlays + closed-form p, no
    RNG); the LIVE CHAIN_CLEAN_START applies."""

    @staticmethod
    def _cell(rows, template):
        return next(r for r in rows if r['template'] == template and r['ticker'] == 'NVDA')

    def test_batch_all_scored_none_invalid(self, nvda_structure_campaign) -> None:
        rows = nvda_structure_campaign
        assert len(rows) == len(STRUCTURE_TEMPLATES)   # 8 cells: NVDA x every template
        assert all(r['ticker'] == 'NVDA' for r in rows)
        # NVDA's chain DOES list far calls at the near strike, so the calendar trades here
        # (unlike MSFT in the full campaign) — every NVDA cell is a real measurement.
        assert sum(r.get('measurement_invalid', False) for r in rows) == 0

    def test_no_survivor(self, nvda_structure_campaign) -> None:
        """The decisive output: no NVDA structure candidate is flagged by e-LOND (the
        control), nor survives the BY diagnostic."""
        rows = nvda_structure_campaign
        assert sum(r['elond_survivor'] for r in rows) == 0   # e-LOND control
        assert sum(r['clean_survivor'] for r in rows) == 0   # BY diagnostic
        assert sum(r['by_survivor'] for r in rows) == 0

    def test_every_cell_wrong_signed(self, nvda_structure_campaign) -> None:
        """Every template predicts positive premium (t_NW>0); on NVDA every cell —
        short-vol, straddle, iron-condor, strangle, the SKEW risk reversal, the CARRY credit
        spread, AND the TERM calendar — is wrong-signed (t_NW<0). No structure carries an edge
        on NVDA's runaway chain."""
        rows = nvda_structure_campaign
        assert all(r['t_stat_newey_west'] < 0 for r in rows)
        assert all(r['sign_ok'] is False for r in rows)

    def test_cell_t_nw_and_p_values(self, nvda_structure_campaign) -> None:
        """Pin each cell's HAC-t and its asymptotic (one-sided upper-tail) p."""
        rows = nvda_structure_campaign
        sc25 = self._cell(rows, 'short_call_25')
        assert sc25['t_stat_newey_west'] == pytest.approx(-0.96, abs=0.05)
        assert sc25['p_value'] == pytest.approx(0.8315, abs=0.01)
        scatm = self._cell(rows, 'short_call_atm')
        assert scatm['t_stat_newey_west'] == pytest.approx(-0.96, abs=0.05)
        assert scatm['p_value'] == pytest.approx(0.8315, abs=0.01)
        strad = self._cell(rows, 'straddle')
        assert strad['t_stat_newey_west'] == pytest.approx(-1.22, abs=0.05)
        assert strad['p_value'] == pytest.approx(0.8888, abs=0.01)
        ic = self._cell(rows, 'iron_condor')
        assert ic['t_stat_newey_west'] == pytest.approx(-1.47, abs=0.05)
        assert ic['p_value'] == pytest.approx(0.9292, abs=0.01)
        strangle = self._cell(rows, 'strangle')                    # widening 1
        assert strangle['t_stat_newey_west'] == pytest.approx(-1.33, abs=0.05)
        assert strangle['p_value'] == pytest.approx(0.9082, abs=0.01)
        rr = self._cell(rows, 'risk_reversal')                     # widening 2 (SKEW)
        assert rr['t_stat_newey_west'] == pytest.approx(-0.19, abs=0.05)
        assert rr['p_value'] == pytest.approx(0.5753, abs=0.01)
        cs = self._cell(rows, 'credit_spread')                     # widening 3 (CARRY)
        assert cs['t_stat_newey_west'] == pytest.approx(-0.06, abs=0.05)
        assert cs['p_value'] == pytest.approx(0.5239, abs=0.01)
        cal = self._cell(rows, 'calendar')                         # widening 4 (TERM)
        assert cal['t_stat_newey_west'] == pytest.approx(-1.88, abs=0.05)
        assert cal['p_value'] == pytest.approx(0.9699, abs=0.01)


class TestLlmCliRefusal:
    """The read-gate CLI refusal (interlock #5): `propose --llm` must FAIL CLOSED unless BOTH
    (a) the engine is unimportable from cwd AND (b) a model author is configured. Always-run,
    synthetic — no datasets, no engine. The core pin is that an LLM author cannot be activated
    from the engine checkout; the only sanctioned home is proposer_client inside the
    oracle_server sandbox (docs/read_gate.md), where C-1 makes `import edge_search` raise."""

    def test_no_model_author_configured_yet(self) -> None:
        # item 4 (the real Claude client) is a later PR — backstop (b) is live until then
        assert _resolve_llm_author() is None

    def test_engine_importable_from_repo_cwd(self) -> None:
        # the subprocess probe sees edge_search from the repo checkout -> precondition (a) fails
        assert _engine_importable_from_cwd() is True

    def test_engine_unimportable_from_clean_cwd(self, monkeypatch, tmp_path) -> None:
        # from a directory that is NOT the engine checkout, the scrubbed-env probe can't
        # import edge_search (no PYTHONPATH leak) -> precondition (a) PASSES
        monkeypatch.chdir(tmp_path)
        assert _engine_importable_from_cwd() is False

    def test_probe_reports_reachable_when_engine_present_but_deps_broken(
            self, monkeypatch, tmp_path) -> None:
        # The SERIOUS-fix regression pin: the probe measures PRESENCE (find_spec), not
        # importability. A PRESENT engine whose body fails to import (missing dep / syntax error)
        # must still read as reachable (True -> refuse), never as "absent". A stand-in edge_search.py
        # whose first line imports a nonexistent module would make a bare `import edge_search` probe
        # exit non-zero (and the old `returncode == 0` logic return False -> spuriously pass (a));
        # find_spec locates the file WITHOUT running it, so it correctly reports reachable.
        (tmp_path / 'edge_search.py').write_text(
            'import a_module_that_surely_does_not_exist_zzz\n')
        monkeypatch.chdir(tmp_path)
        assert _engine_importable_from_cwd() is True

    def test_guard_refuses_from_repo_naming_a(self, capsys) -> None:
        # (a) fails from the repo: the guard refuses and the message names precondition (a)
        with pytest.raises(SystemExit) as exc:
            _assert_llm_boundary()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert 'REFUSED' in err
        assert '(a)' in err and 'importable' in err
        assert 'docs/read_gate.md' in err and 'oracle_server' in err

    def test_a_and_b_independent_b_is_the_backstop(self, monkeypatch, tmp_path, capsys) -> None:
        # in a clean cwd (a) PASSES, but (b) still refuses (no model) -> the two are independent
        # and (b) is the current backstop. The message names (b), NOT (a).
        monkeypatch.chdir(tmp_path)
        assert _engine_importable_from_cwd() is False             # (a) passes here
        with pytest.raises(SystemExit) as exc:
            _assert_llm_boundary()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert '(b)' in err and 'no model author' in err
        assert '(a) the engine IS importable' not in err          # (a) did NOT fail here

    def test_a_fails_even_when_b_passes(self, capsys) -> None:
        # supply a configured author (simulating item 4): (b) passes, but (a) still fails from
        # the repo -> wiring a model never unlocks running it from the engine checkout
        def author(menu, corpus, onboarded):                      # a stand-in LLMProposer
            return None
        with pytest.raises(SystemExit) as exc:
            _assert_llm_boundary(author=author)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert '(a) the engine IS importable' in err              # (a) is the failure here
        assert 'no model author' not in err                       # (b) passed

    def test_propose_llm_fails_closed_via_cli(self, monkeypatch, capsys) -> None:
        # the core interlock pin, driven through the CLI entry: `propose --llm` from the repo
        # exits non-zero with the boundary message before any author/engine runs.
        called = []
        monkeypatch.setattr('edge_search.run_proposer_round',
                            lambda *a, **k: called.append((a, k)) or {})
        monkeypatch.setattr(sys, 'argv', ['edge_search.py', 'propose', '--llm'])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert 'REFUSED' in err and '(a)' in err and '(b)' in err
        assert called == []                                       # never reached the proposer

    def test_menu_walker_propose_path_unaffected(self, monkeypatch) -> None:
        # the default `propose` (no --llm) still routes to the menu-walker: run_proposer_round
        # is called with author unset (None), and no boundary guard fires.
        captured = {}

        def _stub(*args, **kwargs):
            captured['author'] = kwargs.get('author', None)
            captured['run'] = kwargs.get('run')
            captured['record'] = kwargs.get('record')
            return {'proposed': 0, 'recorded': 0, 'needs_onboard': [], 'rows': []}

        monkeypatch.setattr('edge_search.run_proposer_round', _stub)
        # if the guard ran on this path it would explode the test
        monkeypatch.setattr('edge_search._assert_llm_boundary',
                            lambda *a, **k: pytest.fail('guard must not fire without --llm'))
        monkeypatch.setattr(sys, 'argv', ['edge_search.py', 'propose'])
        main()
        assert captured['author'] is None                         # menu-walker path
        assert captured['run'] is False and captured['record'] is False  # default = preview
