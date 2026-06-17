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

import numpy as np
import pytest

from edge_search import (
    COOLDOWN_NS,
    DEFAULT_CAMPAIGN,
    SEALED_TICKERS,
    SEARCH_TICKERS,
    TREND_WINDOWS,
    Campaign,
    Candidate,
    CycleData,
    _add_one_p,
    _vol_confound,
    benjamini_yekutieli,
    build_cycle_data,
    enumerate_candidates,
    kill_gate,
    load_search_runs,
    run_batch,
    run_campaign,
)
from test_real_cc_backtest import _HAVE_MSFT_DAILIES, _HAVE_SPY_DAILIES

_HAVE_SEARCH = _HAVE_MSFT_DAILIES and _HAVE_SPY_DAILIES


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
        lose less), the same sign the pooled three-ticker scout found."""
        _, rows = campaign
        cool = [r for r in rows if r['template'] == 'cooldown']
        assert all(r['D_A'] > 0 for r in cool)
        assert all(r['sign_ok'] is False for r in cool)
        k = self._by_key(rows)
        assert k[('cooldown', 30)]['D_A'] == pytest.approx(604.51, abs=1.0)
        assert k[('cooldown', 30)]['p_value'] == pytest.approx(0.867, abs=0.01)
        assert k[('cooldown', 90)]['D_A'] == pytest.approx(1405.68, abs=1.0)

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
        gate, D_A>0 at p~0.08) is exactly the documented trap: it is the
        low-vol confound (rich-IV entries cluster in calm markets, so
        vol_confound<0), and it does NOT survive campaign-wide BY."""
        _, rows = campaign
        iv = next(r for r in rows if r['template'] == 'iv_rich')
        assert iv['sign_ok'] is True
        assert iv['D_A'] == pytest.approx(747.62, abs=1.0)
        assert iv['p_value'] == pytest.approx(0.080, abs=0.01)
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
