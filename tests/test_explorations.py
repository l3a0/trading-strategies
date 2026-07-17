"""Tests for explorations.py — the pinned exploration log.

Two layers, like the other real-chain suites:

- Pure-logic unit tests (always run): cycle reconstruction + rip flagging,
  the PER-TICKER post-rip shadow (a rip on one name must not cool down
  another), the D_A statistic, the annualized-vol helper, and the vendor-IV
  reader — all on hand-computable fixtures.
- TestCooldownScout (dataset-gated): pins the killed post-rip-cooldown scout.
  The verdict is double — the per-cycle effect is wrong-signed (post-rip
  cycles lose LESS — D_A > 0 at every horizon, real arrangement in the HIGH
  tail of the trigger-placement null), and there is no return memory to set
  the cooldown length to.
- TestIvRichnessScout (dataset-gated): pins the killed IV-richness
  (volatility-risk-premium) gate. The ex-post VRP at the sold contract is ~0,
  entry richness doesn't predict cycle P&L (Spearman ~0), and the one
  positive-looking number (a binary IV>RV split) is the low-vol confound.

All pins are EXPLORATORY numbers, not registered verdicts — pinned so the
dead ends are not re-explored. See docs/explorations.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from search.explorations import (
    portfolio_scout,
    random_entry_scout,
    random_entry_selector,
    SCOUT_TICKERS,
    _ann_vol,
    _d_a,
    _ord,
    cooldown_scout,
    iv_richness_scout,
    load_entry_ivs,
    load_naked_run,
    post_rip_mask,
    reconstruct_cycles,
)
from test_real_cc_backtest import (
    _HAVE_DAILIES,
    _HAVE_MSFT_DAILIES,
    _HAVE_SPY_DAILIES,
    _SPY_DAILIES,
)

_HAVE_ALL = _HAVE_MSFT_DAILIES and _HAVE_DAILIES and _HAVE_SPY_DAILIES


# ---- always-run: cycle/rip logic and the per-ticker shadow ----

class TestCooldownScoutMechanics:
    def test_reconstruct_cycles_flags_rips(self) -> None:
        """A rip = close_itm OR a loss-making expiration; wins are not rips."""
        trades = [
            {'action': 'sell', 'date': '2024-01-02'},
            {'action': 'close', 'date': '2024-01-05', 'pnl': 100.0},      # win
            {'action': 'sell', 'date': '2024-01-08'},
            {'action': 'close_itm', 'date': '2024-01-10', 'pnl': -300.0}, # rip
            {'action': 'sell', 'date': '2024-01-11'},
            {'action': 'expiration', 'date': '2024-01-20', 'pnl': -50.0}, # rip (assignment loss)
            {'action': 'sell', 'date': '2024-01-22'},
            {'action': 'expiration', 'date': '2024-01-30', 'pnl': 25.0},  # profitable expiry
        ]
        cycles = reconstruct_cycles(trades)
        assert [c['rip'] for c in cycles] == [False, True, True, False]

    def test_post_rip_mask_is_per_ticker(self) -> None:
        """A rip on ticker A must NOT cool down a same-day entry on ticker B
        (the cross-ticker tagging bug the pinned scout avoids)."""
        entry_ords = [_ord('2024-01-15'), _ord('2024-01-15')]
        rip_ords = {'A': [_ord('2024-01-10')], 'B': []}
        mask = post_rip_mask(entry_ords, ['A', 'B'], rip_ords, horizon=30)
        assert list(mask) == [True, False]

    def test_post_rip_mask_horizon_and_strict_prior(self) -> None:
        """Within N calendar days AND strictly after the rip. Rip on
        2024-01-10: +30d (2024-02-09) is in, +31d is out, same-day is out
        (strictly prior), before-the-rip is out."""
        rip = {'A': [_ord('2024-01-10')]}
        ents = [_ord('2024-01-15'), _ord('2024-02-09'),
                _ord('2024-02-10'), _ord('2024-01-10'), _ord('2024-01-09')]
        mask = post_rip_mask(ents, ['A'] * 5, rip, horizon=30)
        assert list(mask) == [True, True, False, False, False]

    def test_d_a(self) -> None:
        pnls = np.array([100.0, -300.0, 50.0])
        assert _d_a(pnls, np.array([True, False, False])) == pytest.approx(225.0)
        assert _d_a(pnls, np.array([False, False, False])) is None  # empty cell

    def test_ann_vol(self) -> None:
        # annualized realized vol = sample std (ddof=1) of log returns × √252
        r = np.array([0.01, -0.01, 0.02, -0.02])
        assert _ann_vol(r) == pytest.approx(float(np.std(r, ddof=1)) * np.sqrt(252))

    def test_load_entry_ivs(self, tmp_path) -> None:
        """The IV reader picks the vendor implied_volatility for exactly the
        wanted (date, contractID) rows, skips others, and tolerates blanks."""
        header = ('date,expiration,dte,strike,bid,ask,mark,last,volume,'
                  'open_interest,implied_volatility,delta,contractID')
        p = tmp_path / 'xyz_option_dailies.csv'
        p.write_text('\n'.join([
            header,
            '2024-01-02,2024-02-02,31,110,2.0,2.2,2.1,0,0,0,0.2150,0.25,WANT',
            '2024-01-02,2024-02-02,31,120,0.4,0.6,0.5,0,0,0,0.1800,0.10,SKIP',
            '2024-01-03,2024-02-02,30,110,1.0,1.2,1.1,0,0,0,,0.25,BLANK',
        ]) + '\n')
        out = load_entry_ivs('XYZ', {('2024-01-02', 'WANT'),
                                     ('2024-01-03', 'BLANK')}, path=str(p))
        assert out == {('2024-01-02', 'WANT'): pytest.approx(0.2150)}


# ---- dataset-gated: the pinned scouts ----

@pytest.fixture(scope='module')
def naked_runs():
    """The three naked baseline runs, loaded once and shared by both scouts."""
    if not _HAVE_ALL:
        pytest.skip('needs MSFT + QQQ + SPY option dailies (or .gz twins)')
    return [load_naked_run(t) for t in SCOUT_TICKERS]


@pytest.fixture(scope='module')
def scout(naked_runs):
    return cooldown_scout(naked_runs)


@pytest.fixture(scope='module')
def iv(naked_runs):
    return iv_richness_scout(naked_runs)


@pytest.mark.skipif(
    not _HAVE_ALL,
    reason='needs MSFT + QQQ + SPY option dailies (or their committed .gz twins)',
)
class TestCooldownScout:
    """Pin the killed post-rip-cooldown scout (docs/explorations.md).

    EXPLORATORY, not a registered verdict — pinned so the dead end is not
    re-derived. Deterministic: naked runs on the clean canonical chains
    (CHAIN_CLEAN_START applied), per-ticker rip tagging, seed-20260613
    trigger-placement permutation.
    """

    def test_pool(self, scout) -> None:
        """705 naked cycles across MSFT/QQQ/SPY, 243 rip triggers."""
        assert scout['tickers'] == list(SCOUT_TICKERS)
        assert scout['n_cycles'] == 705
        assert scout['n_rips'] == 243

    def test_wrong_signed_at_every_horizon(self, scout) -> None:
        """The hypothesis predicts D_A < 0 (post-rip entries do worse). The
        data says the OPPOSITE at every horizon: D_A > 0 (post-rip cycles
        lose less), and the real arrangement sits in the HIGH tail of the
        trigger-placement null (perm percentile well above 0.5) — never the
        low tail a real effect needs. So no horizon supports the cooldown."""
        g = {row['N_days']: row for row in scout['grid']}
        assert all(row['D_A'] > 0 for row in scout['grid'])
        assert g[30]['D_A'] == pytest.approx(376.17, abs=1.0)
        assert g[60]['D_A'] == pytest.approx(623.31, abs=1.0)
        assert g[90]['D_A'] == pytest.approx(1770.21, abs=1.0)
        assert g[30]['perm_percentile'] == pytest.approx(0.936, abs=0.02)
        assert g[60]['perm_percentile'] == pytest.approx(0.954, abs=0.02)
        assert all(row['perm_percentile'] >= 0.5 for row in scout['grid'])
        # the kill condition: NO horizon shows D_A<0 in the low (significant) tail
        assert not any(row['D_A'] < 0 and row['perm_percentile'] <= 0.10
                       for row in scout['grid'])

    def test_no_return_memory(self, scout) -> None:
        """No principled cooldown N exists: forward returns after a rip sit
        BELOW the unconditional baseline at every horizon (a rip is weakly
        mean-reverting, not momentum-igniting), and the pooled daily-return
        lag-1 autocorrelation is negative — no momentum for a cooldown to
        ride, so any nonzero N is pure abstinence."""
        mem = scout['memory']
        fwd = {row['horizon_days']: row for row in mem['forward']}
        assert all(row['diff_pct'] < 0 for row in mem['forward'])
        assert fwd[30]['diff_pct'] == pytest.approx(-0.628, abs=0.01)
        assert fwd[60]['diff_pct'] == pytest.approx(-0.879, abs=0.01)
        assert mem['daily_return_acf_lag1'] == pytest.approx(-0.126, abs=0.005)
        assert mem['daily_return_acf_lag1'] < 0

    def test_abstinence_confound_visible(self, scout) -> None:
        """The naive net-P&L 'improvement' from skipping post-rip cycles is
        large and positive and rises monotonically with N — purely because
        the naked strategy loses money, so skipping any growing slice 'helps'.
        This is why the per-cycle D_A (above), not net P&L, is the honest
        statistic: sweeping N against net P&L would 'find' a bogus edge."""
        deltas = [row['net_pnl_delta_if_skipped'] for row in scout['grid']]
        assert all(d > 0 for d in deltas)
        assert deltas == sorted(deltas)  # monotonically rising with N


@pytest.mark.skipif(
    not _HAVE_ALL,
    reason='needs MSFT + QQQ + SPY option dailies (or their committed .gz twins)',
)
class TestIvRichnessScout:
    """Pin the killed IV-richness (volatility-risk-premium) gate
    (docs/explorations.md).

    EXPLORATORY, not a registered verdict. The idea: sell only when implied
    vol is rich vs realized, to harvest the volatility premium. KILLED: there
    is no premium to gate on. Reads the vendor implied_volatility at entry
    (the engine's loader discards it), fail-closed below IV_FLOOR.
    """

    def test_assessed(self, iv) -> None:
        """694 cycles carry a usable entry IV; 11 dropped by the IV<0.05 guard."""
        assert iv['tickers'] == list(SCOUT_TICKERS)
        assert iv['n_assessed'] == 694
        assert iv['n_dropped_iv_guard'] == 11

    def test_no_premium_to_gate_on(self, iv) -> None:
        """The ex-post VRP at the sold ~25-delta/30-day contract (entry IV
        minus the realized vol over the option's life) is ~0 / negative — the
        options were NOT systematically overpriced vs what actually realized,
        so there is no premium to harvest."""
        assert iv['vrp_median_pct'] == pytest.approx(-0.273, abs=0.05)
        assert iv['vrp_mean_pct'] == pytest.approx(-2.304, abs=0.1)
        assert iv['vrp_median_pct'] < 0.5  # not richly positive

    def test_richness_does_not_predict_pnl(self, iv) -> None:
        """The entry-richness signal (the thing you could gate on) has ~zero
        rank-correlation with cycle P&L."""
        assert iv['spearman_richness_pnl'] == pytest.approx(0.036, abs=0.02)
        assert abs(iv['spearman_richness_pnl']) < 0.1

    def test_binary_split_is_the_low_vol_confound(self, iv) -> None:
        """The one positive-looking number — a binary IV>RV split separates
        P&L by +$646/cycle at the 95th permutation percentile — is NOT a
        premium. "Rich" entries cluster where trailing vol is low (0.15 vs
        0.23), i.e. calm markets, where covered calls do better regardless.
        With the ex-post VRP ~0 and the rank-correlation ~0, this split is the
        vol-level confound, not a tradable edge."""
        assert iv['D_A_rich_vs_not'] == pytest.approx(646.17, abs=2.0)
        assert iv['perm_percentile'] == pytest.approx(0.946, abs=0.02)
        assert iv['mean_trailing_rv_rich'] == pytest.approx(0.1455, abs=0.002)
        assert iv['mean_trailing_rv_not'] == pytest.approx(0.2333, abs=0.002)
        assert iv['mean_trailing_rv_rich'] < iv['mean_trailing_rv_not']


def _jitter_scenario(n_days: int, first: str = '2021-01-04',
                     skip_chain: set[int] | None = None,
                     bid: float = 1.0):
    """A minimal one-strike synthetic market for the Gap F selector: every
    chain day offers one 30-DTE 0.25-delta call candidate; `skip_chain`
    indices get NO chain (the day never invokes the selector). The candidate
    tuple order is load_chain_store's: (dte, delta, bid, ask, mid,
    expiration, strike, contractID)."""
    from datetime import date as _date, timedelta
    d0 = _date.fromisoformat(first)
    dates = [(d0 + timedelta(days=i)).isoformat() for i in range(n_days)]
    exp = dates[-1]
    store: dict[str, dict] = {}
    for i, d in enumerate(dates):
        if skip_chain and i in skip_chain:
            continue
        cand = (30, 0.25, bid, bid + 0.1, bid + 0.05, exp, 100.0, 'C1')
        store[d] = {'candidates': [cand],
                    'marks': {'C1': (bid, bid + 0.1, bid + 0.05, 0.25)}}
    return dates, [100.0] * n_days, store


def _run_jitter(dates, prices, store, select):
    from realchains.vol_premium import STRUCTURE_SPECS, run_real_structure_overlay
    spec = STRUCTURE_SPECS['short_vol']
    return run_real_structure_overlay(
        dates, prices, store,
        {'target_delta': 0.25, 'dte': 30, 'capital': 100_000,
         'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0},
        select=select, entry_guard=spec['entry_guard'],
        hedge_mode=spec['hedge_mode'], management=spec['management'])


class TestRandomEntryMechanics:
    """Always-run synthetic layer for the Gap F jitter selector
    (docs/van_tharp_gap_f.md) — the k=0 anchor, chain-day counting,
    per-stretch redraws, delegation, and the emission-keyed desync
    convention, all on crafted days."""

    def test_k0_reproduces_baseline_trade_for_trade(self) -> None:
        """k=0 => J=0 always: the career IS the deterministic baseline."""
        from realchains.vol_premium import STRUCTURE_SPECS
        dates, prices, store = _jitter_scenario(12)
        s_base, t_base, eq_base = _run_jitter(dates, prices, store,
                                              STRUCTURE_SPECS['short_vol']['select'])
        s_rand, t_rand, eq_rand = _run_jitter(dates, prices, store,
                                              random_entry_selector(seed=1, k=0))
        assert t_rand == t_base
        assert eq_rand.equals(eq_base)

    def test_wait_is_chain_day_counted(self) -> None:
        """A career whose first draw is J=2 enters on its third flat CHAIN
        day; a chainless day mid-wait consumes none of the wait."""
        seed = next(s for s in range(1000)
                    if __import__('random').Random(s).randint(0, 10) == 2)
        # chainless day at index 1: the wait must stretch one calendar day longer.
        dates, prices, store = _jitter_scenario(12, skip_chain={1})
        _, trades, _ = _run_jitter(dates, prices, store,
                                   random_entry_selector(seed=seed, k=10))
        # chain days are indices 0,2,3,...; J=2 burns 0 and 2; entry on index 3.
        assert trades[0]['action'] == 'enter'
        assert trades[0]['date'] == dates[3]

    def test_new_stretch_draws_new_j_and_seed_determinism(self) -> None:
        """Same-seed engine determinism (the career reproduces exactly), and
        the career RNG's first two draws differ — the per-stretch redraw the
        guard-rejection test exercises through the engine itself."""
        rng_probe = __import__('random').Random
        # pick a seed whose first two draws differ
        seed = next(s for s in range(1000)
                    if (lambda r: r.randint(0, 5) != r.randint(0, 5))(rng_probe(s)))
        dates, prices, store = _jitter_scenario(10, first='2021-02-01')
        run1 = _run_jitter(dates, prices, store, random_entry_selector(seed=seed, k=5))
        run2 = _run_jitter(dates, prices, store, random_entry_selector(seed=seed, k=5))
        assert run1[1] == run2[1]                       # same seed, same trades
        r = rng_probe(seed)
        j1, j2 = r.randint(0, 5), r.randint(0, 5)
        assert j1 != j2                                 # the career's two draws differ

    def test_post_wait_pick_equals_baseline_pick(self) -> None:
        """After the wait expires the emitted leg equals the baseline
        selector's leg field for field (delegation, not reimplementation)."""
        from realchains.vol_premium import STRUCTURE_SPECS
        dates, prices, store = _jitter_scenario(8)
        params = {'target_delta': 0.25, 'dte': 30, 'fill': 'bid_ask'}
        day = store[dates[0]]
        base_leg = STRUCTURE_SPECS['short_vol']['select'](day, params)
        sel = random_entry_selector(seed=1, k=0)
        assert sel(day, params) == base_leg

    def test_guard_rejected_emission_rearms(self) -> None:
        """The emission-keyed desync convention, pinned synthetically: a
        sub-penny-bid day makes the guard reject the emission, and the
        closure re-arms a fresh J for the same engine-side stretch."""
        rng_probe = __import__('random').Random
        # seed whose draws are J1=0 then J2>=1: emission day 0 (rejected),
        # re-armed wait pushes the real entry past day 1.
        seed = next(s for s in range(1000)
                    if (lambda r: r.randint(0, 3) == 0 and r.randint(0, 3) >= 1)(
                        rng_probe(s)))
        dates, prices, store = _jitter_scenario(10, bid=0.005)   # sub-penny: entry_net < 0
        # make later days quotable so the career can eventually enter
        good = _jitter_scenario(10)[2]
        for d in dates[2:]:
            store[d] = good[d]
        _, trades, _ = _run_jitter(dates, prices, store,
                                   random_entry_selector(seed=seed, k=3))
        r = rng_probe(seed)
        j1, j2 = r.randint(0, 3), r.randint(0, 3)
        assert j1 == 0 and j2 >= 1
        # emission on day 0 rejected by the guard; fresh J2 burns chain days
        # 1..j2; first possible entry index is 1 + j2 (and >= 2 where quotes turn sane).
        first_enter = next(t for t in trades if t['action'] == 'enter')
        assert first_enter['date'] == dates[max(1 + j2, 2)]

    def test_post_wait_band_empty_day_spends_no_redraw(self) -> None:
        """Spec bullet 5: a post-wait day whose only candidate is out of band
        yields None with the wait already spent — no redraw (exactly one RNG
        draw consumed), and entry lands on the first in-band day after it."""
        seed = next(s for s in range(1000)
                    if __import__('random').Random(s).randint(0, 10) == 1)
        dates, prices, store = _jitter_scenario(10)
        exp = dates[-1]
        # days 1-2: only an OUT-OF-BAND candidate (delta 0.02) — band-empty
        for d in (dates[1], dates[2]):
            store[d] = {'candidates': [(30, 0.02, 1.0, 1.1, 1.05, exp, 100.0, 'C1')],
                        'marks': {'C1': (1.0, 1.1, 1.05, 0.02)}}
        _, trades, _ = _run_jitter(dates, prices, store,
                                   random_entry_selector(seed=seed, k=10))
        # J=1 burns day 0; days 1-2 delegate to a band-empty None (wait spent,
        # no redraw); entry on day 3, the first in-band day.
        assert trades[0]['action'] == 'enter'
        assert trades[0]['date'] == dates[3]


@pytest.mark.skipif(not _HAVE_SPY_DAILIES,
                    reason='needs spy_option_dailies.csv or its .gz twin')
class TestRandomEntryScout:
    """Experiment 2's pinned verdict (docs/van_tharp_gap_f.md). No envelope
    exclusion on either metric — no entry-skill claim in either direction.
    The two locates, both pinned: on raw per-cycle expectancy_r the baseline
    sits at the 5th percentile of its own band (worse than 19/20 jittered
    careers; the band spans -0.58R..-0.03R, so the raw option-cycle number
    is PLACEMENT-FRAGILE); on the hedged NW t it sits inside at the 85th
    (2.54 in a 0.98..3.58 band) — the premium isolator is placement-robust.
    The low expectancy locate is recorded WITHOUT a mechanism story (the
    pre-stated cooldown-texture mechanism predicted the opposite tail and
    is thereby not supported). EXPLORATORY — a locate, not significance;
    per-trade / per-day units only. Pre-committed: N=20, K=10,
    RANDOM_ENTRY_SEED=20260714."""

    @pytest.fixture(scope='class')
    def scout(self):
        from realchains.real_cc_backtest import (
            REGISTERED_CLEAN_START,
            load_chain_store,
            load_unadjusted_prices,
        )
        store = load_chain_store(_SPY_DAILIES, start=REGISTERED_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return random_entry_scout([d for d, _ in pairs], [p for _, p in pairs], store)

    def test_baseline_reproduced(self, scout) -> None:
        """The harness re-derives the published pins before any percentile."""
        assert scout['baseline']['n'] == 174
        assert scout['baseline']['expectancy_r'] == pytest.approx(-0.5407, abs=0.005)
        assert scout['baseline']['nw_t'] == pytest.approx(2.54, abs=0.01)

    def test_ensemble_and_percentiles(self, scout) -> None:
        assert scout['ensemble_expectancy']['mean'] == pytest.approx(-0.2711, abs=0.005)
        assert scout['ensemble_expectancy']['min'] == pytest.approx(-0.5768, abs=0.005)
        assert scout['ensemble_expectancy']['max'] == pytest.approx(-0.0313, abs=0.005)
        assert scout['career_n_range'] == [141, 147]
        assert scout['baseline_pct_expectancy'] == pytest.approx(0.05, abs=0.01)
        assert scout['baseline_pct_nw_t'] == pytest.approx(0.85, abs=0.01)
        assert scout['ensemble_nw_t']['min'] == pytest.approx(0.98, abs=0.02)
        assert scout['ensemble_nw_t']['max'] == pytest.approx(3.58, abs=0.02)


def _mini_stream(dates: list[str], equities: list[float],
                 rf_credits: list[float] | None = None):
    """A minimal daily_equity DataFrame in the engines' schema."""
    import pandas as pd
    data: dict = {'date': dates, 'equity': equities,
                  'price': [100.0] * len(dates)}
    if rf_credits is not None:
        data['rf_credit'] = rf_credits
    return pd.DataFrame(data)


class TestPortfolioMechanics:
    """Always-run synthetic layer for the Gap G harness
    (docs/van_tharp_gap_g.md, common/portfolio.py) — alignment, the rf
    switch, correlation exactness, combination arithmetic, and the
    drawdown-subadditivity fixture, all hand-computable."""

    D = ['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08']

    def test_alignment_inner_join_and_missing_day(self) -> None:
        """Mismatched spans join to their overlap; a leg-specific missing
        day drops from the join (never interpolated)."""
        from common.portfolio import align_streams
        a = _mini_stream(self.D, [100_000, 100_100, 100_200, 100_300, 100_400])
        b = _mini_stream(self.D[1:4], [100_000, 100_050, 100_150])   # shorter span
        panel = align_streams({'a': a, 'b': b})
        # b's diffs exist for 01-06 and 01-07 only -> the join is those two days.
        assert list(panel.index) == self.D[2:4]
        assert list(panel.columns) == ['a', 'b']
        assert panel['a'].iloc[0] == pytest.approx(0.001)   # $100 on $100K
        assert panel['b'].iloc[0] == pytest.approx(0.0005)

    def test_rf_column_is_the_switch(self) -> None:
        """A leg carrying rf_credit is netted with the off-by-one honored; a
        leg without it passes through raw."""
        from common.portfolio import align_streams
        eq = [100_000, 100_110, 100_220]
        rf = [0.0, 10.0, 10.0]     # $10 credited each later day
        with_rf = _mini_stream(self.D[:3], eq, rf_credits=rf)
        without = _mini_stream(self.D[:3], eq)
        panel = align_streams({'net': with_rf, 'raw': without})
        # raw diff $110/day; netted = (110 - 10)/100K = 0.001
        assert panel['net'].iloc[0] == pytest.approx(0.001)
        assert panel['raw'].iloc[0] == pytest.approx(0.0011)

    def test_correlation_exactness(self) -> None:
        """Crafted co-moving / anti-moving / orthogonal pairs measure
        exactly +1, −1, and 0."""
        from common.portfolio import align_streams, stream_correlations
        base = [100_000, 100_100, 100_050, 100_200, 100_150]
        co = _mini_stream(self.D, [100_000 + 2 * (e - 100_000) for e in base])
        anti = _mini_stream(self.D, [100_000 - (e - 100_000) for e in base])
        a = _mini_stream(self.D, base)
        panel = align_streams({'a': a, 'co': co, 'anti': anti})
        corr = stream_correlations(panel)
        assert corr.loc['a', 'co'] == pytest.approx(1.0)
        assert corr.loc['a', 'anti'] == pytest.approx(-1.0)
        # orthogonal: +x then -x vs -x then +x over 4 diffs
        o1 = _mini_stream(self.D, [100_000, 100_100, 100_000, 100_100, 100_000])
        o2 = _mini_stream(self.D, [100_000, 100_100, 100_200, 100_100, 100_000])
        panel2 = align_streams({'o1': o1, 'o2': o2})
        assert stream_correlations(panel2).loc['o1', 'o2'] == pytest.approx(0.0, abs=1e-12)

    def test_combination_arithmetic_and_weight_guard(self) -> None:
        """A 50/50 combine reproduces hand arithmetic row for row; weights
        must cover the legs and sum to 1."""
        from common.portfolio import align_streams, combine_streams
        a = _mini_stream(self.D[:3], [100_000, 100_500, 100_200])
        b = _mini_stream(self.D[:3], [100_000, 99_700, 99_900])
        panel = align_streams({'a': a, 'b': b})
        combo = combine_streams(panel, {'a': 0.5, 'b': 0.5})
        assert combo.iloc[0] == pytest.approx(0.5 * 0.005 + 0.5 * (-0.003))  # +0.001
        assert combo.iloc[1] == pytest.approx(0.5 * (-0.003) + 0.5 * 0.002)
        with pytest.raises(ValueError, match='sum to'):
            combine_streams(panel, {'a': 0.6, 'b': 0.6})
        with pytest.raises(ValueError, match='legs'):
            combine_streams(panel, {'a': 1.0})

    def test_drawdown_and_subadditivity_fixture(self) -> None:
        """The V-shaped curve's max DD equals the hand-computed value, and
        an anti-correlated pair's combined DD sits strictly below the
        weighted average of leg DDs — the gap-size fixture."""
        from common.portfolio import align_streams, combine_streams, max_drawdown_pct
        # V shape: up to 102K, down to 98K, recover. Peak 102K -> trough 98K.
        v = _mini_stream(self.D, [100_000, 102_000, 98_000, 99_000, 101_000])
        panel_v = align_streams({'v': v})
        assert max_drawdown_pct(panel_v['v']) == pytest.approx(
            (102_000 - 98_000) / 102_000 * 100, abs=0.01)
        # anti-correlated legs: each has a real DD; the 50/50 combo is flat.
        a = _mini_stream(self.D, [100_000, 101_000, 99_000, 101_000, 99_000])
        b = _mini_stream(self.D, [100_000, 99_000, 101_000, 99_000, 101_000])
        panel = align_streams({'a': a, 'b': b})
        combo = combine_streams(panel, {'a': 0.5, 'b': 0.5})
        dd_a = max_drawdown_pct(panel['a'])
        dd_b = max_drawdown_pct(panel['b'])
        assert max_drawdown_pct(combo) == pytest.approx(0.0, abs=1e-9)
        assert max_drawdown_pct(combo) < 0.5 * dd_a + 0.5 * dd_b


_GAP_G_FILES = [
    'spy_option_dailies.csv', 'qqq_option_dailies.csv',
    'qqq_option_dailies_2011_2016.csv', 'msft_option_dailies.csv',
    'msft_option_dailies_2008_2016.csv',
]


def _have_gap_g() -> bool:
    import os
    from common.paths import data_path
    return all(os.path.exists(str(data_path(f))) or os.path.exists(str(data_path(f)) + '.gz')
               for f in _GAP_G_FILES)


@pytest.mark.skipif(not _have_gap_g(),
                    reason='needs SPY/QQQ/MSFT dailies + era backfills (data-2026-06 release)')
class TestPortfolioCombos:
    """Experiment 6's pinned verdicts (docs/van_tharp_gap_g.md). EXPLORATORY
    — descriptive, kill-or-justify; the scout's internal drift alarm asserts
    every leg's published pin before any combo number is computed.

    Combo A (noncorrelated systems, Loc 1932): the correlation came back LOW
    (0.20 — the different-drivers construction held) but the dead leg earned
    nothing: combined NW t 1.54 vs the better leg's common-span 2.47, and
    the combined percent-of-peak DD (34.31%) sits ABOVE the 50/50 weighted
    average (24.04%) — the dollar-space subadditivity did NOT survive the
    percent transform, because the CC leg's compounding beta dominates the
    book's later, larger peaks. Variance reduction did not earn the
    negative-edge leg its place.

    Combo B (independent markets, Loc 1929): SPY~QQQ correlation 0.656 (the
    shared vol factor, as pre-stated), the MSFT pairs ~0.2; combined NW t
    1.01 vs best-single 2.56 — the claim killed on this cross-section, with
    the correlation and the one negative leg each costing as predicted. The
    combined DD (19.53%) does sit under the weighted average (27.07%) here —
    the subadditive gap is visible when no leg compounds a stock position —
    but stays above the best single leg (6.94%).
    """

    @pytest.fixture(scope='class')
    def scout(self):
        return portfolio_scout()

    def test_combo_a_systems_claim(self, scout) -> None:
        a = scout['combo_a']
        assert a['span'] == ['2016-04-12', '2026-04-10']
        assert a['n_days'] == 2514
        assert a['correlations']['spy_sv~msft_cc'] == pytest.approx(0.1975, abs=0.005)
        assert a['legs']['spy_sv']['nw_t'] == pytest.approx(2.47, abs=0.02)
        assert a['legs']['spy_sv']['max_dd_pct'] == pytest.approx(7.08, abs=0.05)
        assert a['legs']['msft_cc']['nw_t'] == pytest.approx(1.43, abs=0.02)
        assert a['legs']['msft_cc']['max_dd_pct'] == pytest.approx(40.99, abs=0.05)
        assert a['combined']['nw_t'] == pytest.approx(1.54, abs=0.02)
        assert a['combined']['max_dd_pct'] == pytest.approx(34.31, abs=0.05)
        assert a['weighted_avg_leg_dd'] == pytest.approx(24.04, abs=0.05)
        # the pre-stated what-counts, recorded as booleans:
        assert a['combined']['nw_t'] < a['legs']['spy_sv']['nw_t']      # no improvement
        assert a['combined']['max_dd_pct'] > a['weighted_avg_leg_dd']   # percent form broke subadditivity

    def test_combo_b_markets_claim(self, scout) -> None:
        b = scout['combo_b']
        assert b['span'] == ['2011-03-24', '2026-04-10']
        assert b['n_days'] == 3784
        assert b['correlations']['spy_sv~qqq_sv'] == pytest.approx(0.656, abs=0.005)
        assert b['correlations']['spy_sv~msft_sv'] == pytest.approx(0.1925, abs=0.005)
        assert b['correlations']['qqq_sv~msft_sv'] == pytest.approx(0.2248, abs=0.005)
        assert b['legs']['spy_sv']['nw_t'] == pytest.approx(2.56, abs=0.02)
        assert b['legs']['qqq_sv']['nw_t'] == pytest.approx(2.45, abs=0.02)
        assert b['legs']['msft_sv']['nw_t'] == pytest.approx(-0.28, abs=0.02)
        assert b['combined']['nw_t'] == pytest.approx(1.01, abs=0.02)
        assert b['combined']['max_dd_pct'] == pytest.approx(19.53, abs=0.05)
        assert b['weighted_avg_leg_dd'] == pytest.approx(27.07, abs=0.05)
        assert b['best_single_leg_dd'] == pytest.approx(6.94, abs=0.05)
        # the pre-stated kill condition: combined at or below the best single leg.
        assert b['combined']['nw_t'] < b['legs']['spy_sv']['nw_t']
        # and the subadditive gap IS visible here (no compounding stock leg):
        assert b['combined']['max_dd_pct'] < b['weighted_avg_leg_dd']
