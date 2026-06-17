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

from explorations import (
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
