"""Tests for search/wing_premium.py (docs/wing_premium_diagnostic_plan.md).

Two layers, the house pattern:

- An always-run synthetic layer for the frozen machinery: the IV round-trip
  guard, the point-in-time property of the conditioning percentile
  (assignment at t must never change when future days are appended), the
  non-overlap invariant of the cycle sampler, the seeded placebo null, and
  the small numerics (Spearman, RSV+, N(d2)) against hand values.
- A dataset-gated `TestWingPremiumDiagnostic` regression pinning the one
  measurement run's decisive numbers (added with the run, plan §12).

Epistemic status: EXPLORATORY measurement — kill-or-justify, no strategy
sample, no idea-ledger rows; a LIVE read licenses a registration only.
"""

from __future__ import annotations

import math
import os

import pytest

from common.paths import data_path
from realchains.vol_premium import bs_price
import search.wing_premium as wp
from search.wing_premium import (
    PCTL_MIN,
    WING_PLACEBO_SEED,
    WING_RF,
    breach_prob_n_d2,
    pit_percentile,
    placebo_p,
    quintile_table,
    realized_vol,
    rsv_plus,
    sample_cycles,
    select_legs,
    spearman,
    _leg_iv,
)

_SPY = data_path('spy_option_dailies.csv')
_QQQ = data_path('qqq_option_dailies.csv')
_HAVE_DATA = all(os.path.exists(p) or os.path.exists(p + '.gz')
                 for p in (_SPY, _QQQ))


def _cand(dte: int, delta: float, strike: float, sigma: float, spot: float = 100.0,
          cid: str | None = None, exp: str = '2020-02-15') -> tuple:
    """A candidate tuple priced exactly at Black-Scholes: mid == bs_price, the
    quote straddling it by a nickel."""
    px = bs_price('call', spot, strike, dte / 365.0, WING_RF, sigma)
    return (dte, delta, px - 0.05, px + 0.05, px, exp, strike,
            cid or f'C{strike:g}D{dte}')


class TestNumerics:
    def test_spearman_hand_values(self) -> None:
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
        # Ties share average ranks: x has a tie, monotone y — rho < 1 but > 0.
        rho = spearman([1, 1, 2, 3], [1, 2, 3, 4])
        assert 0.7 < rho < 1.0
        assert spearman([1, 2, 3], [5, 5, 5]) == 0.0     # zero-variance guard

    def test_rsv_plus_and_rv_hand_values(self) -> None:
        rets = [0.01, -0.02, 0.03, 0.0]
        assert rsv_plus(rets) == pytest.approx(
            math.sqrt(252 / 4 * (0.01 ** 2 + 0.03 ** 2)))
        assert realized_vol(rets) == pytest.approx(
            math.sqrt(252 / 4 * (0.01 ** 2 + 0.02 ** 2 + 0.03 ** 2)))

    def test_n_d2_hand_value(self) -> None:
        # S=K=100, T=1, rf=0, sigma=0.2: d2 = -0.1, N(-0.1) ~ 0.4602.
        assert breach_prob_n_d2(100, 100, 1.0, 0.0, 0.2) == pytest.approx(
            0.4602, abs=0.0005)


class TestLegSelection:
    def test_frozen_rules(self) -> None:
        day = {'candidates': [
            _cand(28, 0.52, 100, 0.25), _cand(28, 0.27, 110, 0.22),
            _cand(28, 0.10, 120, 0.24),
            _cand(63, 0.50, 100, 0.25, cid='FAR'),
        ]}
        atm, wing = select_legs(day)
        assert atm[6] == 100 and wing[6] == 110      # nearest deltas, 28-DTE cohort
        # DTE tie toward the smaller: 28 vs 32 straddle 30 equally.
        day2 = {'candidates': [_cand(28, 0.5, 100, 0.25), _cand(28, 0.25, 110, 0.22),
                               _cand(32, 0.5, 100, 0.25), _cand(32, 0.25, 110, 0.22)]}
        atm2, _ = select_legs(day2)
        assert atm2[0] == 28
        # bid > 0 hygiene: a zero-bid wing drops out of the cohort.
        day3 = {'candidates': [_cand(28, 0.5, 100, 0.25),
                               (28, 0.25, 0.0, 0.4, 0.2, '2020-02-15', 110, 'Z')]}
        assert select_legs(day3) is None             # one strike matches both
        # One strike matching both targets -> None.
        assert select_legs({'candidates': [_cand(28, 0.4, 100, 0.25)]}) is None

    def test_hygiene_applies_within_the_chosen_expiry(self) -> None:
        # Plan section-3 step order: the nearest LISTED expiry is chosen
        # before hygiene — a fully zero-bid nearest expiry fails the day,
        # never falling through to a farther expiry.
        near_dead = [(28, 0.5, 0.0, 1.0, 0.5, '2020-02-15', 100, 'A'),
                     (28, 0.25, 0.0, 0.5, 0.25, '2020-02-15', 110, 'B')]
        far_live = [_cand(45, 0.5, 100, 0.25, cid='C'),
                    _cand(45, 0.25, 110, 0.22, cid='D')]
        assert select_legs({'candidates': near_dead + far_live}) is None

    def test_iv_roundtrip_guard(self, monkeypatch) -> None:
        leg = _cand(30, 0.27, 110, 0.22)
        iv = _leg_iv(leg, 100.0)
        assert iv == pytest.approx(0.22, abs=0.001)  # recovers the pricing vol
        # A mid below intrinsic has no IV -> failed leg.
        bad = (30, 0.9, 1.0, 1.2, 1.1, '2020-02-15', 50, 'BAD')
        assert _leg_iv(bad, 100.0) is None
        # The section-3.6 repricing guard itself: force implied_vol to
        # return a WRONG vol — the round-trip must reject the leg.
        # (Bisection at tol=1e-7 makes this branch unreachable through the
        # real solver; the guard is frozen defense-in-depth, so pin it
        # directly.)
        monkeypatch.setattr(wp, 'implied_vol', lambda *a, **k: 0.99)
        assert _leg_iv(leg, 100.0) is None


class TestPointInTimePercentile:
    def test_min_history_and_window(self) -> None:
        series = list(range(300))
        out = pit_percentile(series, window=756, min_obs=PCTL_MIN)
        assert all(v is None for v in out[:PCTL_MIN - 1])
        assert out[PCTL_MIN - 1] == pytest.approx(1.0)   # max of its window
        # A strictly increasing series pins every later value at 1.0.
        assert out[-1] == pytest.approx(1.0)

    def test_appending_future_never_changes_the_past(self) -> None:
        base = [float((i * 37) % 101) for i in range(400)]
        a = pit_percentile(base, window=300, min_obs=50)
        b = pit_percentile(base + [999.0, -999.0, 3.0], window=300, min_obs=50)
        assert a == b[:len(a)]

    def test_window_caps_history(self) -> None:
        # The window must genuinely roll old values OUT: a value that ranks
        # mid-pack against full history ranks BOTTOM in its window.
        series = [float(v) for v in range(200)] + [100.5]
        windowed = pit_percentile(series, window=50, min_obs=10)
        full = pit_percentile(series, window=10_000, min_obs=10)
        assert windowed[-1] == pytest.approx(0.0)     # smallest of [151..199]
        assert full[-1] == pytest.approx(101 / 200, abs=0.01)

    def test_positions_window_counts_trading_days(self) -> None:
        # Plan section-3.5: the window is TRADING days, not observations. Two
        # observations 100 trading days apart fall out of a 50-day window
        # even though they are adjacent in the series.
        series = [1.0, 2.0, 3.0]
        dense = pit_percentile(series, positions=[0, 1, 2], window=50, min_obs=2)
        sparse = pit_percentile(series, positions=[0, 100, 200], window=50,
                                min_obs=1)
        assert dense[2] == pytest.approx(1.0)         # sees all three
        # Alone in its window: a percentile needs >= 2 values, so the
        # observation is starved to None (never a 0/0), whatever min_obs.
        assert sparse[2] is None
        wide = pit_percentile(series, positions=[0, 100, 200], window=300,
                              min_obs=2)
        assert wide[2] == pytest.approx(1.0)          # window admits all three


class TestCycleSampler:
    DATES = [f'2020-01-{d:02d}' for d in range(1, 32)]   # calendar-like labels

    def _signal(self, entry_dates: list[str], exp: str | None = None,
                strike: float = 110.0) -> dict[str, dict]:
        """Each entry carries a rolling ~9-day-out expiration (the realistic
        shape) unless a fixed `exp` is forced."""
        out = {}
        for d in entry_dates:
            i = self.DATES.index(d)
            e = exp if exp is not None else self.DATES[min(i + 9, len(self.DATES) - 1)]
            out[d] = {'spread': -0.03, 'atm_iv': 0.25, 'wing_iv': 0.22,
                      'atm_delta': 0.5, 'wing_delta': 0.25,
                      'atm_miss': 0.0, 'wing_miss': 0.0,
                      'wing_strike': strike,
                      'wing_dte': 10, 'expiration': e, 'atm_cid': 'A',
                      'spot': 100.0}
        return out

    def test_non_overlap_and_settlement(self) -> None:
        closes = {d: 100.0 + i for i, d in enumerate(self.DATES)}
        signal = self._signal(self.DATES[:20])
        pcts = {d: 0.5 for d in self.DATES[:20]}
        cycles, skips, tail = sample_cycles(signal, pcts, self.DATES,
                                            closes, {})
        assert len(cycles) >= 2 and skips == 0 and tail == 0
        for a, b in zip(cycles, cycles[1:]):
            assert b['entry'] > a['settle']          # strict non-overlap
        assert cycles[0]['settle'] == self.DATES[9]  # last day <= expiration

    def test_gap_skips_are_counted(self) -> None:
        closes = {d: 100.0 for d in self.DATES}
        del closes[self.DATES[5]]                    # a hole in the closes
        signal = self._signal([self.DATES[0]], exp=self.DATES[9])
        cycles, skips, _ = sample_cycles(signal, {self.DATES[0]: 0.5},
                                         self.DATES, closes, {})
        assert cycles == [] and skips == 1

    def test_signal_failure_days_count_as_skips_after_warmup(self) -> None:
        """Plan section-4: once the first valid entry exists, a trading day
        sought for entry without a clean signal is a counted skip; the
        pre-warm-up days never are."""
        closes = {d: 100.0 + i for i, d in enumerate(self.DATES)}
        # Valid signal only on days 3 and 20; the walk seeks entry on days
        # 13..19 (post-settlement of the first cycle) and finds nothing.
        signal = self._signal([self.DATES[3], self.DATES[20]])
        pcts = {self.DATES[3]: 0.5, self.DATES[20]: 0.5}
        cycles, skips, _ = sample_cycles(signal, pcts, self.DATES, closes, {})
        assert len(cycles) == 2
        # First cycle settles DATES[12]; days 13..19 are 7 failed attempts.
        assert skips == 7

    def test_truncated_tail_cycle_is_dropped(self) -> None:
        """An entry whose expiration lies beyond the data end is DROPPED
        (tail_dropped), never measured on a truncated window."""
        closes = {d: 100.0 for d in self.DATES}
        beyond = '2020-02-15'
        signal = self._signal([self.DATES[0]], exp=beyond)
        cycles, skips, tail = sample_cycles(signal, {self.DATES[0]: 0.5},
                                            self.DATES, closes, {})
        assert cycles == [] and tail == 1 and skips == 0

    def test_breach_flags_use_strike_space_highs(self) -> None:
        closes = {d: 100.0 for d in self.DATES}
        closes[self.DATES[9]] = 111.0                # settle above the strike
        signal = self._signal([self.DATES[0]], exp=self.DATES[9])
        # Split-scaled highs: a 10:1-style factor applied upstream — here a
        # high of 115 in strike space mid-window.
        highs = {self.DATES[4]: 115.0}
        cycles, _, _ = sample_cycles(signal, {self.DATES[0]: 0.5},
                                     self.DATES, closes, highs)
        assert cycles[0]['breach_terminal'] is True
        assert cycles[0]['breach_maxhigh'] is True

    def test_quintile_boundaries(self) -> None:
        cycles = [{'pct': p, 'premium': 1.0, 'implied_breach': 0.2,
                   'breach_terminal': False, 'breach_maxhigh': False}
                  for p in (0.0, 0.19, 0.2, 0.99, 1.0)]
        tab = quintile_table(cycles)
        assert tab['Q1']['n'] == 2 and tab['Q2']['n'] == 1
        assert tab['Q5']['n'] == 2                   # 0.99 and the pct==1.0 edge


class TestPlaceboNull:
    def test_monotone_relation_is_extreme(self) -> None:
        pcts = [i / 99 for i in range(100)]
        prem = [i * 0.001 for i in range(100)]
        rho, p = placebo_p(pcts, prem, n_shifts=200, seed=WING_PLACEBO_SEED)
        assert rho == pytest.approx(1.0)
        assert p == pytest.approx(1 / 201, abs=1e-9)  # no shift ties |rho|=1

    def test_noise_is_unremarkable_and_deterministic(self) -> None:
        rng_vals = [((i * 73) % 97) / 97 for i in range(80)]
        prem = [((i * 41) % 89) / 89 for i in range(80)]
        rho1, p1 = placebo_p(rng_vals, prem, n_shifts=200)
        rho2, p2 = placebo_p(rng_vals, prem, n_shifts=200)
        assert (rho1, p1) == (rho2, p2)              # seeded determinism
        assert p1 > 0.05                             # unstructured -> flat


@pytest.mark.skipif(not _HAVE_DATA,
                    reason='needs spy/qqq option dailies (or their .gz twins)')
class TestWingPremiumDiagnostic:
    """The one measurement run of docs/wing_premium_diagnostic_plan.md,
    pinned (2026-07-20). Verdict under the frozen §5 rule: **H-FLAT — the
    conditioning family closes.**

    - SPY alone shows a strongly negative state-dependence (rho −0.265,
      placebo-p 0.001: RICHER wings forecast SMALLER premiums — the
      informed-flow direction, refuting the rich-wing/overreaction gate),
      robust across its §8 variants (wing20 −0.195/p .023, wing30
      −0.276/.001, window504 −0.259/.001; weaker on full RV −0.134/.118).
      QQQ does NOT confirm (−0.105, p 0.389), and the two-ticker rule exists
      precisely so a single-ticker signal cannot become a belief.
    - The wing premium EXISTS on average everywhere (QQQ +3.0 / SPY +1.8
      vol points; MSFT +4.5; NVDA +10.5) — consistent with the +2.54
      family — but its STATE-dependence is unconfirmed, which is what the
      gates needed.
    - The probability table kills the cheap-wing gate by itself: Q1 (cheap
      wing) realized terminal breach 54.1% (QQQ) / 55.6% (SPY) against
      ~24% implied — cheap upside insurance is run over at TWICE its priced
      rate on both verdict tickers.
    - NVDA runs the OPPOSITE sign (+0.217, p 0.001, d_rich +0.048) — the
      single-name overreaction/speculative-flow contrast, exploratory only.

    All §9 rails passed: ATM-vs-vendor cross-checks 0.986–0.998, coverage
    ≥ 0.97, delta misses ≤ 0.02 median, no demotions; verdict rails_ok.
    Runtime note: the class fixture re-runs the full diagnostic (four store
    loads dominate, ~13 minutes) — the `wing` CI bucket is the widest.
    """

    @pytest.fixture(scope='class')
    def res(self) -> dict:
        from search.wing_premium import run_diagnostic
        return run_diagnostic()

    def test_verdict_h_flat(self, res) -> None:
        v = res['verdict']
        assert v['live'] is False
        assert v['frozen_rule_live'] is False
        assert v['rails_ok'] is True
        assert 'H-flat' in v['reading']

    def test_qqq_primary(self, res) -> None:
        r = res['tickers']['QQQ']
        assert r['cycles'] == 153 and r['tail_dropped'] == 5
        assert r['rho'] == pytest.approx(-0.1051, abs=0.002)
        assert r['placebo_p'] == pytest.approx(0.3886, abs=0.02)
        assert r['mean_premium'] == pytest.approx(0.0297, abs=0.001)
        assert r['d_rich'] == pytest.approx(-0.0143, abs=0.001)
        assert r['atm_cross_check'] >= 0.99
        assert r['demoted'] is False
        assert r['span'] == ['2012-03-21', '2026-05-29']

    def test_spy_primary_and_robustness(self, res) -> None:
        r = res['tickers']['SPY']
        assert r['cycles'] == 168
        assert r['rho'] == pytest.approx(-0.2651, abs=0.002)
        assert r['placebo_p'] == pytest.approx(0.001, abs=0.003)
        assert r['mean_premium'] == pytest.approx(0.0176, abs=0.001)
        assert r['demoted'] is False
        rob = res['robustness']['SPY']
        assert rob['wing30']['rho'] == pytest.approx(-0.276, abs=0.005)
        assert rob['window504']['rho'] == pytest.approx(-0.259, abs=0.005)
        assert rob['rv']['placebo_p'] > 0.05          # weaker on full RV
        # QQQ's variants never clear 0.05 — the non-confirmation is robust.
        assert all(v['placebo_p'] > 0.05 for v in res['robustness']['QQQ'].values())

    def test_cheap_wing_breach_miscalibration(self, res) -> None:
        """Q1 realized breach ~2x implied on BOTH verdict tickers — the
        cheap-wing gate's own funeral, independent of rho."""
        for t, breach in (('QQQ', 0.5405), ('SPY', 0.5556)):
            q1 = res['tickers'][t]['quintiles']['Q1']
            assert q1['real_breach'] == pytest.approx(breach, abs=0.01)
            assert q1['implied_breach'] == pytest.approx(0.24, abs=0.01)
            assert q1['real_breach'] > 2 * q1['implied_breach'] * 0.95

    def test_exploratory_contrast(self, res) -> None:
        msft, nvda = res['tickers']['MSFT'], res['tickers']['NVDA']
        assert msft['rho'] == pytest.approx(0.033, abs=0.002)
        assert msft['placebo_p'] > 0.5
        assert nvda['rho'] == pytest.approx(0.2167, abs=0.002)
        assert nvda['placebo_p'] == pytest.approx(0.001, abs=0.003)
        assert nvda['d_rich'] == pytest.approx(0.0482, abs=0.002)

    def test_premium_exists_on_average_everywhere(self, res) -> None:
        for t in ('QQQ', 'SPY', 'MSFT', 'NVDA'):
            assert res['tickers'][t]['mean_premium'] > 0
