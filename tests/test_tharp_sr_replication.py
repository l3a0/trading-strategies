"""Tests for engine/tharp_sr_replication.py — the frozen S/R replication.

Always-run synthetic layer (plan §8): range-position edge cases and tie
handling, each signal's firing rule on hand-built series (warm-up
exclusion, the Wilder recurrence against hand-computed values), the
flat-only book, the fixed-at-entry Turtle Soup level, null determinism,
and the small statistical helpers against hand-derived values. The
result-pin class runs on the committed OHLC CSVs — in CI, no dataset
gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.tharp_sr_replication import (
    PRIMARY_TICKERS,
    addone_p_one_sided,
    addone_p_two_sided,
    build_trades,
    cell_run,
    era_index,
    fisher_exact_one_sided,
    phase1_measurements,
    phase1_verdicts,
    phase2_survival,
    signal_fires,
    turtle_soup,
    wilder_rsi,
    wilson_interval,
)


def mk(dates, o, h, low, c):
    return {'dates': np.array(dates), 'open': np.array(o, dtype=float),
            'high': np.array(h, dtype=float), 'low': np.array(low, dtype=float),
            'close': np.array(c, dtype=float)}


def flat_dates(n, years=('2005', '2015', '2025')):
    """n dates spread across the three era panels."""
    per = n // 3 + 1
    out = []
    for y in years:
        for k in range(per):
            out.append(f'{y}-{1 + k // 28:02d}-{1 + k % 28:02d}')
    return out[:n]


def p1(d, ticker='SYN'):
    return phase1_measurements(d, ticker)['measurements']


class TestPhase1Counting:
    def test_degenerate_range_excluded_and_tie_counts_against(self):
        # day 0: top-quartile close, next open EXACTLY at the high (tie ->
        # against the strict claim); day 1: high == low (excluded)
        d = mk(['2005-01-03', '2005-01-04', '2005-01-05'],
               o=[100, 110, 100], h=[110, 110, 101],
               low=[100, 110, 99], c=[109, 110, 100])
        m = p1(d)
        assert m['C1_strict_top']['n'] == 1          # only day 0 conditions
        assert m['C1_strict_top']['rate'] == 0.0     # 110 > 110 is False
        assert m['C1_loose_top']['rate'] == 1.0      # 110 > 109

    def test_c3_trending_day_classification(self):
        # day 0 opens in the bottom quartile and closes in the top ->
        # trending-up; next session closes below its open -> headline hit
        d = mk(['2005-01-03', '2005-01-04', '2005-01-05'],
               o=[101, 112, 108], h=[110, 113, 109],
               low=[100, 107, 107], c=[109, 108, 108])
        m = p1(d)
        assert m['C3_up']['n'] == 1
        assert m['C3_up']['rate'] == 1.0             # 108 < 112
        # gap-continuation variant conditions on next open > close: 112 > 109
        assert m['C3_up_gapcont']['n'] == 1

    def test_era_stratified_null_is_deterministic(self):
        n = 90
        dates = flat_dates(n)
        rng = np.random.default_rng(7)
        c = 100 + np.cumsum(rng.normal(0, 1, n))
        h = c + 1.0
        low = c - 1.0
        o = c - 0.5
        d = mk(dates, o, h, low, c)
        m1 = p1(d)
        m2 = p1(d)
        assert m1['C1_strict_top']['p'] == m2['C1_strict_top']['p']

    def test_era_assignment_by_conditioning_date(self):
        assert era_index('1999-11-01') == 0
        assert era_index('2009-12-31') == 0
        assert era_index('2010-01-01') == 1
        assert era_index('2026-06-30') == 2

    def test_bottom_mirrors_and_c2_counting(self):
        # day 0: bottom-quartile close; next open EXACTLY at the low (tie
        # -> against strict); next close below today's close -> C2_bot hit
        d = mk(['2005-01-03', '2005-01-04', '2005-01-05'],
               o=[109, 100, 100], h=[110, 104, 101],
               low=[100, 99, 99], c=[101, 100.5, 100])
        m = p1(d)
        assert m['C1_strict_bot']['n'] == 1          # day 1's rp is 0.3: excluded
        assert m['C1_strict_bot']['rate'] == 0.0     # 100 < 100 is False (tie)
        assert m['C1_loose_bot']['rate'] == 1.0      # 100 < 101
        assert m['C2_bot']['rate'] == 1.0            # 100.5 < 101
        assert m['C2_top']['n'] == 0                 # no top-quartile day

    def test_c3_down_trending_day(self):
        # day 0 opens top-quartile, closes bottom-quartile -> trending-down;
        # next session closes above its open -> C3_dn hit
        d = mk(['2005-01-03', '2005-01-04', '2005-01-05'],
               o=[109, 95, 100], h=[110, 99, 101],
               low=[100, 94, 99], c=[101, 98, 100])
        m = p1(d)
        assert m['C3_dn']['n'] == 1
        assert m['C3_dn']['rate'] == 1.0             # 98 > 95

    def test_degenerate_diagnostic_counted(self):
        d = mk(['2005-01-03', '2005-01-04', '2005-01-05'],
               o=[100, 110, 100], h=[110, 110, 101],
               low=[100, 110, 99], c=[109, 110, 100])
        out = phase1_measurements(d, 'SYN')
        assert out['n_degenerate'] == 1              # day 1: high == low
        assert out['era_starts'][0] == '2005-01-03'

    def test_null_is_era_stratified(self):
        # era 0 outcomes all True, era 2 all False; the conditional set sits
        # entirely in era 0 -> a stratified null reproduces rate 1.0 every
        # resample (p capped at 1.0); an unstratified null would mix zeros
        n = 60
        dates = flat_dates(n, years=('2005', '2005', '2025'))
        c = np.full(n, 109.0)
        c[30:] = 101.0                               # later days: low rp
        h = np.full(n, 110.0)
        low = np.full(n, 100.0)
        o = np.full(n, 105.0)
        # EVERY era-0 day's next open exceeds the high (days 0..41 are the
        # 2005 panel under flat_dates), so the era-0 universe is all-True;
        # the era-2 outcomes stay False and must never be drawn
        o2 = o.copy()
        o2[1:43] = 111.0
        d = mk(dates, o2, h, low, c)
        m = p1(d)
        assert m['C1_strict_top']['n'] == 30
        assert m['C1_strict_top']['rate'] == 1.0
        assert m['C1_strict_top']['p'] == 1.0        # stratified null matches exactly

    def test_null_stream_folds_the_ticker(self):
        from engine.tharp_sr_replication import derived_rng
        a = derived_rng('QQQ|C1_strict_top').permutation(50)
        b = derived_rng('SPY|C1_strict_top').permutation(50)
        assert not np.array_equal(a, b)


class TestStatHelpers:
    def test_addone_two_sided(self):
        null = np.array([0.1, 0.2, 0.3, 0.4])
        # observed above all four: hi=0, lo=4 -> p = 2 * (1/5) = 0.4
        assert addone_p_two_sided(null, 0.9) == pytest.approx(0.4)

    def test_addone_one_sided(self):
        null = np.array([0.1, 0.2, 0.3, 0.4])
        assert addone_p_one_sided(null, 0.35) == pytest.approx((1 + 1) / 5)

    def test_wilson_hand_value(self):
        lo, hi = wilson_interval(8, 10)
        assert lo == pytest.approx(0.4901, abs=1e-3)
        assert hi == pytest.approx(0.9433, abs=1e-3)

    def test_fisher_hand_value(self):
        # margins 5/5, k1=1, k2=4: P(K2 >= 4) = (25 + 1) / 252
        assert fisher_exact_one_sided(1, 5, 4, 5) == pytest.approx(26 / 252)


class TestSignals:
    def test_cb_excludes_entry_day_high_and_warmup(self):
        n = 30
        h = np.full(n, 100.0)
        c = np.full(n, 99.0)
        h[25] = 105.0
        c[25] = 104.0      # close above the PRIOR 20-day high (100)
        d = mk(flat_dates(n), c - 0.5, h, c - 2.0, c)
        fires = signal_fires(d, 'CB-20', 'long')
        assert list(fires) == [25]                   # its own 105 high not used
        # warm-up: an early breakout day below index 20 cannot fire
        h2 = h.copy()
        c2 = c.copy()
        h2[10] = 105.0
        c2[10] = 104.0
        d2 = mk(flat_dates(n), c2 - 0.5, h2, c2 - 2.0, c2)
        assert 10 not in signal_fires(d2, 'CB-20', 'long')

    def test_ma200_cross_fires_once(self):
        n = 260
        c = np.full(n, 100.0)
        c[250:] = 130.0                              # jump above the SMA once
        d = mk(flat_dates(n), c, c + 1, c - 1, c)
        fires = signal_fires(d, 'MA-200', 'long')
        assert list(fires) == [250]

    def test_wilder_rsi_hand_recurrence(self):
        # 14 unit gains: RSI[14] = 100; one unit loss:
        # avg_g = 13/14, avg_l = 1/14 -> RSI = 100 - 100/14
        c = np.array([100.0 + i for i in range(15)] + [113.0])
        r = wilder_rsi(c)
        assert np.isnan(r[13])
        assert r[14] == pytest.approx(100.0)
        assert r[15] == pytest.approx(100.0 - 100.0 / 14.0)

    def test_rsi_cross_is_the_30_recross(self):
        rng = np.random.default_rng(3)
        c = 100 * np.cumprod(1 + np.concatenate([
            rng.normal(0, 0.002, 30), np.full(20, -0.03),   # dive: RSI -> low
            np.full(10, 0.03), rng.normal(0, 0.002, 30)]))   # rebound through 30
        d = mk(flat_dates(len(c)), c, c + 0.5, c - 0.5, c)
        fires = signal_fires(d, 'RSI-30', 'long')
        r = wilder_rsi(c)
        expected = [i for i in range(1, len(c))
                    if not np.isnan(r[i]) and not np.isnan(r[i - 1])
                    and r[i - 1] < 30.0 <= r[i]]
        assert list(fires) == expected and len(fires) >= 1

    def test_gx_cross_fires_once(self):
        # flat forever, then a ramp: SMA50 crosses above SMA200 exactly once
        n = 320
        c = np.full(n, 100.0)
        c[250:] = np.linspace(101, 180, n - 250)
        d = mk(flat_dates(n), c, c + 1, c - 1, c)
        fires = signal_fires(d, 'GX', 'long')
        assert len(fires) == 1
        from engine.tharp_sr_replication import sma
        f, s = sma(c, 50), sma(c, 200)
        i = fires[0]
        assert f[i] > s[i] and f[i - 1] <= s[i - 1]

    @pytest.mark.parametrize('nb', [20, 40, 100])
    def test_cb_warmup_boundary(self, nb):
        # a breakout at exactly index nb fires; at nb-1 it cannot
        n = nb + 10
        for spike, should_fire in ((nb, True), (nb - 1, False)):
            h = np.full(n, 100.0)
            c = np.full(n, 99.0)
            h[spike] = 105.0
            c[spike] = 104.0
            d = mk(flat_dates(n), c - 0.5, h, c - 2.0, c)
            fires = signal_fires(d, f'CB-{nb}', 'long')
            assert (spike in fires) is should_fire

    def test_short_mirrors(self):
        # CB-20 short: close below the prior 20-day low (entry-day excluded)
        n = 30
        low = np.full(n, 100.0)
        c = np.full(n, 101.0)
        low[25] = 94.0
        c[25] = 95.0
        d = mk(flat_dates(n), c + 0.5, c + 1.0, low, c)
        assert list(signal_fires(d, 'CB-20', 'short')) == [25]
        # MA-200 down-cross fires once
        n2 = 260
        c2 = np.full(n2, 100.0)
        c2[250:] = 70.0
        d2 = mk(flat_dates(n2), c2, c2 + 1, c2 - 1, c2)
        assert list(signal_fires(d2, 'MA-200', 'short')) == [250]
        # RSI short: the 70 recross-down, checked against the scalar rule
        rng = np.random.default_rng(5)
        c3 = 100 * np.cumprod(1 + np.concatenate([
            rng.normal(0, 0.002, 30), np.full(20, 0.03),
            np.full(10, -0.03), rng.normal(0, 0.002, 30)]))
        d3 = mk(flat_dates(len(c3)), c3, c3 + 0.5, c3 - 0.5, c3)
        r = wilder_rsi(c3)
        expected = [i for i in range(1, len(c3))
                    if not np.isnan(r[i]) and not np.isnan(r[i - 1])
                    and r[i - 1] > 70.0 >= r[i]]
        assert list(signal_fires(d3, 'RSI-30', 'short')) == expected
        assert len(expected) >= 1


class TestTradeBuilder:
    def test_flat_only_lockout_and_reentry_at_exit(self):
        # fires at 10, 12, 15, 30 with H=5: 12 is locked out, 15 == 10+5
        # re-enters at the exit close, 30 enters
        assert build_trades([10, 12, 15, 30], 5, 100) == [10, 15, 30]

    def test_end_of_span_skip(self):
        assert build_trades([96], 5, 100) == []      # exit would be day 101
        assert build_trades([94], 5, 100) == [94]

    def test_cell_no_verdict_floor(self):
        n = 260
        c = np.full(n, 100.0)
        c[250:] = 130.0
        d = mk(flat_dates(n), c, c + 1, c - 1, c)
        cell = cell_run(d, 'MA-200', 'long', 5, 'SYN', b=50)
        assert cell['n_trades'] == 1
        assert cell['verdict'] == 'NO-VERDICT'

    def test_cell_null_deterministic_and_wins_on_planted_edge(self):
        # a rising staircase: each tooth breaks the prior 20-day high and
        # is followed by a guaranteed five-session rise
        n = 400
        c = np.zeros(n)
        level = 100.0
        for i in range(n):
            if i >= 30 and (i - 30) % 12 == 0 and i + 6 < n:
                level += 60.0
                c[i] = level + 30.0                  # the only firing day
            else:
                # drifts just under the tooth's high (+30.5): no follow-
                # through fires, and the H=5 exit lands above the entry
                c[i] = level + (30.4 if level > 100.0 else 0.0)
        d = mk(flat_dates(n), c, c + 0.5, c - 0.5, c)
        c1 = cell_run(d, 'CB-20', 'long', 5, 'SYN', b=300)
        c2 = cell_run(d, 'CB-20', 'long', 5, 'SYN', b=300)
        assert c1['p'] == c2['p']                    # derived seed determinism
        assert c1['n_trades'] >= 15
        assert c1['win_rate'] == 1.0
        # the null must track the drift wall, and a planted 100% edge on a
        # mostly-flat tape must BEAT it — the null engine's end-to-end pin
        assert c1['null_rate_mean'] == pytest.approx(c1['base_rate'], abs=0.05)
        assert c1['verdict'] == 'BEATS' and c1['p'] <= 0.01

    def test_short_cell_negates_returns(self):
        # a falling staircase: every short entry is followed by a fall
        n = 400
        c = np.zeros(n)
        level = 10000.0
        for i in range(n):
            if i >= 30 and (i - 30) % 12 == 0 and i + 6 < n:
                level -= 60.0
                c[i] = level - 30.0                  # breakdown day
            else:
                c[i] = level - (30.4 if level < 10000.0 else 0.0)
        d = mk(flat_dates(n), c, c + 0.5, c - 0.5, c)
        cell = cell_run(d, 'CB-20', 'short', 5, 'SYN', b=300)
        assert cell['n_trades'] >= 15
        assert cell['win_rate'] == 1.0               # falls -> short wins
        assert cell['verdict'] == 'BEATS'

    def test_survival_excludes_shorts_and_robustness_tickers(self):
        def cellrow(ticker, side, verdict='BEATS'):
            return {'signal': 'X', 'side': side, 'ticker': ticker,
                    'verdict': verdict, 'h': 5}
        # three beating SHORT cells + beating GLD cells must not count
        cells = [cellrow('QQQ', 'short'), cellrow('SPY', 'short'),
                 cellrow('QQQ', 'short'), cellrow('GLD', 'long'),
                 cellrow('GLD', 'long'), cellrow('GLD', 'long'),
                 cellrow('QQQ', 'long', 'no'), cellrow('SPY', 'long', 'no')]
        assert phase2_survival(cells)['X']['survives'] is False

    def test_survival_rule(self):
        def cellrow(sig, ticker, verdict):
            return {'signal': sig, 'side': 'long', 'ticker': ticker,
                    'verdict': verdict, 'h': 5}
        # 3 beats spanning both primaries -> survives
        cells = [cellrow('X', 'QQQ', 'BEATS'), cellrow('X', 'QQQ', 'BEATS'),
                 cellrow('X', 'SPY', 'BEATS'), cellrow('X', 'SPY', 'no')]
        assert phase2_survival(cells)['X']['survives'] is True
        # 3 beats on one ticker only -> closed
        cells = [cellrow('Y', 'QQQ', 'BEATS')] * 3 + [cellrow('Y', 'SPY', 'no')]
        assert phase2_survival(cells)['Y']['survives'] is False
        # 2 beats -> closed
        cells = [cellrow('Z', 'QQQ', 'BEATS'), cellrow('Z', 'SPY', 'BEATS')]
        assert phase2_survival(cells)['Z']['survives'] is False


class TestTurtleSoup:
    def test_level_fixed_at_entry_and_strict_fail(self):
        n = 40
        h = np.full(n, 100.0)
        c = np.full(n, 99.0)
        h[25] = 106.0
        c[25] = 105.0                                # breakout over level 100
        c[26:31] = 101.0                             # holds above the level...
        c[27] = 99.5                                 # ...except one fall-back -> fail
        d = mk(flat_dates(n), c - 0.5, h, c - 2.0, c)
        soup = turtle_soup(d)
        assert sum(r['fails'] for r in soup['per_era']) == 1
        # NOT a fail if the pullback stops exactly at the level (strict)
        c2 = c.copy()
        c2[27] = 100.0
        d2 = mk(flat_dates(n), c2 - 0.5, h, c2 - 2.0, c2)
        soup2 = turtle_soup(d2)
        assert sum(r['fails'] for r in soup2['per_era']) == 0

    def test_tail_skip_counted(self):
        n = 30
        h = np.full(n, 100.0)
        c = np.full(n, 99.0)
        h[27] = 106.0
        c[27] = 105.0                                # fires 2 sessions from end
        d = mk(flat_dates(n), c - 0.5, h, c - 2.0, c)
        assert turtle_soup(d)['skipped_tail'] == 1

    def test_decay_test_direction(self):
        def staircase(era_fail):
            """One breakout per era block; era_fail[j] decides whether the
            follow-through falls back below the level."""
            n = 90
            dates = flat_dates(n)
            h = np.full(n, 100.0)
            c = np.full(n, 99.0)
            for j, base_i in enumerate((25, 55, 85)):
                # keep entries clear of the tail window
                i = min(base_i, n - 7)
                h[i] = 106.0 + j
                c[i] = 105.0 + j
                c[i + 1:i + 6] = 90.0 if era_fail[j] else 104.0 + j
            return mk(dates, c - 0.5, h, c - 2.0, c)

        # earliest clean, latest fails -> decay direction, small p expected
        soup = turtle_soup(staircase([False, False, True]))
        # reversed: earliest fails, latest clean -> p near 1
        soup_rev = turtle_soup(staircase([True, False, False]))
        assert soup['decay_p'] < soup_rev['decay_p']


class TestPhase1Verdicts:
    def _meas(self, p, gap, era_signs):
        return {'p': p, '_p_raw': p, 'gap': gap, '_gap_raw': gap, 'n': 100,
                'per_era': [{'era': j, 'n': 10, 'rate': 0.5 + s * 0.1,
                             'base_rate': 0.5} for j, s in enumerate(era_signs)]}

    def test_ambiguity_rule(self):
        # strict survives on both tickers, loose does not -> AMBIGUOUS
        good = self._meas(0.001, 0.1, (1, 1, 1))
        bad = self._meas(0.5, 0.1, (1, 1, 1))
        primary = {t: {'C1_strict_top': good, 'C1_loose_top': bad,
                       'C2_top': bad, 'C3_up': bad,
                       'C3_up_gapcont': bad, 'C3_up_closeclose': bad}
                   for t in PRIMARY_TICKERS}
        v = phase1_verdicts(primary)
        assert v['C1'] == 'AMBIGUOUS'
        assert v['C2'] == 'CLOSED' and v['C3'] == 'CLOSED'

    def test_era_agreement_requirement(self):
        # significant p but era signs disagree (only 1 of 3 matches) -> CLOSED
        flaky = self._meas(0.001, 0.1, (1, -1, -1))
        primary = {t: {'C1_strict_top': flaky, 'C1_loose_top': flaky,
                       'C2_top': flaky, 'C3_up': flaky,
                       'C3_up_gapcont': flaky, 'C3_up_closeclose': flaky}
                   for t in PRIMARY_TICKERS}
        v = phase1_verdicts(primary)
        assert v == {'C1': 'CLOSED', 'C2': 'CLOSED', 'C3': 'CLOSED'}

    def test_empty_conditional_set_closes(self):
        empty = {'p': None, '_p_raw': None, 'gap': None, '_gap_raw': None,
                 'n': 0, 'per_era': []}
        primary = {t: {'C1_strict_top': empty, 'C1_loose_top': empty,
                       'C2_top': empty, 'C3_up': empty,
                       'C3_up_gapcont': empty, 'C3_up_closeclose': empty}
                   for t in PRIMARY_TICKERS}
        v = phase1_verdicts(primary)
        assert v == {'C1': 'CLOSED', 'C2': 'CLOSED', 'C3': 'CLOSED'}


class TestHelperEdges:
    def test_addone_two_sided_caps_at_one(self):
        null = np.array([0.4, 0.5, 0.6])
        assert addone_p_two_sided(null, 0.5) == 1.0

    def test_wilson_n_zero(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_fisher_k2_at_max(self):
        # latest panel all-fails: single table in the tail
        p = fisher_exact_one_sided(0, 5, 5, 5)
        assert p == pytest.approx(1 / 252)

    def test_build_trades_empty_fires(self):
        assert build_trades([], 5, 100) == []

    def test_binom_diagnostic_present(self):
        d = mk(['2005-01-03', '2005-01-04', '2005-01-05'],
               o=[100, 111, 100], h=[110, 112, 101],
               low=[100, 110, 99], c=[109, 111, 100])
        m = p1(d)
        assert m['C1_strict_top']['binom_p'] is not None
        assert 0.0 < m['C1_strict_top']['binom_p'] <= 1.0


# ------------------------------------------------------------- result pins

# The 2026-07-20 run's decisive numbers (plan §9 step 3). These run in CI:
# the OHLC inputs are committed CSVs, no dataset gate. Primary tickers only
# (the full robustness sweep is the module's __main__), per plan §8.
class TestSrReplicationPins:
    """Verdicts: C1 AMBIGUOUS (the strict quote-bearing reading survives
    mechanically — proximity geometry — while the loose reading is
    wrong-signed and fails), C2 CLOSED (the close odds are NOT 'much
    less' than the open odds — prior 2 contradicted), C3 CLOSED (flat).
    Phase 2: ZERO beating cells — every signal closes at 0-of-8, the
    LeBeau-Lucas conclusion replicated wholesale. Turtle Soup: 'most
    fail' true in the modern panel, the decay claim false (the earliest
    era already failed at 55%/65%)."""

    @pytest.fixture(scope='class')
    def phase1(self):
        from engine.tharp_sr_replication import run_phase1
        return run_phase1(list(PRIMARY_TICKERS))

    @pytest.fixture(scope='class')
    def phase2(self):
        from engine.tharp_sr_replication import run_phase2
        return run_phase2(list(PRIMARY_TICKERS))

    def test_phase1_verdicts(self, phase1):
        assert phase1['verdicts'] == {'C1': 'AMBIGUOUS', 'C2': 'CLOSED',
                                      'C3': 'CLOSED'}

    def test_c1_strict_is_proximity_geometry(self, phase1):
        q = phase1['measurements']['QQQ']['measurements']['C1_strict_top']
        s = phase1['measurements']['SPY']['measurements']['C1_strict_top']
        assert (q['n'], q['rate'], q['base_rate']) == (2286, 0.4038, 0.2143)
        assert (s['n'], s['rate'], s['base_rate']) == (2352, 0.392, 0.2224)
        assert q['p'] == 0.0002 and s['p'] == 0.0002
        # nowhere near the book's quoted 70-80%
        assert q['rate'] < 0.5 and s['rate'] < 0.5

    def test_c1_loose_is_wrong_signed(self, phase1):
        q = phase1['measurements']['QQQ']['measurements']['C1_loose_top']
        s = phase1['measurements']['SPY']['measurements']['C1_loose_top']
        assert q['gap'] == -0.0186 and s['gap'] == -0.0278
        assert q['p'] == 0.0186          # fails the p < 0.01 bar on QQQ
        assert s['p'] == 0.001

    def test_c2_not_much_less(self, phase1):
        # prior 2 contradicted: close odds ~= open odds, no conditional signal
        q = phase1['measurements']['QQQ']['measurements']['C2_top']
        s = phase1['measurements']['SPY']['measurements']['C2_top']
        assert (q['rate'], q['p']) == (0.5376, 0.34857)
        assert (s['rate'], s['p']) == (0.5361, 0.49755)

    def test_c3_flat(self, phase1):
        q = phase1['measurements']['QQQ']['measurements']['C3_up']
        s = phase1['measurements']['SPY']['measurements']['C3_up']
        assert (q['n'], q['gap'], q['p']) == (1093, 0.0019, 0.83892)
        assert (s['n'], s['gap'], s['p']) == (1079, 0.0036, 0.76772)

    def test_diagnostics(self, phase1):
        for t in PRIMARY_TICKERS:
            tick = phase1['measurements'][t]
            assert tick['n_degenerate'] == 0
            assert tick['era_starts'] == ['1999-11-01', '2010-01-04', '2020-01-02']

    def test_phase2_zero_beating_cells(self, phase2):
        judged = [c for c in phase2['cells'] if c.get('verdict') != 'NO-VERDICT']
        assert all(c['verdict'] == 'no' for c in judged)
        assert all(not v['survives'] for v in phase2['survival'].values())
        assert set(phase2['survival']) == {'CB-20', 'CB-40', 'CB-100',
                                           'MA-200', 'GX', 'RSI-30'}

    def test_phase2_exemplar_cell(self, phase2):
        c = next(x for x in phase2['cells']
                 if (x['ticker'], x['signal'], x['side'], x['h'])
                 == ('QQQ', 'CB-20', 'long', 5))
        assert c['n_trades'] == 377
        assert c['win_rate'] == 0.5915
        assert c['base_rate'] == 0.5734          # the drift wall
        assert c['null_rate_mean'] == 0.5732     # the null tracks it
        assert c['p'] == 0.22738

    def test_turtle_soup(self, phase2):
        q = phase2['turtle_soup']['QQQ']
        s = phase2['turtle_soup']['SPY']
        assert [r['rate'] for r in q['per_era']] == [0.55, 0.4792, 0.5272]
        assert [r['rate'] for r in s['per_era']] == [0.6515, 0.5408, 0.5473]
        assert q['modern_rate_above_half'] and s['modern_rate_above_half']
        # the decay claim FAILS: breakouts always mostly failed here
        assert q['decay_p'] == 0.72009 and not q['decay_confirmed']
        assert s['decay_p'] == 0.98979 and not s['decay_confirmed']
