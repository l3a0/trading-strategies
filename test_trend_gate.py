"""Tests for trend_gate.py — the registered experiment's analysis machinery.

Two layers, mirroring test_real_cc_backtest.py:

- Pure-logic unit tests (always run, including CI): the §5.1 sequence
  generator's mechanics, the §5.2/§5.3 statistics and the add-one p-value,
  §6.1 short-call-day counting and the statistic T, the §6.4 LOYO
  recomputation, and the engine's §10 suspension seam (entry-only,
  byte-identical when off) — all on hand-computable synthetic fixtures.
- Dataset-gated pins (skip without the chain/price files): the §3.1 span
  table, the §3.3 signal-side characterization (the registered F* and
  per-ticker suspension structure), and the determinism of the accepted
  sequence stream. These touch TREATMENT-SIDE data only — no engine runs,
  no cycle P&L, no exceedance series — so they produce no Stage 1 number
  and respect the registration's §10 ordering.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from real_cc_backtest import run_real_cc_overlay
from test_real_cc_backtest import (
    _HAVE_DAILIES,
    _HAVE_MSFT_BACKFILL,
    _HAVE_MSFT_DAILIES,
    _HAVE_QQQ_BACKFILL,
    _HAVE_SPY_DAILIES,
    _PARAMS,
)
from trend_gate import (
    ACCEPT_HI,
    ACCEPT_LO,
    ENGINE_PARAMS,
    F_STAR,
    TICKERS,
    add_one_p,
    characterize,
    complement_suspension,
    d_a,
    d_b,
    draw_raw_sequence,
    family_s_offsets,
    load_market,
    loyo_t,
    master_calendar,
    placebo_statistics,
    pooled_fraction,
    reconstruct_cycles,
    record_suspension,
    run_length_multisets,
    shifted_signal_suspension,
    shifted_states,
    short_call_days,
    statistic_t,
    vol_ablation_suspension,
)

_HAVE_ALL_DATA = (_HAVE_DAILIES and _HAVE_MSFT_DAILIES and _HAVE_SPY_DAILIES
                  and _HAVE_MSFT_BACKFILL and _HAVE_QQQ_BACKFILL)


# ---- pure logic: §2.1 signal ----

class TestShiftedStates:
    def test_shift_and_warmup(self) -> None:
        # 205 flat closes then a jump: index 205 classifies 'bull'
        # (120 > 1.05 x SMA200 of ~100s), so the SHIFTED series goes bull
        # one day later, and day 0 is 'unknown' (no prior close).
        closes = [100.0] * 205 + [120.0] * 5
        s = shifted_states(closes)
        assert len(s) == len(closes)
        assert s[0] == 'unknown'
        assert s[199] == 'unknown'   # SMA defined at 199; shifted → 200
        assert s[200] == 'sideways'  # close 199 vs SMA through 199
        assert s[205] == 'sideways'  # close 204 still flat
        assert s[206] == 'bull'      # close 205 = 120 broke the band


# ---- pure logic: §5.1 generator mechanics ----

class TestSequenceGenerator:
    def test_deterministic(self) -> None:
        a = draw_raw_sequence(np.random.default_rng(7), 50, [2, 5], [3])
        b = draw_raw_sequence(np.random.default_rng(7), 50, [2, 5], [3])
        assert np.array_equal(a, b)

    def test_alternating_runs_from_multisets(self) -> None:
        seq = draw_raw_sequence(np.random.default_rng(7), 50, [2, 5], [3])
        assert len(seq) == 50
        runs: list[tuple[bool, int]] = []
        for v in seq:
            if runs and runs[-1][0] == bool(v):
                runs[-1] = (bool(v), runs[-1][1] + 1)
            else:
                runs.append((bool(v), 1))
        # states alternate by construction
        assert all(runs[i][0] != runs[i + 1][0] for i in range(len(runs) - 1))
        # every full run's length comes from its multiset (last may truncate)
        for v, n in runs[:-1]:
            assert n in ((2, 5) if v else (3,))
        last_v, last_n = runs[-1]
        assert last_n <= max((2, 5) if last_v else (3,))

    def test_pooled_fraction(self) -> None:
        seq = np.array([True, True, False, False, True, False])
        m1 = np.array([True, True, True, False, False, False])   # 2/3 susp
        m2 = np.array([False, False, False, True, True, True])   # 1/3 susp
        assert pooled_fraction(seq, [m1, m2]) == pytest.approx(3 / 6)

    def test_acceptance_band_is_the_registered_one(self) -> None:
        # ±5% relative around F* (§5.1), endpoints as registered.
        assert ACCEPT_LO == pytest.approx(F_STAR * 0.95, abs=5e-6)
        assert ACCEPT_HI == pytest.approx(F_STAR * 1.05, abs=5e-6)

    def test_family_s_offsets(self) -> None:
        offs = family_s_offsets()
        assert len(offs) == 500
        assert min(offs) >= 250 and max(offs) <= 3_374
        assert offs == family_s_offsets()  # seed 42, deterministic


class _FakePool:
    """SequencePool stand-in: a fixed pool plus an indexable tail."""

    def __init__(self, pool: list[np.ndarray], tail: list[np.ndarray]) -> None:
        self.pool = pool
        self._tail = tail

    def tail(self, j: int) -> np.ndarray:
        return self._tail[j]


class TestPlaceboReplacement:
    def test_degenerate_pool_draw_replaced_from_tail(self) -> None:
        # Calendar of 2 days; stat is None unless both labels appear.
        cal = ['d1', 'd2']

        def stat(susp: set[str]) -> float | None:
            if len(susp) in (0, 2):
                return None
            return 1.0 if 'd1' in susp else 2.0

        pool = _FakePool(
            pool=[np.array([True, False]),   # ok → 1.0
                  np.array([True, True]),    # degenerate
                  np.array([False, True])],  # ok → 2.0
            tail=[np.array([False, False]),  # degenerate replacement too
                  np.array([False, True])],  # ok → 2.0
        )
        stats, replacements = placebo_statistics(stat, pool, cal)
        assert stats == [1.0, 2.0, 2.0]  # order preserved, slot 2 replaced
        assert replacements == 2          # both tail draws consumed


# ---- pure logic: §5.2 / §5.3 statistics ----

class TestStage1Statistics:
    TRADES = [
        {'action': 'sell', 'date': '2024-01-02', 'pnl': 0},
        {'action': 'close', 'date': '2024-01-05', 'pnl': 100.0},
        {'action': 'sell', 'date': '2024-01-08', 'pnl': 0},
        {'action': 'close_itm', 'date': '2024-01-10', 'pnl': -300.0},
        {'action': 'sell', 'date': '2024-01-11', 'pnl': 0},
        {'action': 'expiration', 'date': '2024-01-20', 'pnl': 50.0},
        {'action': 'sell', 'date': '2024-01-22', 'pnl': 0},  # still open
    ]

    def test_reconstruct_cycles_drops_open(self) -> None:
        cycles = reconstruct_cycles(self.TRADES)
        assert [(c['entry_date'], c['pnl']) for c in cycles] == [
            ('2024-01-02', 100.0), ('2024-01-08', -300.0),
            ('2024-01-11', 50.0)]

    def test_d_a_by_real_tag_and_by_suspension(self) -> None:
        cycles = reconstruct_cycles(self.TRADES)
        # The real statistic reads each cycle's own per-ticker 'bull' flag
        # (precomputed at pooling time); placebo re-tagging is by date.
        for c, bull in zip(cycles, (True, False, False)):
            c['bull'] = bull
        # bull mean 100; non-bull mean (-300+50)/2 = -125 → D_A = 225
        assert d_a(cycles) == pytest.approx(225.0)
        assert d_a(cycles, suspended={'2024-01-02'}) == pytest.approx(225.0)
        assert d_a(cycles, suspended=set()) is None          # empty bull cell
        assert d_a(cycles, suspended={'2024-01-02', '2024-01-08',
                                      '2024-01-11'}) is None  # empty non-bull

    def test_d_a_real_tag_is_per_item_not_shared(self) -> None:
        # Two cycles on the SAME date from different tickers can carry
        # different real tags — the cross-ticker collision the spec's
        # per-ticker §2.1 state forbids cannot occur by construction.
        cycles = [
            {'entry_date': '2024-01-02', 'pnl': 100.0, 'bull': True},
            {'entry_date': '2024-01-02', 'pnl': -40.0, 'bull': False},
        ]
        assert d_a(cycles) == pytest.approx(140.0)

    def test_d_b(self) -> None:
        obs = [('d1', 1, True), ('d2', 0, False),
               ('d3', 1, True), ('d4', 1, False)]
        # bull = d1,d3 → rate 1.0; non-bull = d2,d4 → 0.5; D_B = 0.5
        assert d_b(obs) == pytest.approx(0.5)
        assert d_b(obs, suspended={'d1', 'd3'}) == pytest.approx(0.5)
        # placebo re-tag can move a day across cells
        assert d_b(obs, suspended={'d1', 'd4'}) == pytest.approx(0.5)
        assert d_b(obs, suspended=set()) is None

    def test_add_one_p(self) -> None:
        # add-one convention: real counted among candidates, p never 0
        assert add_one_p(5.0, [1.0, 2.0, 3.0], 'ge') == pytest.approx(1 / 4)
        assert add_one_p(0.0, [1.0, 2.0, 3.0], 'ge') == pytest.approx(4 / 4)
        assert add_one_p(0.0, [1.0, 2.0, 3.0], 'le') == pytest.approx(1 / 4)
        assert add_one_p(2.0, [1.0, 2.0, 3.0], 'le') == pytest.approx(3 / 4)


# ---- pure logic: §6.1 / §6.4 ----

class TestStage2Arithmetic:
    SPAN = [f'2024-01-{d:02d}' for d in range(2, 12)]  # 10 trading days

    def test_short_call_days(self) -> None:
        trades = [
            {'action': 'sell', 'date': '2024-01-02', 'pnl': 0},
            {'action': 'close', 'date': '2024-01-05', 'pnl': 1.0},  # 3 days
            {'action': 'sell', 'date': '2024-01-08', 'pnl': 0},     # open: 4 days
        ]
        # closed: entry incl → terminal excl = 3; open: entry incl → span
        # end incl = 10 - 6 = 4
        assert short_call_days(trades, self.SPAN) == 7
        assert short_call_days([], self.SPAN) == 0

    def test_statistic_t_equal_weights(self) -> None:
        per = {'A': {'net_overlay_pnl': 100.0, 'short_call_days': 10},
               'B': {'net_overlay_pnl': -300.0, 'short_call_days': 100},
               'C': {'net_overlay_pnl': 50.0, 'short_call_days': 50}}
        assert statistic_t(per) == pytest.approx((10.0 - 3.0 + 1.0) / 3)

    def test_engine_params_match_published_baseline(self) -> None:
        """§3.2: every arm runs the published baseline configuration (a
        constant comparison — no dataset needed, so it always runs)."""
        assert ENGINE_PARAMS == _PARAMS

    def test_loyo_t(self) -> None:
        cycles = {'A': [['2022-03-01', 10, 500.0], ['2023-02-01', 10, -100.0]],
                  'B': [['2022-05-01', 20, 200.0], ['2023-06-01', 20, 400.0]]}
        # drop 2022: A → -100/10, B → 400/20 → mean of (-10, 20) = 5
        assert loyo_t(cycles, '2022') == pytest.approx(5.0)
        # dropping 2023 leaves both tickers with days → fine
        assert loyo_t(cycles, '2023') == pytest.approx((50.0 + 10.0) / 2)
        # a year whose removal empties a ticker → None
        one = {'A': [['2022-03-01', 10, 500.0]], 'B': [['2023-01-01', 5, 50.0]]}
        assert loyo_t(one, '2022') is None

    def test_record_complement_partition(self) -> None:
        market = {'span_dates': ['d1', 'd2', 'd3'],
                  'span_states': ['bull', 'bear', 'sideways']}
        rec = record_suspension(market)
        comp = complement_suspension(market)
        assert rec == {'d1'}
        assert comp == {'d2', 'd3'}
        assert rec | comp == set(market['span_dates'])
        assert not rec & comp

    def test_shifted_signal_suspension(self) -> None:
        market = {'span_dates': ['d1', 'd2', 'd3', 'd4'],
                  'span_states': ['bull', 'bull', 'bear', 'bear']}
        # offset 1: rotated[i] = flags[i-1] → suspension moves right one day
        assert shifted_signal_suspension(market, 1) == {'d2', 'd3'}
        # full rotation is identity
        assert shifted_signal_suspension(market, 4) == {'d1', 'd2'}


# ---- pure logic: the §10 engine seam ----

def _day(cands: list[tuple[Any, ...]],
         marks: dict[str, tuple[float, float, float, float]]) -> dict[str, Any]:
    return {'candidates': cands, 'marks': marks}


def _c(cid: str, strike: float, bid: float, ask: float, mid: float,
       delta: float = 0.25, dte: int = 30,
       exp: str = '2099-01-01') -> tuple[Any, ...]:
    return (dte, delta, bid, ask, mid, exp, strike, cid)


class TestSuspensionSeam:
    DATES = ['2024-01-02', '2024-01-03', '2024-01-04']
    PRICES = [100.0, 102.0, 101.0]

    @staticmethod
    def _store() -> dict[str, dict[str, Any]]:
        c1 = _c('C1', 110.0, 2.00, 2.20, 2.10)
        c2 = _c('C2', 112.0, 1.50, 1.70, 1.60)
        return {
            '2024-01-02': _day([c1], {'C1': (2.00, 2.20, 2.10, 0.25)}),
            '2024-01-03': _day([c2], {'C1': (2.50, 2.70, 2.60, 0.35),
                                      'C2': (1.50, 1.70, 1.60, 0.25)}),
            '2024-01-04': _day([], {'C1': (0.10, 0.30, 0.20, 0.05),
                                    'C2': (0.10, 0.30, 0.20, 0.05)}),
        }

    PARAMS = {**_PARAMS, 'capital': 10_000}

    def test_suspended_day_defers_entry(self) -> None:
        s, trades, eq = run_real_cc_overlay(
            self.DATES, self.PRICES, self._store(), self.PARAMS,
            suspended_dates={'2024-01-02'})
        sells = [t for t in trades if t['action'] == 'sell']
        assert [t['contract'] for t in sells] == ['C2']
        assert sells[0]['date'] == '2024-01-03'
        # the suspended day still gets an equity row: shares + cash, no call
        assert eq['equity'].iloc[0] == pytest.approx(10_000.0)

    def test_gate_is_entry_only(self) -> None:
        # A position opened before suspension still closes on a suspended
        # day (§2.3: the gate never triggers an exit).
        store = self._store()
        store['2024-01-03']['marks']['C1'] = (9.0, 9.4, 9.2, 0.95)  # deep ITM
        _, trades, _ = run_real_cc_overlay(
            self.DATES, self.PRICES, store, self.PARAMS,
            suspended_dates={'2024-01-03', '2024-01-04'})
        actions = [t['action'] for t in trades]
        assert actions[0] == 'sell'
        assert 'close_itm' in actions
        close = next(t for t in trades if t['action'] == 'close_itm')
        assert close['date'] == '2024-01-03'

    def test_off_is_byte_identical(self) -> None:
        base_s, base_t, base_eq = run_real_cc_overlay(
            self.DATES, self.PRICES, self._store(), self.PARAMS)
        for susp in (None, set(), frozenset()):
            s, t, eq = run_real_cc_overlay(
                self.DATES, self.PRICES, self._store(), self.PARAMS,
                suspended_dates=susp)
            assert s == base_s
            assert t == base_t
            assert eq.equals(base_eq)


# ---- dataset-gated: §3.1 spans, §3.3 characterization, stream determinism ----

@pytest.mark.skipif(
    not _HAVE_ALL_DATA,
    reason='needs all three tickers\' option dailies (and the MSFT/QQQ '
           'backfills) or their committed .gz twins',
)
class TestRegisteredSignalSide:
    """Pins the treatment-assignment quantities the registration states
    (§3.1, §3.3). No engine run, no cycle P&L, no exceedance series — these
    tests produce no Stage 1 number (§10 ordering).
    """

    @pytest.fixture(scope='class')
    def markets(self) -> dict[str, dict[str, Any]]:
        return {t: load_market(t) for t in TICKERS}

    def test_analysis_spans(self, markets) -> None:
        """§3.1 table: span starts, ends, and trading-day counts."""
        expected = {'MSFT': ('2010-05-10', '2026-04-10', 4_005),
                    'SPY': ('2010-12-01', '2026-06-05', 3_901),
                    'QQQ': ('2012-01-06', '2026-06-05', 3_624)}
        for t, (lo, hi, n) in expected.items():
            m = markets[t]
            assert m['span_dates'][0] == lo, t
            assert m['span_dates'][-1] == hi, t
            assert len(m['span_dates']) == n, t

    def test_characterization(self, markets) -> None:
        """§3.3 table: suspension structure and the registered F*."""
        rep = characterize(markets)
        per = rep['per_ticker']
        assert per['MSFT']['suspended'] == 2_506
        assert per['SPY']['suspended'] == 2_207
        assert per['QQQ']['suspended'] == 2_510
        assert per['MSFT']['bull_fraction'] == pytest.approx(0.626, abs=0.001)
        assert per['SPY']['bull_fraction'] == pytest.approx(0.566, abs=0.001)
        assert per['QQQ']['bull_fraction'] == pytest.approx(0.693, abs=0.001)
        # the registered table's sideways/bear columns (§3.3)
        assert per['MSFT']['sideways_fraction'] == pytest.approx(0.239, abs=0.001)
        assert per['MSFT']['bear_fraction'] == pytest.approx(0.136, abs=0.001)
        assert per['SPY']['sideways_fraction'] == pytest.approx(0.356, abs=0.001)
        assert per['SPY']['bear_fraction'] == pytest.approx(0.078, abs=0.001)
        assert per['QQQ']['sideways_fraction'] == pytest.approx(0.222, abs=0.001)
        assert per['QQQ']['bear_fraction'] == pytest.approx(0.085, abs=0.001)
        assert per['MSFT']['episodes'] == 69
        assert per['SPY']['episodes'] == 86
        assert per['QQQ']['episodes'] == 85
        assert per['MSFT']['episode_len_min_med_max'] == (1, 3, 531)
        assert per['SPY']['episode_len_min_med_max'] == (1, 5, 229)
        assert per['QQQ']['episode_len_min_med_max'] == (1, 4, 277)
        assert rep['suspended_days'] == 7_223
        assert rep['span_days'] == 11_530
        assert rep['pooled_fraction'] == pytest.approx(F_STAR, abs=5e-7)
        agree = rep['pairwise_bull_agreement']
        assert agree['MSFT/SPY'] == pytest.approx(0.681, abs=0.001)
        assert agree['MSFT/QQQ'] == pytest.approx(0.812, abs=0.001)
        assert agree['SPY/QQQ'] == pytest.approx(0.802, abs=0.001)

    def test_master_calendar(self, markets) -> None:
        cal = master_calendar(markets)
        assert cal[0] == '2010-05-10'
        assert cal[-1] == '2026-06-05'
        assert len(cal) >= 4_005  # at least the longest span

    def test_multiset_totals(self, markets) -> None:
        """Run lengths partition the pooled span days (§5.1)."""
        bull, nonbull = run_length_multisets(markets)
        assert sum(bull) == 7_223           # suspended days
        assert sum(bull) + sum(nonbull) == 11_530
        assert len(bull) == 69 + 86 + 85    # one length per episode

    def test_sequence_stream_determinism(self, markets) -> None:
        """§5.1: a single seeded stream — two instantiations agree, accepted
        fractions sit in the registered band, and the first sequence's
        fingerprint locks the RNG call pattern of record."""
        from trend_gate import accepted_sequences, span_masks
        cal = master_calendar(markets)
        masks = list(span_masks(markets, cal).values())
        s1 = accepted_sequences(markets)
        s2 = accepted_sequences(markets)
        first = next(s1)
        assert np.array_equal(first, next(s2))
        for seq in (first, next(s1), next(s1)):
            assert ACCEPT_LO <= pooled_fraction(seq, masks) <= ACCEPT_HI
        assert hashlib.sha256(first.tobytes()).hexdigest() == (
            'fca1c092d31efce109ee82d327e05b3e397b9e630b73e689eabf8e66af419a6b')

    def test_vol_ablation_matches_exposure(self, markets) -> None:
        """§4 arm 4: suspension fraction matches the bull fraction by
        construction (quantile at the bull-fraction level)."""
        for t, m in markets.items():
            susp = vol_ablation_suspension(m)
            flags = [s == 'bull' for s in m['span_states']]
            level = sum(flags) / len(flags)
            frac = len(susp) / len(m['span_dates'])
            assert frac == pytest.approx(level, abs=0.01), t
