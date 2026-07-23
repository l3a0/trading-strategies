"""Tests for engine/cup_handle_scan.py — the frozen cup-and-handle build.

The §4 synthetic battery (always-run), rebuilt after the adversarial
review's mutation pass proved several first-draft checks toothless: each
rejection now isolates ITS rule (verified by the reviewers' mutation
counterexamples), the shortest-window rule pins the exact recorded
anatomy, and the cluster-null machinery is pinned on hand-derived
fixtures. The dataset-gated result pins land in the run PR after the
archive fetch completes (plan §10 step 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.cup_handle_scan import (
    DETECT_FLOOR,
    _try_window,
    build_clusters,
    build_trades,
    cluster_null_p,
    detect_cup_handle,
    quadratic_roundness,
    spans_return_break,
    stratum_of,
)


# ------------------------------------------------------- the textbook cup

def textbook(depth_frac=0.20, handle_drop=4.0, handle_len=10, v_shape=None,
             breakout_vol=2000.0, uptrend=True, overshoot=False,
             pad=240, handle_low_override=None, handle_vol=500.0,
             left_rim=140.0, cup_spike=None):
    """A parametric textbook formation. Defaults fire the detector; each
    keyword breaks exactly one frozen rule."""
    rim = left_rim
    pre = ([100.0] * pad if uptrend else [139.0] * pad)
    if uptrend:
        pre = pre + [100.0 + (rim - 100.0) * k / 99 for k in range(100)]
    else:
        pre = pre + [139.0] * 100
    cup_len = 100
    bottom = 140.0 * (1 - depth_frac)          # depth vs the nominal 140 rim
    amp = (140.0 - bottom) / 2
    mid = (140.0 + bottom) / 2
    if v_shape == 'sharp':
        cup = [140.0 - 1.0] * cup_len
        mid_i = cup_len // 2
        for k in range(10):
            cup[mid_i - 10 + k] = 139.0 - (139.0 - bottom) * k / 9
            cup[mid_i + k] = bottom + (139.0 - bottom) * k / 9
        cup[-1] = 138.0
    elif v_shape == 'slow':
        half = cup_len // 2
        down = list(np.linspace(140.0, bottom, half, endpoint=False))
        up = list(np.linspace(bottom, 138.0, cup_len - half))
        cup = down + up
    else:
        cup = [mid + amp * np.cos(2 * np.pi * k / (cup_len - 1))
               for k in range(cup_len)]
        cup[-1] = 138.0
    if overshoot:
        cup[-10] = 148.0
    if cup_spike is not None:
        cup[10] = cup_spike                    # an early in-cup spike
    r_close = cup[-1]
    h_low = handle_low_override if handle_low_override is not None \
        else r_close - handle_drop
    handle = list(np.linspace(r_close, h_low, handle_len))[1:]
    breakout = [r_close + 1.0]
    closes = np.array(pre + cup + handle + breakout + [r_close] * 5)
    n = len(closes)
    vol = np.full(n, 1000.0)
    h_start = pad + 100 + cup_len
    vol[h_start:h_start + handle_len - 1] = handle_vol
    t_break = h_start + handle_len - 1
    vol[t_break] = breakout_vol
    return closes, vol, t_break


class TestDetectorFires:
    def test_textbook_cup_fires_once_with_pinned_anatomy(self):
        c, v, t_break = textbook()
        hits = detect_cup_handle(c, v)
        assert len(hits) == 1
        hit = hits[0]
        assert hit['t'] == t_break
        # the pinned anatomy: L=9 is the SHORTEST passing window (shorter
        # ones die on the interior rule — their "rim" is a mid-handle day
        # and the cup's recovery top at ~139.97 overshoots it), the handle
        # high is the right rim, the 140 peak is the left rim
        assert hit['h0'] == t_break - 9
        assert c[hit['l']] == pytest.approx(140.0)
        assert c[hit['r']] == pytest.approx(137.5556, abs=1e-3)
        assert 0.12 <= hit['depth'] <= 0.33
        assert hit['roundness'] >= 0.15

    def test_shortest_window_wins_discriminates(self):
        # L=10 ALSO passes in isolation, so recording h0 at t-9 proves
        # ascending-first-wins (longest-first would record a longer window)
        c, v, t = textbook()
        assert _try_window(c, v, t, 10) is not None
        (hit,) = detect_cup_handle(c, v)
        assert hit['h0'] == t - 9

    def test_deterministic(self):
        c, v, _ = textbook()
        assert detect_cup_handle(c, v) == detect_cup_handle(c, v)

    def test_detect_floor_matches_null_floor(self):
        # no detection can precede DETECT_FLOOR, and the iteration starts
        # there — the null's eligibility floor is the same constant
        assert DETECT_FLOOR == 330
        c, v, _ = textbook(pad=0)                  # too short: nothing fires
        assert detect_cup_handle(c, v) == []


class TestDetectorRejections:
    def test_sharp_v_bottom_rejected(self):
        c, v, _ = textbook(v_shape='sharp')
        assert detect_cup_handle(c, v) == []

    def test_slow_pointed_bottom_is_a_known_admission(self):
        # A DOCUMENTED BOUNDARY of the frozen detector (the §0 "a detector,
        # not THE pattern" cap): a slow linear V spends ~25% of its sessions
        # in the bottom quartile — above the frozen 0.15 floor — so the
        # detector ADMITS pointed-but-slow bottoms. Pinned so the admission
        # is a stated fact; changing the floor is a §4 owner-signed
        # amendment, never a silent retune.
        c, v, _ = textbook(v_shape='slow')
        assert len(detect_cup_handle(c, v)) == 1

    def test_too_shallow_isolated(self):
        # handle_drop=2.0 keeps the handle ABOVE the shallow cup's
        # upper-half floor, so ONLY the depth minimum rejects (the review's
        # mutation check: deleting DEPTH_MIN would let this fire)
        c, v, _ = textbook(depth_frac=0.08, handle_drop=2.0)
        assert detect_cup_handle(c, v) == []

    def test_too_deep_rejected(self):
        c, v, _ = textbook(depth_frac=0.45)
        assert detect_cup_handle(c, v) == []

    def test_handle_below_mid_cup(self):
        c, v, _ = textbook(handle_low_override=120.0)
        assert detect_cup_handle(c, v) == []

    def test_upward_sloping_handle(self):
        c, v, t = textbook()
        c2 = c.copy()
        c2[t - 9:t] = np.linspace(130.0, 137.0, 9)
        assert detect_cup_handle(c2, v) == []

    def test_quiet_volume_breakout(self):
        c, v, _ = textbook(breakout_vol=900.0)
        assert detect_cup_handle(c, v) == []

    def test_handle_volume_above_cup_volume_rejected(self):
        # the rule-1 volume-dry-up clause in isolation: everything else
        # passes, but the handle trades HEAVIER than the cup
        c, v, _ = textbook(handle_vol=1500.0)
        assert detect_cup_handle(c, v) == []

    def test_missing_prior_uptrend(self):
        c, v, _ = textbook(uptrend=False)
        assert detect_cup_handle(c, v) == []

    def test_interior_overshoot(self):
        c, v, _ = textbook(overshoot=True)
        assert detect_cup_handle(c, v) == []

    def test_rim_band_rejects_tall_left_rim(self):
        # left rim 170: c[r]=138 < 0.85 x 170 — the rim band rejects
        c, v, _ = textbook(left_rim=170.0)
        assert detect_cup_handle(c, v) == []

    def test_rim_band_no_second_best_left(self):
        # an early in-cup spike at 170 becomes the argmax left rim; the
        # band fails against IT, and the frozen no-second-best rule means
        # the (otherwise valid) 140 rim is never retried
        c, v, _ = textbook(cup_spike=170.0)
        assert detect_cup_handle(c, v) == []

    def test_volume_ablation_readmits_quiet_breakout(self):
        c, v, _ = textbook(breakout_vol=900.0)
        assert len(detect_cup_handle(c, v, use_volume_trigger=False)) == 1

    def test_dedup_skip_suppresses_adjacent_retrigger(self):
        # after the detection at t, a second qualifying trigger inside the
        # 25-session skip is suppressed (a t+=1 iteration would record it):
        # the post-breakout tail re-crosses the SAME handle high on a surge
        c, v, t = textbook()
        c2 = c.copy()
        v2 = v.copy()
        c2[t + 1] = 137.0                          # dip back under the rim
        c2[t + 2] = 139.5                          # re-break within the skip
        v2[t + 2] = 2000.0
        hits = detect_cup_handle(c2, v2)
        assert [h['t'] for h in hits] == [t]


class TestQuadraticVariant:
    def test_round_cup_passes_and_sharp_v_fails(self):
        c, v, _ = textbook()
        (hit,) = detect_cup_handle(c, v)
        q = quadratic_roundness(c, hit['l'], hit['r'])
        assert q['passes'] and q['r2'] >= 0.70
        # the sharp crash-and-snap fits a parabola badly (mostly flat with
        # a narrow spike) — the variant rejects it
        c2, _, _ = textbook(v_shape='sharp')
        q2 = quadratic_roundness(c2, hit['l'], hit['r'])
        assert q2['r2'] < 0.70 and not q2['passes']

    def test_slow_pointed_bottom_admitted_by_both_definitions(self):
        # ANOTHER documented boundary: a symmetric slow V fits a parabola
        # WELL (measured r2 ~0.91, above the cosine cup's ~0.90), so the
        # quadratic variant admits slow pointed bottoms just like the
        # primary time-near-the-bottom gate. Both frozen roundness
        # formalizations share this limitation — pinned as a stated fact.
        c, v, _ = textbook(v_shape='slow')
        (hit,) = detect_cup_handle(c, v)
        q = quadratic_roundness(c, hit['l'], hit['r'])
        assert q['passes'] and q['r2'] >= 0.70


# --------------------------------------------------------------- §2 helpers

    def test_run_scan_skips_ticker_moving_underfoot(self, monkeypatch):
        # a read that still catches the archive mid-move must not kill the
        # whole pass — the ticker is skipped and reported for next time
        import engine.cup_handle_scan as chs
        monkeypatch.setattr(chs, 'failed_tickers', lambda: set())
        monkeypatch.setattr(chs, 'load_splits', lambda: {})

        def boom(ticker, splits):
            raise EOFError('compressed file ended early')
        monkeypatch.setattr(chs, 'scan_ticker', boom)
        res = chs.run_scan(['RACE'])
        assert res['skipped_in_flight'] == ['RACE']
        assert res['tickers_scanned'] == 0


# --------------------------------------------------------------- §5 helpers

def synth_data(ticker_days, drift=0.05):
    """{ticker: n_days} -> per-ticker data dicts on a shared calendar,
    seeded stably (no builtin hash()) so fixtures reproduce cross-process."""
    out = {}
    for ticker, n in ticker_days.items():
        dates = [f'{2020 + d // 240}-{1 + (d % 240) // 20:02d}-{1 + d % 20:02d}'
                 for d in range(n)]
        seed = sum(ord(ch) for ch in ticker)
        rng = np.random.default_rng(seed)
        closes = 100 + np.cumsum(rng.normal(drift, 1.0, n))
        out[ticker] = {'dates': np.array(dates), 'close': closes}
    return out


class TestEvaluation:
    def test_build_trades_lockout_and_end_skip(self):
        assert build_trades([10, 12, 15, 30], 5, 100) == [10, 15, 30]
        assert build_trades([96], 5, 100) == []

    def test_stratum_widths_and_boundaries(self):
        assert stratum_of('2024-05-07', 20) == '2024-05'
        assert stratum_of('2024-05-07', 60) == '2024Q2'
        assert stratum_of('2024-06-30', 120) == '2024H1'   # inclusive June
        assert stratum_of('2024-07-01', 120) == '2024H2'
        assert stratum_of('2024-03-31', 60) == '2024Q1'
        assert stratum_of('2024-04-01', 60) == '2024Q2'

    def test_clusters_equal_weight_mean(self):
        # hand-set closes: AAA +10%, BBB -4% on the shared date
        data = {
            'AAA': {'dates': np.array(['2024-01-02', '2024-01-03']),
                    'close': np.array([100.0, 110.0])},
            'BBB': {'dates': np.array(['2024-01-02', '2024-01-03']),
                    'close': np.array([50.0, 48.0])},
        }
        clusters, dropped = build_clusters({'AAA': [0], 'BBB': [0]}, data, 1)
        assert dropped == 0
        (cl,) = clusters
        assert sorted(cl['members']) == ['AAA', 'BBB']
        assert cl['ret'] == pytest.approx((0.10 - 0.04) / 2)

    def test_cluster_null_monotone_pins_p_one(self):
        # every ticker rises every session past the floor: real clusters and
        # every null draw are wins -> null_rate_mean == 1.0 and p == 1.0
        n = 600
        dates = [f'{2020 + d // 240}-{1 + (d % 240) // 20:02d}-{1 + d % 20:02d}'
                 for d in range(n)]
        data = {t: {'dates': np.array(dates),
                    'close': np.linspace(100, 200, n) * m}
                for t, m in (('AAA', 1.0), ('BBB', 2.0))}
        trades = {'AAA': [400, 450], 'BBB': [400]}
        clusters, _ = build_clusters(trades, data, 5)
        r = cluster_null_p(clusters, data, 5, b=200)
        assert r['win_rate'] == 1.0
        assert r['null_rate_mean'] == 1.0
        assert r['p'] == 1.0

    def test_cluster_null_deterministic_dilution_and_underpowered(self):
        data = synth_data({'AAA': 700, 'BBB': 400})    # BBB short: dilutes
        trades = {'AAA': [520, 540], 'BBB': [380]}
        # BBB's entry 380 needs 380+5 < 400 (ok) but many null draws for
        # BBB land before DETECT_FLOOR=330 or past 395 -> dropped
        clusters, _ = build_clusters(trades, data, 5)
        r1 = cluster_null_p(clusters, data, 5, b=200)
        r2 = cluster_null_p(clusters, data, 5, b=200)
        assert r1['p'] == r2['p']
        assert r1['underpowered'] is True
        assert r1['dilution_per_resample'] > 0

    def test_run_scan_without_evaluate_computes_no_returns(self, monkeypatch):
        import engine.cup_handle_scan as chs
        monkeypatch.setattr(chs, 'failed_tickers', lambda: set())
        monkeypatch.setattr(chs, 'load_splits', lambda: {})
        monkeypatch.setattr(chs, 'archive_path', lambda t: None)
        out = chs.run_scan(['ZZZ'], evaluate=False)
        assert 'evaluation' not in out and 'survives' not in out
        assert out['missing'] == ['ZZZ']

class TestReturnBreakGuard:
    """A value detachment (spin-off, purge dividend) is a REAL price move
    and a FAKE return. The guard drops any window measured across one —
    from the real book and from the null alike, since applying it to only
    one side would bias the comparison."""

    def test_window_spanning_a_break_is_flagged(self):
        breaks = np.array([50])
        # break strictly after entry and at or before exit -> contaminated
        assert spans_return_break(breaks, 45, 5) is True
        assert spans_return_break(breaks, 49, 1) is True
        assert spans_return_break(breaks, 30, 20) is True

    def test_entry_on_the_break_day_is_clean(self):
        # close[t] already reflects the detachment, so the forward return
        # is honest — this is the boundary that must NOT be over-excluded
        assert spans_return_break(np.array([50]), 50, 10) is False

    def test_window_ending_before_or_starting_after_is_clean(self):
        breaks = np.array([50])
        assert spans_return_break(breaks, 40, 9) is False   # exits at 49
        assert spans_return_break(breaks, 51, 10) is False  # starts past it
        assert spans_return_break(np.empty(0, dtype=int), 45, 20) is False

    def test_build_clusters_drops_and_counts_the_contaminated_entry(self,
                                                                   monkeypatch):
        import engine.cup_handle_scan as chs
        # AAA detaches on index 1: $100 -> $38 with the holder made whole.
        # Ungated, that books a -62% "loss"; the guard must drop it.
        data = {
            'AAA': {'dates': np.array(['2024-01-02', '2024-01-03']),
                    'close': np.array([100.0, 38.0])},
            'BBB': {'dates': np.array(['2024-01-02', '2024-01-03']),
                    'close': np.array([50.0, 55.0])},
        }
        monkeypatch.setattr(chs, 'return_break_indices',
                            lambda t, d: (np.array([1]) if t == 'AAA'
                                          else np.empty(0, dtype=int)))
        clusters, dropped = chs.build_clusters({'AAA': [0], 'BBB': [0]},
                                               data, 1)
        assert dropped == 1
        (cl,) = clusters
        assert cl['members'] == ['BBB']          # AAA's fake loss excluded
        assert cl['ret'] == pytest.approx(0.10)  # BBB alone, not the mean

    def test_guard_is_off_by_default_for_tickers_without_breaks(self):
        # the whole S&P universe minus three names must be untouched
        data = {
            'AAA': {'dates': np.array(['2024-01-02', '2024-01-03']),
                    'close': np.array([100.0, 110.0])},
        }
        clusters, dropped = build_clusters({'AAA': [0]}, data, 1)
        assert dropped == 0
        assert clusters[0]['ret'] == pytest.approx(0.10)

    def test_real_table_indices_resolve_against_a_real_series(self):
        from pipeline.minute_archive import (RETURN_BREAKS,
                                             return_break_indices)
        # every listed date must be findable, and a ticker with no entry
        # must yield nothing — the table is keyed on the retained series
        dates = np.array(['2010-07-19', '2010-07-20', '2010-07-21'])
        assert return_break_indices('WY', dates).tolist() == [1]
        assert return_break_indices('AAPL', dates).tolist() == []
        # RETURN_BREAKS holds the pure detachments (MO/ROK/WY) plus the
        # blended reverse-split-and-spin-off events (HLT/MSI); the wider
        # detachment set lives in the committed CSV
        assert {'MO', 'ROK', 'WY'} <= set(RETURN_BREAKS)
