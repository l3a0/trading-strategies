"""Tests for pipeline/minute_archive.py — the minute-archive data layer.

Aggregation, split adjustment, the owner-signed hygiene rulings, and the
reference cross-check. These were split out of
``tests/test_cup_handle_scan.py`` when the data layer moved out of the
cup-and-handle study: the rulings are facts about the vendor tape, not
about any one hypothesis, so they are tested where they live.

All synthetic — no archive and no network required, so these run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.minute_archive import (
    aggregate_daily,
    cliff_flags,
    split_adjust,
)


class TestAggregation:
    def test_session_bounds_dedup_and_absence(self, tmp_path):
        p = tmp_path / 'x_intraday_1min.csv'
        p.write_text(
            'timestamp,open,high,low,close,volume\n'
            '2024-01-02 15:59:00,100,102,99,101,20\n'    # UNSORTED input
            '2024-01-02 09:29:00,99,99,99,99,50\n'       # pre-market: excluded
            '2024-01-02 09:30:00,100,101,100,100,10\n'
            '2024-01-02 16:00:00,101,101,100,100.5,30\n'
            '2024-01-02 16:00:00,101,101,100,100.7,31\n'  # dup ts: LAST wins
            '2024-01-02 19:00:00,90,90,90,90,5\n'        # after-hours: excluded
            '2024-01-03 04:05:00,98,98,98,98,7\n'        # extended only -> absent
        )
        d = aggregate_daily(str(p), cache_dir=None)
        assert list(d['dates']) == ['2024-01-02']
        assert d['open'][0] == 100.0                     # time-sorted despite input
        assert d['high'][0] == 102.0
        assert d['close'][0] == 100.7
        assert d['volume'][0] == pytest.approx(10 + 20 + 31)

    def test_cache_round_trip(self, tmp_path):
        p = tmp_path / 'y_intraday_1min.csv'
        p.write_text('timestamp,open,high,low,close,volume\n'
                     '2024-01-02 10:00:00,1,2,0.5,1.5,100\n')
        d1 = aggregate_daily(str(p), cache_dir=str(tmp_path))
        cache = tmp_path / 'y_intraday_1min_daily_cache.csv'
        assert cache.exists()
        d2 = aggregate_daily(str(p), cache_dir=str(tmp_path))  # cache hit
        assert list(d1['dates']) == list(d2['dates'])
        assert d1['close'][0] == d2['close'][0]

    def test_split_adjust_multi_split_compounding(self):
        d = {'dates': np.array(['2024-01-02', '2024-01-03', '2024-01-04']),
             'open': np.array([400.0, 200.0, 100.0]),
             'high': np.array([400.0, 200.0, 100.0]),
             'low': np.array([400.0, 200.0, 100.0]),
             'close': np.array([400.0, 200.0, 100.0]),
             'volume': np.array([10.0, 20.0, 40.0])}
        adj = split_adjust(d, [('2024-01-03', 2.0), ('2024-01-04', 2.0)])
        # day 1 divides by 2*2, day 2 by 2, day 3 untouched
        assert list(adj['close']) == [100.0, 100.0, 100.0]
        assert list(adj['volume']) == [40.0, 40.0, 40.0]

    def test_cliff_guard_on_adjusted_series(self):
        dates = np.array(['2024-01-02', '2024-01-03', '2024-01-04'])
        # a correctly-committed split leaves NO cliff after adjustment
        raw = {'dates': dates,
               'open': np.array([300.0, 100.0, 100.0]),
               'high': np.array([300.0, 100.0, 100.0]),
               'low': np.array([300.0, 100.0, 100.0]),
               'close': np.array([300.0, 100.0, 100.0]),
               'volume': np.array([1.0, 3.0, 3.0])}
        adj = split_adjust(raw, [('2024-01-03', 3.0)])
        assert cliff_flags(adj['close'], dates, [('2024-01-03', 3.0)]) == []
        # a MISSING split leaves the cliff — flagged, no exemption
        assert cliff_flags(raw['close'], dates, []) == ['2024-01-03']
        # a cliff remaining ON a committed split day is STILL flagged (the
        # committed ratio failed to explain the move)
        assert cliff_flags(raw['close'], dates,
                           [('2024-01-03', 1.5)]) == ['2024-01-03']

    def test_resolved_cliff_scans_unresolved_excludes(self, monkeypatch):
        import pipeline.minute_archive as ma
        dates = np.array(['2024-01-02', '2024-01-03', '2024-01-04'])
        d = {'dates': dates,
             'open': np.array([100.0, 300.0, 300.0]),
             'high': np.array([100.0, 300.0, 300.0]),
             'low': np.array([100.0, 300.0, 300.0]),
             'close': np.array([100.0, 300.0, 300.0]),
             'volume': np.array([1.0, 1.0, 1.0])}
        monkeypatch.setattr(ma, 'archive_path', lambda t: '/synthetic')
        monkeypatch.setattr(ma, 'aggregate_daily', lambda p, cache_dir=None: d)
        # unresolved cliff -> no detections attempted (excluded)
        adj, cov = ma.load_clean_daily('XXX', {})
        assert cov['cliff_flags'] == ['2024-01-03']
        # owner-signed resolution -> the flag clears and the scan proceeds
        monkeypatch.setitem(ma.RESOLVED_CLIFFS, ('XXX', '2024-01-03'), 'test')
        adj, cov = ma.load_clean_daily('XXX', {})
        assert cov['cliff_flags'] == []
        assert cov['resolved_cliffs'] == ['2024-01-03']

    def test_start_clip_drops_predecessor_era(self, monkeypatch):
        # the AMCR shape: pre-clip rows are a predecessor company's prices
        # (~$56) wearing the ticker; the real entity starts at ~$11, so the
        # seam looks like an unexplained one-day cliff until the clip
        import pipeline.minute_archive as ma
        dates = np.array(['2019-06-07', '2019-06-10', '2019-06-11',
                          '2019-06-12'])
        px = np.array([56.0, 57.0, 11.0, 11.2])
        d = {'dates': dates, 'open': px, 'high': px, 'low': px,
             'close': px, 'volume': np.ones(4)}
        monkeypatch.setattr(ma, 'archive_path', lambda t: '/synthetic')
        monkeypatch.setattr(ma, 'aggregate_daily', lambda p, cache_dir=None: d)
        # without the clip: the seam is an unresolved cliff -> excluded
        adj, cov = ma.load_clean_daily('YYY', {})
        assert cov['cliff_flags'] == ['2019-06-11']
        assert cov['start_clip'] is None
        # owner-signed start clip: predecessor rows dropped, no cliff, scan
        # proceeds, and the clip is recorded in the coverage row
        monkeypatch.setitem(ma.TICKER_START_CLIPS, 'YYY', '2019-06-11')
        adj, cov = ma.load_clean_daily('YYY', {})
        assert cov['cliff_flags'] == [] and cov['start_clip'] == '2019-06-11'
        assert adj['dates'][0] == '2019-06-11' and len(adj['dates']) == 2

    def test_drop_window_removes_corrupt_patch(self, monkeypatch):
        # the ECL/ELV shape: a mid-history patch at the wrong scale (a
        # different security's day, or a mis-adjusted span) cliffs on BOTH
        # edges; dropping the window (inclusive) leaves a seamless join
        import pipeline.minute_archive as ma
        dates = np.array(['2019-02-01', '2019-02-04', '2019-02-05',
                          '2019-02-06', '2019-02-07'])
        px = np.array([159.0, 159.13, 18.92, 159.16, 160.0])
        d = {'dates': dates, 'open': px, 'high': px, 'low': px,
             'close': px, 'volume': np.ones(5)}
        monkeypatch.setattr(ma, 'archive_path', lambda t: '/synthetic')
        monkeypatch.setattr(ma, 'aggregate_daily', lambda p, cache_dir=None: d)
        # without the ruling: both edges of the patch flag -> excluded
        adj, cov = ma.load_clean_daily('WWW', {})
        assert cov['cliff_flags'] == ['2019-02-05', '2019-02-06']
        assert cov['drop_windows'] == []
        # owner-signed drop: the patch vanishes, the join is seamless,
        # and the ruling is recorded in the coverage row
        monkeypatch.setitem(ma.TICKER_DROP_WINDOWS, 'WWW',
                            [('2019-02-05', '2019-02-05')])
        adj, cov = ma.load_clean_daily('WWW', {})
        assert cov['cliff_flags'] == []
        assert cov['drop_windows'] == [('2019-02-05', '2019-02-05')]
        assert '2019-02-05' not in adj['dates'] and len(adj['dates']) == 4

    def test_snapshot_deviation_pins(self):
        # the owner-signed edits to the committed split snapshot must
        # survive any future regeneration from the vendor: NVDA's 2001
        # split is re-dated to the 9/11-shifted reopen, and EXPE's
        # reverse-split row is REMOVED (the TripAdvisor spin-off offset
        # it — applying the bare factor to an as-traded tape fabricates
        # a 2x break Expedia's own 10-K disproves)
        import pipeline.minute_archive as ma
        splits = ma.load_splits()
        assert ('2001-09-17', 2.0) in [(d, f) for d, f in splits['NVDA']]
        assert '2001-09-10' not in [d for d, _ in splits['NVDA']]
        assert not [d for d, _ in splits.get('EXPE', [])
                    if d.startswith('2011')]
        # WST's 2:1 is re-dated to its true ex-date: the as-traded
        # halving is on 2004-09-30 ($41.00 -> $20.74), and the vendor's
        # 09-29 made the ADJUSTED series spike instead of stepping down
        assert ('2004-09-30', 2.0) in [(d, f) for d, f in splits['WST']]
        assert '2004-09-29' not in [d for d, _ in splits['WST']]

    def test_elv_ruling_is_start_clip_not_drop_window(self):
        # the 2026-07-21 re-ruling: the whole pre-2011 ELV tape is
        # vendor fiction (2000-2005 matches no lineage security;
        # 2006-2010 is a smooth blend up to 49% off) — the clip
        # supersedes the earlier 2005 drop-window
        import pipeline.minute_archive as ma
        assert ma.TICKER_START_CLIPS['ELV'] == '2010-12-17'
        assert 'ELV' not in ma.TICKER_DROP_WINDOWS


class TestReferenceCrosscheck:
    def test_unsplit_restores_as_traded(self):
        # a 2:1 split before day 3: the reference shows adjusted
        # [50, 50, 100]; as-traded was flat 100 throughout
        import pipeline.minute_archive as ma
        dates = np.array(['2024-01-02', '2024-01-03', '2024-01-04'])
        out = ma.unsplit_reference(np.array([50.0, 50.0, 100.0]), dates,
                                    [('2024-01-04', 2.0)])
        assert list(out) == [100.0, 100.0, 100.0]

    def test_matching_series_not_flagged(self):
        import pipeline.minute_archive as ma
        dates = np.array([f'2024-01-{i:02d}' for i in range(2, 12)])
        px = np.linspace(100, 110, 10)
        r = ma.crosscheck_series(dates, px, dates, px, [])
        assert r['flagged'] is False and r['mismatch_days'] == 0
        assert r['compared'] == 10

    def test_fake_era_flagged_run_counted(self):
        # the ELV shape: a smooth block at 2x the reference — no cliff,
        # but every day mismatches; the run must count across a
        # reference hole (a missing ref day cannot split the era)
        import pipeline.minute_archive as ma
        dates = np.array([f'2024-01-{i:02d}' for i in range(1, 32)])
        ref_px = np.full(31, 100.0)
        ours = ref_px.copy()
        ours[10:22] = 200.0          # a 12-day fake era (run >= 10 flags)
        hole = '2024-01-15'          # ref missing a day inside the era
        ref_keep = np.array([x != hole for x in dates])
        r = ma.crosscheck_series(dates, ours, dates[ref_keep],
                                  ref_px[ref_keep], [])
        assert r['flagged'] is True
        assert r['mismatch_days'] == 11 and r['max_run'] == 11
        assert r['unreferenced_days'] == 1
        assert r['mismatch_first'] == '2024-01-11'
        assert r['mismatch_last'] == '2024-01-22'

    def test_crisis_noise_below_bar_not_flagged(self):
        # the AIG/F/DD shape from the calibration sweep: short bursts of
        # close-vs-last-trade disagreement in wild markets (run <= 5)
        # stay unflagged
        import pipeline.minute_archive as ma
        dates = np.array([f'2024-01-{i:02d}' for i in range(1, 32)])
        ref_px = np.full(31, 100.0)
        ours = ref_px.copy()
        ours[5:10] = 104.0           # five straight days ~4% off
        r = ma.crosscheck_series(dates, ours, dates, ref_px, [])
        assert r['flagged'] is False
        assert r['mismatch_days'] == 5 and r['max_run'] == 5
        assert r['severe_days'] == 0   # 4% is noise-band, not scale-level

    def test_scattered_severe_days_flag_via_backstop(self):
        # the backstop: scale-level (>5%) days scattered so no run forms
        # still flag once there are enough of them
        import pipeline.minute_archive as ma
        dates = np.array([f'2024-{m:02d}-{i:02d}'
                          for m in range(1, 13) for i in range(1, 29)])
        n = len(dates)
        ref_px = np.full(n, 100.0)
        ours = ref_px.copy()
        ours[::14] = 120.0           # every 14th day 20% off (24 severe days)
        r = ma.crosscheck_series(dates, ours, dates, ref_px, [])
        assert r['severe_days'] == 24 and r['max_run'] == 1
        assert r['flagged'] is True

    def test_single_day_glitch_not_flagged(self):
        # one bad print in either source is tolerated — reporting
        # noise, not a fake era
        import pipeline.minute_archive as ma
        dates = np.array([f'2024-01-{i:02d}' for i in range(2, 12)])
        px = np.full(10, 100.0)
        ours = px.copy()
        ours[4] = 150.0
        r = ma.crosscheck_series(dates, ours, dates, px, [])
        assert r['flagged'] is False and r['mismatch_days'] == 1

    def test_crosscheck_applies_owner_clip_first(self, monkeypatch):
        # a ruled-out predecessor era must not re-flag: garbage before
        # the clip, exact match after
        import pipeline.minute_archive as ma
        dates = np.array([f'2024-01-{i:02d}' for i in range(1, 32)])
        real = np.full(31, 100.0)
        ours = real.copy()
        ours[:15] = 7.0   # predecessor-company scale, 15 days (run >= 10)
        d = {'dates': dates, 'open': ours, 'high': ours, 'low': ours,
             'close': ours, 'volume': np.ones(31)}
        monkeypatch.setattr(ma, 'archive_path', lambda t: '/synthetic')
        monkeypatch.setattr(ma, 'aggregate_daily', lambda p, cache_dir=None: d)
        monkeypatch.setattr(ma, 'fetch_reference', lambda t: {
            'dates': dates, 'closes': real, 'splits': []})
        r = ma.crosscheck_ticker('VVV')
        assert r['flagged'] is True
        monkeypatch.setitem(ma.TICKER_START_CLIPS, 'VVV', '2024-01-16')
        r = ma.crosscheck_ticker('VVV')
        assert r['flagged'] is False and r['compared'] == 16

    def test_av_reference_membership(self):
        # the names whose yfinance history carries an un-undoable phantom
        # back-adjustment (owner-signed) must route to the AV reference
        import pipeline.minute_archive as ma
        assert {'BLDR', 'CCI', 'HWM'} <= ma.CROSSCHECK_AV_REFERENCE

    def test_av_reference_tickers_use_av_fetch(self, monkeypatch):
        # BLDR/CCI-class names (yfinance carries a phantom adjustment its
        # own event feed can't undo) must be checked against the Alpha
        # Vantage daily reference instead of yfinance
        import pipeline.minute_archive as ma
        dates = np.array(['2024-01-02', '2024-01-03'])
        px = np.array([100.0, 101.0])
        d = {'dates': dates, 'open': px, 'high': px, 'low': px,
             'close': px, 'volume': np.ones(2)}
        monkeypatch.setattr(ma, 'archive_path', lambda t: '/synthetic')
        monkeypatch.setattr(ma, 'aggregate_daily', lambda p, cache_dir=None: d)
        calls = []
        monkeypatch.setattr(ma, 'fetch_reference', lambda t: calls.append('yf') or {
            'dates': dates, 'closes': px, 'splits': []})
        monkeypatch.setattr(ma, 'fetch_reference_av', lambda t: calls.append('av') or {
            'dates': dates, 'closes': px, 'splits': []})
        monkeypatch.setattr(ma, 'CROSSCHECK_AV_REFERENCE', {'UUU'})
        ma.crosscheck_ticker('UUU')
        ma.crosscheck_ticker('TTT')
        assert calls == ['av', 'yf']

    def test_partial_archive_refused(self, tmp_path, monkeypatch):
        import pipeline.minute_archive as ma
        ws = tmp_path / 'sp500_intraday_1min'
        ws.mkdir()
        (ws / 'zzz_intraday_1min.csv').write_text('timestamp\n')  # no .done
        monkeypatch.setattr(ma, 'data_path', lambda p: str(tmp_path / p))
        assert ma.archive_path('ZZZ') is None
        (ws / 'zzz_intraday_1min.csv.months.done').write_text('complete\n')
        assert ma.archive_path('ZZZ') is not None

    def test_gzip_race_prefers_intact_csv(self, tmp_path, monkeypatch):
        # the batch gzips in place: mid-compression BOTH files exist and
        # the gz is truncated — the marker-vouched csv must win; once the
        # csv is unlinked the (now complete) gz is trusted via the marker
        import pipeline.minute_archive as ma
        ws = tmp_path / 'sp500_intraday_1min'
        ws.mkdir()
        monkeypatch.setattr(ma, 'data_path', lambda p: str(tmp_path / p))
        csv_f = ws / 'yyy_intraday_1min.csv'
        csv_f.write_text('timestamp\n')
        (ws / 'yyy_intraday_1min.csv.months.done').write_text('complete\n')
        (ws / 'yyy_intraday_1min.csv.gz').write_bytes(b'\x1f\x8b partial')
        assert ma.archive_path('YYY') == str(csv_f)   # csv wins mid-gzip
        csv_f.unlink()
        assert ma.archive_path('YYY').endswith('.csv.gz')  # gz after
        (ws / 'yyy_intraday_1min.csv.months.done').unlink()
        assert ma.archive_path('YYY') is None  # gz alone: unvouched



class TestHygieneUniverse:
    """The hygiene sweep and a study's population are different questions.
    They were accidentally sharing one answer, so while the S&P 400 landed
    the sweep kept re-checking the same S&P 500 names and reporting clean."""

    def test_universe_stays_the_frozen_study_population(self):
        import pipeline.minute_archive as ma
        # the cup-and-handle design freezes this at the S&P 500 snapshot
        assert ma.TICKERS_PATH == 'sp500_tickers_2026-07.txt'
        assert len(ma.universe()) == 502

    def test_archived_tickers_spans_every_snapshot(self, tmp_path, monkeypatch):
        import pipeline.minute_archive as ma
        monkeypatch.setattr(ma, 'data_path', lambda p: str(tmp_path / p))
        (tmp_path / 'a.txt').write_text('AAA\nBBB\n')
        (tmp_path / 'b.txt').write_text('BBB\nCCC\n')   # BBB in both
        monkeypatch.setattr(ma, 'UNIVERSE_SNAPSHOTS', ('a.txt', 'b.txt'))
        monkeypatch.setattr(ma, 'archive_path',
                            lambda t: None if t == 'CCC' else '/x')
        # deduped, snapshot order, and CCC excluded for having no archive
        assert ma.archived_tickers() == ['AAA', 'BBB']

    def test_missing_snapshot_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        import pipeline.minute_archive as ma
        monkeypatch.setattr(ma, 'data_path', lambda p: str(tmp_path / p))
        (tmp_path / 'a.txt').write_text('AAA\n')
        monkeypatch.setattr(ma, 'UNIVERSE_SNAPSHOTS', ('a.txt', 'gone.txt'))
        monkeypatch.setattr(ma, 'archive_path', lambda t: '/x')
        assert ma.archived_tickers() == ['AAA']


class TestValueDetachments:
    """Convention: adjust only SHARE-COUNT changes. A spin-off leaves the
    share count alone, so applying it as a split rescales every earlier
    price — Abbott's real 2012-12-31 close was $65.47, rendered $31.41."""

    # An exact simple fraction — a split (2:1, 7:5, 1:32) or a stock
    # dividend (1.05 = 21/20) — is SUFFICIENT to be a share-count change
    # but NOT necessary. These nine were researched to primary filings and
    # are share-count changes whose ratio is market-derived, because how
    # many shares got issued depended on a price: elective cash-or-stock
    # dividends (SPG, MAR, HST, IRM's REIT distribution), Lennar's Class B
    # stock dividend, and Google's 2015 Class C adjustment payment. Listed
    # rather than pattern-matched, so adding a tenth is a deliberate act.
    RESEARCHED_INEXACT = {
        ('GOOG', '2015-04-27'), ('HST', '2009-11-04'), ('IRM', '2014-09-26'),
        ('LEN', '2017-11-09'), ('MAR', '2009-06-23'), ('MAR', '2009-08-18'),
        ('MAR', '2009-11-17'), ('SPG', '2009-08-13'), ('SPG', '2009-11-12'),
    }

    def test_split_table_holds_no_unvetted_market_derived_factors(self):
        from fractions import Fraction
        import pipeline.minute_archive as ma
        bad = [(t, d, r) for t, rows in ma.load_splits().items()
               for d, r in rows
               if float(Fraction(r).limit_denominator(100)) != r
               and (t, d) not in self.RESEARCHED_INEXACT]
        assert bad == [], f'unvetted market-derived factors in the split table: {bad}'

    def test_the_inexact_exceptions_are_all_still_present(self):
        import pipeline.minute_archive as ma
        s = ma.load_splits()
        missing = [(t, d) for t, d in self.RESEARCHED_INEXACT
                   if d not in {x for x, _ in s.get(t, [])}]
        assert missing == [], f'researched share-count changes lost: {missing}'

    def test_detachments_and_splits_are_disjoint(self):
        import pipeline.minute_archive as ma
        sp = {(t, d) for t, rows in ma.load_splits().items() for d, _ in rows}
        det = {(t, d) for t, ds in ma.load_detachments().items() for d in ds}
        assert not (sp & det), f'a row cannot be both: {sp & det}'

    def test_reverse_splits_survived_the_reclassification(self):
        import pipeline.minute_archive as ma
        s = ma.load_splits()
        # EQIX 1-for-32 and MNST 1-for-50: exact fractions whose
        # denominators exceed the triage bound. Evicted by the first pass
        # and restored — the bug this pins.
        assert ('2002-12-31', 0.03125) in s['EQIX']
        assert ('1988-02-29', 0.02) in s['MNST']

    def test_return_breaks_read_both_sources(self, monkeypatch):
        import pipeline.minute_archive as ma
        dates = np.array(['2013-01-01', '2013-01-02', '2013-01-03'])
        monkeypatch.setattr(ma, 'RETURN_BREAKS', {'XXX': ('2013-01-01',)})
        monkeypatch.setattr(ma, 'load_detachments',
                            lambda: {'XXX': ('2013-01-03',)})
        # hand-signed table AND the committed snapshot, merged and sorted
        assert ma.return_break_indices('XXX', dates).tolist() == [0, 2]

    def test_a_date_in_both_sources_is_not_double_counted(self, monkeypatch):
        import pipeline.minute_archive as ma
        dates = np.array(['2013-01-01', '2013-01-02'])
        monkeypatch.setattr(ma, 'RETURN_BREAKS', {'XXX': ('2013-01-02',)})
        monkeypatch.setattr(ma, 'load_detachments',
                            lambda: {'XXX': ('2013-01-02',)})
        assert ma.return_break_indices('XXX', dates).tolist() == [1]

    def test_hand_signed_breaks_and_blended_events_resolve(self):
        import pipeline.minute_archive as ma
        # the pure detachments plus the blended reverse-split+spin-off pair
        assert {'MO', 'ROK', 'WY'} <= set(ma.RETURN_BREAKS)
        assert ma.RETURN_BREAKS.get('HLT') == ('2017-01-04',)   # + 1-for-3
        assert ma.RETURN_BREAKS.get('MSI') == ('2011-01-04',)   # + 1-for-7
        d = ma.load_detachments()
        assert 'ABT' in d and '2013-01-02' in d['ABT']   # AbbVie

    def test_blended_events_are_in_both_split_table_and_return_breaks(self):
        import pipeline.minute_archive as ma
        # a blended reverse-split-and-spin-off adjusts (split table) AND is
        # a return break (RETURN_BREAKS) — the two halves of one event
        s = ma.load_splits()
        for t, d in (('HLT', '2017-01-04'), ('MSI', '2011-01-04')):
            assert d in {x for x, _ in s[t]}, f'{t} reverse split missing'
            assert d in ma.RETURN_BREAKS[t], f'{t} return break missing'
            assert d not in ma.load_detachments().get(t, ())  # not double-filed


class TestHygieneRulingsDataFile:
    """The owner-signed rulings moved from hardcoded dict literals to
    data/hygiene_rulings.csv (2026-07-23) when the S&P 400 onboarding pushed
    them past ~180 rulings. The module rebuilds the identical dicts at
    import; this pins the loader and the file's integrity."""

    def test_loader_rebuilds_four_tables(self):
        import pipeline.minute_archive as ma
        resolved, clips, drops, breaks = ma.load_hygiene_rulings()
        # the module-level tables ARE the loader's output
        assert resolved == ma.RESOLVED_CLIFFS
        assert clips == ma.TICKER_START_CLIPS
        assert drops == ma.TICKER_DROP_WINDOWS
        assert breaks == ma.RETURN_BREAKS

    def test_value_types_match_the_old_dicts(self):
        import pipeline.minute_archive as ma
        # resolved_cliff value is the reason string; the rest are the engine
        # values the tables always held
        (k, v) = next(iter(ma.RESOLVED_CLIFFS.items()))
        assert isinstance(k, tuple) and isinstance(v, str)
        assert all(isinstance(v, str) for v in ma.TICKER_START_CLIPS.values())
        assert all(isinstance(w, list) and isinstance(w[0], tuple)
                   for w in ma.TICKER_DROP_WINDOWS.values())
        assert all(isinstance(v, tuple) for v in ma.RETURN_BREAKS.values())

    def test_every_row_has_a_reason_and_a_valid_kind(self):
        import csv as _csv
        from pipeline.minute_archive import HYGIENE_RULINGS_PATH
        from common.paths import data_path
        kinds = {'resolved_cliff', 'start_clip', 'drop_window', 'return_break'}
        with open(data_path(HYGIENE_RULINGS_PATH)) as f:
            rows = list(_csv.DictReader(f))
        assert rows, 'rulings file is empty'
        for r in rows:
            assert r['kind'] in kinds, r
            assert r['reason'].strip(), f"no reason for {r['ticker']} {r['date']}"
            if r['kind'] == 'drop_window':
                assert r['end'], f"drop_window needs an end: {r}"

    def test_unknown_kind_raises(self, tmp_path, monkeypatch):
        import pipeline.minute_archive as ma
        bad = tmp_path / 'bad.csv'
        bad.write_text('kind,ticker,date,end,reason\nbogus,X,2020-01-01,,r\n')
        monkeypatch.setattr(ma, 'data_path', lambda p: str(bad))
        with pytest.raises(ValueError, match='unknown ruling kind'):
            ma.load_hygiene_rulings()

    def test_known_rulings_survived_the_migration(self):
        import pipeline.minute_archive as ma
        assert ma.TICKER_START_CLIPS['AA'] == '2016-11-01'      # Alcoa/Arconic
        assert ('AAPL', '2000-09-29') in ma.RESOLVED_CLIFFS     # profit warning
        assert ma.TICKER_DROP_WINDOWS['ASML'] == [('2007-10-01', '2007-10-26')]
        assert ma.RETURN_BREAKS['MO'] == ('2008-03-31',)        # PMI spin-off
