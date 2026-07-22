#!/bin/bash
# Third stage of the minute-archive fetch queue: wait for the S&P 500 run
# AND the chained Nasdaq-100 net-new run to finish, then fetch the S&P 400
# mid-caps.
#
# WHY CHAINED: the fetch is API-RATE-BOUND, not machine-bound — the
# measured run sustains ~69 requests/min against Alpha Vantage's ~75/min
# cap (318 month-calls per ticker), so concurrent stages cannot buy
# throughput, only throttle backoff. Stages run back-to-back for the same
# total wall-clock.
#
# WHY IT WAITS ON BOTH: between the S&P workers exiting and the Nasdaq
# workers starting there is a window with no fetch_sp500_intraday process
# alive, but chain_nasdaq100_after_sp500.sh is still running as their
# parent. Waiting on the union of both patterns closes that race without
# editing the already-running Nasdaq script.
#
# WHY stage_alive AND NOT pgrep: `pgrep -f` matches the WHOLE command line,
# so a bare `fetch_sp500_intraday` fires on any process that merely
# MENTIONS the name — including a monitor loop whose own text greps for
# these names. That self-match deadlocked this queue on 2026-07-22: the S&P
# workers exited at 12:28 and the Nasdaq stage waited on the watcher.
# stage_alive compares argv[0]/argv[1] rather than scanning the line, which
# closes the class; scripts/stage_alive.sh records the two narrower
# substring patterns that were tried first and exactly how each leaked.
#
# The S&P 400 is 100% NET NEW: S&P's indices are mutually exclusive (500 +
# 400 + 600 = the Composite 1500), verified — zero overlap against both
# the committed S&P 500 and Nasdaq-100 snapshots.
#
# Measured expectation: ~400 tickers x ~4.6 min = ~1.3 days, ~4-5 GB
# (mid-cap median ~5,500 minute bars/month vs ~8,500 for the S&P 500).
#
# Usage: ALPHAVANTAGE_API_KEY=... scripts/chain_sp400_after_nasdaq100.sh
set -u
cd "$(dirname "$0")/.."
DIR=data/sp500_intraday_1min
LIST=data/sp400_tickers_2026-07.txt
WORKERS=${WORKERS:-3}

. "$(dirname "$0")/stage_alive.sh"

echo "== waiting for the S&P 500 + Nasdaq-100 stages to finish ($(date))"
while stage_alive 'scripts/fetch_sp500_intraday.sh|scripts/chain_nasdaq100_after_sp500.sh'; do
  sleep 300
done
echo "== prior stages done ($(date)); starting the S&P 400 mid-cap fetch"

for i in $(seq 0 $((WORKERS - 1))); do
  WORKER_INDEX=$i WORKER_COUNT=$WORKERS TICKERS="$LIST" \
    AV_FETCH_SLEEP=${AV_FETCH_SLEEP:-1.5} \
    scripts/fetch_sp500_intraday.sh > "$DIR/batch.sp400.w$i.log" 2>&1 &
done
wait
echo "== s&p 400 batch complete ($(date))"
