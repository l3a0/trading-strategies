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
# WHY THE PATTERNS CARRY 'scripts/' AND '.sh': a bare `pgrep -f
# fetch_sp500_intraday` also matches any OTHER process whose command line
# merely CONTAINS that string — including the session's own monitor loop,
# whose script text greps for exactly these names. That self-match made
# the first version of this waiter block forever on a process that is not
# a fetch. Anchoring on the invocation path 'scripts/<name>.sh' matches
# only real script invocations (verified: 3 fetch workers + 1 chain, and
# NOT the monitor).
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

echo "== waiting for the S&P 500 + Nasdaq-100 stages to finish ($(date))"
while pgrep -f 'scripts/fetch_sp500_intraday.sh|scripts/chain_nasdaq100_after_sp500.sh' >/dev/null; do
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
