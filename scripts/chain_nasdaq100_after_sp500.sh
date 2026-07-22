#!/bin/bash
# Wait for the in-flight S&P 500 minute fetch to finish, then fetch the
# Nasdaq-100's net-new names.
#
# WHY CHAINED, not concurrent: the fetch is API-RATE-BOUND, not machine-
# bound. The measured S&P run sustains ~69 requests/min against Alpha
# Vantage's ~75/min premium cap (318 month-calls per ticker), so adding
# workers cannot buy throughput — it only risks throttle backoff. Running
# the two batches back-to-back costs the same wall-clock as running them
# together, without the throttling.
#
# The batch script skips any ticker already archived, so handing it the
# full 103-symbol Nasdaq-100 snapshot fetches ONLY the ~15 names absent
# from the S&P 500 universe.
#
# Usage: ALPHAVANTAGE_API_KEY=... scripts/chain_nasdaq100_after_sp500.sh
set -u
cd "$(dirname "$0")/.."
DIR=data/sp500_intraday_1min
LIST=data/nasdaq100_tickers_2026-07.txt
WORKERS=${WORKERS:-3}

# NOT `pgrep -f fetch_sp500_intraday`: that matches any command line
# MENTIONING the script, including the session's monitor loop, which
# deadlocked this stage on 2026-07-22 — the S&P workers exited at 12:28 and
# this script kept waiting on the watcher. stage_alive compares argv[0]/
# argv[1] instead of scanning the whole line; see scripts/stage_alive.sh
# for the two narrower patterns that were tried first and why both leaked.
. "$(dirname "$0")/stage_alive.sh"

echo "== waiting for the S&P 500 fetch workers to exit ($(date))"
while stage_alive 'scripts/fetch_sp500_intraday.sh'; do sleep 300; done
echo "== S&P fetch done ($(date)); starting Nasdaq-100 net-new fetch"

for i in $(seq 0 $((WORKERS - 1))); do
  WORKER_INDEX=$i WORKER_COUNT=$WORKERS TICKERS="$LIST" \
    AV_FETCH_SLEEP=${AV_FETCH_SLEEP:-1.5} \
    scripts/fetch_sp500_intraday.sh > "$DIR/batch.ndx.w$i.log" 2>&1 &
done
wait
echo "== nasdaq-100 net-new batch complete ($(date))"
