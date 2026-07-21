#!/bin/bash
# Sequential S&P 500 1-minute archive fetch — one ticker to completion
# before the next starts (the standing sequential-fetch preference), with
# a per-ticker retry loop around the resumable fetcher and ROLLING GZIP:
# each completed ticker is gzip -9'd and its raw csv removed, so the
# on-disk footprint stays ~25 GB instead of ~150 GB.
#
# Resumable at both levels: re-running skips gzipped tickers and resumes
# partial ones from their .months sidecar. Tickers already archived at
# the data/ root (MSFT, NVDA from the nine-ticker fetch) are skipped.
#
# Concurrency (owner-relaxed 2026-07-20 from the sequential default):
# WORKER_INDEX/WORKER_COUNT stride-partition the ticker list, so N workers
# each stay one-ticker-to-completion on DISJOINT tickers. Launch each
# worker with AV_FETCH_SLEEP raised so the aggregate request rate stays
# near the 75/min cap (3 workers -> 1.5s works; the fetcher's throttle
# backoff self-regulates any overshoot).
#
# Usage: ALPHAVANTAGE_API_KEY=... [WORKER_INDEX=0 WORKER_COUNT=3] \
#        scripts/fetch_sp500_intraday.sh
set -u
PY=./.venv/bin/python; [ -x "$PY" ] || PY=python3
DIR=data/sp500_intraday_1min
# the committed frozen universe; override to fetch another committed list
# (e.g. TICKERS=data/nasdaq100_tickers_2026-07.txt for the Nasdaq-100 —
# already-archived tickers are skipped, so a list that overlaps an earlier
# run fetches only its net-new names)
TICKERS=${TICKERS:-data/sp500_tickers_2026-07.txt}
START=2000-01
END=2026-07
WORKER_INDEX=${WORKER_INDEX:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
FAILED="$DIR/failed_tickers.txt"

LINE=0
while read -r T; do
  [ -z "$T" ] && continue
  LINE=$((LINE+1))
  [ $(( (LINE - 1) % WORKER_COUNT )) -ne "$WORKER_INDEX" ] && continue
  F=$(echo "$T" | tr '[:upper:]' '[:lower:]' | tr '.' '-')
  if [ -f "data/${F}_intraday_1min.csv.gz" ] || [ -f "data/${F}_intraday_1min.csv" ]; then
    echo "== $T: already archived at data/ root, skipping"
    continue
  fi
  if [ -f "$DIR/${F}_intraday_1min.csv.gz" ]; then
    continue
  fi
  echo "== $T ($(date '+%H:%M'))"
  n=0
  rc=1
  while :; do
    "$PY" -m pipeline.download_intraday --symbol "$T" --start "$START" --end "$END" --out-dir "$DIR"
    rc=$?
    [ $rc -eq 0 ] && break                       # complete
    [ $rc -eq 3 ] && { echo "$T NO_DATA" >> "$FAILED"; break; }
    n=$((n+1))
    [ $n -ge 8 ] && { echo "$T RETRIES_EXHAUSTED" >> "$FAILED"; break; }
    echo "== $T: fetcher died (rc=$rc), retry $n" >&2
    sleep 30
  done
  if [ $rc -eq 0 ] && [ -f "$DIR/${F}_intraday_1min.csv" ]; then
    gzip -9 "$DIR/${F}_intraday_1min.csv"
  fi
done < "$TICKERS"
echo "== batch complete ($(date))"
[ -f "$FAILED" ] && { echo "== failed tickers:"; cat "$FAILED"; }
