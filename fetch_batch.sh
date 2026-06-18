#!/bin/zsh
# Generic sequential both-wing option-dailies fetcher for the edge-search onboarding
# pipeline — the parameterized replacement for the per-batch fetch_*.sh wrappers.
#
#   ./fetch_batch.sh GLD TLT XLE EEM
#
# For each ticker, IN ORDER (one to completion before the next — shared Alpha-Vantage
# rate budget, standing sequential preference): ensure a trading-day price calendar
# exists (download_prices.py, free from yfinance), then fetch the canonical both-wing
# option dailies (download_option_dailies.py), retry-wrapped — the fetcher resumes
# (skips days already present) but dies on socket timeouts, so we relaunch until a
# pass reports nothing left. A pkill-first guard keeps a single writer. The Alpha
# Vantage key is parsed from ~/.zshrc and never printed.
#
# Env overrides: START (2010-12-01 — past the placeholder-greeks era), END (2026-06-05).
# Next step after a fetch: ./onboard_ticker.sh <TICKER>  (validate, then publish).
cd "${0:A:h}" || exit 1
[ $# -ge 1 ] || { echo "usage: $0 TICKER [TICKER...]" >&2; exit 2; }
START=${START:-2010-12-01}; END=${END:-2026-06-05}
LOG=batch_fetch.log
PY=./.venv/bin/python; [ -x "$PY" ] || PY=python3

export ALPHAVANTAGE_API_KEY=$(grep -E '^[[:space:]]*export[[:space:]]+ALPHAVANTAGE_API_KEY=' ~/.zshrc \
  | head -1 | sed -E 's/.*ALPHAVANTAGE_API_KEY=//; s/^["'"'"']//; s/["'"'"'].*$//' | tr -d '[:space:]')
[ -n "$ALPHAVANTAGE_API_KEY" ] || { echo "[ERR] no AV key parsed from ~/.zshrc" | tee -a $LOG >&2; exit 1; }
echo "[ALL] START $(date +%H:%M:%S): $* (START=$START END=$END, key loaded ok)" >> $LOG

fetch_one() {  # $1 = ticker
  local tk=${1:u} lc=${1:l} n=0 tmp=/tmp/fetch_${1:l}.txt
  local cal=${lc}_20yr_prices.csv out=${lc}_option_dailies.csv
  if [ ! -f "$cal" ]; then
    echo "[$tk] price calendar $cal absent — downloading (yfinance, free)" >> $LOG
    $PY download_prices.py --ticker $tk --period 20y --output $cal >> $LOG 2>&1 \
      || { echo "[$tk][ERR] price calendar fetch failed" | tee -a $LOG >&2; return 1; }
  fi
  while true; do
    n=$((n+1)); pkill -9 -f "download_option_dailies.py" 2>/dev/null; sleep 1
    echo "[$tk #$n $(date +%H:%M:%S)] launch" >> $LOG
    $PY download_option_dailies.py --ticker $tk --out $out --dates-from $cal \
      --keep both --start $START --end $END 2>>$LOG | tee -a $LOG > $tmp
    if grep -qE '0 day\(s\) to fetch|: 0 row\(s\) appended' $tmp; then
      echo "[$tk] DONE $(date +%H:%M:%S) — remaining todo are Alpha-Vantage gap days" >> $LOG
      break
    fi
    sleep 2
  done
}

for tk in "$@"; do fetch_one $tk; done
echo "[ALL] FETCH COMPLETE $(date +%H:%M:%S): $*" >> $LOG
echo "fetched: $*"
echo "next: ./onboard_ticker.sh <TICKER>   (validate -> sign-off -> publish)"
