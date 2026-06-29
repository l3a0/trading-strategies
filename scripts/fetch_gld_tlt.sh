#!/bin/zsh
# Sequential, self-healing fetch of the GLD + TLT BOTH-WING chains (the
# structurally-different next batch: gold + long Treasuries). One request per
# trading day returns the full chain, so both wings cost the same API budget as
# calls-only — fetched into the canonical {ticker}_option_dailies.csv (matches
# IWM's both-wing canonical; enables the covered-call campaign AND future VRP).
# GLD to completion, then TLT (shared rate budget; standing sequential-fetch
# preference). pkill-first guard => single writer. Terminates a ticker when a
# pass reports "0 day(s) to fetch" (all done) or "0 row(s) appended" (remaining
# todo are Alpha-Vantage gap days). Key parsed from ~/.zshrc, never printed.
# Resumable: download_option_dailies.py skips days already in the output, so a
# re-run picks up where a socket timeout left off.
cd "${0:A:h}/.." || exit 1  # repo root (data/, packages on path)
LOG=logs/gld_tlt_fetch.log

export ALPHAVANTAGE_API_KEY=$(grep -E '^[[:space:]]*export[[:space:]]+ALPHAVANTAGE_API_KEY=' ~/.zshrc \
  | head -1 | sed -E 's/.*ALPHAVANTAGE_API_KEY=//; s/^["'"'"']//; s/["'"'"'].*$//' | tr -d '[:space:]')
if [ -z "$ALPHAVANTAGE_API_KEY" ]; then echo "[ERR] no AV key parsed from ~/.zshrc" >> $LOG; exit 1; fi
echo "[ALL] START $(date +%H:%M:%S): GLD -> TLT both wings (key loaded ok)" >> $LOG

fetch_one() {  # $1=ticker $2=out $3=dates-from $4=start $5=end
  local tk=$1 out=$2 cal=$3 start=$4 end=$5 n=0 tmp=/tmp/lastrun_$1.txt
  while true; do
    n=$((n+1))
    pkill -9 -f "download_option_dailies" 2>/dev/null
    sleep 1
    echo "[$tk #$n $(date +%H:%M:%S)] launch" >> $LOG
    ./.venv/bin/python -m pipeline.download_option_dailies --ticker $tk \
      --out $out --dates-from $cal --start $start --end $end 2>>$LOG | tee -a $LOG > $tmp
    if grep -qE '0 day\(s\) to fetch|: 0 row\(s\) appended' $tmp; then
      echo "[$tk] DONE $(date +%H:%M:%S) — remaining todo are AV gaps" >> $LOG
      break
    fi
    sleep 2
  done
}

fetch_one GLD data/gld_option_dailies.csv data/gld_20yr_prices.csv 2010-12-01 2026-06-05
fetch_one TLT data/tlt_option_dailies.csv data/tlt_20yr_prices.csv 2010-12-01 2026-06-05
echo "[ALL] FETCH COMPLETE $(date +%H:%M:%S): GLD + TLT both wings" >> $LOG
