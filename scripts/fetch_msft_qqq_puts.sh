#!/bin/zsh
# Sequential, self-healing fetch of the MSFT + QQQ PUT wings (separate puts files,
# never appended to the calls canonical). MSFT to completion, then QQQ (shared rate
# budget; standing sequential-fetch preference). pkill-first guard => single writer.
# Terminates a ticker when a pass reports "0 day(s) to fetch" (all done) or appends
# "0 row(s)" (remaining todo are Alpha-Vantage gap days). Key parsed from ~/.zshrc,
# never printed.
cd "${0:A:h}/.." || exit 1  # repo root (data/, packages on path)
LOG=logs/fetch_progress.log

export ALPHAVANTAGE_API_KEY=$(grep -E '^[[:space:]]*export[[:space:]]+ALPHAVANTAGE_API_KEY=' ~/.zshrc \
  | head -1 | sed -E 's/.*ALPHAVANTAGE_API_KEY=//; s/^["'"'"']//; s/["'"'"'].*$//' | tr -d '[:space:]')
if [ -z "$ALPHAVANTAGE_API_KEY" ]; then echo "[ERR] no AV key parsed from ~/.zshrc" >> $LOG; exit 1; fi
echo "[ALL] START $(date +%H:%M:%S): MSFT puts -> QQQ puts (key loaded ok)" >> $LOG

fetch_one() {  # $1=ticker $2=out $3=dates-from $4=start $5=end
  local tk=$1 out=$2 cal=$3 start=$4 end=$5 n=0 tmp=/tmp/lastrun_$1.txt
  while true; do
    n=$((n+1))
    pkill -9 -f "download_option_dailies" 2>/dev/null
    sleep 1
    echo "[$tk #$n $(date +%H:%M:%S)] launch" >> $LOG
    ./.venv/bin/python -m pipeline.download_option_dailies --ticker $tk --keep put \
      --out $out --dates-from $cal --start $start --end $end 2>>$LOG | tee -a $LOG > $tmp
    if grep -qE '0 day\(s\) to fetch|: 0 row\(s\) appended' $tmp; then
      echo "[$tk] DONE $(date +%H:%M:%S) — remaining todo are AV gaps" >> $LOG
      break
    fi
    sleep 2
  done
}

fetch_one MSFT data/msft_option_dailies_puts.csv data/msft_20yr_prices.csv 2010-05-10 2026-04-10
fetch_one QQQ  data/qqq_option_dailies_puts.csv  data/qqq_20yr_prices.csv  2011-03-23 2026-06-05
echo "[ALL] FETCH COMPLETE $(date +%H:%M:%S): MSFT puts + QQQ puts" >> $LOG
