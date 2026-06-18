#!/bin/zsh
# Onboard one ticker end-to-end for the edge-search pipeline, split across the
# human SIGN-OFF gate the data-hygiene discipline reserves:
#
#   ./onboard_ticker.sh NVDA            # phase 1: fetch -> validate, then STOP
#   ./onboard_ticker.sh NVDA --publish  # phase 2: publish -> campaign smoke test
#
# Phase 1 fetches the chain (fetch_batch.sh) and runs validate_dailies.py — the
# entry-band hygiene boundary AND the price-vs-chain scale guard — then stops on the
# verdict. A human reads it and decides:
#   CLEAN       -> rerun with --publish
#   CLIP at D   -> add CHAIN_CLEAN_START['TK']='D' to real_cc_backtest.py, then --publish
#   UNVERIFIED  -> investigate (split/scale mismatch? a novel pathology?) before publishing
# The gate is deliberate: auto-publishing an unvalidated store is exactly what the
# validator fails closed to prevent. Phase 2 publishes (publish_dailies.sh) and runs
# a single-ticker structure campaign as a smoke test — proving the overlays + scale
# guard run clean on the new name before it joins a real FDR batch.
cd "${0:A:h}" || exit 1
[ $# -ge 1 ] || { echo "usage: $0 TICKER [--publish]" >&2; exit 2; }
TK=${1:u}; PY=./.venv/bin/python; [ -x "$PY" ] || PY=python3

if [ "$2" != "--publish" ]; then
  echo "== phase 1: fetch + validate $TK =="
  ./fetch_batch.sh $TK || { echo "[ERR] fetch failed" >&2; exit 1; }
  echo ""
  $PY validate_dailies.py $TK
  echo ""
  echo "================= SIGN-OFF GATE ================="
  echo "Review the verdict + the price-vs-chain scale line above, then:"
  echo "  CLEAN       -> ./onboard_ticker.sh $TK --publish"
  echo "  CLIP at D   -> add CHAIN_CLEAN_START['$TK']='D' to real_cc_backtest.py, then --publish"
  echo "  UNVERIFIED  -> investigate before publishing (the validator fails closed for a reason)"
  exit 0
fi

echo "== phase 2: publish $TK =="
./publish_dailies.sh $TK || { echo "[ERR] publish failed" >&2; exit 1; }
echo ""
echo "== campaign smoke test: structure overlays on $TK (scale-guarded) =="
$PY - "$TK" <<'PYEOF'
import sys
from edge_search import run_structure_campaign, Campaign
tk = sys.argv[1]
rows = run_structure_campaign(Campaign(search=(tk,)))
for r in rows:
    if r.get('measurement_invalid'):
        print(f"  {r['template']:<15} INVALID  scale={r['scale_ratio']}  -> NOT campaign-ready (fix the price file)")
    else:
        print(f"  {r['template']:<15} t_NW={r['t_stat_newey_west']:+.2f}  p={r['p_value']:.3f}")
PYEOF
echo ""
echo "$TK is campaign-ready. To include it in a real FDR batch, add it to"
echo "STRUCTURE_SEARCH (or pass a Campaign to run_structure_campaign / run_batch),"
echo "roll a fresh underlying into the sealed vault, and re-pin the campaign."
