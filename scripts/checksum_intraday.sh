#!/bin/bash
# Integrity baseline for the minute archive: sha256 every completed
# ``*_intraday_1min.csv.gz`` into ``data/intraday_checksums.sha256``.
#
# WHY A SEPARATE MANIFEST from data/data_checksums.sha256: that file is
# consumed by CI two ways — `sha256sum -c` on every chain-data job, and
# `hashFiles()` as the chain-data CACHE KEY. The minute archives are
# gitignored, unpublished, and absent in CI, so folding them in would (a)
# fail the verify step on missing files and (b) invalidate the option-chain
# cache on every ticker that lands, re-downloading gigabytes per increment.
# Two datasets, two lifecycles, two manifests.
#
# WHY THIS EXISTS AT ALL: the archive is ~10 GB, gitignored, and in no
# release. Checksums do not protect against LOSS — only cold storage does
# that — but they are the only thing that detects silent corruption, and
# without them a bit flip is indistinguishable from real market history.
#
# ONLY COMPLETED TICKERS ARE HASHED. A ticker qualifies when its
# ``.months.done`` marker exists AND no plain ``.csv`` remains beside the
# ``.gz``. The batch driver gzips in place, so mid-compression BOTH files
# exist and the ``.gz`` is a truncated prefix — hashing it would pin a
# checksum that changes the moment compression finishes. Skipped tickers
# are reported, never silently dropped.
#
# Safe to run against a live fetch: it is a snapshot of what is finished,
# and re-running after more tickers land is the intended workflow.
#
# Usage:  scripts/checksum_intraday.sh          # write the manifest
#         VERIFY=1 scripts/checksum_intraday.sh # check files against it
set -u
cd "$(dirname "$0")/.."
DIR=data/sp500_intraday_1min
OUT=data/intraday_checksums.sha256

SHA=$(command -v sha256sum || command -v shasum)
case "$SHA" in *shasum) SHA="$SHA -a 256";; esac

if [ "${VERIFY:-0}" = "1" ]; then
  [ -f "$OUT" ] || { echo "no manifest at $OUT — run without VERIFY first"; exit 1; }
  echo "== verifying $(wc -l < "$OUT" | tr -d ' ') archives against $OUT"
  ( cd "$DIR" && $SHA -c "../../$OUT" )
  exit $?
fi

tmp=$(mktemp)
skipped=0
total=0
for gz in "$DIR"/*_intraday_1min.csv.gz; do
  [ -e "$gz" ] || continue
  base=${gz%.gz}
  total=$((total + 1))
  # in-flight guard: marker present, and the source .csv already removed
  if [ ! -f "$base.months.done" ] || [ -f "$base" ]; then
    echo "  skip (in flight): $(basename "$gz")"
    skipped=$((skipped + 1))
    continue
  fi
  ( cd "$DIR" && $SHA "$(basename "$gz")" ) >> "$tmp"
done

LC_ALL=C sort -k2 "$tmp" > "$OUT"
rm -f "$tmp"
echo "== wrote $(wc -l < "$OUT" | tr -d ' ') checksums to $OUT"
echo "   ($total archives seen, $skipped skipped as in flight)"
echo "   verify with: VERIFY=1 scripts/checksum_intraday.sh"
