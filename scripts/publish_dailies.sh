#!/bin/zsh
# Publish a fetched + validated canonical option-dailies store to the data-2026-06
# GitHub release, per the Option-Chain Data Pipeline ceremony in CLAUDE.md:
#
#   ./scripts/publish_dailies.sh GLD TLT      # publish gld_option_dailies.csv, tlt_...
#   DRY_RUN=1 ./scripts/publish_dailies.sh GLD # print every step, mutate nothing
#
# Automates the safe, deterministic half:
#   gzip -k -9  ->  sha256 into data/data_checksums.sha256  ->  gh release upload
#   ->  round-trip verify (re-download with CI's glob + shasum -c).
# It does NOT auto-edit ci.yml's cache lists (fragile YAML; reviewed by hand) — it
# PRINTS the exact lines to add. Cold storage: copied iff COLD_STORAGE_DIR is set
# (the path is private and deliberately not in any tracked file).
#
# The data store lives under data/; the checksum manifest's entries stay BARE
# basenames (CI verifies with `cd data && shasum -c`, and edge_search keys its
# lineage on the bare names), so the manifest format is unchanged by the reorg.
#
# PRECONDITION (the human-review gate): only publish a store validate_dailies.py
# read CLEAN, or whose CHAIN_CLEAN_START you have set. Publishing extends the
# release; never re-fetch into a canonical file (it re-pins every pinned number).
cd "${0:A:h}/.." || exit 1   # repo root (data/ holds the stores + manifest)
[ $# -ge 1 ] || { echo "usage: $0 TICKER [TICKER...]   (DRY_RUN=1 to rehearse)" >&2; exit 2; }
REL=data-2026-06; SUMS=data/data_checksums.sha256; GLOB='*_option_dailies*.csv.gz'
DRY=${DRY_RUN:-0}
run() { if [ "$DRY" = "1" ]; then echo "  DRY: $*"; else eval "$@"; fi; }

published=()   # BARE .gz basenames (the manifest + release asset names)
for tk in "$@"; do
  lc=${tk:l}; bn=${lc}_option_dailies.csv.gz; csv=data/${lc}_option_dailies.csv; gz=data/$bn
  [ -f "$csv" ] || { echo "[$tk][ERR] $csv not found — fetch first (./scripts/fetch_batch.sh $tk)" >&2; exit 1; }
  echo "== $tk =="
  run "gzip -k -9 -f '$csv'"
  if grep -q "  ${bn}\$" "$SUMS" 2>/dev/null; then
    echo "  sha256 already in $SUMS"
  elif [ "$DRY" = "1" ]; then
    echo "  DRY: (cd data && shasum -a 256 $bn) >> $SUMS"
  else
    [ -n "$(tail -c1 "$SUMS" 2>/dev/null)" ] && printf '\n' >> "$SUMS"
    ( cd data && shasum -a 256 "$bn" ) >> "$SUMS" && echo "  appended sha256 for $bn"
  fi
  run "gh release upload '$REL' '$gz' --clobber"
  published+=("$bn")
done

echo "== round-trip verify (download with CI's glob, shasum -c) =="
if [ "$DRY" = "1" ]; then
  echo "  DRY: (skipped)"
else
  tmp=$(mktemp -d)
  ( cd "$tmp" && gh release download "$REL" --pattern "$GLOB" --clobber >/dev/null 2>&1 )
  ok=1
  for bn in "${published[@]}"; do
    if [ -f "$tmp/$bn" ] && grep "  ${bn}\$" "$SUMS" | (cd "$tmp" && shasum -a 256 -c >/dev/null 2>&1); then
      echo "  $bn: OK"
    else echo "  $bn: VERIFY FAILED" >&2; ok=0; fi
  done
  rm -rf "$tmp"
  [ "$ok" = "1" ] || { echo "[ERR] round-trip verify failed" >&2; exit 1; }
fi

echo "== cold storage =="
if [ -n "$COLD_STORAGE_DIR" ] && [ "$DRY" != "1" ]; then
  cp "${published[@]/#/data/}" "$SUMS" "$COLD_STORAGE_DIR"/ && echo "  copied ${#published[@]} .gz + checksums to cold storage"
else
  echo "  set COLD_STORAGE_DIR to copy the .gz + $SUMS there (path kept private; or do it by hand)"
fi

echo ""
echo "MANUAL FOLLOW-UP (the human-review steps the ceremony reserves):"
echo "  1. Add to BOTH chain-data cache 'path:' lists in .github/workflows/ci.yml:"
for tk in "$@"; do echo "         data/${tk:l}_option_dailies.csv.gz"; done
echo "  2. Commit data/data_checksums.sha256 + ci.yml on a feat/ branch -> PR (data files stay gitignored)."
echo "  3. Refresh the cold-storage provenance README (spans, gap days, wing)."
