#!/usr/bin/env bash
# Fetch the large option-chain datasets from the repo's data release into data/.
# They are deliberately NOT in git history (raw CSVs run 77-281MB,
# 17-65MB gzipped); CI runs this too.
# Requires: gh CLI authenticated with repo access.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (data/ holds the stores + manifest)
gh release download data-2026-06 --pattern '*_option_dailies*.csv.gz' --clobber --dir data
( cd data && { sha256sum -c data_checksums.sha256 2>/dev/null || shasum -a 256 -c data_checksums.sha256; } )
echo "OK. Optional: gunzip -k data/*_option_dailies.csv.gz for the faster raw-CSV local path."
