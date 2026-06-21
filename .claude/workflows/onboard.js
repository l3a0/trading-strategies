export const meta = {
  name: 'onboard',
  description: 'Onboard tickers for the edge-search pipeline ONE AT A TIME — each ticker runs its complete lifecycle (fetch -> clean gate -> publish -> single-ticker structure campaign -> triage) to completion before the next ticker starts. SAFE (default): read-only validate gate -> campaign -> triage. LIVE (args.live=true): adds the fetch front-half (sequential, per the shared API rate budget), auto-applies KNOWN data repairs (a proposed CHAIN_CLEAN_START clip; a split-driven price-scale mismatch), and publishes. Novel pathologies and campaign survivors ALWAYS flag to a human. args: {tickers:[...], live:false}.',
  phases: [
    { title: 'Fetch', detail: 'live only — fetch_batch.sh (sequential, one ticker at a time)' },
    { title: 'Clean gate', detail: 'validate; live auto-applies known repairs' },
    { title: 'Publish', detail: 'live only — publish_dailies.sh' },
    { title: 'Campaign', detail: 'single-ticker structure smoke test (this ticker alone, TLT sealed)' },
    { title: 'Triage', detail: 'kill, or adversarially vet + flag a survivor' },
    { title: 'Publish-ready', detail: 'live only — assemble the cross-ticker PR bundle (staged)' },
  ],
}

const PY = './.venv/bin/python'
// the Workflow tool delivers `args` as a JSON string — parse it (tolerate object/undefined too)
const A = typeof args === 'string' ? JSON.parse(args) : (args && typeof args === 'object' ? args : {})
const LIVE = !!A.live
const TICKERS = (Array.isArray(A.tickers) && A.tickers.length)
  ? A.tickers : ['GLD', 'XLE', 'EEM']   // safe default: already-onboarded sample
log(`mode=${LIVE ? 'LIVE (fetch + auto-apply known repairs + publish)' : 'SAFE (read-only)'}  tickers=[${TICKERS.join(', ')}]  (each ticker end-to-end before the next)`)

const GATE = {
  type: 'object', additionalProperties: false,
  properties: {
    ticker: { type: 'string' },
    status: { type: 'string', enum: ['CLEAN', 'CLIP', 'UNVERIFIED'] },
    scale_ok: { type: 'boolean' },
    // proceed = CLEAN; repaired = a known fix was applied + re-validated CLEAN (live);
    // auto-clean = a known fix WOULD apply but safe mode didn't (excluded from the batch);
    // human-flag = unverified / novel pathology / no known repair.
    action: { type: 'string', enum: ['proceed', 'repaired', 'auto-clean', 'human-flag'] },
    applied: { type: 'string' },
    detail: { type: 'string' },
  },
  required: ['ticker', 'status', 'scale_ok', 'action', 'detail'],
}
const CAMP = {
  type: 'object', additionalProperties: false,
  properties: {
    rows: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          template: { type: 'string' }, ticker: { type: 'string' },
          t_nw: { type: ['number', 'null'] }, p_value: { type: ['number', 'null'] },
          elond_survivor: { type: 'boolean' },   // e-LOND, the FDR control of record (#3b)
          by_survivor: { type: 'boolean' },       // BY, retained as a diagnostic
          measurement_invalid: { type: 'boolean' },
        },
        required: ['template', 'ticker', 'elond_survivor', 'by_survivor'],
      },
    },
  },
  required: ['rows'],
}
const VERDICT = {
  type: 'object', additionalProperties: false,
  properties: { refuted: { type: 'boolean' }, reason: { type: 'string' } },
  required: ['refuted', 'reason'],
}

function cleanPrompt(tk) {
  const run = `Run:  ${PY} validate_dailies.py ${tk}
   Read the VERDICT (CLEAN / CLIP / UNVERIFIED) and the "price-vs-chain scale" line (scale_ok).`
  if (!LIVE) {
    return `SAFE MODE — strictly read-only. Do NOT edit files, fetch, gzip, or publish.
   ${run}
   Route (do NOT apply anything): CLEAN -> action "proceed".  CLIP -> action "auto-clean"
   (the live loop would set CHAIN_CLEAN_START and re-validate; safe mode leaves it, so it is
   excluded from this campaign).  UNVERIFIED or a SCALE MISMATCH -> action "human-flag".
   ticker="${tk}".`
  }
  return `LIVE MODE — you MAY apply KNOWN, bounded repairs only, then re-validate. Loop <=4 attempts.
   ${run}
   - CLEAN -> action "proceed".
   - CLIP at date D -> APPLY: add '${tk}': 'D' to the CHAIN_CLEAN_START dict in
     real_cc_backtest.py, then re-run validate to confirm CLEAN. action "repaired",
     applied="CHAIN_CLEAN_START['${tk}']='D'".
   - SCALE MISMATCH caused by a stock split -> the repair already exists in
     load_unadjusted_prices (_unsplit_factor backs the split out); regenerate the price file:
     rm ${tk.toLowerCase()}_*_unadjusted.csv, then re-run validate (it refetches split-corrected).
     action "repaired", applied="regenerated split-corrected price file".
   - UNVERIFIED, or ANY pathology with no known repair -> DO NOT edit anything. action
     "human-flag", with a diagnosis + a DRAFT fix in detail. NEVER invent a new data-cleaning
     rule that would pin numbers without human review.
   ticker="${tk}".`
}

// One ticker's COMPLETE lifecycle: fetch -> clean gate -> publish -> single-ticker structure
// campaign -> triage. The sequential loop below runs this to completion for one ticker before the
// next begins, so a ticker is finished end-to-end first and the LIVE fetch is serialized (the
// shared API rate budget — one ticker to completion before the next — is a standing preference).
async function onboardOne(tk) {
  if (LIVE) {
    await agent(`LIVE — run ./fetch_batch.sh ${tk} to completion (resumable; can take a while on a
     cold ticker). Report the store path + row count.`,
      { phase: 'Fetch', label: `fetch:${tk}` })
  }
  const gate = await agent(cleanPrompt(tk), { phase: 'Clean gate', label: `clean:${tk}`, schema: GATE })
  const ready = gate && (gate.action === 'proceed' || gate.action === 'repaired')
  if (LIVE && ready) {
    await agent(`LIVE — run ./publish_dailies.sh ${tk}; confirm the round-trip verify passed.`,
      { phase: 'Publish', label: `publish:${tk}` })
  }

  // Campaign + triage are part of THIS ticker's lifecycle. Only a clean/repaired ticker is
  // campaigned (it has a sound store to score); an auto-clean (safe mode) or human-flag ticker
  // stops at the gate. The campaign is the SINGLE-TICKER onboarding smoke test — this ticker
  // ALONE, TLT sealed (the per-ticker check pinned by TestNvdaStructureCampaign) — NOT the
  // cross-sectional batch; folding the ticker into STRUCTURE_SEARCH (which re-pins the whole
  // 28-cell campaign under one FDR pass) stays the deliberate human step flagged below.
  let rows = [], survivors = [], critiques = []
  if (ready) {
    const camp = await agent(
      `SAFE — read-only. Run the engine-re-run STRUCTURE campaign on ${tk} ALONE (the single-ticker
       onboarding smoke test, TLT sealed by omission — a 1-ticker search never includes it), and
       return every cell. This mirrors the pinned per-ticker check (TestNvdaStructureCampaign).
       JSON-emitting command:
         ${PY} -c "import json,edge_search as e; rows=e.run_structure_campaign(e.Campaign(search=('${tk}',))); print(json.dumps([{k:r.get(k) for k in ('template','ticker','t_stat_newey_west','p_value','elond_survivor','by_survivor','measurement_invalid')} for r in rows]))"
       Parse the JSON array; rename t_stat_newey_west -> t_nw. Return rows. (~10-20s; let it finish.)`,
      { phase: 'Campaign', label: `campaign:${tk}`, schema: CAMP })
    rows = (camp && camp.rows) || []
    // Flag a cell if EITHER FDR gate fires: elond_survivor is the control of record (#3b,
    // run_structure_campaign), by_survivor the retained BY diagnostic. The union is the conservative
    // direction for a HUMAN-review gate — over-flagging is cheap (a human dismisses it), missing a
    // real survivor is the danger — and the two are not guaranteed to coincide (e-LOND's (R+1) reward
    // can flag a cell BY does not). This is the opposite asymmetry from the proposer corpus, which
    // excludes on the control ALONE so a non-control cell stays proposable.
    survivors = rows.filter(r => r.elond_survivor || r.by_survivor)
    // Triage: adversarially vet this ticker's survivor(s) before any reaches a human. Index-aligned
    // with `survivors` (no filter, and parallel() preserves input order), so critiques[i] is
    // survivors[i]'s verdict even if one returns null.
    if (survivors.length) {
      critiques = await parallel(survivors.map(s => () =>
        agent(`Adversarially REFUTE this campaign survivor before a human sees it: ${JSON.stringify(s)}.
           Real variance-risk premium, or an artifact (regime-specific / cost-fragile / trailing-vol
           confound)? Default to refuted if uncertain.`,
          { phase: 'Triage', label: `refute:${s.ticker}:${s.template}`, schema: VERDICT }))
    }
  }
  log(`${tk}: gate=${gate ? gate.action : 'null'}` +
      (ready ? `, campaigned ${rows.length} cells, ${survivors.length} survivor(s)` : ' — not campaigned'))
  return { ticker: tk, gate, ready, rows, survivors, critiques }
}

// SEQUENTIAL: each ticker runs its complete lifecycle before the next begins (depth-first per
// ticker). The deliberate trade vs. the old all-tickers-in-parallel barrier — it finishes a ticker
// end-to-end first, and serializes the LIVE fetch to respect the shared API rate budget. (Price for
// it: no overlap on the read-only stages — acceptable for an onboarding tool, and a pipeline that
// overlapped them would also overlap the rate-budgeted fetch, which the preference forbids.)
const onboarded = []
for (const tk of TICKERS) {
  onboarded.push(await onboardOne(tk))
}

const clean = onboarded.filter(o => o.ready).map(o => o.ticker)
const autoClean = onboarded.filter(o => o.gate && o.gate.action === 'auto-clean').map(o => o.gate)
const dataFlags = onboarded.filter(o => o.gate && o.gate.action === 'human-flag').map(o => o.gate)
const rows = onboarded.flatMap(o => o.rows)
const survivors = onboarded.flatMap(o =>
  o.survivors.map((s, i) => ({ ...s, critique: o.critiques[i] || null })))
log(`onboarded ${onboarded.length} ticker(s) -> ${clean.length} clean/campaigned, ` +
    `${autoClean.length} auto-cleanable, ${dataFlags.length} human-flag; ` +
    `${rows.length} cells, ${survivors.length} survivor(s)`)

// ---- PUBLISH-READY (live only): assemble the review-ready PR bundle, STAGED not committed ----
// Deliberately assembled ONCE over all clean tickers, NOT per-ticker: the three target surfaces
// (ci.yml's cache lists, the shared docs/edge_search.md addendum, test_edge_search.py) are
// cross-ticker files, so per-ticker assembly would race N agents on the same edits. The per-ticker
// requirement governs the data/evidence lifecycle above; PR packaging is a downstream step over the
// already-finished set, reviewed as one PR.
// Everything here is mechanical transcription of the deterministic campaign output, so the agents
// auto-apply it; the human reviews the staged PR and CI validates the edits. The two judgments stay
// human: folding the ticker into the main STRUCTURE_SEARCH (which re-pins the whole campaign) and
// promoting any survivor are flagged as deliberate decisions, not done.
const STEP = {
  type: 'object', additionalProperties: false,
  properties: { surface: { type: 'string' }, ok: { type: 'boolean' }, summary: { type: 'string' } },
  required: ['surface', 'ok', 'summary'],
}
let prBundle = null
if (LIVE && clean.length) {
  phase('Publish-ready')
  const cells = JSON.stringify(rows)
  const findings = JSON.stringify(Object.fromEntries(onboarded.map(o => [o.ticker, o.gate])))
  const tks = clean.join(', ')
  prBundle = (await parallel([
    () => agent(`LIVE, STAGE-ONLY (never commit). Wire CI for [${tks}]: add each
       {ticker}_option_dailies.csv.gz to BOTH chain-data cache 'path:' lists in
       .github/workflows/ci.yml, then confirm it still parses:
       ${PY} -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"
       surface="ci.yml".`,
      { phase: 'Publish-ready', label: 'wire-ci', schema: STEP }),
    () => agent(`LIVE, STAGE-ONLY. Append a "Campaign 2 addendum — ${tks} (live-onboarded)" section to
       docs/edge_search.md recording the structure result (cells below) and any clean-gate data-hygiene
       finding (e.g. a split-driven scale repair). Match the doc's voice; markdownlint-clean (real
       headings, single trailing newline). Cells: ${cells}. Findings: ${findings}. surface="docs/edge_search.md".`,
      { phase: 'Publish-ready', label: 'document', schema: STEP }),
    () => agent(`LIVE, STAGE-ONLY. Add a CI-reproducible pin to test_edge_search.py for each onboarded
       ticker's single-ticker structure campaign — dataset-gated on the now-published store, asserting
       0 survivors and the cell t_NW signs. Cells: ${cells}. Follow the existing dataset-gated structure
       tests' style, then RUN the new test to confirm it passes before finishing. surface="test_edge_search.py".`,
      { phase: 'Publish-ready', label: 'pin', schema: STEP }),
    () => agent(`LIVE, STAGE-ONLY. git add the committed split-corrected price files for [${tks}]
       ({ticker}_*_unadjusted.csv) so they ship with the PR. surface="price files".`,
      { phase: 'Publish-ready', label: 'stage-prices', schema: STEP }),
  ])).filter(Boolean)
}

return {
  mode: LIVE ? 'LIVE' : 'SAFE',
  outcome: clean.length === 0
    ? `no clean tickers to campaign (${dataFlags.length} human-flagged, ${autoClean.length} auto-cleanable)`
    : survivors.length === 0
      ? `KILLED — ${clean.length} clean ticker(s) campaigned, 0 survivors / ${rows.length} cells`
      : `${survivors.length} survivor(s) flagged for human promotion`,
  human_queue: {
    data_flags: dataFlags,                                              // novel pathologies
    auto_cleanable: autoClean,                                          // known clips (safe mode only)
    survivor_flags: survivors,                                          // already enriched with the refute critique
  },
  publish_ready: prBundle ? {
    assembled: prBundle,
    review: 'STAGED, not committed — review the ci.yml / docs / test-pin edits + the released data, then /create-pr.',
    deliberate_decisions: [
      'Fold the onboarded ticker(s) into STRUCTURE_SEARCH (re-pins the whole campaign) — left for you, not auto-done.',
      ...(survivors.length ? ['Pre-register the flagged survivor before promoting.'] : []),
      ...(dataFlags.length ? ['Investigate the human-flagged ticker(s) before publishing them.'] : []),
    ],
  } : null,
  note: LIVE
    ? 'LIVE run published to the release and STAGED the PR bundle (ci.yml, docs, test pin, price files) — review + /create-pr.'
    : 'SAFE run — read-only; no fetch, edits, or uploads.',
}
