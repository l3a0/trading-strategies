export const meta = {
  name: 'onboard',
  description: 'Onboard tickers end-to-end for the edge-search pipeline. SAFE (default): read-only validate gate -> structure campaign -> triage. LIVE (args.live=true): adds the fetch front-half, auto-applies KNOWN data repairs (a proposed CHAIN_CLEAN_START clip; a split-driven price-scale mismatch), and publishes. Novel pathologies and campaign survivors ALWAYS flag to a human. args: {tickers:[...], live:false}.',
  phases: [
    { title: 'Fetch', detail: 'live only — fetch_batch.sh' },
    { title: 'Clean gate', detail: 'validate; live auto-applies known repairs' },
    { title: 'Publish', detail: 'live only — publish_dailies.sh' },
    { title: 'Campaign', detail: 'structure campaign over the clean set' },
    { title: 'Triage', detail: 'kill, or adversarially vet + flag a survivor' },
  ],
}

const PY = './.venv/bin/python'
// the Workflow tool delivers `args` as a JSON string — parse it (tolerate object/undefined too)
const A = typeof args === 'string' ? JSON.parse(args) : (args && typeof args === 'object' ? args : {})
const LIVE = !!A.live
const TICKERS = (Array.isArray(A.tickers) && A.tickers.length)
  ? A.tickers : ['GLD', 'XLE', 'EEM']   // safe default: already-onboarded sample
log(`mode=${LIVE ? 'LIVE (fetch + auto-apply known repairs + publish)' : 'SAFE (read-only)'}  tickers=[${TICKERS.join(', ')}]`)

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
          by_survivor: { type: 'boolean' }, measurement_invalid: { type: 'boolean' },
        },
        required: ['template', 'ticker', 'by_survivor'],
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
  return { ticker: tk, gate }
}

// Per-ticker onboarding in parallel (each independent); barrier before the batch campaign.
const onboarded = (await parallel(TICKERS.map(tk => () => onboardOne(tk)))).filter(o => o && o.gate)
const clean = onboarded.filter(o => o.gate.action === 'proceed' || o.gate.action === 'repaired').map(o => o.ticker)
const autoClean = onboarded.filter(o => o.gate.action === 'auto-clean').map(o => o.gate)
const dataFlags = onboarded.filter(o => o.gate.action === 'human-flag').map(o => o.gate)
log(`clean gate -> ${clean.length} ready, ${autoClean.length} auto-cleanable, ${dataFlags.length} human-flag`)

if (!clean.length) {
  return { mode: LIVE ? 'LIVE' : 'SAFE', outcome: 'no clean tickers to campaign',
           human_queue: { data_flags: dataFlags, auto_cleanable: autoClean, survivor_flags: [] } }
}

// One structure campaign over the clean set (TLT sealed).
const camp = await agent(
  `SAFE — read-only. Run the engine-re-run STRUCTURE campaign on the clean tickers
   [${clean.join(', ')}] with TLT sealed, and return every cell. JSON-emitting command:
     ${PY} -c "import json,edge_search as e; rows=e.run_structure_campaign(e.Campaign(search=tuple('${clean.join(',')}'.split(',')),sealed=('TLT',))); print(json.dumps([{k:r.get(k) for k in ('template','ticker','t_stat_newey_west','p_value','by_survivor','measurement_invalid')} for r in rows]))"
   Parse the JSON array; rename t_stat_newey_west -> t_nw. Return rows. (~40-70s; let it finish.)`,
  { phase: 'Campaign', label: 'structure-campaign', schema: CAMP })
const rows = camp.rows || []
const survivors = rows.filter(r => r.by_survivor)
log(`campaign -> ${rows.length} cells, ${survivors.length} survivor(s)`)

// Triage: kill, or adversarially vet a survivor before it reaches a human.
let critiques = []
if (survivors.length) {
  critiques = (await parallel(survivors.map(s => () =>
    agent(`Adversarially REFUTE this campaign survivor before a human sees it: ${JSON.stringify(s)}.
       Real variance-risk premium, or an artifact (regime-specific / cost-fragile / trailing-vol
       confound)? Default to refuted if uncertain.`,
      { phase: 'Triage', label: `refute:${s.ticker}:${s.template}`, schema: VERDICT })))).filter(Boolean)
}

// ---- PUBLISH-READY (live only): assemble the review-ready PR bundle, STAGED not committed ----
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
  outcome: survivors.length === 0
    ? `KILLED — ${clean.length} clean tickers campaigned, 0 survivors / ${rows.length} cells`
    : `${survivors.length} survivor(s) flagged for human promotion`,
  human_queue: {
    data_flags: dataFlags,                                              // novel pathologies
    auto_cleanable: autoClean,                                          // known clips (safe mode only)
    survivor_flags: survivors.map((s, i) => ({ ...s, critique: critiques[i] || null })),
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
