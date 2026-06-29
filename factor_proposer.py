"""factor_proposer.py — the factor LLM-author front-end (H2a of docs/integration_plan.md).

The factor analog of edge_search.py's option proposer: it swaps the option coordinate schema
(`{overlay, ticker, params}`) for the factor one (`{expr, universe, predicted_sign}`) and the
`StructureCandidate` grammar gate for the `Expr` grammar gate, while the seal + the score/judge/record
loop stay byte-identical in SHAPE. OPTION-INDEPENDENT: the whole factor domain imports nothing from
`edge_search` (which pulls the option chains) — this module reaches only the dependency-free wire
(`read_gate_wire`), the shared transports (`proposer_clients`), the factor modules, and the FDR control
(`evalue_fdr`, which holds the shared `_asymptotic_p` convention). The two pre-existing edge_search imports
in `factor_engine`/`factor_backend` (`STRUCTURE_END`, `_asymptotic_p`) were extracted — `_asymptotic_p`
moved to `evalue_fdr`, and the factor panel's as-of date is the factor domain's own `FACTOR_END`.

THE SEAL IS THE SAME THREE LAYERS (docs/read_gate.md), reused not reinvented:
  1. `build_factor_proposer_prompt` runs `assert_numberless` on the scrubbed-corpus INPUT before
     composing any prompt — the #1 builder-bug defense (a raw ledger row carries banned result keys).
  2. `score_and_record_factor` (the oracle seam) ALWAYS records before replying — every look is a
     recorded look, committed to the lifetime e-LOND stream.
  3. The reply carries ONLY `FACTOR_PROPOSAL_FIELDS` + a one-bit verdict, never a result statistic —
     `rejected[].proposal` is re-scrubbed and the whole reply is `assert_numberless`-checked.

OFF BY DEFAULT + PROMOTION CLOSED. `_resolve_factor_llm_author` returns None until a real factor client
is wired (H2b) AND env-gated on, so the default proposer is the deterministic menu-walker (no model).
A flagged cell escalates to manual pre-registration + the Phase-C time-axis holdout — never auto-promoted.

H2a builds the gate + the seal + the loop, STUB-author tested (no real LLM). H2b wires the real Claude
clients (the option `ClaudeProposer`/`ClaudeCodeProposer` mechanics, with this module's prompt + parse).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

from evalue_fdr import online_fdr_survivors
from factor_engine import ExprFactor
from factor_grammar import (
    BINARY_OPS,
    MAX_DEPTH,
    OPERANDS,
    TS_OPS,
    UNARY_OPS,
    WINDOWS,
    Expr,
    ExprGrammarError,
    canonical_expr_key,
    enumerate_exprs,
    validate_expr,
)
from factor_search import FACTOR_LEDGER_PATH, _record_factor_cells
from proposer_clients import (   # the domain-agnostic Claude transports (H2b-pre, shared w/ the option author)
    ClaudeCodeProposer as _ClaudeCodeBase,
    ClaudeProposer as _ClaudeApiBase,
)
from read_gate_wire import (
    FACTOR_PROPOSAL_FIELDS,
    REQUIRED_MODEL_FIELDS,
    WIRE_VERSION,
    ProposalBatch,
    _parse_proposal_array,
    assert_numberless,
)

FACTOR_PROVENANCE_PATH = 'factor_proposal_provenance.jsonl'

# (grammar_menu, scrubbed_corpus, onboarded_universes) -> ProposalBatch — the factor author's contract.
# Its own type (NOT edge_search.LLMProposer, which is typed to StructureTemplate): the factor menu is the
# grammar-space description, not a list of option templates.
FactorProposer = Callable[[dict[str, Any], list[dict[str, Any]], tuple[str, ...]], ProposalBatch]


# --- Expr <-> JSON coordinate serialization (the one thing genuinely new for an LLM author) -----------
def expr_to_dict(e: Expr) -> dict[str, Any]:
    """Serialize an `Expr` tree to a nested JSON coordinate dict the LLM emits/reads:
    `{op: 'field', operand: 'close'}` for a leaf, else `{op, args: [...], window?}` (window only for a
    time-series op). Pure coordinates — op / operand / window — so the result is numberless by
    construction (no result statistic can appear here)."""
    if e.op == 'field':
        return {'op': 'field', 'operand': e.operand}
    d: dict[str, Any] = {'op': e.op, 'args': [expr_to_dict(a) for a in e.args]}
    if e.op in TS_OPS:
        d['window'] = e.window
    return d


def dict_to_expr(d: Any) -> Expr:
    """Parse a coordinate dict back into an `Expr`, RAISING `ExprGrammarError` on a structurally
    malformed reply (not a dict, missing/!str `op`, non-list `args`, non-int `window`). This guarantees
    a well-FORMED `Expr` to hand the grammar gate; grammar VALIDITY (the operator alphabet, arity,
    windows, depth) is `validate_expr`'s job downstream, so a structurally-fine but off-grammar tree
    reaches the gate and is rejected there with a reason rather than crashing the round."""
    if not isinstance(d, dict) or not isinstance(d.get('op'), str):
        raise ExprGrammarError(f'not a coordinate dict with a str op: {d!r}')
    op = d['op']
    if op == 'field':
        operand = d.get('operand', '')
        if not isinstance(operand, str):
            raise ExprGrammarError(f'field operand must be a str, got {operand!r}')
        return Expr('field', operand=operand)
    raw_args = d.get('args', [])
    if not isinstance(raw_args, (list, tuple)):
        raise ExprGrammarError(f'args must be a list, got {type(raw_args).__name__}')
    args = tuple(dict_to_expr(a) for a in raw_args)
    window = d.get('window', 0)
    if type(window) is not int:               # type-strict (no bool, no float) — matches validate_expr
        raise ExprGrammarError(f'window must be an int, got {window!r}')
    return Expr(op, args=args, window=window)


# --- the scrubbed scoreboard (FACTOR_SAFE_FIELDS allow-list — the proposer-visible projection) --------
# The factor analog of edge_search.SAFE_FIELDS. The coordinates the proposer MAY see: the readable Expr
# (`expr`), the canonical identity (`key`), the universe (`ticker` slot), the bet (`predicted_sign`), the
# phase + span. NOTABLY ABSENT vs the option corpus: `family`. The option family is DECLARED a-priori by
# the grammar (a coordinate); the FACTOR family is DERIVED by the loading regression (a measurement), so
# it stays out of the proposer's view — only the one-bit KILLED/INVALID verdict crosses, never the typing.
FACTOR_SAFE_FIELDS: tuple[str, ...] = ('phase', 'key', 'expr', 'ticker', 'predicted_sign', 'end')


def scrub_factor_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one factor-ledger row to FACTOR_SAFE_FIELDS + a one-bit verdict. Allow-list (not
    deny-list), so no result statistic — and not the derived `family` — can leak through a forgotten
    field. `expr` is defensively deep-copied (a nested coordinate tree) so a consumer mutating the corpus
    cannot reach the source row. The verdict keys off `elond_survivor` — the e-LOND control of record,
    NOT the `by_survivor` diagnostic — exactly as the option scrub does, so the corpus exclusion tracks
    the control. `measurement_invalid` surfaces as INVALID (a per-UNIVERSE data state, not a result)."""
    out = {f: (json.loads(json.dumps(row.get(f))) if f == 'expr' else row.get(f))
           for f in FACTOR_SAFE_FIELDS}
    out['verdict'] = ('INVALID' if row.get('measurement_invalid')
                      else 'SURVIVED' if row.get('elond_survivor') else 'KILLED')
    return out


def build_factor_proposer_corpus(ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The scrubbed view of the lifetime factor ledger an automated proposer may read — coordinates +
    verdict, no numbers and no derived family. SURVIVED rows are EXCLUDED (a survivor escalates to manual
    pre-registration out-of-band, never back into automated proposal), so the corpus is the duds (KILLED)
    plus unmeasurable universes (INVALID). `assert_numberless`-checked before return — defense-in-depth
    behind the allow-list, identical to the option corpus."""
    corpus = [s for r in ledger_rows
              if (s := scrub_factor_ledger_row(r))['verdict'] != 'SURVIVED']
    assert_numberless(corpus, 'factor_proposer_corpus')
    return corpus


def render_factor_proposer_corpus(scrubbed: list[dict[str, Any]]) -> str:
    """A markdown table of the scrubbed corpus — the exprs tried, on which universe, and the verdict,
    every result statistic absent. Safe to hand to a proposer."""
    if not scrubbed:
        return '(no factor comparisons recorded yet)'
    lines = ['| expr | universe | predicted | verdict |', '| --- | --- | --- | --- |']
    for r in scrubbed:
        sign = '+1' if r['predicted_sign'] > 0 else '-1'
        lines.append(f'| `{_expr_str(r["expr"])}` | {r["ticker"]} | {sign} | {r["verdict"]} |')
    return '\n'.join(lines)


def _expr_str(d: Any) -> str:
    """A compact human-readable spelling of a coordinate-dict Expr, e.g. `rank(ts_mean(ret,20))`."""
    if not isinstance(d, dict):
        return str(d)
    if d.get('op') == 'field':
        return str(d.get('operand'))
    inner = ','.join(_expr_str(a) for a in d.get('args', []))
    win = f',{d["window"]}' if 'window' in d else ''
    return f'{d.get("op")}({inner}{win})'


def factor_grammar_menu() -> dict[str, Any]:
    """The proposable grammar SPACE — the legal coordinate alphabet the author draws from (operands, the
    three operator buckets, the window menu, the depth cap). The factor analog of the option
    `_render_grammar_menu`'s grid: numberless by construction (it is the grammar definition, never an
    engine output). The author emits an `Expr` coordinate tree from THIS alphabet; the gate validates it."""
    return {
        'operands': list(OPERANDS),
        'ts_ops': list(TS_OPS), 'unary_ops': list(UNARY_OPS), 'binary_ops': list(BINARY_OPS),
        'windows': list(WINDOWS), 'max_depth': MAX_DEPTH,
    }


# --- the lineage-free dedup identity (matches the scrubbed corpus's coordinates) ----------------------
def _factor_proposer_key(key: str, universe: str) -> tuple[str, str]:
    """The proposer's LINEAGE-FREE dedup identity: (canonical_expr_key, universe), matching the scrubbed
    corpus's coordinates exactly (scrub drops the lineage hash — it is not in FACTOR_SAFE_FIELDS). ONE
    canonicalizer shared by the candidate side and the corpus side, so the proposer's skip and the corpus
    cannot desync. Lineage-free is deliberate — within the published lineage it skips what's tried; a data
    refresh is a separate path, exactly as the option `_proposer_key`."""
    return (key, universe)


def _candidate_key(c: ExprFactor, universe: str) -> tuple[str, str]:
    """A factor candidate's proposer dedup key (see _factor_proposer_key)."""
    return _factor_proposer_key(canonical_expr_key(c.expr), universe)


# --- the deterministic menu-walker (the author=None default) ------------------------------------------
def propose_factor_candidates(backend: Any, tried: set[tuple[str, str]]) -> tuple[list[ExprFactor], list[str]]:
    """The deterministic Phase-1 proposer: walk the bounded grammar slice (`enumerate_exprs`) on the
    backend's ONE universe, minus what the corpus says was tried. Returns `(candidates, needs_onboard)`;
    the menu-walker never needs onboarding (it proposes only for the loaded panel). The future LLM author
    swaps its JSON output for this enumerator while the gate/judge/record stay identical."""
    cands = [ExprFactor(e, 1) for e in enumerate_exprs()
             if _factor_proposer_key(canonical_expr_key(e), backend.universe) not in tried]
    return cands, []


# --- the LLM-author front-end (the gate on coordinate-only proposals) ---------------------------------
def llm_propose_factor_candidates(
    author: FactorProposer,
    backend: Any,
    *,
    search: frozenset[str],
    sealed: frozenset[str],
    corpus: list[dict[str, Any]] | None = None,
    tried_keys: set[tuple[str, str]] | None = None,
    max_batch: int = 16,
) -> tuple[list[ExprFactor], list[str], list[dict[str, Any]], ProposalBatch]:
    """The factor LLM-author front-end — the drop-in for `propose_factor_candidates`. Hand the author
    exactly what the menu-walker sees (the grammar menu, the scrubbed corpus, the onboarded universe),
    then GATE its raw output:

      * `dict_to_expr` + `validate_expr` resolve each `expr` coordinate tree to a grammar-valid `Expr` —
        a structurally-broken or off-grammar tree is REJECTED with a reason (the grammar gate);
      * `predicted_sign` must be the committed +1 (the harvesting bet; type-strict, no bool);
      * reject SEALED universes (the holdout must never run) and universes off the committed `search`
        set (a universe edit is human-gated); route an un-onboarded search universe (not the loaded
        panel) to `needs_onboard`;
      * drop already-tried cells (`_candidate_key` vs the corpus coordinates) and within-batch
        duplicates; cap accepted at `max_batch` (an LLM must not burn the e-LOND budget).

    Returns `(candidates, needs_onboard, rejected, batch)`. `batch` carries the model identity for the
    provenance log; it is NOT mixed into the comparison rows. A malformed/off-grammar proposal is dropped
    WITH A REASON, never crashing the round."""
    tried = tried_keys or set()
    menu = factor_grammar_menu()
    onboarded = (backend.universe,) if backend.universe in search else ()
    batch = author(menu, list(corpus or []), onboarded)
    cands: list[ExprFactor] = []
    needs_onboard: list[str] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    # Guard a malformed author: `proposals` must be an iterable of dicts. A non-list/None (a broken author,
    # not a single bad proposal) yields zero candidates rather than crashing the round; an individual
    # non-dict proposal is caught per-item below (the `except` on the coordinate unpack).
    proposals = batch.proposals if isinstance(batch.proposals, (list, tuple)) else ()
    for p in proposals:
        try:
            expr_coord, universe, sign = p['expr'], p['universe'], p['predicted_sign']
        except (KeyError, TypeError) as exc:
            rejected.append({'proposal': p, 'reason': f'malformed: {exc!r}'})
            continue
        if universe in sealed:
            rejected.append({'proposal': p, 'reason': 'sealed universe — must never run'})
            continue
        if universe not in search:
            rejected.append({'proposal': p, 'reason': 'off-search universe (universe edit is human-gated)'})
            continue
        try:
            expr = validate_expr(dict_to_expr(expr_coord))      # parse + grammar-gate the coordinate tree
        except ExprGrammarError as exc:
            rejected.append({'proposal': p, 'reason': f'off-grammar: {exc}'})
            continue
        if type(sign) is not int or sign != 1:       # type-strict, like validate (no bool); +1-only menu
            rejected.append({'proposal': p, 'reason': f'predicted_sign {sign!r} != menu +1'})
            continue
        if universe != backend.universe:              # a search universe, but not the loaded panel
            if universe not in needs_onboard:
                needs_onboard.append(universe)
            continue
        cand = ExprFactor(expr, sign)                 # sign now known-good +1; construct the candidate
        key = _candidate_key(cand, universe)
        if key in tried or key in seen:
            continue                                  # already tried, or duplicated within this batch
        seen.add(key)
        cands.append(cand)
        if len(cands) >= max_batch:
            break
    return cands, needs_onboard, rejected, batch


# --- the numberless prompt (the seal's first layer) ---------------------------------------------------
def build_factor_proposer_prompt(
    menu: dict[str, Any],
    scrubbed_corpus: list[dict[str, Any]],
    onboarded_universes: tuple[str, ...],
    *,
    max_proposals: int = 16,
) -> str:
    """Assemble the NUMBERLESS prompt the oracle-side factor author sees — the load-bearing seal of the
    in-process design. A pure function of the THREE author inputs (grammar menu, scrubbed corpus,
    onboarded universes) plus static instruction text; it NEVER reads the answer-key ledger or any engine
    output, so no result statistic is in scope to format. THE SEAL is `assert_numberless` on the
    scrubbed-corpus INPUT (run first): the effective guard against the #1 builder bug — a raw ledger row
    (with banned result keys) passed where the scrubbed corpus belongs fails loudly here, before the
    model sees anything."""
    assert_numberless(scrubbed_corpus, 'factor_proposer_prompt_corpus')
    ops = (f"time-series (need a window): {menu['ts_ops']}; cross-sectional/sign (no window): "
           f"{menu['unary_ops']}; binary (two args): {menu['binary_ops']}")
    return (
        'Propose alpha-factor expressions to test for cross-sectional predictive power on an equity '
        'panel. You are GENERATING HYPOTHESES from economic reasoning — you are NOT shown any result '
        '(no IC, t-stat, p-value, or mechanism verdict is in scope, by construction).\n\n'
        '## The grammar (the only legal coordinate alphabet)\n'
        f"- base operands: {menu['operands']} (ret == daily pct-change of close)\n"
        f'- operators: {ops}\n'
        f"- windows (time-series only): {menu['windows']}; max operator depth: {menu['max_depth']}\n\n"
        '## Coordinate format\n'
        'Emit each factor as a nested JSON `expr` tree: a leaf is {"op":"field","operand":"close"|"ret"}; '
        'an operator node is {"op":<operator>,"args":[<expr>,...]} with "window":<int> ADDED iff the op '
        'is time-series. Example — rank of 20-day mean return:\n'
        '  {"op":"rank","args":[{"op":"ts_mean","args":[{"op":"field","operand":"ret"}],"window":20}]}\n\n'
        f'## Universes you may target (onboarded)\n{list(onboarded_universes)}\n\n'
        '## Already tried (avoid re-proposing these)\n'
        f'{render_factor_proposer_corpus(scrubbed_corpus)}\n\n'
        '## Output\n'
        f'Return ONLY a JSON array of at most {max_proposals} objects, each '
        '{"expr":<tree>,"universe":<id>,"predicted_sign":1,"reasoning":<one line: the a-priori economic '
        'story>}. predicted_sign is +1 (the harvesting bet). No prose outside the array.')


def parse_factor_proposals(text: str) -> list[dict[str, Any]]:
    """Parse a model reply into the factor coordinate-dict list the gate consumes — the shared
    `_parse_proposal_array` (tolerant of a ```json fence, RAISES on an unrecoverable reply). Per-proposal
    validity is `llm_propose_factor_candidates`' job."""
    return _parse_proposal_array(text)


# --- provenance (the lineage-adjacent audit log) ------------------------------------------------------
def record_factor_provenance(
    batch: ProposalBatch,
    accepted: list[tuple[str, str]],
    *,
    round_id: str,
    path: str = FACTOR_PROVENANCE_PATH,
) -> None:
    """Append ONE row to the factor proposal-provenance audit trail — a SEPARATE artifact from the
    comparison ledger. Carries the exact model identity + the round's proposals (coordinates + the
    model's `reasoning`) + the accepted cell keys. Lineage-ADJACENT: never read by the ledger, the
    scrubbed corpus, or the prompt, so a model change re-records HERE but does not re-key or re-spend the
    model-blind comparison ledger. The `reasoning` rides this audit row ONLY (excluded from the ledger /
    corpus / oracle reply), so it is auditable — INSIGHT, never EVIDENCE — but never re-enters the loop."""
    row = {
        'round_id': round_id,
        'model_requested': batch.model_requested,
        'model_served': batch.model_served,
        'temperature': batch.temperature,
        'transport': batch.transport,
        'prompt_sha': batch.prompt_sha,
        'n_proposed': len(batch.proposals),
        'proposals': [{k: p.get(k) for k in (*FACTOR_PROPOSAL_FIELDS, 'reasoning')}
                      for p in batch.proposals if isinstance(p, dict)],
        'accepted': [list(k) for k in accepted],
    }
    with open(path, 'a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')


# --- the loop the LLM plugs into ----------------------------------------------------------------------
def _load_factor_ledger(path: str) -> list[dict[str, Any]]:
    """The raw committed factor-ledger rows (the lifetime stream to judge against). Carries the answer
    key — used by the oracle/loop, NEVER by the proposer (which reads only the scrubbed corpus)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _judge_factor_lifetime(new_rows: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    """Judge `new_rows` over the LIFETIME e-LOND stream: place the committed prior ledger AHEAD of the new
    batch and run one `online_fdr_survivors` pass, returning the fresh tail. The factor analog of
    `judge_against_lifetime_stream` — so an appended batch never restarts the discount sequence at t=1.

    DEDUP IS LOAD-BEARING, not cosmetic (it mirrors the option judge): a row already in the prior ledger
    or repeated within the batch (same `(key, universe)` — exactly what `_record_factor_cells` dedups on)
    is NOT a fresh look and must not be re-appended. The menu-walker RE-PROPOSES every round's survivors
    (the corpus excludes them), so without this dedup those rows would re-enter the stream and double-spend
    the e-LOND budget within the round. Deduped, the fresh tail is judged on a duplicate-free stream and is
    what `_record_factor_cells` then appends; the pass order == the record order == the file order (no
    sort), so a future re-judge of the committed file reproduces the verdict. Keeps the judged schema
    (incl. `e_value`/`elond_level`) consistent with `run_factor_search`."""
    prior = _load_factor_ledger(path)
    seen = {(r['key'], r['ticker']) for r in prior}
    fresh: list[dict[str, Any]] = []
    for r in new_rows:
        k = (r['key'], r['ticker'])
        if k not in seen:
            seen.add(k)
            fresh.append(r)
    return online_fdr_survivors(prior + fresh)[len(prior):]


def run_factor_proposer_round(
    backend: Any,
    *,
    path: str = FACTOR_LEDGER_PATH,
    search: frozenset[str] | None = None,
    sealed: frozenset[str] = frozenset(),
    author: FactorProposer | None = None,
    run: bool = True,
    record: bool = False,
    max_batch: int = 16,
    round_id: str | None = None,
    provenance_path: str | None = None,
) -> dict[str, Any]:
    """One factor proposer round — the loop the LLM plugs into. `author` selects the proposer:

      * `author is None` -> the deterministic MENU-WALKER (walk the grammar slice on this universe, minus
        tried).
      * a `FactorProposer` -> the LLM-author front-end (`llm_propose_factor_candidates`), capped at
        `max_batch`, with the model identity written to the provenance audit log on `record=True`.

    Either way the path downstream is identical: READ scrubbed corpus -> PROPOSE -> GRAMMAR-GATE
    (`ExprFactor`) -> SCORE (the backend, each row enriched with its readable `expr` coordinate) -> JUDGE
    over the lifetime e-LOND stream -> RECORD -> next round re-reads the corpus and skips them. `run=False`
    is a cheap preview (propose only); `run=True, record=False` scores + judges without writing; `record`
    appends to the lifetime ledger. The proposer reads ONLY the scrubbed corpus, never the answer-key
    ledger. Promotion stays CLOSED — a survivor escalates to manual pre-reg + the Phase-C holdout."""
    search = search if search is not None else frozenset({backend.universe})
    corpus = build_factor_proposer_corpus(_load_factor_ledger(path))
    tried = {_factor_proposer_key(r['key'], r['ticker']) for r in corpus}
    if author is None:
        cands, needs_onboard = propose_factor_candidates(backend, tried)
        batch, rejected = None, []
    else:
        cands, needs_onboard, rejected, batch = llm_propose_factor_candidates(
            author, backend, search=search, sealed=sealed, corpus=corpus,
            tried_keys=tried, max_batch=max_batch)
    # No extra cap here: the LLM gate already caps its batch at `max_batch` internally (so an LLM can't
    # burn the e-LOND budget), and the menu-walker is bounded by the finite grammar slice — capping both
    # at max_batch would wrongly truncate the deterministic menu-walker's full untried sweep.
    result: dict[str, Any] = {'proposed': len(cands), 'recorded': 0, 'needs_onboard': needs_onboard,
                              'rejected': rejected, 'candidates': cands, 'rows': [], 'ledger_rows': [],
                              'batch': batch}
    if record and run and batch is not None:
        prov = provenance_path or os.path.join(os.path.dirname(path) or '.', FACTOR_PROVENANCE_PATH)
        from datetime import datetime
        record_factor_provenance(batch, [_candidate_key(c, backend.universe) for c in cands],
                                 round_id=round_id or datetime.now().isoformat(), path=prov)
    if not cands or not run:
        return result
    rows = [{**backend.score(c), 'expr': expr_to_dict(c.expr)} for c in cands]
    ledger_rows = _judge_factor_lifetime(rows, path)
    result['rows'] = rows
    result['ledger_rows'] = ledger_rows
    if record:
        result['recorded'] = _record_factor_cells(ledger_rows, path)
    return result


# --- the read-gate oracle seam (the one-bit entry point across the boundary) --------------------------
def score_and_record_factor(
    proposals: list[dict[str, Any]],
    *,
    round_id: str,
    model: dict[str, Any],
    backend: Any,
    search: frozenset[str] | None = None,
    sealed: frozenset[str] = frozenset(),
    path: str = FACTOR_LEDGER_PATH,
    provenance_path: str | None = None,
) -> dict[str, Any]:
    """The factor ORACLE's single entry point — the ONLY way across the read-gate to reach the factor
    engine. Takes COORDINATE-ONLY proposals (`{expr, universe, predicted_sign}`) + the proposer's
    self-reported `model` identity (REQUIRED_MODEL_FIELDS), runs the full gate -> score -> lifetime-judge
    -> record chain, and returns ONLY the scrubbed one-bit scoreboard — never the t-stat-bearing rows.

    "Every look is a recorded look" holds BY CONSTRUCTION: the engine is reachable only through here, and
    here always runs with `record=True`. Two guards keep the reply clean: `rejected[].proposal` is
    re-scrubbed to FACTOR_PROPOSAL_FIELDS (a proposer cannot smuggle an extra key back through its echoed
    input), and the whole reply is `assert_numberless`-checked. A survivor is excluded from the reply's
    corpus and escalates to manual pre-registration — so `recorded > 0` with an empty `corpus` means a
    cell survived."""
    missing = [f for f in REQUIRED_MODEL_FIELDS if f not in model]
    if missing:
        raise ValueError(f'score_and_record_factor: model identity missing {missing} '
                         f'(required: {list(REQUIRED_MODEL_FIELDS)})')

    def _echo(_menu: Any, _corpus: Any, _onboarded: Any) -> ProposalBatch:
        return ProposalBatch(tuple(proposals), model_requested=model['model_requested'],
                             model_served=model['model_served'], temperature=model['temperature'],
                             prompt_sha=model['prompt_sha'])
    result = run_factor_proposer_round(
        backend, path=path, search=search, sealed=sealed, author=_echo,
        run=True, record=True, round_id=round_id, provenance_path=provenance_path)
    rejected = [{'proposal': ({k: r['proposal'].get(k) for k in FACTOR_PROPOSAL_FIELDS}
                              if isinstance(r['proposal'], dict) else {}),
                 'reason': r['reason']} for r in result['rejected']]
    reply = {
        'wire_version': WIRE_VERSION,
        'recorded': result['recorded'],
        'needs_onboard': result['needs_onboard'],
        'rejected': rejected,
        'corpus': build_factor_proposer_corpus(result['ledger_rows']),
    }
    assert_numberless(reply)
    return reply


# --- env-gating: OFF by default (the real factor client is H2b) ---------------------------------------
class FactorClaudeProposer(_ClaudeApiBase):
    """The FACTOR proposer's API client (H2b) — the shared `proposer_clients.ClaudeProposer` (metered
    Anthropic API, seal gold-standard) bound to this module's `build_factor_proposer_prompt`. The transport
    + the numberless-seal contract are the base class's; only the prompt (factor coordinates from the Expr
    grammar) differs from the option author."""

    def __init__(self, model: str = 'claude-opus-4-8', **kw: Any) -> None:
        super().__init__(model, prompt_builder=build_factor_proposer_prompt, **kw)


class FactorClaudeCodeProposer(_ClaudeCodeBase):
    """The FACTOR proposer's subscription transport (H2b) — the shared `proposer_clients.ClaudeCodeProposer`
    (hardened `claude -p`) bound to `build_factor_proposer_prompt`. The seal-critical invocation hardening
    (all tools denied, neutral cwd, API-key scrub) is the base class's, inherited unchanged."""

    def __init__(self, model: str = 'claude-opus-4-8', **kw: Any) -> None:
        super().__init__(model, prompt_builder=build_factor_proposer_prompt, **kw)


def _resolve_factor_llm_author() -> FactorProposer | None:
    """The activated factor LLM author, or None (the default). Gated on EDGE_SEARCH_LLM_MODEL (the repo's
    single LLM-author opt-in, shared with the option author) and EDGE_SEARCH_LLM_TRANSPORT (default
    `claude_code`, the Claude.ai subscription; `api` selects the metered client). With EDGE_SEARCH_LLM_MODEL
    UNSET this returns None, so the default — the deterministic menu-walker, no model — is the proposer.
    Activation never relaxes the standing limits: promotion stays CLOSED and survivors stay EXPLORATORY
    until the Phase-C time-axis holdout exists."""
    import os
    model = os.environ.get('EDGE_SEARCH_LLM_MODEL')
    if not model:
        return None
    transport = os.environ.get('EDGE_SEARCH_LLM_TRANSPORT', 'claude_code')
    if transport == 'api':
        return FactorClaudeProposer(model=model)
    if transport == 'claude_code':
        return FactorClaudeCodeProposer(model=model)
    raise ValueError(
        f"unknown EDGE_SEARCH_LLM_TRANSPORT={transport!r} (expected 'api' or 'claude_code')")


def _assert_factor_llm_boundary(author: FactorProposer | None = None) -> FactorProposer:
    """FAIL CLOSED unless a factor model author is activated, returning the author when one is. The
    backstop that keeps NO LLM running by default — with no real client wired/activated,
    `_resolve_factor_llm_author()` returns None and this raises, leaving the menu-walker as the proposer.
    For a wired-and-activated author the load-bearing seal is the oracle-side correctness argument (a
    numberless prompt, coordinate-only output, every score recorded), NOT engine-absence."""
    import sys
    resolved = author if author is not None else _resolve_factor_llm_author()
    if resolved is None:
        print('REFUSED: no factor LLM author is configured/wired (H2b wires the env-gated client; the '
              'menu-walker is the default). See docs/integration_plan.md.', file=sys.stderr)
        raise SystemExit(2)
    return resolved
