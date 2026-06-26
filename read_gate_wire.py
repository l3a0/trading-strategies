"""read_gate_wire.py — the frozen seal contract for the read-gate (docs/read_gate.md).

Dependency-free (stdlib only) ON PURPOSE. This is the read-gate's one source of truth for
the coordinate/model field allow-lists, the banned result set, and the numberless guard —
kept free of the engine (edge_search / vol_premium / numpy / the chains) so the contract is
shareable and testable in isolation, and so a future in-process LLM author can carry it
without importing the engine. (A separate sandboxed-proposer transport once shared this
contract across a process boundary over NDJSON; that container/transport was removed — the
decided LLM author is oracle-side and IN-PROCESS, sealed by INFORMATION not isolation,
docs/llm_proposer_plan.md.)

THE CONTRACT (frozen at WIRE_VERSION):

  proposals (coordinate-only): {"overlay","ticker","params","predicted_sign"}  (PROPOSAL_FIELDS)
  generative proposals (coordinate-only): {"legs":[...],"ticker","predicted_sign"}  (GEN_PROPOSAL_FIELDS)
  model identity (audit provenance):
    {"model_requested","model_served","temperature","prompt_sha"}  (REQUIRED_MODEL_FIELDS)
  the oracle seam's reply (score_and_record): one-bit verdicts only, NEVER a result statistic
    {"wire_version": 1, "recorded": <int>, "needs_onboard": [<str>],
     "rejected": [{"proposal": {<coords>}, "reason": <str>}],
     "corpus": [<scrubbed row: template/ticker/params/predicted_sign/verdict>]}

The reply's `corpus` is THIS round's scrubbed verdicts (deltas); the *lifetime* scrubbed
corpus is the seeded proposer-facing projection, not carried by the reply (docs/read_gate.md).
`assert_numberless` (below) is the load-bearing guard on any oracle-authored content.
"""
from __future__ import annotations

from typing import Any

WIRE_VERSION = 1

# The required keys of the proposer's self-reported model identity (audit provenance).
REQUIRED_MODEL_FIELDS: tuple[str, ...] = (
    'model_requested', 'model_served', 'temperature', 'prompt_sha')

# A proposal is COORDINATES ONLY — these keys and nothing else cross into the oracle, and
# the oracle echoes only these back in `rejected[].proposal` (so a proposer can't smuggle
# extra keys through the reply, nor crash the numberless check with a banned-named key).
PROPOSAL_FIELDS: tuple[str, ...] = ('overlay', 'ticker', 'params', 'predicted_sign')

# The GENERATIVE author's proposal (Phase 4, generative_proposer.py) is also COORDINATES ONLY, but
# drawn from the production grammar (generative_grammar) rather than the closed (overlay, params)
# lattice: a composition is {legs: [{side, right, delta|strike, dte}, ...], ticker, predicted_sign}.
# Same role as PROPOSAL_FIELDS — the oracle echoes only these back in `rejected[].proposal`. The nested
# leg dicts carry ONLY grammar coordinates (no result-bearing key), so `assert_numberless` stays the
# belt behind this allow-list, exactly as for the closed grammar's `params` dict.
GEN_PROPOSAL_FIELDS: tuple[str, ...] = ('legs', 'ticker', 'predicted_sign')

# Result statistics that must NEVER cross the read-gate to the proposer. The scrub
# (build_proposer_corpus' SAFE_FIELDS allow-list) is the primary control — it drops
# everything not allow-listed. `assert_numberless` is the belt to that suspenders: a
# KEY-NAME guard over oracle-authored content, so a future code path that routed a
# kill-gate / ledger row into a reply (or — under the oracle-side prompt builder of
# docs/llm_proposer_plan.md, where there is NO kernel backstop — into the model's prompt)
# fails loudly instead of leaking.
#
# COMPLETENESS IS LOAD-BEARING. This set must name EVERY result-bearing key any oracle row
# can carry — a forgotten key is a silent leak, because the guard only fires on names it
# knows. The union below is the result keys produced by structure_kill_gate (both the
# trade and no-trade branches: t_stat_newey_west / nw_lag / sharpe / ann_excess_return_pct
# / sign_ok / p_value / n_days / no_trades / measurement_invalid), run_structure_campaign
# (the scale-invalid branch's scale_ratio + the online_fdr_survivors additions e_value /
# elond_level / elond_survivor + the Benjamini-Yekutieli additions fdr_q / by_survivor /
# clean_survivor), and structure_ledger_rows (statistic / statistic_kind / data_lineage_hash).
# The owner-facing search_saturation readout (edge_search.py) is ALSO a producer: it is
# display-only and reaches no proposer input today (the SAFE_FIELDS scrub strips it on any
# mis-route), but its result keys (best_p / ceiling_t / required_t / required_p / e_required /
# survivors / past_bar) are banned anyway so the belt — assert_numberless — catches the whole
# dict if a future change ever routes it across the gate. Pure hypothesis-coordinate keys
# (phase / template / overlay / ticker / params / predicted_sign / end) and the public ledger
# size `n` are NOT banned — they are the scrubbed corpus's legitimate content. When a new
# result field is added to any of those producers, add it here in the same change; the
# always-run test_read_gate_wire.py::TestBannedSetCompleteness pins this set against the live
# engine's result keys, and TestSearchSaturation pins the saturation readout's keys.
BANNED_RESULT_FIELDS: frozenset[str] = frozenset({
    't_stat_newey_west', 'nw_lag', 'p_value', 'e_value', 'elond_level', 'statistic',
    'statistic_kind', 'sign_ok', 'scale_ratio', 'sharpe', 'ann_excess_return_pct',
    'n_days', 'no_trades', 'data_lineage_hash', 'measurement_invalid',
    'elond_survivor', 'by_survivor', 'clean_survivor', 'fdr_q',
    # search_saturation (owner-facing readout) — display-only, banned for belt-completeness
    'best_p', 'ceiling_t', 'required_t', 'required_p', 'e_required', 'survivors', 'past_bar',
})


def assert_numberless(obj: Any, _path: str = 'reply') -> None:
    """Walk `obj` and raise if any dict ANYWHERE inside it carries a BANNED_RESULT_FIELDS
    KEY — a result statistic that must never cross the read-gate to the proposer (the wire
    reply today; the model's assembled prompt under the oracle-side builder of
    docs/llm_proposer_plan.md, where this is the SOLE seal with no kernel backstop).

    This is a key-NAME guard, not a number detector: a statistic smuggled as a string VALUE
    under a safe key (e.g. `{'note': 't_stat=2.1'}`) is the allow-list scrub's job, not this.
    The guard is defense-in-depth BEHIND that scrub — a future code path that routes a raw
    kill-gate / ledger row (rather than the scrubbed projection) into oracle-authored content
    fails loudly here instead of leaking a number.

    Recursion is EXHAUSTIVE over the JSON container shapes: it descends into every dict value
    AND every element of lists/tuples, at arbitrary depth and in any combination — a dict
    nested in a list, a list nested in a dict, dicts inside tuples inside lists, deeply nested.
    A banned key is caught regardless of its value's type (a banned key with a scalar, dict, or
    list value all raise). Scalars and strings are inspected only as dict KEYS, never as values,
    so a banned NAME appearing as a plain string value does not (and should not) trip the guard.

    LEAF-TYPE GUARD: every leaf must be a JSON primitive, checked by EXACT type (str / int / float
    / bool / None) — not `isinstance`. A subclass of a primitive (notably `numpy.float64`, which
    subclasses `float` and so would slip an `isinstance` check) is REJECTED, so a statistic smuggled
    as a numpy scalar under a safe key cannot ride through. A non-primitive leaf — a dataclass or
    custom object whose ATTRIBUTES (not dict keys) could carry a banned name, or a set/frozenset —
    is likewise rejected. KEY GUARD: dict keys must be `str` (JSON object keys are strings); a
    non-str key (e.g. `bytes`) would slip the `BANNED_RESULT_FIELDS` set-intersection, so a banned
    name could otherwise hide on one. Together these keep the guard self-sufficient as the SOLE
    seal, not leaning on a downstream json.dumps failure. (A namedtuple is a tuple, descended above
    as values; its field names don't survive JSON, so it is not a banned-NAME vector.)"""
    if isinstance(obj, dict):
        bad = BANNED_RESULT_FIELDS & obj.keys()
        if bad:
            raise ValueError(f'read-gate numberless violation at {_path}: {sorted(bad)}')
        for k, v in obj.items():
            if not isinstance(k, str):
                # JSON object keys are strings; a non-str key (e.g. bytes) would slip the
                # BANNED_RESULT_FIELDS set-intersection above, so a banned name could ride it.
                raise ValueError(
                    f'read-gate numberless violation at {_path}: non-string key of type '
                    f'{type(k).__name__} (the wire is JSON; a banned name could hide on a non-str key)')
            assert_numberless(v, f'{_path}.{k}')
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_numberless(v, f'{_path}[{i}]')
    elif obj is not None and type(obj) not in (str, int, float, bool):
        # Leaf-type guard — the SOLE seal must be self-sufficient, not lean on a downstream
        # json.dumps failure (docs/llm_proposer_plan.md: oracle-side builder, no kernel backstop).
        # EXACT type, not isinstance: a numeric/str SUBCLASS (numpy.float64 subclasses float, so
        # isinstance would pass it) is rejected uniformly — matching the JSON-only rationale and the
        # way numpy.int64 (not an int subclass) is already rejected. A non-primitive leaf — a
        # dataclass / custom object whose ATTRIBUTES could carry a banned name, or a set/frozenset —
        # is rejected here too. (A namedtuple is a tuple, descended above; its fields don't survive JSON.)
        raise ValueError(
            f'read-gate numberless violation at {_path}: non-primitive leaf of type '
            f'{type(obj).__name__} (a banned field could hide in its attributes; the wire is JSON)')
