"""read_gate_wire.py — the frozen wire contract for the read-gate (docs/read_gate.md).

Dependency-free (stdlib only) ON PURPOSE. The sandboxed PROPOSER process must not import
the engine (edge_search / vol_premium / numpy / the chains), so the contract it shares
with the trusted ORACLE lives here — importable by BOTH sides without pulling in the
engine. This is what lets the oracle server and the proposer client be built in parallel
against one source of truth instead of a vendored copy that drifts.

THE CONTRACT (frozen at WIRE_VERSION):

  request  (proposer -> oracle): coordinate-only proposals + the proposer's model identity
    {"wire_version": 1, "round_id": <str>,
     "model": {"model_requested","model_served","temperature","prompt_sha"},
     "proposals": [{"overlay","ticker","params","predicted_sign"}]}

  reply    (oracle -> proposer): one-bit verdicts only, NEVER a result statistic
    {"wire_version": 1, "recorded": <int>, "needs_onboard": [<str>],
     "rejected": [{"proposal": {<coords>}, "reason": <str>}],
     "corpus": [<scrubbed row: template/ticker/params/predicted_sign/verdict>]}

The reply's `corpus` is THIS round's scrubbed verdicts (deltas); the proposer holds the
*lifetime* scrubbed corpus as a seeded sandbox file, not via the reply (docs/read_gate.md).
`wire_version` is stamped on every message; version negotiation/validation is the
transport PR's job (advisory here).
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
# Pure hypothesis-coordinate keys (phase / template / overlay / ticker / params /
# predicted_sign / end) are NOT banned — they are the scrubbed corpus's legitimate content.
# When a new result field is added to any of those producers, add it here in the same change;
# the always-run test_read_gate_wire.py::TestBannedSetCompleteness pins this set against the
# live engine's result keys, so an unbanned result field fails CI.
BANNED_RESULT_FIELDS: frozenset[str] = frozenset({
    't_stat_newey_west', 'nw_lag', 'p_value', 'e_value', 'elond_level', 'statistic',
    'statistic_kind', 'sign_ok', 'scale_ratio', 'sharpe', 'ann_excess_return_pct',
    'n_days', 'no_trades', 'data_lineage_hash', 'measurement_invalid',
    'elond_survivor', 'by_survivor', 'clean_survivor', 'fdr_q',
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

    LEAF-TYPE GUARD: every leaf must be a JSON primitive (str / int / float / bool / None). A
    non-primitive leaf — a dataclass or custom object whose ATTRIBUTES (not dict keys) could carry
    a banned name, or a set/frozenset — is REJECTED, so the guard is self-sufficient as the sole
    seal and does not rely on a downstream json.dumps failure to catch a smuggled object. (A
    namedtuple is a tuple, descended above as values; its field names don't survive JSON, so it is
    not a banned-NAME vector.)"""
    if isinstance(obj, dict):
        bad = BANNED_RESULT_FIELDS & obj.keys()
        if bad:
            raise ValueError(f'read-gate numberless violation at {_path}: {sorted(bad)}')
        for k, v in obj.items():
            assert_numberless(v, f'{_path}.{k}')
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_numberless(v, f'{_path}[{i}]')
    elif obj is not None and not isinstance(obj, (str, int, float, bool)):
        # Leaf-type guard — the SOLE seal must be self-sufficient, not lean on a downstream
        # json.dumps failure (docs/llm_proposer_plan.md: oracle-side builder, no kernel backstop).
        # The numberless object crosses the wire as JSON, so every leaf must be a JSON primitive.
        # A non-primitive leaf — a dataclass / custom object whose ATTRIBUTES (not dict keys) could
        # carry a banned field name the key-guard never inspects — is rejected here. (A namedtuple
        # is a tuple, descended above as values; its field names don't survive JSON, so a banned
        # NAME cannot ride it.)
        raise ValueError(
            f'read-gate numberless violation at {_path}: non-primitive leaf of type '
            f'{type(obj).__name__} (a banned field could hide in its attributes; the wire is JSON)')
