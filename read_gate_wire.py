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
# kill-gate / ledger row into a reply fails loudly instead of leaking. Keep this in sync
# with the result keys structure_kill_gate / structure_ledger_rows produce.
BANNED_RESULT_FIELDS: frozenset[str] = frozenset({
    't_stat_newey_west', 'nw_lag', 'p_value', 'e_value', 'elond_level', 'statistic',
    'statistic_kind', 'sign_ok', 'scale_ratio', 'sharpe', 'ann_excess_return_pct',
    'n_days', 'no_trades', 'data_lineage_hash',
    'elond_survivor', 'by_survivor', 'clean_survivor', 'fdr_q',
})


def assert_numberless(obj: Any, _path: str = 'reply') -> None:
    """Walk `obj` and raise if any dict carries a BANNED_RESULT_FIELDS KEY — a result
    statistic that must never cross the read-gate. This is a key-NAME guard, not a
    number detector (a statistic smuggled as a string value under a safe key is the
    allow-list scrub's job, not this); it is defense-in-depth behind that scrub, so a
    future leak of a banned-named field fails loudly here. Recurses dicts, lists, and
    tuples (including dicts nested inside tuples inside lists)."""
    if isinstance(obj, dict):
        bad = BANNED_RESULT_FIELDS & obj.keys()
        if bad:
            raise ValueError(f'read-gate numberless violation at {_path}: {sorted(bad)}')
        for k, v in obj.items():
            assert_numberless(v, f'{_path}.{k}')
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_numberless(v, f'{_path}[{i}]')
