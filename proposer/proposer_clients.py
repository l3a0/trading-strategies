"""proposer_clients.py — the domain-agnostic Claude LLM-author transports (shared by every proposer).

The option proposer (edge_search.py) and the factor proposer (factor_proposer.py) emit DIFFERENT
coordinate schemas from DIFFERENT grammar menus, but the transport that turns a numberless prompt into a
`ProposalBatch` is identical — a stateless Anthropic API call, or a hardened `claude -p` subscription run.
This module holds that transport ONCE, parameterized by a `prompt_builder`, so the security-critical
mechanics (especially the Claude-Code hardening below) live in a single place. Each domain subclasses these
and binds its own prompt builder; the menu is opaque here (passed straight to the builder).

Dependency-light: imports only the dependency-free wire (`read_gate_wire`) + stdlib; `anthropic` is an
OPTIONAL dependency imported lazily inside the API client's `__call__` (never pulled by the engine suite).
Both transports preserve the seal proper — the numberless prompt (the builder runs `assert_numberless` on
its corpus input), coordinate-only output, every-look-recorded downstream — regardless of domain.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

from proposer.read_gate_wire import ProposalBatch, _parse_proposal_array

# (menu, scrubbed_corpus, onboarded, *, max_proposals) -> a numberless prompt string. The menu type is
# DOMAIN-specific (option StructureTemplates / a factor grammar-menu dict) and opaque to the transport.
PromptBuilder = Callable[..., str]


class ClaudeProposer:
    """The oracle-side, in-process LLM author over the metered Anthropic API — the SEAL GOLD-STANDARD: one
    stateless, numberless completion with zero context and zero tools. Callable with
    `(menu, corpus, onboarded) -> ProposalBatch`, the same slot the deterministic menu-walker fills, so the
    gate -> score -> lifetime-judge -> record path downstream is byte-identical.

    THE SEAL is the injected `prompt_builder`'s `assert_numberless` on the corpus input, which runs inside
    the prompt build below — this object adds NO result-bearing context of its own. The model sees only the
    grammar menu + the scrubbed corpus + the onboarded universe, and emits only coordinates; the domain's
    gate validates every proposal afterward.

    NOT ACTIVATED by merely existing: a domain's resolver constructs one only when its model env var is set,
    and the API call needs ANTHROPIC_API_KEY in the environment + `anthropic` installed (an OPTIONAL
    dependency, imported lazily in `__call__` — not in requirements.txt, so the engine suite never pulls it
    in). Construction is cheap and import-free so the contract is testable with a stub `client=` and no
    network. Promotion stays CLOSED and survivors stay EXPLORATORY until the Phase-C holdout — activation
    changes neither.

    Model defaults to Claude Opus 4.8. The 4.8 family takes NO `temperature` parameter (sending one is a
    400) — depth is governed by adaptive thinking + `effort` (default `max`), so none is sent. The frozen
    wire still carries a `temperature` field, so the recorded value is a documented sentinel (`0.0`); the
    reconstructable identity is `model_served` + `prompt_sha`."""

    def __init__(self, model: str = 'claude-opus-4-8', *, prompt_builder: PromptBuilder,
                 client: Any | None = None, max_proposals: int = 16, effort: str = 'max',
                 max_tokens: int = 16000) -> None:
        self.model = model
        self.prompt_builder = prompt_builder
        self._client = client          # injectable for tests; None -> lazily build anthropic.Anthropic()
        self.max_proposals = max_proposals
        self.effort = effort
        self.max_tokens = max_tokens

    def _make_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic   # optional dependency — only needed when an LLM round actually runs
        return anthropic.Anthropic()   # resolves ANTHROPIC_API_KEY / profile from the environment

    def __call__(self, menu: Any, corpus: list[dict[str, Any]],
                 onboarded: tuple[str, ...]) -> ProposalBatch:
        # the prompt_builder runs the numberless SEAL on `corpus` (raises before any API call if a raw
        # answer-key row slipped in). No result statistic is in scope here.
        prompt = self.prompt_builder(menu, corpus, onboarded, max_proposals=self.max_proposals)
        prompt_sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        resp = self._make_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={'type': 'adaptive'},          # 4.8: adaptive only; NO temperature/sampling params
            output_config={'effort': self.effort},  # low|medium|high|xhigh|max
            messages=[{'role': 'user', 'content': prompt}],
        )
        if getattr(resp, 'stop_reason', None) == 'refusal':
            raise RuntimeError(
                f'LLM proposer refused (stop_details={getattr(resp, "stop_details", None)}); '
                'no proposals produced')
        text = ''.join(b.text for b in resp.content if getattr(b, 'type', None) == 'text')
        return ProposalBatch(
            tuple(_parse_proposal_array(text)),
            model_requested=self.model,
            model_served=resp.model,      # the EXACT served snapshot the API reports — the audit id
            temperature=0.0,              # sentinel: the 4.8 family sends no temperature (see class docstring)
            prompt_sha=prompt_sha,
        )


class ClaudeCodeProposer:
    """ALT-TRANSPORT: an LLM author that drives Claude Code (`claude -p`) under a Claude.ai SUBSCRIPTION
    (Max/Pro) instead of the metered API. Same numberless prompt, same coordinate-only output, same
    downstream gate -> score -> judge -> record — only the transport differs. Records
    `transport='claude_code'` to the provenance log.

    THE SEAL IS A LARGER SURFACE HERE — read this. Claude Code is an AGENT, not a stateless API call: by
    default it (a) offers tools (bash/read/edit/web) and (b) auto-loads working-dir + ~/.claude context
    (CLAUDE.md, settings, MCP). BOTH are seal-hostile — a tool-enabled run could `cat` the answer-key
    ledger, and THIS repo's CLAUDE.md carries pinned result numbers, so loading it would feed the proposer
    the very statistics the numberless prompt withholds. So the invocation is HARDENED into a pure
    prompt->text completion equivalent to the API call (`_build_invocation`, unit-tested):

      * `--disallowedTools "*"` removes EVERY tool from the model's context (deny-first precedence — the
        model never even sees a tool, so there is no Read/Bash/`cat`) plus `--strict-mcp-config` (no
        `--mcp-config` => zero MCP servers load) and `--max-turns 1`;
      * the subprocess runs from a NEUTRAL temp cwd, so the repo's CLAUDE.md (and its pinned numbers) is
        never in scope — and no global `~/.claude/CLAUDE.md` exists;
      * `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` are SCRUBBED from the child env, both so the
        subscription OAuth is used (a set key would override it) and so no metered call is made. NOT
        `--bare`: bare skips OAuth/keychain and would force an API key, defeating the subscription.

    RESIDUAL SURFACE (named, not hidden): because subscription auth needs non-`--bare`, Claude Code still
    reads `~/.claude` global settings — a global hook could run. That is user-controlled config, not an
    attacker vector, but it makes this a LARGER trusted surface than the API path. The API `ClaudeProposer`
    remains the seal gold-standard (one stateless numberless call, zero context, zero tools). The seal
    proper — numberless prompt, coordinate-only output, every-look-recorded — is preserved on either
    transport. Effort/thinking knobs are NOT exposed the way the raw API is (Claude Code applies its own
    defaults), so this transport does not honor `effort='max'`; record which transport ran so the two are
    not conflated. `anthropic` is NOT needed here (no API client) — only the `claude` CLI on PATH and a
    subscription login. Promotion stays CLOSED, survivors EXPLORATORY pending Phase C."""

    _DENY_ALL_TOOLS = '*'   # --disallowedTools "*" removes every tool from the model's context

    def __init__(self, model: str = 'claude-opus-4-8', *, prompt_builder: PromptBuilder,
                 runner: Any | None = None, max_proposals: int = 16, timeout: int = 600) -> None:
        self.model = model
        self.prompt_builder = prompt_builder
        self._runner = runner       # injectable (prompt -> json dict) for tests; None -> real subprocess
        self.max_proposals = max_proposals
        self.timeout = timeout

    def _build_invocation(self, prompt: str) -> tuple[list[str], dict[str, str]]:
        """The `claude -p` argv + the SCRUBBED child env. Factored out so the seal-critical construction
        (api key scrubbed, ALL tools denied, single turn, no `--bare`) is unit-testable without spawning a
        subprocess. The neutral cwd is applied separately in `_run`."""
        import os
        env = {k: v for k, v in os.environ.items()
               if k not in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')}   # force subscription OAuth
        cmd = ['claude', '-p', prompt,
               '--model', self.model,
               '--output-format', 'json',
               '--disallowedTools', self._DENY_ALL_TOOLS,   # remove EVERY tool from the model's context
               '--strict-mcp-config',                       # + no --mcp-config => zero MCP servers
               '--max-turns', '1']                          # single completion; errors if it tries to loop
        return cmd, env

    def _run(self, prompt: str) -> dict[str, Any]:
        if self._runner is not None:
            return self._runner(prompt)
        import json as _json
        import subprocess
        import tempfile
        cmd, env = self._build_invocation(prompt)
        with tempfile.TemporaryDirectory() as neutral_cwd:   # no repo CLAUDE.md in scope
            out = subprocess.run(cmd, env=env, cwd=neutral_cwd,
                                 capture_output=True, text=True, timeout=self.timeout)
        try:
            payload = _json.loads(out.stdout) if (out.stdout or '').strip() else {}
        except _json.JSONDecodeError:
            payload = {}
        # `claude -p --output-format json` reports API/auth errors in STDOUT json (`is_error` + `result` +
        # `api_error_status`), NOT stderr — so surface that, falling back to stderr/stdout. A 401 here means
        # the standalone CLI isn't authenticated: run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN
        # (this transport passes it through), or `claude login`.
        if out.returncode != 0 or payload.get('is_error'):
            detail = (payload.get('result') or (out.stderr or '').strip()
                      or (out.stdout or '').strip() or '(no output)')
            raise RuntimeError(f'claude -p failed (rc={out.returncode}): {detail[:500]}')
        return payload

    def __call__(self, menu: Any, corpus: list[dict[str, Any]],
                 onboarded: tuple[str, ...]) -> ProposalBatch:
        # the prompt_builder runs the numberless SEAL on `corpus` (raises before any subprocess if a raw
        # answer-key row slipped in). No result statistic is in scope here.
        prompt = self.prompt_builder(menu, corpus, onboarded, max_proposals=self.max_proposals)
        prompt_sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        payload = self._run(prompt)
        text = payload.get('result', '')
        served = payload.get('model') or self.model   # the served snapshot if reported, else the alias
        return ProposalBatch(
            tuple(_parse_proposal_array(text)),
            model_requested=self.model,
            model_served=served,
            temperature=0.0,              # sentinel: no temperature on this path either
            prompt_sha=prompt_sha,
            transport='claude_code',
        )
