"""test_read_gate_container.py — the read-gate proposer CONTAINER seal (docs/read_gate.md).

DOCKER-GATED: every test skips when docker is unavailable (the dev machine has none), so this
file collects and "passes" (skips) locally and actually RUNS in CI, where the ubuntu runner
has docker. THE NEGATIVE TESTS ARE THE VERIFICATION — a misconfigured Dockerfile (an
accidentally-copied engine file, a missing `--network none`) is caught here, in CI, rather than
by trusting the image by inspection.

What is pinned — the kernel-enforced absence the supervised MVP could not provide:

  * the ENGINE is absent from the image (`import edge_search` / `vol_premium` / `cc_backtest`
    raise ModuleNotFoundError) — so no `import` recompute, and none is reachable by any path;
  * the engine SOURCE and the answer-key LEDGER are not reachable by their host absolute paths
    (no bind mount of the repo) -> `open()` raises FileNotFoundError. This is the abspath-read
    residual the soft sandbox left open, now closed by the mount namespace;
  * the NETWORK is unreachable (`--network none`) -> a recomputed/scraped number can't be
    exfiltrated;
  * the proposer's OWN engine-free code IS present and imports cleanly, dragging in no engine —
    so the sealed image is still a working proposer, not a brick.

The image is built once per module (a session-expensive `docker build`) and removed in teardown.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which('docker') is None,
    reason='docker not available (the seal test runs in CI, where the ubuntu runner has docker)')

_REPO = os.path.dirname(os.path.abspath(__file__))
_TAG = 'read-gate-proposer-pytest:latest'

# The HARDENED run config for an untrusted workload — the seal is more than "engine absent".
# The integration PR's `launch` MUST spawn the proposer with EXACTLY these flags (promote this
# constant to the launch path and add a test that launch uses it); the seal test runs the
# container under them, so it certifies the HARDENED config, not a softer one a real
# untrusted-LLM deployment would never use.
_SEAL_FLAGS = (
    '--network', 'none',                     # no egress: a recomputed/scraped number can't leave
    '--read-only',                           # immutable image filesystem
    '--tmpfs', '/tmp',                       # the only writable surface (per-container memory)
    '--cap-drop', 'ALL',                     # zero Linux capabilities (no NET_RAW / SETUID / MKNOD)
    '--security-opt', 'no-new-privileges',   # no setuid privilege escalation
    '--ipc', 'none',                         # no shared IPC namespace
    '--pids-limit', '128',                   # bound process creation (anti fork-bomb)
    '--memory', '512m',                      # bound memory (anti OOM-the-host)
    '--cpus', '1',                           # bound CPU
)


@pytest.fixture(scope='module')
def proposer_image():
    """Build the sealed proposer image once from the repo build context; remove it after.

    The build context is the repo root so the Dockerfile's `COPY proposer_client.py
    read_gate_wire.py` resolves — but the Dockerfile copies ONLY those two files, so nothing
    else from the context lands in the image (that is exactly what the seal tests assert)."""
    build = subprocess.run(
        ['docker', 'build', '-f', os.path.join(_REPO, 'Dockerfile.proposer'),
         '-t', _TAG, _REPO],
        capture_output=True, text=True, timeout=600)
    assert build.returncode == 0, f'docker build failed:\n{build.stderr}'
    yield _TAG
    subprocess.run(['docker', 'rmi', '-f', _TAG], capture_output=True)


def _run(image, code):
    """Run `python -c code` inside the image under the full hardened seal (`_SEAL_FLAGS`) — no
    egress, immutable FS, zero capabilities, no privilege escalation — and NO bind mount of the
    repo (so the host engine/ledger are absent). This is EXACTLY the config the integration PR
    must launch the proposer with. Returns the CompletedProcess."""
    return subprocess.run(
        ['docker', 'run', '--rm', *_SEAL_FLAGS, image, 'python', '-c', code],
        capture_output=True, text=True, timeout=60)


class TestProposerImageSeal:
    """The image is sealed: engine absent (by import AND by abspath), network dead, but the
    proposer's own code still runs."""

    def test_engine_not_importable(self, proposer_image) -> None:
        # the engine module is not in the image -> a bare import raises (and, since it is
        # genuinely ABSENT, the message names edge_search itself, not a failed dependency).
        proc = _run(proposer_image, 'import edge_search')
        assert proc.returncode != 0, f'edge_search WAS importable in the container:\n{proc.stdout}'
        assert "No module named 'edge_search'" in proc.stderr, proc.stderr

    def test_other_engine_modules_not_importable(self, proposer_image) -> None:
        for mod in ('vol_premium', 'cc_backtest', 'real_cc_backtest'):
            proc = _run(proposer_image, f'import {mod}')
            assert proc.returncode != 0 and f"No module named '{mod}'" in proc.stderr, (
                f'{mod} WAS importable in the container:\n{proc.stderr}')

    def test_engine_source_not_readable_by_host_abspath(self, proposer_image) -> None:
        # the host's real path to edge_search.py does not resolve inside the container (no repo
        # bind mount) -> open() raises FileNotFoundError. This is the abspath-read residual the
        # soft sandbox left open, closed by the mount namespace.
        host_engine = os.path.join(_REPO, 'edge_search.py')
        proc = _run(proposer_image, f'open({host_engine!r})')
        assert proc.returncode != 0, 'the host engine source WAS readable in the container'
        assert 'FileNotFoundError' in proc.stderr, proc.stderr

    def test_answer_key_ledger_not_readable_by_host_abspath(self, proposer_image) -> None:
        # the answer-key ledger is the thing the whole read-gate protects; its host path must
        # not resolve inside the container.
        host_ledger = os.path.join(_REPO, 'idea_ledger.jsonl')
        proc = _run(proposer_image, f'open({host_ledger!r})')
        assert proc.returncode != 0, 'the host answer-key ledger WAS readable in the container'
        assert 'FileNotFoundError' in proc.stderr, proc.stderr

    def test_network_is_unreachable(self, proposer_image) -> None:
        # --network none: a recomputed/scraped statistic can't be exfiltrated. Assert the failure
        # is a NETWORK-class error (urllib URLError — DNS/connect fails with no interface), not an
        # unrelated crash: dropping --network none would let urlopen REACH example.com (rc 0),
        # flipping the first assert; the URLError check stops a transient non-network failure from
        # passing this vacuously.
        proc = _run(
            proposer_image,
            'import urllib.request; urllib.request.urlopen("https://example.com", timeout=5)')
        assert proc.returncode != 0, 'the container reached the network despite --network none'
        assert 'URLError' in proc.stderr, proc.stderr

    def test_proposer_client_runs_and_is_engine_free(self, proposer_image) -> None:
        # the sealed image is still a WORKING proposer: its own code imports, and pulls in no
        # engine (proposer_client imports only read_gate_wire + the stdlib).
        code = (
            'import sys, proposer_client, read_gate_wire\n'
            "leaked = [m for m in ('edge_search', 'vol_premium', 'numpy', 'pandas') "
            'if m in sys.modules]\n'
            "print('LEAKED:' + ','.join(leaked))\n"
        )
        proc = _run(proposer_image, code)
        assert proc.returncode == 0, f'proposer_client did not import in the container:\n{proc.stderr}'
        assert proc.stdout.strip() == 'LEAKED:', f'an engine module leaked in: {proc.stdout!r}'
