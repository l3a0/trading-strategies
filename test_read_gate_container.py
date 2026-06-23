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

`TestContainerRoundTrip` adds the live COMPOSITION: `oracle_server.launch_in_container` spawns a
real `proposer_client` INSIDE the sealed image (under `CONTAINER_SEAL_FLAGS`, a read-only seed
mount, no host env), round-trips one request, and the oracle scores+records it — proving the
proposer reaches the engine ONLY through the recording oracle while running engine-free in a
kernel-sealed box. Plus two hardening pins: the seed mount is READ-ONLY (a write fails) and the
docker socket is absent (no container escape).

The image is built once per module (a session-expensive `docker build`) and removed in teardown.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess

import pytest

from edge_search import Campaign, load_idea_ledger
from oracle_server import CONTAINER_SEAL_FLAGS, launch_in_container

pytestmark = pytest.mark.skipif(
    shutil.which('docker') is None,
    reason='docker not available (the seal test runs in CI, where the ubuntu runner has docker)')

_REPO = os.path.dirname(os.path.abspath(__file__))
_TAG = 'read-gate-proposer-pytest:latest'

# The HARDENED run config for an untrusted workload — the seal is more than "engine absent". This
# PR PROMOTED the flag set to `oracle_server.CONTAINER_SEAL_FLAGS`, the SINGLE definition the launch
# path (`launch_in_container`) AND this seal test now share — so the seal test certifies EXACTLY the
# flags `launch_in_container` spawns the proposer under, not a softer set a real untrusted-LLM
# deployment would never run. (The launch-uses-the-container composition is pinned by
# `TestContainerRoundTrip` below.)
_SEAL_FLAGS = CONTAINER_SEAL_FLAGS


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

    def test_seed_mount_is_read_only(self, proposer_image, tmp_path) -> None:
        # `launch_in_container` mounts the seed dir READ-ONLY (type=bind,...,readonly), so an
        # untrusted proposer cannot write into the dir the host seeds. Run the image with the same
        # mount shape and assert a write to /sandbox fails (OSError); /tmp is the only writable
        # surface (the --tmpfs).
        seed = tmp_path / 'seed'
        seed.mkdir()
        proc = subprocess.run(
            ['docker', 'run', '--rm', *_SEAL_FLAGS,
             '--mount', f'type=bind,src={seed},dst=/sandbox,readonly', '-w', '/sandbox',
             proposer_image, 'python', '-c', "open('/sandbox/x', 'w').write('nope')"],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode != 0, 'a write to the READ-ONLY seed mount SUCCEEDED'
        assert 'OSError' in proc.stderr or 'Read-only' in proc.stderr, proc.stderr

    def test_docker_socket_absent(self, proposer_image) -> None:
        # `launch_in_container` mounts ONLY the read-only seed dir — never /var/run/docker.sock.
        # Without the socket the container cannot spawn sibling containers / escape; assert the
        # socket path does not exist inside the container.
        proc = _run(proposer_image,
                    "import os, sys; sys.exit(0 if not os.path.exists('/var/run/docker.sock') else 1)")
        assert proc.returncode == 0, 'the docker socket /var/run/docker.sock WAS present in the container'


# A synthetic per-candidate scorer mirroring test_oracle_server._scorer: a KILLED verdict (t=0.5
# is nowhere near the e-LOND bar) carrying the banned result keys, so the oracle's scrub/numberless
# guards are genuinely exercised while the engine is NOT run inside the container.
def _scorer(cand):
    return {'phase': 'structure', 'template': cand.template, 'ticker': cand.ticker,
            'params': cand.params_dict(), 'predicted_sign': cand.predicted_sign,
            't_stat_newey_west': 0.5, 'sign_ok': True, 'p_value': 0.3}


class TestContainerRoundTrip:
    """The live COMPOSITION the soft `launch` can't reach: `launch_in_container` spawns a real
    `proposer_client` INSIDE the sealed image (under `CONTAINER_SEAL_FLAGS`, a read-only seed mount,
    NO host env), round-trips one request over the container's stdio, and the oracle scores+records
    the comparison. This proves the proposer reaches the engine ONLY through the recording oracle
    while running ENGINE-FREE in a kernel-sealed box — the whole point of the integration.

    A SIGALRM(30s) guard fails fast on a transport deadlock (a missing-newline hang), mirroring
    test_oracle_server.TestLaunchEndToEnd. The engine is NOT run inside the container: the synthetic
    `_scorer` stands in for the overlay, so this pins the COMPOSITION (spawn -> wire -> record), not
    a real backtest."""

    def test_real_proposer_client_round_trips_in_container(self, proposer_image, tmp_path) -> None:
        led = str(tmp_path / 'idea_ledger.jsonl')
        sandbox = str(tmp_path / 'sandbox')
        # A real proposer running PURELY from the read-only seed mount (cwd=/sandbox). It proposes
        # one COMMITTED grammar cell (short_call_25 on AAA) via a stub author and drives one round.
        # `import proposer_client` resolves to the COPY launch_in_container seeded into the mount
        # (sys.path[0] = cwd = /sandbox); `import edge_search` would fail (no engine in the box).
        stub = (
            "import sys\n"
            "import proposer_client as pc\n"
            "author = lambda menu, corpus, onboarded: (\n"
            "    [{'overlay': 'short_vol', 'ticker': 'AAA',\n"
            "      'params': {'target_delta': 0.25, 'dte': 30}, 'predicted_sign': 1}],\n"
            "    {'model_requested': 'stub', 'model_served': 'stub',\n"
            "     'temperature': 0.0, 'prompt_sha': 'x'})\n"
            "def w(s):\n"
            "    sys.stdout.write(s); sys.stdout.flush()\n"
            "pc.run_proposer_loop(sys.stdin.readline, w, author, rounds=1)\n"
        )

        def _bark(signum, frame):
            raise TimeoutError('read-gate container e2e exceeded 30s — likely a transport deadlock')
        old = signal.signal(signal.SIGALRM, _bark)
        signal.alarm(30)
        try:
            code = launch_in_container(
                proposer_image, ['python', '-c', stub], sandbox_dir=sandbox, path=led,
                campaign=Campaign(search=('AAA',)), scorer=_scorer)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        assert code == 0, f'the container proposer exited {code}'
        assert len(load_idea_ledger(led)) == 1, (
            'the oracle did not record exactly one comparison from the container round-trip')
