"""E-value false-discovery control for the edge-search loop (interlock #3b).

Implements the procedure REGISTERED in docs/prereg_fdr_budget.md:

  1. Per-cell: calibrate the existing HAC-t p-value to an e-value via a Vovk-Wang
     (2021) calibrator, e = kappa * p^(kappa-1), kappa=0.5 -> e = 1/(2*sqrt p).
  2. The single FDR control: e-LOND (Xu & Ramdas 2024) over the lifetime STREAM of
     cell e-values — proven online FDR under ARBITRARY dependence, peek-whenever.
  3. e-BH (Wang & Ramdas 2022) is a within-campaign DIAGNOSTIC only, not the control.

PORTED, not depended (the repo is from-scratch minimal-dependency Python). The
recurrences are ported from the published papers; the always-run tests pin the
calibrator, the e-LOND recurrence, and e-BH against the `online-fdr` package
(its GitHub-main e-value module, which parity-tests itself against the
R/Bioconductor `onlineFDR`) — the oracle values are hardcoded in test_evalue_fdr.py
so nothing here depends on `online-fdr` at runtime. See test_evalue_fdr.py for the
optional live-parity check (skips unless `online-fdr` is installed, Python 3.10+).
This module is `evalue_fdr` (not `online_fdr`) so it does not shadow the `online-fdr`
package's import name when both are present.

The e-value route is LESS powerful than BY (calibration is lossy); the win is
arbitrary-dependence robustness + online/peek-whenever, not power — the trade the
prereg's S0 states. The guarantee is exact in the dependence structure but inherits
the per-cell HAC-t asymptotics; the deferred betting e-process is the finite-sample
endpoint. Promotion stays CLOSED: e-LOND SURFACES survivors, never crowns them.
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple, Sequence

# --- registered constants (prereg S3) ----------------------------------------
ONLINE_FDR_ALPHA = 0.10        # target FDR — owner risk-appetite choice
CALIBRATOR_KAPPA = 0.5         # Vovk-Wang exponent: e = kappa * p^(kappa-1)


# --- per-cell calibration (axis A) -------------------------------------------
def calibrate_p_to_e(p: float | None, kappa: float = CALIBRATOR_KAPPA) -> float:
    """Vovk-Wang (2021) admissible p-to-e calibrator f(p) = kappa * p^(kappa-1),
    kappa in (0,1) (decreasing in p, integral over [0,1] equals 1 — so it maps a
    valid p-value to a valid e-value). Default kappa=0.5 gives e = 1/(2*sqrt p).

    A measurement_invalid cell (p is None) calibrates to e = 0: it enters the stream
    but 0 can never clear any e-LOND / e-BH threshold, so it counts yet can never be
    rejected — the e-value analogue of the p=None N-shrink defense pinned in #46."""
    if p is None:
        return 0.0
    if not 0.0 < kappa < 1.0:
        raise ValueError(f'kappa must be in (0, 1), got {kappa}')
    p = min(max(float(p), 1e-300), 1.0)       # a valid p-value lies in (0, 1]
    return kappa * p ** (kappa - 1.0)


def _calibrator_integral(kappa: float, grid: int = 100_000) -> float:
    """Midpoint estimate of integral_0^1 kappa*p^(kappa-1) dp (analytically = 1 for
    kappa in (0,1)). The integrand has an integrable singularity at p=0, so the
    midpoint rule UNDER-estimates near 0 — used only for the admissibility direction
    (integral <= 1), per the prereg's 'implementation asserts integral <= 1'."""
    h = 1.0 / grid
    return h * sum(kappa * ((i + 0.5) * h) ** (kappa - 1.0) for i in range(grid))


def _assert_calibrator_admissible(kappa: float, tol: float = 1e-3) -> None:
    if not 0.0 < kappa < 1.0:
        raise ValueError(f'calibrator kappa must be in (0, 1), got {kappa}')
    total = _calibrator_integral(kappa)
    if total > 1.0 + tol:                      # admissibility: integral f <= 1
        raise ValueError(f'calibrator kappa={kappa} not admissible: integral≈{total:.4f} > 1')


_assert_calibrator_admissible(CALIBRATOR_KAPPA)


# --- the registered e-LOND discount sequence (prereg S3) ----------------------
def _registered_gamma_raw(t: int) -> float:
    """Unnormalized registered weight 1/(t * log^2(t+1)), t 1-based (prereg S3)."""
    return 1.0 / (t * math.log(t + 1) ** 2)


def _registered_gamma_norm(horizon: int = 100_000) -> float:
    """Normalizer c with Sum_{t>=1} c*raw(t) <= 1 over the INFINITE stream (the
    e-LOND requirement). Finite sum to H + a VALID conservative tail bound: for
    t>=2, raw(t) = 1/(t log^2(t+1)) <= 1/(t log^2 t), and integral_H^inf dx/(x log^2 x)
    = 1/log(H) exactly, so Sum_{t>H} raw(t) <= 1/log(H). (1/log(H+1) is the integral
    of the SHIFTED integrand 1/((x+1) log^2(x+1)) — strictly SMALLER, so NOT an upper
    bound; using it makes the infinite sum 1.000000008 > 1. Hence 1/log(H).)"""
    s = sum(_registered_gamma_raw(t) for t in range(1, horizon + 1))
    tail = 1.0 / math.log(horizon)
    return 1.0 / (s + tail)


ELOND_GAMMA_C = _registered_gamma_norm()       # pinned normalization constant


def registered_gamma(t: int) -> float:
    """The committed e-LOND discount sequence gamma_t (prereg S3), normalized so
    Sum_{t>=1} gamma_t <= 1. Non-increasing. `t` is 1-based."""
    return ELOND_GAMMA_C * _registered_gamma_raw(t)


# --- e-LOND: the single FDR control (axis B + C) ------------------------------
class ElondStep(NamedTuple):
    rejected: bool       # flagged for a human (a survivor); never auto-promoted
    level: float         # alpha_t = alpha * gamma_t * (R_{t-1} + 1)
    e_value: float


def elond(e_values: Sequence[float], alpha: float = ONLINE_FDR_ALPHA,
          gamma: Callable[[int], float] = registered_gamma) -> list[ElondStep]:
    """e-LOND (Xu & Ramdas 2024) — the single FDR control over the lifetime stream
    of cell e-values, in committed arrival order. Hypothesis t (1-based) is assigned
    level alpha_t = alpha * gamma(t) * (R_{t-1} + 1), where R_{t-1} is the number of
    discoveries before t; it is REJECTED (flagged) iff e_t >= 1/alpha_t. Controls
    online FDR <= alpha under ARBITRARY dependence at any stopping time (peek-whenever).

    `gamma` defaults to the registered sequence; pass a callable t->gamma_t to
    override (the always-run test feeds online-fdr's gamma for recurrence parity)."""
    out: list[ElondStep] = []
    r = 0
    for t, e in enumerate(e_values, start=1):
        level = alpha * gamma(t) * (r + 1)
        rejected = level > 0.0 and float(e) >= 1.0 / level
        out.append(ElondStep(rejected, level, float(e)))
        if rejected:
            r += 1
    return out


# --- e-BH: the within-campaign diagnostic (NOT the control) -------------------
def e_bh(e_values: Sequence[float], alpha: float = ONLINE_FDR_ALPHA) -> list[bool]:
    """e-BH (Wang & Ramdas 2022) — the within-campaign DIAGNOSTIC (e-LOND is the FDR
    control of record). Reject the top k* cells, where k* is the largest k with the
    k-th largest e-value e_(k) >= n/(k*alpha). FDR <= alpha under ARBITRARY dependence,
    no penalty. Returns a per-cell boolean in the INPUT order. Reported as an
    isolated-batch view; it never flags and is not part of the lifetime guarantee."""
    n = len(e_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: float(e_values[i]), reverse=True)
    kstar = 0
    for k in range(1, n + 1):
        if float(e_values[order[k - 1]]) >= n / (k * alpha):
            kstar = k
    rejects = [False] * n
    for i in range(kstar):
        rejects[order[i]] = True
    return rejects


# --- the ledger-stream runner: the FDR control over the lifetime ledger -------
def online_fdr_survivors(ledger_rows: Sequence[dict], alpha: float = ONLINE_FDR_ALPHA,
                         kappa: float = CALIBRATOR_KAPPA) -> list[dict]:
    """Run the registered e-value control over the lifetime ledger stream: calibrate
    each cell's `p_value` to an e-value (measurement_invalid / None -> e=0), then
    e-LOND over the stream in committed ledger (arrival) order. Returns the rows
    annotated with `e_value`, `elond_level`, and `elond_survivor` (the flag). This is
    the single FDR guarantee; a survivor is SURFACED for a human, never auto-promoted."""
    e_values = [calibrate_p_to_e(r.get('p_value'), kappa) for r in ledger_rows]
    steps = elond(e_values, alpha)
    return [{**r,
             'e_value': round(e, 6),     # always finite: calibrate clamps p >= 1e-300
             'elond_level': s.level,
             'elond_survivor': s.rejected}
            for r, e, s in zip(ledger_rows, e_values, steps)]
