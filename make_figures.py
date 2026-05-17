"""Generate the educational figures embedded in the tutorial and blog series.

Produces PNGs in docs/figures/ that visualize:
  1. Equity curves: covered-call overlay vs. buy-and-hold over the
     bundled MSFT sample. (tutorial; blog Post 1 + Post 4)
  2. Histogram of daily excess returns from the same sample, with the
     sample mean and the Newey-West t-stat annotated. (tutorial; Post 4)
  3. Bias-variance tradeoff for the Newey-West lag cutoff L,
     simulated on an AR(1) process to make the curves smooth.
     (tutorial; Post 4)
  4. Expected t-statistic vs. years of data at fixed Sharpe, showing
     how long it would take to clear conventional and Harvey-Liu-Zhu
     significance thresholds. (tutorial; Post 4)
  5. Realized vs. proxied implied volatility over the sample, with the
     regime-multiplier bands that produce the gap. (blog Post 2)
  6. Call delta vs. moneyness — the "probability dial" a covered-call
     seller turns, with the income band and the 0.25 strike marked.
     (blog Post 2)
  7. Walk-forward schematic: the real train/lock/test/roll windows
     across all 15 cycles on the MSFT data. (blog Post 3)
  8. In-sample vs. honest out-of-sample vs. buy-and-hold growth over
     the walk-forward span. (blog Post 3)
  9. Monte Carlo shuffle: the real ordered path's return against the
     distribution of 500 scrambled-sequence returns. (blog Post 3)
 10. Overlay P&L per day by market regime — the defensive asymmetry.
     (blog Post 3)
 11. Autocorrelation of daily excess P&L — the dependence Newey-West
     corrects, and why it's mild and negative here. (blog Post 4)
 12. Where the $268K comes from: gross premium → costs → net overlay
     P&L, as a waterfall. (blog Post 1)

Run: python make_figures.py
"""

# matplotlib's bundled stubs leave Axes/pyplot member return types and
# **kwargs as partially-Unknown. Suppress those categories at file level
# rather than annotating every plot()/hist()/set_xlabel() call.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence
from typing import Any, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

from cc_backtest import (
    bs_delta,
    calc_rolling_volatility,
    compute_statistics,
    detect_regime,
    estimate_iv,
    find_strike_for_delta,
    monte_carlo_shuffle,
    regime_analysis,
    run_cc_overlay,
    walk_forward_optimization,
)


OUT = "docs/figures"
FIGSIZE = (16, 9)
SAVE_DPI = 100  # 16x9 inches @ 100 dpi → 1600x900px PNG


# Colorblind-safe palette pulled from matplotlib's tab10
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
GRAY = "#888888"
LIGHTGRAY = "#cccccc"


def load_msft_csv(path: str) -> tuple[list[str], NDArray[np.float64]]:
    """Mirror the parser used by cc_backtest.py's __main__."""
    dates: list[str] = []
    prices: list[float] = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or not row[0][:4].isdigit():
                continue
            dates.append(row[0])
            prices.append(float(row[1]))
    return dates, np.array(prices, dtype=np.float64)


def fig1_equity_curves(
    daily_equity: pd.DataFrame, summary: dict[str, Any]
) -> Figure:
    """Overlay vs. buy-and-hold equity curves over the 10-year window."""
    shares = summary["num_contracts"] * 100
    cash = summary["cash"]

    date_strs = cast("list[str]", daily_equity["date"].tolist())
    overlay = cast(
        "NDArray[np.float64]",
        daily_equity["equity"].to_numpy(dtype=float),
    )
    prices = cast(
        "NDArray[np.float64]",
        daily_equity["price"].to_numpy(dtype=float),
    )
    bh = shares * prices + cash

    dates = np.array(date_strs, dtype="datetime64[D]")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, overlay, color=BLUE, linewidth=2, label="Covered-call overlay")
    ax.plot(dates, bh, color=GRAY, linewidth=2, linestyle="--", label="Buy-and-hold MSFT")

    # Endpoint labels at the right edge
    ax.annotate(
        f"Overlay: ${overlay[-1] / 1000:.0f}K",
        xy=(dates[-1], overlay[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=11,
        color=BLUE,
        fontweight="bold",
    )
    ax.annotate(
        f"Buy & Hold: ${bh[-1] / 1000:.0f}K",
        xy=(dates[-1], bh[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=11,
        color="#555555",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio value", fontsize=12)
    ax.set_title(
        f"Covered call overlay vs. buy-and-hold — MSFT 2016–2026, ${summary['capital'] / 1000:.0f}K start",
        fontsize=14,
        pad=15,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x / 1000:,.0f}K"))
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig2_excess_histogram(
    daily_equity: pd.DataFrame,
    summary: dict[str, Any],
    stats: dict[str, Any],
) -> Figure:
    """Histogram of daily excess returns with sample-mean and t-stat annotation."""
    shares = summary["num_contracts"] * 100
    cash = summary["cash"]

    equity = cast(
        "NDArray[np.float64]",
        daily_equity["equity"].to_numpy(dtype=float),
    )
    prices = cast(
        "NDArray[np.float64]",
        daily_equity["price"].to_numpy(dtype=float),
    )
    bh = shares * prices + cash

    overlay_ret = np.diff(equity) / equity[:-1]
    bh_ret = np.diff(bh) / bh[:-1]
    excess = overlay_ret - bh_ret
    excess_bps = excess * 10_000

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(
        excess_bps,
        bins=60,
        color=BLUE,
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
    )

    mean_bps = float(np.mean(excess_bps))
    std_bps = float(np.std(excess_bps, ddof=1))

    ax.axvline(0, color=GRAY, linestyle="-", linewidth=1, alpha=0.6)
    ax.axvline(
        mean_bps,
        color=RED,
        linestyle="-",
        linewidth=2,
        label=f"Sample mean: {mean_bps:+.3f} bps/day",
    )

    ann_text = (
        f"Mean:    {mean_bps:+.3f} bps/day  "
        f"({stats['ann_excess_return_pct']:+.2f}% annualized)\n"
        f"Std dev: {std_bps:.1f} bps/day\n"
        f"NW t-stat: {stats['t_stat_newey_west']:+.2f}  "
        f"(naive: {stats['t_stat_naive']:+.2f})"
    )
    ax.text(
        0.98,
        0.97,
        ann_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=11,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor=LIGHTGRAY,
            alpha=0.95,
        ),
    )

    ax.set_xlabel("Daily excess return (basis points)", fontsize=12)
    ax.set_ylabel("Count of days", fontsize=12)
    ax.set_title(
        f"Daily excess returns: overlay − buy-and-hold ({len(excess_bps):,} days)",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig3_bias_variance(
    n: int = 2500, phi: float = 0.3, n_paths: int = 2000, L_max: int = 30
) -> Figure:
    """Bias-variance tradeoff for the Newey-West variance-of-the-mean estimator.

    Simulates an AR(1) process y_t = phi * y_{t-1} + eps_t with eps_t ~ N(0,1)
    and computes the NW estimator at each L across `n_paths` independent
    realizations. Compares against the true long-run variance of the mean,
    which for AR(1) is 1 / [n · (1−phi)²].
    """
    rng = np.random.default_rng(seed=42)

    # True long-run variance of the mean for AR(1) with innovation variance 1
    true_var_mean = 1.0 / (n * (1.0 - phi) ** 2)

    estimates = np.zeros((n_paths, L_max + 1))

    for p in range(n_paths):
        eps = rng.standard_normal(n)
        y = np.zeros(n)
        # Initialize at stationary distribution
        y[0] = eps[0] / math.sqrt(1.0 - phi ** 2)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + eps[t]

        y_demean = y - y.mean()
        gamma_0 = float(np.var(y, ddof=1))

        # Precompute autocovariances 1..L_max
        autocov = np.zeros(L_max + 1)
        autocov[0] = gamma_0
        for k in range(1, L_max + 1):
            autocov[k] = float(np.mean(y_demean[:-k] * y_demean[k:]))

        # NW variance-of-the-mean at each L
        for L in range(L_max + 1):
            nw_sum = 0.0
            for k in range(1, L + 1):
                w = 1.0 - k / (L + 1)
                nw_sum += w * autocov[k]
            estimates[p, L] = (gamma_0 + 2.0 * nw_sum) / n

    mean_est = estimates.mean(axis=0)
    var_est = estimates.var(axis=0, ddof=1)
    bias2 = (mean_est - true_var_mean) ** 2
    mse = bias2 + var_est

    # Normalize to max MSE so y-axis is dimensionless and readable
    norm = float(mse.max())
    bias2_n = bias2 / norm
    var_n = var_est / norm
    mse_n = mse / norm

    Ls = np.arange(L_max + 1)
    optimal_L = int(Ls[int(np.argmin(mse))])
    andrews_L = int(4 * (n / 100) ** (2 / 9))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(Ls, bias2_n, color=BLUE, linestyle="--", linewidth=2, label="bias²")
    ax.plot(Ls, var_n, color=ORANGE, linestyle=":", linewidth=2, label="variance")
    ax.plot(Ls, mse_n, color="black", linewidth=2.5, label="MSE = bias² + variance")

    ax.axvline(
        andrews_L,
        color=GREEN,
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label=f"Andrews/NW formula (n={n}): L={andrews_L}",
    )
    if optimal_L != andrews_L:
        ax.axvline(
            optimal_L,
            color="#aaaaaa",
            linestyle="-",
            linewidth=1,
            alpha=0.6,
            label=f"Empirical MSE minimum: L={optimal_L}",
        )

    ax.set_xlabel("Lag cutoff L", fontsize=12)
    ax.set_ylabel("Contribution to MSE (normalized to max=1)", fontsize=12)
    ax.set_title(
        "Why the lag cutoff L isn't free — bias-variance tradeoff",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    ax.set_xlim(0, L_max)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def fig4_t_stat_vs_years(sharpe: float) -> Figure:
    """Expected t-statistic vs. years of data at a fixed Sharpe ratio."""
    years = cast('NDArray[np.float64]', np.logspace(0, np.log10(500), 200))
    t_stats = cast('NDArray[np.float64]', sharpe * np.sqrt(years))

    years_for_t2 = (2.0 / sharpe) ** 2
    years_for_t3 = (3.0 / sharpe) ** 2
    our_t = sharpe * math.sqrt(10)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(years, t_stats, color="black", linewidth=2.5)

    ax.axhline(
        2,
        color=ORANGE,
        linestyle="--",
        linewidth=1.5,
        label=f"Conventional bar (t=2): ~{years_for_t2:.0f} years needed",
    )
    ax.axhline(
        3,
        color=RED,
        linestyle="--",
        linewidth=1.5,
        label=f"HLZ multiple-testing bar (t=3): ~{years_for_t3:.0f} years needed",
    )

    ax.scatter(
        [10],
        [our_t],
        color=BLUE,
        s=120,
        zorder=10,
        edgecolors="white",
        linewidths=1.5,
    )
    ax.annotate(
        f"Our MSFT sample\n(10 years, expected t≈{our_t:.2f})",
        xy=(10, our_t),
        xytext=(14, our_t + 0.55),
        fontsize=11,
        color=BLUE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, alpha=0.7),
    )

    ax.set_xlabel("Years of data", fontsize=12)
    ax.set_ylabel("Expected t-statistic", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(1, 500)
    ax.set_ylim(0, 4)
    ax.set_title(
        f"How long to clear statistical significance? (Sharpe = {sharpe:.3f})",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="-", linewidth=0.4, color="#dddddd", which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig5_implied_vs_realized_vol(
    dates: list[str], prices: NDArray[np.float64]
) -> Figure:
    """Realized 30-day vol vs. the regime-scaled IV proxy the engine uses.

    The four other Black-Scholes inputs are facts; volatility is the
    guess. The engine measures trailing realized vol and marks it up by
    a regime multiplier (1.5x when vol is low, 1.3x normal, 1.1x high).
    The horizontal bands are the regime cutoffs (15% / 25%); the gap
    between the two lines is the markup that gets assumed into existence.
    """
    rolling = calc_rolling_volatility(prices, window=30)  # len == len(prices) - 1
    rv = np.asarray(rolling, dtype=float) * 100.0
    iv = np.array(
        [
            estimate_iv(v, detect_regime(v)) * 100.0 if np.isfinite(v) else np.nan
            for v in np.asarray(rolling, dtype=float)
        ]
    )
    x = np.array(dates[1:], dtype="datetime64[D]")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ymax = float(np.nanmax(iv)) * 1.05

    # Regime bands: the realized-vol level determines the multiplier.
    ax.axhspan(0, 15, color=GREEN, alpha=0.07)
    ax.axhspan(15, 25, color=ORANGE, alpha=0.07)
    ax.axhspan(25, ymax, color=RED, alpha=0.07)
    for y, label in (
        (7.5, "Low vol (<15%) → 1.5× markup"),
        (20, "Normal (15–25%) → 1.3× markup"),
        (max(27, ymax - 4), "High vol (>25%) → 1.1× markup"),
    ):
        ax.text(
            x[int(len(x) * 0.015)], y, label, fontsize=10, va="center",
            color="#555555", style="italic", zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )

    ax.plot(x, rv, color=GRAY, linewidth=1.6, label="Realized vol (trailing 30-day)")
    ax.plot(x, iv, color=BLUE, linewidth=1.8, label="Proxied implied vol (regime-scaled)")
    ax.fill_between(
        x, rv, iv,
        where=cast("Sequence[bool]", np.isfinite(iv)),
        color=BLUE, alpha=0.12,
        label="Assumed markup",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Annualized volatility", fontsize=12)
    ax.set_title(
        "The one input you can't look up — realized vol vs. the IV proxy (MSFT)",
        fontsize=14,
        pad=15,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig6_delta_dial(prices: NDArray[np.float64]) -> Figure:
    """Call delta vs. how far out-of-the-money the strike is set.

    Delta reads as the rough probability the shares get called away, so
    it's the risk dial a covered-call seller turns. Income sellers live
    in the 0.20–0.40 band; the bundled backtest uses 0.25.
    """
    rolling = calc_rolling_volatility(prices, window=30)
    finite = np.asarray(rolling, dtype=float)
    finite = finite[np.isfinite(finite)]
    base_vol = float(np.median(finite))
    sigma = estimate_iv(base_vol, detect_regime(base_vol))  # representative IV
    S = float(prices[0])
    T = 21.0 / 252.0
    r = 0.045

    otm_pct = np.linspace(0.0, 22.0, 240)
    strikes = S * (1.0 + otm_pct / 100.0)
    deltas = np.array([bs_delta(S, float(K), T, r, sigma, "call") for K in strikes])

    k25 = find_strike_for_delta(S, T, r, sigma, 0.25, "call")
    otm25 = (k25 / S - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhspan(0.20, 0.40, color=GREEN, alpha=0.10, label="Income-seller band (0.20–0.40)")
    ax.plot(otm_pct, deltas, color=BLUE, linewidth=2.5)

    ax.scatter([otm25], [0.25], color=RED, s=120, zorder=10,
               edgecolors="white", linewidths=1.5)
    ax.annotate(
        f"Backtest setting: 0.25 delta\n"
        f"strike ≈ \\${k25:,.0f}  ({otm25:.1f}% above the \\${S:,.0f} stock)\n"
        f"≈ 1-in-4 chance of assignment",
        xy=(otm25, 0.25),
        xytext=(otm25 + 3.5, 0.42),
        fontsize=11,
        color=RED,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, alpha=0.7),
    )

    ax.set_xlabel("How far out-of-the-money the strike is set (%)", fontsize=12)
    ax.set_ylabel("Call delta  ≈  P(shares called away)", fontsize=12)
    ax.set_title(
        f"Delta is a probability dial  "
        f"(MSFT stock ≈ ${S:,.0f},  assumed volatility ≈ {sigma * 100:.0f}%,  "
        f"21 days to expiry)",
        fontsize=14,
        pad=15,
    )
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig7_walk_forward_schematic(records: list[dict[str, Any]]) -> Figure:
    """The rolling exam: train two years, lock, test six months, roll.

    Uses the real period bounds from walk_forward_optimization on the
    bundled MSFT data — 15 train/test cycles, non-overlapping test
    windows stitched into the honest out-of-sample curve.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    most = (0.25, 21, 0.75)  # the configuration the optimizer keeps choosing
    for i, rec in enumerate(records):
        y = i + 1  # period 1 first; y-axis inverted below so it sits on top
        tr0 = mdates.date2num(np.datetime64(rec["train_start"]))
        tr1 = mdates.date2num(np.datetime64(rec["train_end"]))
        te0 = mdates.date2num(np.datetime64(rec["test_start"]))
        te1 = mdates.date2num(np.datetime64(rec["test_end"]))
        ax.barh(y, tr1 - tr0, left=tr0, height=0.6, color=BLUE, alpha=0.35,
                label="Train (2y, search 27 combos)" if i == 0 else None)
        bp = rec["best_params"]
        chosen = (bp["call_delta"], bp["dte"], bp["close_at_pct"])
        ax.barh(y, te1 - te0, left=te0, height=0.6, color=ORANGE,
                label="Test (6mo, rules LOCKED)" if i == 0 else None)
        if chosen != most:
            ax.annotate(
                f"Δ{bp['call_delta']} / {int(bp['dte'])}d / {bp['close_at_pct']}",
                xy=(float(te1), float(y)), xytext=(6, 0),
                textcoords="offset points",
                va="center", fontsize=8, color="#555555",
            )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    # Pad the right edge so the latest period's param annotation fits.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 220)
    ax.set_ylim(0.3, len(records) + 0.7)
    ax.invert_yaxis()  # period 1 at the top, reading downward in time
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Walk-forward cycle", fontsize=12)
    ax.set_yticks(range(1, len(records) + 1))
    ax.set_title(
        "Studying for a different test — 15 train/lock/test/roll cycles (MSFT)",
        fontsize=14,
        pad=15,
    )
    ax.text(
        0.5, -0.12,
        "The strike dial locked onto Δ0.25 in 14 of 15 cycles; the other two "
        "knobs wandered, but stayed in a tight neighborhood of the middle "
        "setting — labels mark cycles that left Δ0.25/21d/0.75.",
        transform=ax.transAxes, ha="center", fontsize=10, color="#555555",
        style="italic",
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig8_is_vs_oos(
    dates: list[str],
    prices: NDArray[np.float64],
    oos_equity: pd.DataFrame,
    records: list[dict[str, Any]],
    params: dict[str, float],
) -> Figure:
    """Optimized vs. honest out-of-sample vs. buy-and-hold growth.

    The optimized fixed-defaults curve and the no-hindsight stitched OOS
    curve both compound far above 1x — no collapse — but buy-and-hold
    over the identical span gets most of the way there on its own. The
    honest edge over doing nothing clever is a sliver.
    """
    oos_lo = records[0]["test_start"]
    oos_hi = records[-1]["test_end"]
    span_idx = [i for i, d in enumerate(dates) if oos_lo <= d < oos_hi]
    s0, s1 = span_idx[0], span_idx[-1] + 1
    span_dates = np.array(dates[s0:s1], dtype="datetime64[D]")
    span_prices = prices[s0:s1]

    # Honest OOS: chain each locked 6-month window's growth. Pull the
    # equity column and dates once as plain arrays (pandas-stubs degrades
    # the .loc[mask, col] form; column-then-boolean-index stays typed).
    oos_dates = np.array(
        cast("list[str]", oos_equity["date"].tolist()),  # pyright: ignore[reportUnknownMemberType]
        dtype="datetime64[D]",
    )
    oos_eq_all = cast(
        "NDArray[np.float64]",
        oos_equity["equity"].to_numpy(dtype=float),  # pyright: ignore[reportUnknownMemberType]
    )
    oos_growth = np.empty(len(oos_equity), dtype=float)
    cum = 1.0
    pos = 0
    for rec in records:
        mask = (oos_dates >= np.datetime64(rec["test_start"])) & (
            oos_dates < np.datetime64(rec["test_end"])
        )
        eq = oos_eq_all[mask]
        seg = cum * (eq / eq[0])
        oos_growth[pos:pos + len(seg)] = seg
        pos += len(seg)
        cum *= eq[-1] / eq[0]

    # Optimized fixed-defaults and buy-and-hold over the identical span.
    fx_summary, _, fx_eq = run_cc_overlay(list(dates[s0:s1]), span_prices, params)
    fx_equity = fx_eq["equity"].to_numpy(dtype=float)
    fx_growth = fx_equity / fx_equity[0]
    shares = fx_summary["num_contracts"] * 100
    bh = shares * span_prices + fx_summary["cash"]
    bh_growth = bh / bh[0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(span_dates, (fx_growth - 1) * 100, color=GREEN, linewidth=2,
            label="Optimized fixed defaults (in-sample flavor)")
    ax.plot(span_dates[:len(oos_growth)], (oos_growth - 1) * 100, color=BLUE,
            linewidth=2, label="Honest out-of-sample (no hindsight)")
    ax.plot(span_dates, (bh_growth - 1) * 100, color=GRAY, linewidth=2,
            linestyle="--", label="Buy-and-hold MSFT")

    for val, color, yoff in (
        ((fx_growth[-1] - 1) * 100, GREEN, 0),
        ((oos_growth[-1] - 1) * 100, BLUE, 0),
        ((bh_growth[-1] - 1) * 100, "#555555", 0),
    ):
        ax.annotate(f"{val:+.0f}%", xy=(span_dates[-1], val),
                    xytext=(8, yoff), textcoords="offset points",
                    va="center", fontsize=11, color=color, fontweight="bold")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative return", fontsize=12)
    ax.set_title(
        "No collapse — but the honest edge over buy-and-hold is a sliver "
        "(walk-forward span)",
        fontsize=14,
        pad=15,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig9_monte_carlo(mc: dict[str, Any]) -> Figure:
    """Real ordered path's return vs. 500 scrambled-sequence returns."""
    mc_returns = np.asarray(mc["mc_returns"], dtype=float)
    real = float(mc["real_return"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(mc_returns, bins=40, color=BLUE, alpha=0.6,
            edgecolor="white", linewidth=0.5,
            label=f"{len(mc_returns)} shuffled paths")
    ax.axvline(float(mc["mc_mean"]), color=GRAY, linestyle=":", linewidth=2,
               label=f"Shuffle mean: {mc['mc_mean']:.0f}%")
    ax.axvline(float(mc["mc_max"]), color=ORANGE, linestyle="--", linewidth=1.5,
               label=f"Best shuffle: {mc['mc_max']:.0f}%")
    ax.axvline(real, color=RED, linewidth=2.5,
               label=f"Real ordered path: {real:.0f}%")

    ax.annotate(
        f"Real path beats every one of\n{len(mc_returns)} shuffles "
        f"(percentile {mc['percentile']:.0f})",
        xy=(real, ax.get_ylim()[1] * 0.6),
        xytext=(real - (real - float(mc['mc_mean'])) * 1.1,
                ax.get_ylim()[1] * 0.78),
        fontsize=11, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, alpha=0.7),
    )

    ax.set_xlabel("Total return of the overlay (%)", fontsize=12)
    ax.set_ylabel("Count of shuffled paths", fontsize=12)
    ax.set_title(
        "Monte Carlo: harvesting a statistical property, not a lucky sequence",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig10_regime_pnl(regimes: dict[str, dict[str, float | int]]) -> Figure:
    """Overlay P&L per day by market regime — the defensive asymmetry."""
    order = ["bull", "sideways", "bear"]
    labels = ["Bull", "Sideways", "Bear"]
    per_day = [float(regimes[k]["avg_pnl_per_day"]) for k in order]
    colors = [GREEN, GRAY, RED]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, per_day, color=colors, alpha=0.75,
                  edgecolor="white", linewidth=1)
    for b, v in zip(bars, per_day):
        ax.annotate(f"${v:,.0f}/day", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Average overlay P&L per day in regime ($)", fontsize=12)
    ax.set_title(
        "Structurally defensive: the overlay earns most of its keep when "
        "the market isn't going straight up",
        fontsize=14,
        pad=15,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig


def fig11_excess_acf(
    daily_equity: pd.DataFrame, summary: dict[str, Any]
) -> Figure:
    """Autocorrelation of daily excess P&L — what Newey-West corrects for.

    The textbook t-stat assumes these bars are all zero. Holding one
    option for weeks makes them non-zero. Here the net is mild and
    slightly negative, which is why the correction nudged 0.40 → 0.46
    instead of demoting a fake winner.
    """
    shares = summary["num_contracts"] * 100
    cash = summary["cash"]
    equity = daily_equity["equity"].to_numpy(dtype=float)
    prices = daily_equity["price"].to_numpy(dtype=float)
    bh = shares * prices + cash
    excess = (np.diff(equity) / equity[:-1]) - (np.diff(bh) / bh[:-1])

    n = len(excess)
    e = excess - excess.mean()
    var = float(np.mean(e * e))
    max_lag = 20
    acf = np.array([float(np.mean(e[:-k] * e[k:])) / var for k in range(1, max_lag + 1)])
    lags = np.arange(1, max_lag + 1)
    ci = 1.96 / math.sqrt(n)
    nw_L = int(4 * (n / 100) ** (2 / 9))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhspan(-ci, ci, color=GRAY, alpha=0.15,
               label=f"95% white-noise band (±{ci:.3f})")
    ax.axvspan(0.5, nw_L + 0.5, color=ORANGE, alpha=0.10,
               label=f"Newey-West window (lags 1–{nw_L})")
    colors = [BLUE if v <= 0 else RED for v in acf]
    ax.bar(lags, acf, width=0.6, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Lag (trading days)", fontsize=12)
    ax.set_ylabel("Autocorrelation of daily excess P&L", fontsize=12)
    ax.set_title(
        "The dependence the textbook formula ignores — mild and net-negative here",
        fontsize=14,
        pad=15,
    )
    ax.set_xticks(lags)
    ax.set_xlim(0.5, max_lag + 0.5)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig12_premium_waterfall(summary: dict[str, Any]) -> Figure:
    """Where the $268K comes from: gross premium → costs → net P&L."""
    gross = float(summary["total_premium_collected"])
    costs = float(summary["overlay_costs"])
    net = float(summary["net_overlay_pnl"])
    retention = float(summary["premium_retention_pct"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Bar 1: gross premium from 0. Bar 2: floating drop for costs.
    # Bar 3: net, from 0.
    ax.bar("Gross premium\ncollected", gross, color=GREEN, alpha=0.75,
           edgecolor="white", linewidth=1, width=0.6)
    ax.bar("Buybacks +\nassignment costs", costs, bottom=net, color=RED,
           alpha=0.75, edgecolor="white", linewidth=1, width=0.6)
    ax.bar("Net overlay\nP&L", net, color=BLUE, alpha=0.85,
           edgecolor="white", linewidth=1, width=0.6)

    # Connector lines between the floating bars.
    ax.plot([0.3, 0.7], [gross, gross], color=GRAY, linewidth=1, linestyle=":")
    ax.plot([1.3, 1.7], [net, net], color=GRAY, linewidth=1, linestyle=":")

    for x, top, label in (
        (0, gross, f"${gross:,.0f}"),
        (1, net + costs, f"−${costs:,.0f}"),
        (2, net, f"${net:,.0f}"),
    ):
        ax.annotate(label, xy=(x, top), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    fontsize=12, fontweight="bold")

    ax.set_ylabel("Dollars", fontsize=12)
    ax.set_title(
        f"The $268K, decomposed — only {retention:.1f}% of gross premium "
        f"survives to the bottom line",
        fontsize=14,
        pad=15,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}K"))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    dates, prices = load_msft_csv("msft_10yr_prices.csv")
    params: dict[str, float] = {
        "call_delta": 0.25,
        "close_at_pct": 0.75,
        "dte": 21,
        "risk_free_rate": 0.045,
        "capital": 100_000,
    }
    summary, trades, daily_equity = run_cc_overlay(dates, prices, params)
    stats = compute_statistics(
        daily_equity,
        num_contracts=summary["num_contracts"],
        cash=summary["cash"],
    )

    # Walk-forward grid mirrors test_cc_backtest.py's pinned grid.
    param_grid: dict[str, list[float]] = {
        "call_delta": [0.15, 0.20, 0.25],
        "dte": [21, 30, 45],
        "close_at_pct": [0.50, 0.75, 1.00],
    }

    print(f"Generating figures into {OUT}/")
    print(f"  Sharpe of excess return: {stats['sharpe_excess']:+.3f}")
    print(f"  Newey-West t-stat:       {stats['t_stat_newey_west']:+.2f}")

    fig1_equity_curves(daily_equity, summary).savefig(
        f"{OUT}/01_equity_curves.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/01_equity_curves.png")
    plt.close("all")

    fig2_excess_histogram(daily_equity, summary, stats).savefig(
        f"{OUT}/02_excess_histogram.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/02_excess_histogram.png")
    plt.close("all")

    print("  running bias-variance simulation (2000 AR(1) paths)...")
    fig3_bias_variance().savefig(
        f"{OUT}/03_bias_variance.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/03_bias_variance.png")
    plt.close("all")

    fig4_t_stat_vs_years(stats["sharpe_excess"]).savefig(
        f"{OUT}/04_t_stat_vs_years.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/04_t_stat_vs_years.png")
    plt.close("all")

    fig5_implied_vs_realized_vol(dates, prices).savefig(
        f"{OUT}/05_implied_vs_realized_vol.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/05_implied_vs_realized_vol.png")
    plt.close("all")

    fig6_delta_dial(prices).savefig(
        f"{OUT}/06_delta_dial.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/06_delta_dial.png")
    plt.close("all")

    print("  running walk-forward optimization (15 cycles × 27 combos)...")
    oos_equity, records = walk_forward_optimization(dates, prices, param_grid)

    fig7_walk_forward_schematic(records).savefig(
        f"{OUT}/07_walk_forward_schematic.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/07_walk_forward_schematic.png")
    plt.close("all")

    fig8_is_vs_oos(dates, prices, oos_equity, records, params).savefig(
        f"{OUT}/08_is_vs_oos.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/08_is_vs_oos.png")
    plt.close("all")

    print("  running Monte Carlo shuffle (500 paths)...")
    mc = monte_carlo_shuffle(dates, prices, params, n_shuffles=500, seed=42)

    fig9_monte_carlo(mc).savefig(
        f"{OUT}/09_monte_carlo.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/09_monte_carlo.png")
    plt.close("all")

    regimes = regime_analysis(dates, prices, trades)
    fig10_regime_pnl(regimes).savefig(
        f"{OUT}/10_regime_pnl.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/10_regime_pnl.png")
    plt.close("all")

    fig11_excess_acf(daily_equity, summary).savefig(
        f"{OUT}/11_excess_acf.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/11_excess_acf.png")
    plt.close("all")

    fig12_premium_waterfall(summary).savefig(
        f"{OUT}/12_premium_waterfall.png", dpi=SAVE_DPI, bbox_inches="tight"
    )
    print(f"  wrote {OUT}/12_premium_waterfall.png")
    plt.close("all")


if __name__ == "__main__":
    main()
