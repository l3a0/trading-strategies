"""Generate the four educational figures embedded in the tutorial.

Produces PNGs in docs/figures/ that visualize:
  1. Equity curves: covered-call overlay vs. buy-and-hold over the
     bundled MSFT sample.
  2. Histogram of daily excess returns from the same sample, with the
     sample mean and the Newey-West t-stat annotated.
  3. Bias-variance tradeoff for the Newey-West lag cutoff L,
     simulated on an AR(1) process to make the curves smooth.
  4. Expected t-statistic vs. years of data at fixed Sharpe, showing
     how long it would take to clear conventional and Harvey-Liu-Zhu
     significance thresholds.

Run: python make_figures.py
"""

from __future__ import annotations

import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

from cc_backtest import compute_statistics, run_cc_overlay


OUT = "docs/figures"
FIGSIZE = (16, 9)
SAVE_DPI = 100  # 16x9 inches @ 100 dpi → 1600x900px PNG


# Colorblind-safe palette pulled from matplotlib's tab10
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
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


def fig1_equity_curves(daily_equity: list[dict], summary: dict) -> Figure:
    """Overlay vs. buy-and-hold equity curves over the 10-year window."""
    shares = summary["num_contracts"] * 100
    cash = summary["cash"]

    date_strs = [d["date"] for d in daily_equity]
    overlay = np.array([d["equity"] for d in daily_equity], dtype=float)
    prices = np.array([d["price"] for d in daily_equity], dtype=float)
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
    daily_equity: list[dict], summary: dict, stats: dict
) -> Figure:
    """Histogram of daily excess returns with sample-mean and t-stat annotation."""
    shares = summary["num_contracts"] * 100
    cash = summary["cash"]

    equity = np.array([d["equity"] for d in daily_equity], dtype=float)
    prices = np.array([d["price"] for d in daily_equity], dtype=float)
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
    years = np.logspace(0, np.log10(500), 200)
    t_stats = sharpe * np.sqrt(years)

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
    summary, _trades, daily_equity = run_cc_overlay(dates, prices, params)
    stats = compute_statistics(
        daily_equity,
        num_contracts=summary["num_contracts"],
        cash=summary["cash"],
    )

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


if __name__ == "__main__":
    main()
