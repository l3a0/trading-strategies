"""Generate the Jupyter notebook companion from the tutorial markdown.

The tutorial (`tutorial_covered_call_backtest.md`) is the single source of
truth. This script parses it into notebook cells so the two never drift:

  * Prose becomes markdown cells, split at H2/H3 headings so each cell is a
    digestible section a reader can run-as-they-read.
  * Fenced ``python`` blocks that actually compile become runnable code
    cells. Signature stubs and pseudo-code (the tutorial's "illustrative,
    not in the codebase" excerpts) don't compile, so they stay rendered as
    markdown — exactly as they read in the tutorial.
  * Every ``![...](docs/figures/NN_*.png)`` embed is replaced by the
    matching chart-generation call from `make_figures.py`, so the chart
    code is visible and the figure renders inline when the cell runs.
  * Sections that *link* to code instead of inlining it (so there's no
    fenced block to convert) get a generated demo cell that runs the
    linked helpers — see ``LINKED_CODE_DEMOS``.

A setup cell at the top clones the repo and pip-installs when running on
Google Colab (no-op locally), then imports the engine's public API.

Run: python make_notebook.py   →   covered_call_backtest.ipynb

CI/consistency note: this is the notebook's only source. Re-run it after
editing the tutorial or the figure script and commit the regenerated
.ipynb (the same contract `make_figures.py` has with the PNGs).
"""

from __future__ import annotations

import json
import re

TUTORIAL = "tutorial_covered_call_backtest.md"
OUT = "covered_call_backtest.ipynb"

# docs/figures/NN_name.png  ->  the make_figures call that regenerates it.
# stats / summary / daily_equity are bound by the data-prep cell below.
FIGURE_CALLS: dict[str, str] = {
    "01_equity_curves.png": "fig1_equity_curves(daily_equity, summary)",
    "02_excess_histogram.png": "fig2_excess_histogram(daily_equity, summary, stats)",
    "03_bias_variance.png": "fig3_bias_variance()",
    "04_t_stat_vs_years.png": 'fig4_t_stat_vs_years(stats["sharpe_excess"])',
    "05_implied_vs_realized_vol.png": "fig5_implied_vs_realized_vol(dates, prices)",
    "06_delta_dial.png": "fig6_delta_dial(prices)",
    "07_walk_forward_schematic.png": "fig7_walk_forward_schematic(records)",
    "09_monte_carlo.png": "fig9_monte_carlo(mc)",
    "10_regime_pnl.png": "fig10_regime_pnl(regimes)",
    "11_excess_acf.png": "fig11_excess_acf(daily_equity, summary)",
}

IMAGE_RE = re.compile(r"^!\[.*\]\((?:\./)?docs/figures/([0-9A-Za-z_]+\.png)\)\s*$")

# H2–H4 starts a new markdown cell (and bounds a LINKED_CODE_DEMOS section).
# Not H1 (the single doc title) and not H5+ (none exist).
HEADING_RE = re.compile(r"#{2,4} ")

# Spans where a literal $ must be left alone when escaping currency dollars
# (see escape_notebook_dollars): single-backtick inline code and $$...$$
# display math. Matched left-to-right so the prose *between* two code spans
# (e.g. the cost paragraph's "$0.65/contract ... $0.0065/share") is treated
# as prose, not as one span.
_DOLLAR_SAFE_RE = re.compile(r"`[^`\n]*`|\$\$[^$\n]*\$\$")

# Some tutorial sections *link* to code in cc_backtest.py instead of inlining
# it (the repo's "linked, test-pinned implementations" convention), so the
# converter sees no fenced block to turn into a runnable cell. For those, map
# the exact section heading -> a short demo that exercises the linked helpers
# on the bundled MSFT data. The demo cell is emitted right after the section's
# prose. Same single-source idea as FIGURE_CALLS: the runnable code lives here
# in the generator, not in the tutorial markdown.
LINKED_CODE_DEMOS: dict[str, str] = {
    "### The IV Proxy: Why a Regime-Based Multiplier Works": '''\
# Run the linked helpers on the bundled MSFT data:
#   calc_rolling_volatility  ->  detect_regime  ->  estimate_iv
# (the three cc_backtest.py functions this section links to)
import collections

rolling_vol = calc_rolling_volatility(prices, window=30)

# Most recent day with a valid (non-NaN warm-up) rolling vol
i = int(np.flatnonzero(~np.isnan(rolling_vol))[-1])
hv = float(rolling_vol[i])
regime = detect_regime(hv)
iv = estimate_iv(hv)
print(
    f"{dates[i]}  30-day HV {hv:6.2%}  ->  regime {regime:<6}"
    f"  ->  IV estimate {iv:6.2%}  ({iv / hv:.1f}x)"
)

# Regime mix across the whole sample (multipliers: high 1.1 / normal 1.3 / low 1.5)
valid = rolling_vol[~np.isnan(rolling_vol)]
mix = collections.Counter(detect_regime(float(v)) for v in valid)
for r in ("low", "normal", "high"):
    print(f"  {r:<6} {mix[r]:4d} days ({mix[r] / len(valid):5.1%})")''',
    "### The State Machine: OPEN → Check → Handle → Reset": '''\
# The real engine: run_cc_overlay inlines exactly this state machine
# (the four transitions above are its if/elif branches).
import collections

summary, trades, _ = run_cc_overlay(dates, prices, params)

# The engine's trade actions map 1:1 onto the diagram's branches:
#   sell        IDLE -> OPEN: sold a 0.25-delta call
#   close       profit target hit (75% of premium captured)
#   close_itm   deep-ITM assignment risk (delta > 0.70)
#   expiration  reached expiry: assigned if ITM, else expired worthless
counts = collections.Counter(t["action"] for t in trades)
for action in ("sell", "close", "close_itm", "expiration"):
    print(f"  {action:<11} {counts[action]:4d}")

print("\\nFirst 4 trade-state transitions:")
for t in trades[:4]:
    extra = (
        f" strike ${t['strike']:.0f}"
        if t["action"] == "sell"
        else f" pnl ${t['pnl']:+.2f}"
    )
    print(f"  {t['date']}  {t['action']:<11}{extra}")''',
    "### Transaction Costs: Commission ($0.65/contract) + Slippage (3% of Premium)": '''\
# Transaction costs aren't a helper — run_cc_overlay inlines them. This is
# the exact sell-side line from cc_backtest.py#L335:
#     net_premium = premium * (1 - 0.03) - 0.0065
# (3% slippage; $0.65/contract commission = $0.0065/share). Run it on a
# range of gross premiums:
SLIPPAGE = 0.03
COMMISSION_PER_SHARE = 0.65 / 100

hdr = f"{'gross $/sh':>10} {'-slippage':>10} {'-commission':>12} {'net $/sh':>10} {'net $/contract':>15}"
print(hdr)
for gross in (0.05, 0.50, 0.89, 1.00, 2.50):
    net = gross * (1 - SLIPPAGE) - COMMISSION_PER_SHARE
    print(
        f"{gross:>10.2f} {gross * SLIPPAGE:>10.4f} {COMMISSION_PER_SHARE:>12.4f}"
        f" {net:>10.4f} {net * 100:>15.2f}"
    )

# The section's worked example: $1.00 gross -> $0.9635 net/share
assert abs((1.00 * (1 - SLIPPAGE) - COMMISSION_PER_SHARE) - 0.9635) < 1e-9
print("\\n$1.00 gross -> $0.9635 net/share  (matches the worked example)")

# Negative-net guard: deep-OTM calls where costs exceed the credit are skipped
tiny = 0.005
print(
    f"${tiny} gross -> {tiny * (1 - SLIPPAGE) - COMMISSION_PER_SHARE:+.4f} net/share"
    f"  -> run_cc_overlay refuses to open (net_premium <= 0)"
)''',
    "### How to Stitch Out-of-Sample Results into a Single Equity Curve": '''\
# The walk-forward search already ran once in the setup cell (it binds
# `records` + `oos_equity` from the 3x3x3 grid pinned by
# test_walk_forward_optimization). Reuse those — re-running here would
# repeat the 405-backtest search just to reprint the same results.
import collections

periods = records

print(
    f"{len(periods)} out-of-sample periods: "
    f"{periods[0]['test_start']} -> {periods[-1]['test_end']}"
)
for key in ("call_delta", "dte", "close_at_pct"):
    tally = collections.Counter(p["best_params"][key] for p in periods)
    spread = "  ".join(f"{v}x{n}" for v, n in sorted(tally.items()))
    print(f"  {key:<12} -> {tally.most_common(1)[0][0]!s:<5} ({spread})")

# Cumulative OOS return = chain each period's 6-month return. The stitched
# curve resets per period, so dividing last by first would be meaningless.
# This is the same chaining test_walk_forward_optimization asserts; it's
# shown once, here, because this is the section that explains stitching.
cumulative = 1.0
for p in periods:
    seg = oos_equity.loc[
        (oos_equity["date"] >= p["test_start"])
        & (oos_equity["date"] < p["test_end"]),
        "equity",
    ]
    cumulative *= 1.0 + (seg.iloc[-1] - seg.iloc[0]) / seg.iloc[0]
print(f"\\nChained OOS compound return: {cumulative - 1.0:+.0%}")''',
    "#### The Code": '''\
# compute_statistics already ran in the data-prep cell — these are its real
# outputs (exactly what the next section, "What MSFT Actually Says", quotes):
import math

print(f"Annualized Excess Return:  {stats['ann_excess_return_pct']:+7.3f}%")
print(f"Annualized Excess Vol:     {stats['ann_excess_vol_pct']:7.2f}%")
print(f"Sharpe of Excess Return:   {stats['sharpe_excess']:+7.3f}")
print(f"t-stat (naive, IID):       {stats['t_stat_naive']:+7.2f}")
print(f"t-stat (Newey-West, L={stats['nw_lag']}): {stats['t_stat_newey_west']:+7.2f}")
print(f"Clears t=2 bar?            {stats['passes_t_2']}")
print(f"Clears t=3 bar (HLZ)?      {stats['passes_t_3']}")

# Cross-check the t ~ Sharpe x sqrt(years) shortcut the tutorial derives:
yrs, sh = stats["years_of_data"], stats["sharpe_excess"]
print(
    f"\\nShortcut: Sharpe x sqrt(years) = {sh:.3f} x sqrt({yrs}) "
    f"= {sh * math.sqrt(yrs):.2f}  (vs naive t {stats['t_stat_naive']:+.2f})"
)''',
    "### The Parameter Grid: What We Search Over and Why": '''\
# Expand the 3x3x3 grid with the real helper the optimizer uses.
from cc_backtest import _param_combinations

grid = {
    "call_delta": [0.15, 0.20, 0.25],
    "dte": [21, 30, 45],
    "close_at_pct": [0.50, 0.75, 1.00],
}
combos = _param_combinations(grid)
sizes = " x ".join(str(len(v)) for v in grid.values())
print(f"{sizes} = {len(combos)} parameter sets")
for c in combos[:3]:
    print("  ", c)
print("   ...")
for c in combos[-2:]:
    print("  ", c)''',
    "### Monte Carlo Simulation: Shuffle Daily Returns, Rebuild Price Paths": '''\
# `mc` already came from cc_backtest.monte_carlo_shuffle in the setup
# cell (500 paths, seed=42 — the exact call test_monte_carlo_shuffle
# pins). Reuse it; rerunning the 500-backtest shuffle here would just
# reproduce the same dict the heavy setup cell already holds.
print(
    f"real {mc['real_return']:.0f}%  vs  MC mean {mc['mc_mean']:.0f}%"
    f"  (max {mc['mc_max']:.0f}%)  ->  percentile {mc['percentile']}"
)
print(
    f"the real ordered path beats {mc['percentile']}%"
    f" of {mc['n_completed']} shuffled paths"
)''',
    "### Sensitivity Analysis: Perturb Each Parameter, See If Results Collapse": '''\
# Reuse the real cc_backtest.sensitivity_analysis — the exact function
# test_sensitivity_perturbations pins, so the notebook can't drift from it.
# Same sweeps: call_delta ±0.05/±0.10, close_at_pct ±0.10/±0.20.
from cc_backtest import sensitivity_analysis

sens = sensitivity_analysis(dates, prices, params)
for name, res in sens.items():
    cells = "  ".join(
        f"{off:+.2f}:{ret:.0f}%" if off != 0.0 else f"base:{ret:.0f}%"
        for off, ret in res["returns"]
    )
    verdict = "robust" if res["worst_drop_pct"] < 10 else "fragile"
    print(
        f"{name:<12} {cells}\\n"
        f"{'':<12} worst drop from base {res['worst_drop_pct']:.1f}%"
        f"  ->  {verdict}"
    )''',
    "### Regime Analysis: Does It Work in Bulls, Bears, and Sideways?": '''\
# Bucket overlay trade P&L by 200-day-SMA regime — same as
# test_regime_analysis.
from cc_backtest import regime_analysis

_, trades, _ = run_cc_overlay(dates, prices, params)
reg = regime_analysis(dates, prices, trades)
print(f"{'regime':<9} {'days':>5} {'total P&L':>14} {'$/day':>8}")
for r in ("bull", "bear", "sideways", "unknown"):
    d = reg[r]
    print(f"{r:<9} {d['days']:>5} {d['total_pnl']:>14,.0f} {d['avg_pnl_per_day']:>8,.0f}")''',
    "#### Risk-Managed Covered Calls: Stripping Out the Equity-Timing Wiggle": '''\
# Same trade flow, two modes: naive vs. delta_hedge=1.0. Reproduces the
# side-by-side table; pinned by TestMsftTenYearRegression.test_significance
# (naive) and TestMsftRiskManagedRegression.test_significance_uplift (hedged).
def _excess_stats(p):
    s, _, eq = run_cc_overlay(dates, prices, p)
    return compute_statistics(eq, num_contracts=s["num_contracts"], cash=s["cash"])

naive = _excess_stats(params)
managed = _excess_stats({**params, "delta_hedge": 1.0})
rows = (
    ("Annualized excess return", "ann_excess_return_pct", "{:+.3f}%"),
    ("Annualized excess vol", "ann_excess_vol_pct", "{:.2f}%"),
    ("Sharpe of excess return", "sharpe_excess", "{:+.3f}"),
    ("t-stat (Newey-West)", "t_stat_newey_west", "{:+.2f}"),
)
print(f"{'Metric':<26} {'Naive CC':>12} {'Risk-Managed':>14}")
for label, key, fmt in rows:
    print(f"{label:<26} {fmt.format(naive[key]):>12} {fmt.format(managed[key]):>14}")
print(f"{'Clears t = 2 bar?':<26} {str(naive['passes_t_2']):>12} "
      f"{str(managed['passes_t_2']):>14}")''',
}

SETUP_CODE = '''\
# === Setup — runs anywhere; clones + installs only on Google Colab ===
import os
import subprocess
import sys

if "google.colab" in sys.modules:
    if not os.path.isdir("covered-call-backtesting"):
        subprocess.run(
            ["git", "clone",
             "https://github.com/l3a0/covered-call-backtesting.git"],
            check=True,
        )
    os.chdir("covered-call-backtesting")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )

%matplotlib inline
import numpy as np

from cc_backtest import (
    bs_delta,
    bs_price,
    calc_rolling_volatility,
    compute_statistics,
    detect_regime,
    estimate_iv,
    find_strike_for_delta,
    monte_carlo_shuffle,
    normal_cdf,
    normal_pdf,
    regime_analysis,
    run_cc_overlay,
    walk_forward_optimization,
)
'''

DATA_PREP_CODE = '''\
# === Run the backtest once — the figure cells below reuse these results ===
# Heads-up: this cell also runs the 15x27 walk-forward and a 500-path
# Monte Carlo so the Part 4/5 figure cells have their inputs. That makes
# this the slow cell (tens of seconds, like the bias-variance figure) —
# it runs once and every figure below reuses the results.
from make_figures import (
    fig1_equity_curves,
    fig2_excess_histogram,
    fig3_bias_variance,
    fig4_t_stat_vs_years,
    fig5_implied_vs_realized_vol,
    fig6_delta_dial,
    fig7_walk_forward_schematic,
    fig9_monte_carlo,
    fig10_regime_pnl,
    fig11_excess_acf,
    load_msft_csv,
)

dates, prices = load_msft_csv("msft_10yr_prices.csv")
params = {
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
param_grid = {
    "call_delta": [0.15, 0.20, 0.25],
    "dte": [21, 30, 45],
    "close_at_pct": [0.50, 0.75, 1.00],
}
oos_equity, records = walk_forward_optimization(dates, prices, param_grid)
mc = monte_carlo_shuffle(dates, prices, params, n_shuffles=500, seed=42)
regimes = regime_analysis(dates, prices, trades)

print(
    f"Backtest ready — Sharpe of excess return: {stats['sharpe_excess']:+.3f}, "
    f"Newey-West t-stat: {stats['t_stat_newey_west']:+.2f}, "
    f"walk-forward periods: {len(records)}, "
    f"MC percentile: {mc['percentile']:.0f}"
)
'''

INTRO_MD = '''\
# Covered Call Backtester — Notebook Companion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l3a0/covered-call-backtesting/blob/main/covered_call_backtest.ipynb)

This notebook is generated from
[`tutorial_covered_call_backtest.md`](https://github.com/l3a0/covered-call-backtesting/blob/main/tutorial_covered_call_backtest.md)
by [`make_notebook.py`](https://github.com/l3a0/covered-call-backtesting/blob/main/make_notebook.py)
— the tutorial is the source of truth; don't hand-edit this file.

Run the two setup cells first, then read top-to-bottom. Code cells are the
tutorial's runnable excerpts plus the chart-generation calls from
`make_figures.py`; signature stubs and pseudo-code stay as formatted text,
just as they appear in the tutorial.
'''


def compiles(src: str) -> bool:
    """True if the snippet is a syntactically valid module on its own.

    Signature-only stubs and pseudo-code (top-level ``return``) raise
    SyntaxError and stay rendered as markdown, which is how the tutorial
    presents them anyway.
    """
    if not src.strip():
        return False
    try:
        compile(src, "<tutorial-cell>", "exec")
    except SyntaxError:
        return False
    return True


def _src(text: str) -> list[str]:
    """nbformat source list: each line keeps its newline except the last."""
    lines = text.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


def escape_notebook_dollars(text: str) -> str:
    """Escape literal currency dollars as ``\\$`` for the notebook only.

    The tutorial keeps dollars bare — correct on GitHub's ``.md`` renderer
    (it only typesets math it tags server-side and never tags dollar prose)
    and on Substack. But GitHub renders ``.ipynb`` with a naive MathJax pass
    that pairs *any* two ``$`` on a line, so bare prose like "$50 ... $52"
    would typeset as math. A ``\\$`` in the cell source survives markdown
    processing as a literal backslash-dollar, which MathJax's processEscapes
    renders as a plain ``$`` — the one form confirmed to work in GitHub's
    notebook viewer.

    Left untouched, because there a ``$`` is already safe and a backslash
    would show literally: fenced code blocks, single-backtick inline code,
    and ``$$...$$`` display math (the SE formula in Part 5).
    """
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(_escape_inline_dollars(line))
    return "\n".join(out)


def _escape_inline_dollars(line: str) -> str:
    """Escape ``$`` -> ``\\$`` outside inline code and ``$$`` display math."""
    parts: list[str] = []
    pos = 0
    for m in _DOLLAR_SAFE_RE.finditer(line):
        parts.append(line[pos:m.start()].replace("$", "\\$"))
        parts.append(m.group(0))  # code span / display math — left as-is
        pos = m.end()
    parts.append(line[pos:].replace("$", "\\$"))
    return "".join(parts)


def build_cells(md: str) -> list[dict]:
    lines = md.split("\n")
    cells: list[dict] = [md_cell(INTRO_MD), code_cell(SETUP_CODE),
                         code_cell(DATA_PREP_CODE)]
    buf: list[str] = []
    pending_demo: str | None = None

    def flush() -> None:
        text = "\n".join(buf).strip("\n")
        if text.strip():
            cells.append(md_cell(escape_notebook_dollars(text)))
        buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            lang = line[3:].strip()
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            src = "\n".join(block)
            if lang == "python" and compiles(src):
                flush()
                cells.append(code_cell(src))
            else:
                buf.append("```" + lang)
                buf.extend(block)
                buf.append("```")
            continue

        m = IMAGE_RE.match(line.strip())
        if m and m.group(1) in FIGURE_CALLS:
            flush()
            call = FIGURE_CALLS[m.group(1)]
            cells.append(code_cell(f"# Regenerates docs/figures/{m.group(1)}\n_ = {call}"))
            i += 1
            continue

        if HEADING_RE.match(line):
            if any(b.strip() for b in buf):
                flush()
            # The just-finished section ended here — emit its demo (if any)
            # after its prose, before the new heading starts accumulating.
            if pending_demo is not None:
                cells.append(code_cell(pending_demo))
                pending_demo = None
            if line in LINKED_CODE_DEMOS:
                pending_demo = LINKED_CODE_DEMOS[line]
        buf.append(line)
        i += 1

    flush()
    if pending_demo is not None:  # demo on the final section
        cells.append(code_cell(pending_demo))
    return cells


def main() -> None:
    with open(TUTORIAL, encoding="utf-8") as f:
        md = f.read()

    notebook = {
        "cells": build_cells(md),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    print(f"Wrote {OUT}: {len(notebook['cells'])} cells ({n_code} code, {n_md} markdown)")


if __name__ == "__main__":
    main()
