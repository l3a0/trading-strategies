"""Figure for docs/gld-gdx-cointegration-lessons.md section 6 -- the GLD/GDX regime map.

A rolling-window CADF scan over the full as-traded (raw) GLD/GDX history: a
one-year window stepped monthly, each window's cointegrating augmented
Dickey-Fuller t-statistic drawn as a time series. Where the statistic dips below
the 10% critical value the pair cointegrates in that window; elsewhere it does
not. The lower panel tracks Chan's through-origin hedge, which climbs from his
~1.64 to well above 4 as the miners detach from gold -- the essay's "the
relationship moved" point, as a picture.

Reuses the reproduction engine (search/pair_cointegration.py) on the COMMITTED
GLD/GDX price CSVs, so it needs no release-sized data and runs anywhere the
reproduction tests run. The rolling scan is pinned by ``TestRollingRegime`` in
tests/test_pair_cointegration.py, so a re-download that shifts the vintage moves
the figure and the pins together. Regenerate after any such change:

    python -m search.make_regime_figure

Colours match the essay's palette (warm cream ground, brass hedge line, green
cointegrating bands, red critical lines), so the PNG reads as part of the essay
on both its light and dark surfaces.
"""

from __future__ import annotations

import matplotlib.dates as mdates
from matplotlib.figure import Figure

from common.paths import FIGURES_DIR
from common.timeseries import EG_CRIT_N2
from search.pair_cointegration import aligned_closes, rolling_cointegration

# Essay palette -- docs/gld-gdx-cointegration-lessons.html :root tokens.
SURFACE = "#FEFDFA"  # figure ground
GROUND = "#F7F5EF"  # axes ground
INK = "#23201A"
MUTED = "#6B6353"
RULE = "#CFC8B4"
ACCENT = "#8B651A"  # brass -- the hedge line (WCAG-AA against the cream ground)
GOOD = "#4A7C59"  # green -- cointegrating windows
LOST = "#A24C3C"  # red -- critical lines / breakdown

REPRO_HEDGE = 1.6379  # the reproduced Chan through-origin hedge the essay quotes


def _style(ax) -> None:
    ax.set_facecolor(GROUND)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=RULE, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def make_regime_figure() -> Figure:
    """Build the two-panel GLD/GDX cointegration regime map."""
    df = aligned_closes("GLD", "GDX", unadjusted=True)
    a = df["GLD"].to_numpy(float)
    b = df["GDX"].to_numpy(float)
    scan = rolling_cointegration(a, b, window=252, step=21, lags=1)
    x = df.index[scan.end_idx]
    adf = scan.adf_stat
    hedge = scan.origin_hedge

    crit10 = EG_CRIT_N2["10%"]  # -3.04
    crit5 = EG_CRIT_N2["5%"]  # -3.34

    fig = Figure(figsize=(11, 7), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.18)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _style(ax1)
    _style(ax2)

    # --- Panel A: rolling CADF t-statistic ---
    coint = adf < crit10
    labelled = False
    i = 0
    while i < len(coint):
        if coint[i]:
            j = i
            while j + 1 < len(coint) and coint[j + 1]:
                j += 1
            ax1.axvspan(
                x[i], x[min(j + 1, len(x) - 1)],
                color=GOOD, alpha=0.16,
                label="cointegrates (10%)" if not labelled else None,
            )
            labelled = True
            i = j + 1
        else:
            i += 1

    ax1.plot(x, adf, color=INK, lw=1.6, zorder=5)
    ax1.axhline(crit10, color=LOST, lw=1.1, ls="--", zorder=4)
    ax1.axhline(crit5, color=LOST, lw=1.1, ls=":", zorder=4)
    # Critical-line labels out in the right margin, at each line's level, so
    # they never cross the data.
    yt = ax1.get_yaxis_transform()
    ax1.text(1.012, crit10 + 0.04, "10% critical  −3.04", color=LOST, transform=yt,
             fontsize=8.5, va="bottom", ha="left")
    ax1.text(1.012, crit5 - 0.04, "5% critical  −3.34", color=LOST, transform=yt,
             fontsize=8.5, va="top", ha="left")
    ax1.tick_params(labelbottom=False)  # shared x-axis -- labels on the lower panel
    ax1.annotate(
        "Chan's 2006–07 window\n(t ≈ −3.18)",
        xy=(x[0], adf[0]), xytext=(x[6], -4.15),
        color=INK, fontsize=8.5, ha="left", va="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    ax1.set_ylabel("rolling CADF t-statistic\n(1-year window)", color=INK, fontsize=10)
    ax1.set_title(
        "GLD / GDX cointegration is a property of the window, not the pair",
        color=INK, fontsize=13.5, loc="left", pad=12, fontweight="bold",
    )
    # Legend out in the right margin so it never sits over the data.
    ax1.legend(loc="upper left", bbox_to_anchor=(1.008, 1.0), frameon=False,
               fontsize=9, labelcolor=INK)

    # --- Panel B: the through-origin hedge drift ---
    ax2.plot(x, hedge, color=ACCENT, lw=1.6)
    ax2.axhline(REPRO_HEDGE, color=MUTED, lw=1.0, ls=":")
    ax2.text(x[-1], REPRO_HEDGE, "  reproduced Chan hedge 1.64", color=MUTED,
             fontsize=8.5, va="center", ha="left")
    ax2.set_ylabel("through-origin\nhedge ratio", color=INK, fontsize=10)
    ax2.set_xlabel("window-end date", color=INK, fontsize=10)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.margins(x=0.01)

    path = FIGURES_DIR / "reproduction_regime_map.png"
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return fig


def main() -> None:
    make_regime_figure()
    print(f"wrote {FIGURES_DIR / 'reproduction_regime_map.png'}")


if __name__ == "__main__":
    main()
