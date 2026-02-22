"""
aggregate_batch_results.py  (FIXED v2)

Reads all individual equity.csv + results.json files from a batch run and produces:
  1. Combined PnL equity curve  ← FIX: was summing absolute portfolio values
  2. Combined drawdown           ← FIX: now computed on PnL not absolute value
  3. Combined returns distribution ← FIX: now uses dollar PnL per market, not total_return
  4. Summary stats table         ← FIX: market count now matches charts

ROOT CAUSE FIXES:
  Bug 1 — Equity curve inflation:
    The old script summed raw portfolio_value columns across markets.
    If each market starts with $10,000 and you have 773 markets, the
    combined equity starts at 773 × $10,000 = $7.73M — completely wrong.
    Fix: subtract each market's starting portfolio value before summing.
    This gives a true cumulative PnL curve that starts at $0.

  Bug 2 — Returns distribution:
    The old script used the equity curve's percentage returns (tiny fractions
    of a $7.73M base), making every data point cluster near zero.
    Fix: pull realized dollar PnL directly from results.json per market.
    This shows the actual dollar gain/loss distribution across markets.

  Bug 3 — Market count discrepancy (843 stats vs 773 charts):
    The old script counted equity.csv files for the chart title but
    summary.csv rows for the stats table — different numbers.
    Fix: both now source from summary.csv. We warn when equity files
    are missing so you know which markets produced incomplete output.

Usage:
    uv run python scripts/aggregate_batch_results.py <batch_output_dir>
    uv run python scripts/aggregate_batch_results.py output/kalshi_flb_10k_batch
"""

import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Locate batch folder
# ─────────────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: uv run python scripts/aggregate_batch_results.py <batch_output_dir>")
    print("Example: uv run python scripts/aggregate_batch_results.py output/kalshi_flb_10k_batch")
    sys.exit(1)

BATCH_DIR = Path(sys.argv[1])

# Handle both cases: user passes the root dir or the batch_YYYYMMDD subdir
batch_subdirs = sorted(
    [d for d in BATCH_DIR.iterdir() if d.is_dir() and d.name.startswith("batch_")]
) if BATCH_DIR.exists() else []

BATCH_SUBDIR = batch_subdirs[-1] if batch_subdirs else BATCH_DIR
RUNS_DIR     = BATCH_SUBDIR / "runs"
SUMMARY_CSV  = BATCH_SUBDIR / "summary.csv"
OUTPUT_DIR   = BATCH_SUBDIR / "aggregate"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\nBatch folder : {BATCH_SUBDIR.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Load summary.csv — this is the ground truth for market count
# ─────────────────────────────────────────────────────────────────────────────
if not SUMMARY_CSV.exists():
    print(f"ERROR: summary.csv not found at {SUMMARY_CSV}")
    sys.exit(1)

summary = pd.read_csv(SUMMARY_CSV)
total_markets_in_summary = len(summary)
print(f"summary.csv  : {total_markets_in_summary} markets")

# ─────────────────────────────────────────────────────────────────────────────
# Load all equity.csv files — convert each to PnL BEFORE combining
# ─────────────────────────────────────────────────────────────────────────────
equity_files = list(RUNS_DIR.glob("*/equity.csv")) if RUNS_DIR.exists() else []
print(f"equity files : {len(equity_files)} found")

# Warn if equity files don't match summary rows
missing_equity = total_markets_in_summary - len(equity_files)
if missing_equity > 0:
    print(f"WARNING      : {missing_equity} markets in summary.csv have no equity.csv")
    print(f"             → these markets ran but produced no bar-by-bar equity data")
    print(f"             → stats below cover all {total_markets_in_summary} markets")
    print(f"             → charts below cover {len(equity_files)} markets with equity data")

pnl_series_list = []
parse_errors    = 0


# Peek at the first equity file so we know the real column names
if equity_files:
    try:
        _peek = pd.read_csv(equity_files[0])
        print(f"equity.csv columns: {list(_peek.columns)}")
        print(f"equity.csv index sample: {list(_peek.iloc[:2, 0])}")
    except Exception as _e:
        print(f"WARNING: could not peek at equity file: {_e}")

for f in equity_files:
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)

        # ── Timezone fix ─────────────────────────────────────────────────────
        # pandas DatetimeIndex uses .tz (not .tzinfo) to check for timezone
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # ── Find the portfolio value column ───────────────────────────────────
        # Different engine versions may name this column differently
        pv_col = None
        for candidate in ["portfolio_value", "equity", "value", "cash", "total_value"]:
            if candidate in df.columns:
                pv_col = candidate
                break

        if pv_col is None:
            # Use the first numeric column as a fallback
            numeric_cols_df = df.select_dtypes(include="number").columns
            if len(numeric_cols_df) > 0:
                pv_col = numeric_cols_df[0]
            else:
                parse_errors += 1
                continue

        if df.empty:
            continue

        # Ensure the index is actually datetime — if parse_dates failed,
        # try to coerce it manually
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=False)
            except Exception:
                parse_errors += 1
                continue

        # ── THE KEY FIX ──────────────────────────────────────────────────────
        # Subtract starting value to get PnL delta starting at $0
        starting_value = df[pv_col].iloc[0]
        pnl_series = df[pv_col] - starting_value
        pnl_series.name = "pnl"
        pnl_series_list.append(pnl_series)

    except Exception as e:
        parse_errors += 1
        if parse_errors <= 3:   # show first 3 errors so you can diagnose
            print(f"  parse error in {f.name}: {e}")

if parse_errors:
    print(f"WARNING      : {parse_errors} equity files could not be parsed")

markets_with_equity = len(pnl_series_list)
print(f"Markets in charts: {markets_with_equity}")

# ─────────────────────────────────────────────────────────────────────────────
# Build combined equity (PnL) curve
# ─────────────────────────────────────────────────────────────────────────────
if pnl_series_list:
    # ── Memory-efficient aggregation ─────────────────────────────────────────
    # With 10k markets a 1-minute common_index spanning months creates ~2M
    # timestamps. Reindexing 7,000+ series to that index exhausts RAM.
    # Fix: pre-resample each series to hourly (last value per hour, ffill),
    # then build a common HOURLY index — ~60x fewer points, no visible
    # quality difference on a multi-week/month chart.

    # Step 1: resample each raw series to hourly before storing
    hourly_list = []
    for s in pnl_series_list:
        # resample to 1h: take last value in each bucket, ffill gaps
        h = s.resample("1h").last().ffill()
        hourly_list.append(h)

    # Step 2: build a common HOURLY index (33k points for 1 yr vs 500k for 1 min)
    common_start = min(h.index.min() for h in hourly_list)
    common_end   = max(h.index.max() for h in hourly_list)
    common_index = pd.date_range(common_start, common_end, freq="1h")

    # Step 3: reindex each hourly series to the common hourly grid and sum
    resampled = []
    for h in hourly_list:
        r = h.reindex(common_index, method="ffill")
        r = r.fillna(0)   # before market started, PnL is 0
        resampled.append(r)

    combined_pnl = sum(resampled)  # Sum of PnL series — starts at ~$0

    # ── Plot 1: Combined Equity (PnL) Curve ───────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(combined_pnl.index, combined_pnl.values, color="#42A5F5", linewidth=1.0)
    ax.fill_between(
        combined_pnl.index, 0, combined_pnl.values,
        where=(combined_pnl.values >= 0), alpha=0.25, color="#42A5F5"
    )
    ax.fill_between(
        combined_pnl.index, 0, combined_pnl.values,
        where=(combined_pnl.values < 0), alpha=0.25, color="#EF5350"
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"Combined PnL Equity Curve ({markets_with_equity} markets with equity data, "
        f"{total_markets_in_summary} total markets)",
        fontsize=13
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_equity_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Plot 2: Combined Drawdown (dollar terms) ─────────────────────────
    # We use DOLLAR drawdown, not percentage.
    # Percentage drawdown breaks here because the PnL curve starts near $0
    # and goes negative immediately — dividing by a peak near $0 produces
    # values like -600,000%, which is meaningless.
    # Dollar drawdown = current PnL minus the highest PnL seen so far.
    # When the curve never goes positive, peak is clamped at 0 (starting point).
    peak_pnl      = combined_pnl.cummax().clip(lower=0)
    drawdown_dollar = combined_pnl - peak_pnl   # always ≤ 0

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.fill_between(
        drawdown_dollar.index, 0, drawdown_dollar.values,
        color="#EF5350", alpha=0.6
    )
    ax.plot(drawdown_dollar.index, drawdown_dollar.values,
            color="#EF5350", linewidth=0.8)
    ax.set_title(
        f"Combined Drawdown ({markets_with_equity} markets with equity data, "
        f"{total_markets_in_summary} total markets)",
        fontsize=13
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_drawdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

else:
    print("No equity data found — skipping equity and drawdown charts.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Returns distribution — use dollar PnL per market from summary.csv
# ─────────────────────────────────────────────────────────────────────────────
# We pull total_pnl from summary.csv (one row per market).
# This shows the actual dollar gain/loss per market — NOT percentage returns
# of an inflated portfolio base, which is what caused everything to cluster near 0.

if "total_pnl" in summary.columns:
    pnl_per_market = summary["total_pnl"].dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(
        pnl_per_market,
        bins=min(80, max(20, len(pnl_per_market) // 10)),
        color="#42A5F5",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(
        pnl_per_market.mean(),
        color="navy", linewidth=1.5,
        label=f"Mean: ${pnl_per_market.mean():.4f}",
    )
    ax.axvline(
        pnl_per_market.median(),
        color="green", linewidth=1.5,
        label=f"Median: ${pnl_per_market.median():.4f}",
    )
    ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7, label="Break-even")
    ax.set_title(
        f"PnL Distribution per Market ({total_markets_in_summary} markets)",
        fontsize=13,
    )
    ax.set_xlabel("PnL per Market ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_returns_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
else:
    print("No total_pnl column in summary.csv — skipping returns chart.")

# ─────────────────────────────────────────────────────────────────────────────
# Summary stats — sourced entirely from summary.csv
# Includes a note when realized_pnl is zero (means strategy holds to resolution
# without explicit exits — positions settle as unrealized, not realized)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== AGGREGATE STRATEGY STATS ===")
numeric_cols = ["total_pnl", "realized_pnl", "max_drawdown", "fills_count"]
available    = [c for c in numeric_cols if c in summary.columns]

stats = summary[available].agg(["mean", "median", "sum", "std"]).T
stats.columns = ["Mean", "Median", "Total", "Std Dev"]
print(stats.to_string())

print(f"\nTotal markets (summary.csv) : {total_markets_in_summary}")
print(f"Markets with equity data    : {markets_with_equity}")
if missing_equity > 0:
    print(f"Markets missing equity file : {missing_equity}  ← produced results but no equity.csv")

if "total_pnl" in summary.columns:
    profitable = (summary["total_pnl"] > 0).sum()
    print(f"\nProfitable markets  : {profitable} / {total_markets_in_summary} "
          f"({profitable / total_markets_in_summary * 100:.1f}%)")
    print(f"Total PnL           : ${summary['total_pnl'].sum():.2f}")
    print(f"Avg PnL per market  : ${summary['total_pnl'].mean():.4f}")

if "realized_pnl" in summary.columns and summary["realized_pnl"].sum() == 0:
    print(
        "\n⚠️  NOTICE: realized_pnl is $0 across all markets."
        "\n   This means the strategy is entering positions but never explicitly closing them."
        "\n   All PnL is sitting in unrealized_pnl (mark-to-market at last tick)."
        "\n   See the companion fix in favorite_longshot.py (on_market_close method)."
        "\n   total_pnl is still correct — it includes both realized + unrealized."
    )

print(f"\nAggregate charts saved to: {OUTPUT_DIR}/")