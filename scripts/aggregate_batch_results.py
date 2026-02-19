"""
aggregate_batch_results.py

Reads all individual equity.csv files from a batch run and produces:
  1. Combined equity curve (sum of all markets)
  2. Combined drawdown
  3. Combined returns distribution
  4. Summary stats table

Usage:
    uv run python scripts/aggregate_batch_results.py output/mean_reversion_batch
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: uv run python scripts/aggregate_batch_results.py <batch_output_dir>")
    print("Example: uv run python scripts/aggregate_batch_results.py output/mean_reversion_batch")
    sys.exit(1)

BATCH_DIR = Path(sys.argv[1])

# Find the batch subfolder (batch_YYYYMMDD_HHMMSS_xxxxx)
batch_subdirs = [d for d in BATCH_DIR.iterdir() if d.is_dir() and d.name.startswith("batch_")]
if not batch_subdirs:
    # Maybe the path IS the batch subdir already
    batch_subdirs = [BATCH_DIR]

BATCH_SUBDIR = sorted(batch_subdirs)[-1]  # most recent
RUNS_DIR = BATCH_SUBDIR / "runs"
SUMMARY_CSV = BATCH_SUBDIR / "summary.csv"
OUTPUT_DIR = BATCH_SUBDIR / "aggregate"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Reading batch: {BATCH_SUBDIR.name}")

# ---------------------------------------------------------------------------
# Load summary.csv
# ---------------------------------------------------------------------------
if not SUMMARY_CSV.exists():
    print(f"ERROR: summary.csv not found at {SUMMARY_CSV}")
    sys.exit(1)

summary = pd.read_csv(SUMMARY_CSV)
print(f"Found {len(summary)} completed markets in summary.csv")

# ---------------------------------------------------------------------------
# Load all equity curves
# ---------------------------------------------------------------------------
equity_files = list(RUNS_DIR.glob("*/equity.csv")) if RUNS_DIR.exists() else []
print(f"Found {len(equity_files)} equity.csv files")

equity_frames = []
for f in equity_files:
    try:
        df = pd.read_csv(f, parse_dates=["ts"])
        df = df.set_index("ts").sort_index()
        equity_frames.append(df["equity"])
    except Exception as e:
        print(f"  Skipping {f.name}: {e}")

# ---------------------------------------------------------------------------
# Aggregate equity curve
# ---------------------------------------------------------------------------
if equity_frames:
    # Align all series to a common time index and sum
    combined = pd.concat(equity_frames, axis=1)

    # Sum equity across all markets (forward-fill gaps)
    combined_ffill = combined.ffill()
    portfolio_equity = combined_ffill.sum(axis=1)

    # Normalize to start at 10,000 for easy reading
    initial = portfolio_equity.iloc[0]
    portfolio_equity_norm = (portfolio_equity / initial) * 10_000

    # Drawdown
    rolling_max = portfolio_equity_norm.cummax()
    drawdown = (portfolio_equity_norm - rolling_max) / rolling_max

    # Returns
    returns = portfolio_equity_norm.pct_change().dropna()

    # ---------------------------------------------------------------------------
    # Plot 1: Combined Equity Curve
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(portfolio_equity_norm.index, portfolio_equity_norm.values, color="#2196F3", linewidth=1.5)
    ax.set_title(f"Combined Equity Curve ({len(equity_frames)} markets)", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_equity_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ---------------------------------------------------------------------------
    # Plot 2: Combined Drawdown
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color="#F44336", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values * 100, color="#F44336", linewidth=0.8)
    ax.set_title(f"Combined Drawdown ({len(equity_frames)} markets)", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_drawdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ---------------------------------------------------------------------------
    # Plot 3: Combined Returns Distribution
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(returns, bins=100, color="#42A5F5", edgecolor="white", linewidth=0.3)
    ax.axvline(returns.mean(), color="navy", linewidth=1.5, label=f"Mean: {returns.mean():.6f}")
    ax.axvline(returns.median(), color="green", linewidth=1.5, label=f"Median: {returns.median():.6f}")
    ax.set_title(f"Combined Returns Distribution ({len(equity_frames)} markets)", fontsize=14)
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "combined_returns_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Summary stats from summary.csv
# ---------------------------------------------------------------------------
print("\n=== AGGREGATE STRATEGY STATS ===")
numeric_cols = ["total_pnl", "realized_pnl", "win_rate", "sharpe_ratio", "max_drawdown", "fills_count"]
available = [c for c in numeric_cols if c in summary.columns]

stats = summary[available].agg(["mean", "median", "sum", "std"]).T
stats.columns = ["Mean", "Median", "Total", "Std Dev"]
print(stats.to_string())

print(f"\nTotal markets completed : {len(summary)}")
if "total_pnl" in summary.columns:
    profitable = (summary["total_pnl"] > 0).sum()
    print(f"Profitable markets      : {profitable} / {len(summary)} ({profitable/len(summary)*100:.1f}%)")
    print(f"Total PnL across all    : ${summary['total_pnl'].sum():.2f}")
    print(f"Avg PnL per market      : ${summary['total_pnl'].mean():.2f}")

print(f"\nAggregate charts saved to: {OUTPUT_DIR}/")