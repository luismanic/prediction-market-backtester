# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportAny=false

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_equity_curve(equity_df: pl.DataFrame, output_path: Path) -> None:
    """Save an equity curve plot (equity + cash) to *output_path*."""
    ts = equity_df["ts"].to_list()
    equity = equity_df["equity"].to_numpy()
    cash = equity_df["cash"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, equity, label="Equity", linewidth=1.2)
    ax.plot(ts, cash, label="Cash", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_drawdown(equity_df: pl.DataFrame, output_path: Path) -> None:
    """Save a drawdown curve plot to *output_path*."""
    equity = equity_df["equity"].to_numpy()
    peak = np.maximum.accumulate(equity)
    # Guard against division by zero when peak is 0.
    safe_peak = np.where(peak > 0, peak, 1.0)
    drawdown = (peak - equity) / safe_peak

    ts = equity_df["ts"].to_list()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(ts, 0, drawdown, color="salmon", alpha=0.6)  # type: ignore[arg-type]
    ax.plot(ts, drawdown, color="firebrick", linewidth=0.8)
    ax.set_title("Drawdown")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (%)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_returns_distribution(equity_df: pl.DataFrame, output_path: Path) -> None:
    """Save a histogram of bar-to-bar equity returns to *output_path*."""
    if equity_df.height < 2:
        # Not enough data for a meaningful histogram.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Returns Distribution (insufficient data)")
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return

    returns = equity_df["equity"].pct_change().drop_nulls().to_numpy()
    returns = returns[np.isfinite(returns)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(returns, bins=min(50, max(10, len(returns) // 5)), edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    if len(returns) > 0:
        ax.axvline(float(np.mean(returns)), color="blue", linewidth=0.8, label="mean")
        ax.axvline(float(np.median(returns)), color="green", linewidth=0.8, label="median")
        ax.legend()
    ax.set_title("Bar-to-Bar Returns Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
