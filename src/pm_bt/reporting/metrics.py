from __future__ import annotations

import math
from typing import cast

import polars as pl

# Approximate number of bars per year by timeframe, used for Sharpe annualization.
_BARS_PER_YEAR: dict[str, float] = {
    "1m": 525_600.0,
    "2m": 262_800.0,
    "5m": 105_120.0,
    "10m": 52_560.0,
    "15m": 35_040.0,
    "30m": 17_520.0,
    "1h": 8_760.0,
    "4h": 2_190.0,
    "1d": 365.0,
}


def compute_sharpe_ratio(equity_df: pl.DataFrame, bar_timeframe: str) -> float:
    """Annualized Sharpe ratio from bar-to-bar equity returns.

    Returns 0.0 when fewer than 2 bars or zero volatility.
    """
    if equity_df.height < 2:
        return 0.0

    returns = equity_df["equity"].pct_change().drop_nulls()
    if returns.len() == 0:
        return 0.0

    # Series is Float64 so std()/mean() return float|None at runtime.
    # Polars stubs expose broader unions; cast to satisfy the type checker.
    std_raw = cast(float | None, returns.std())
    std = float(std_raw) if std_raw is not None else 0.0
    if std == 0.0 or not math.isfinite(std):
        return 0.0

    mean_raw = cast(float | None, returns.mean())
    mean = float(mean_raw) if mean_raw is not None else 0.0
    if not math.isfinite(mean):
        return 0.0

    periods_per_year = _BARS_PER_YEAR.get(bar_timeframe, 525_600.0)
    return mean / std * math.sqrt(periods_per_year)


def compute_win_rate(equity_df: pl.DataFrame) -> float:
    """Fraction of bars where equity increased vs the previous bar.

    This is a pragmatic MVP approximation. A precise win rate would require
    matching buy/sell round-trips via the PnL ledger, which is not exposed
    in the current BacktestArtifacts API.
    """
    if equity_df.height < 2:
        return 0.0

    changes = equity_df["equity"].diff().drop_nulls()
    positive = changes.filter(changes > 0.0).len()
    total = changes.len()
    if total == 0:
        return 0.0
    return float(positive / total)


def compute_max_exposure_by_market(exposure_df: pl.DataFrame) -> dict[str, float]:
    """Maximum cash-at-risk per market from the exposure curve."""
    if exposure_df.is_empty():
        return {}

    grouped = exposure_df.group_by("market_id").agg(
        pl.col("cash_at_risk").max().alias("max_cash_at_risk")
    )
    return dict(
        zip(
            grouped["market_id"].to_list(),
            grouped["max_cash_at_risk"].to_list(),
            strict=True,
        )
    )
