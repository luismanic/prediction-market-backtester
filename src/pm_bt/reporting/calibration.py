"""Prediction-market calibration metrics: Brier score, log loss, ECE.

These metrics evaluate how well market prices correspond to actual outcomes
on resolved markets.  They are computed at *trade execution time* (the price
the participant paid), which is the correct methodology for measuring market
calibration.  Approaches that use near-resolution price snapshots produce
artificially low scores because markets have already converged toward 0 or 1.

Formulas
--------
- **Brier score** = mean((price - outcome)^2)   where outcome ∈ {0, 1}
  A perfectly calibrated market with trades uniformly distributed across
  prices has an expected Brier score of ~0.17.
- **Log loss** = -mean(outcome * ln(price) + (1 - outcome) * ln(1 - price))
- **ECE (Expected Calibration Error)** = weighted mean of |win_rate - price|
  across price buckets.

All functions operate on Polars DataFrames and are fully vectorized.
"""

from __future__ import annotations

import math
from typing import cast

import polars as pl


def compute_brier_score(trades: pl.DataFrame) -> float:
    """Brier score from a DataFrame with ``price`` (0–1) and ``won`` (bool) columns.

    Returns 0.0 when the input is empty.
    """
    if trades.is_empty():
        return 0.0

    result = cast(
        float | None,
        trades.select(
            ((pl.col("price") - pl.col("won").cast(pl.Float64)) ** 2).mean().alias("brier")
        ).item(),
    )

    return float(result) if result is not None and math.isfinite(result) else 0.0


def compute_log_loss(trades: pl.DataFrame, *, epsilon: float = 1e-6) -> float:
    """Log loss from a DataFrame with ``price`` (0–1) and ``won`` (bool) columns.

    Prices are clamped to [epsilon, 1 - epsilon] to avoid log(0).
    Returns 0.0 when the input is empty.
    """
    if trades.is_empty():
        return 0.0

    clamped = trades.with_columns(
        pl.col("price").clip(epsilon, 1.0 - epsilon).alias("p"),
    )

    result = cast(
        float | None,
        clamped.select(
            (
                -(
                    pl.col("won").cast(pl.Float64) * pl.col("p").log()
                    + (1.0 - pl.col("won").cast(pl.Float64)) * (1.0 - pl.col("p")).log()
                )
            )
            .mean()
            .alias("logloss")
        ).item(),
    )

    return float(result) if result is not None and math.isfinite(result) else 0.0


def compute_ece(
    trades: pl.DataFrame,
    *,
    n_bins: int = 100,
) -> float:
    """Expected Calibration Error from price-bucketed win rates.

    Bins trades by price into *n_bins* equal-width buckets (default: 100,
    matching the 1-cent resolution of prediction market contracts).
    Returns the trade-count-weighted mean of |actual_win_rate - bucket_midpoint|.

    Returns 0.0 when the input is empty.
    """
    if trades.is_empty():
        return 0.0

    bin_width = 1.0 / n_bins

    bucketed = (
        trades.with_columns(
            (pl.col("price") / bin_width).floor().cast(pl.Int32).clip(0, n_bins - 1).alias("bin"),
        )
        .group_by("bin")
        .agg(
            pl.col("won").cast(pl.Float64).mean().alias("win_rate"),
            pl.len().alias("count"),
        )
        .with_columns(
            ((pl.col("bin").cast(pl.Float64) + 0.5) * bin_width).alias("bin_midpoint"),
        )
    )

    total = cast(int | None, bucketed["count"].sum())
    if total is None or total == 0:
        return 0.0

    ece = cast(
        float | None,
        bucketed.select(
            ((pl.col("win_rate") - pl.col("bin_midpoint")).abs() * pl.col("count")).sum()
            / pl.lit(total)
        ).item(),
    )

    return float(ece) if ece is not None and math.isfinite(ece) else 0.0


def compute_calibration_metrics(trades: pl.DataFrame) -> dict[str, float]:
    """Compute all calibration metrics at once.

    Parameters
    ----------
    trades
        DataFrame with columns:

        - ``price`` (Float64, 0–1): the trade execution price
        - ``won`` (Boolean): whether the traded outcome won

    Returns
    -------
    dict with keys ``brier_score``, ``log_loss``, ``ece``, ``n_trades``.
    """
    return {
        "brier_score": compute_brier_score(trades),
        "log_loss": compute_log_loss(trades),
        "ece": compute_ece(trades),
        "n_trades": float(trades.height),
    }


def compute_calibration_metrics_from_fills_and_markets(
    fills: pl.DataFrame,
    markets: pl.DataFrame,
) -> dict[str, float]:
    """Compute calibration metrics from executed fills joined to resolved markets.

    Parameters
    ----------
    fills
        DataFrame with columns ``market_id``, ``outcome_id``, ``venue``, ``price_fill``.
    markets
        DataFrame with columns ``market_id``, ``outcome_id``, ``venue``, ``resolved``,
        ``winning_outcome``.

    Notes
    -----
    - Scoring sample is the fill execution price (``price_fill``), per project rules.
    - Currently only yes/no outcome IDs can be scored deterministically.
    """
    if fills.is_empty() or markets.is_empty():
        return {
            "brier_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "n_trades": 0.0,
        }

    required_fill_cols = {"market_id", "outcome_id", "venue", "price_fill"}
    required_market_cols = {"market_id", "outcome_id", "venue", "resolved", "winning_outcome"}
    if not required_fill_cols.issubset(set(fills.columns)):
        return {
            "brier_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "n_trades": 0.0,
        }
    if not required_market_cols.issubset(set(markets.columns)):
        return {
            "brier_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "n_trades": 0.0,
        }

    resolved_markets = markets.filter(
        pl.col("resolved").cast(pl.Boolean, strict=False).fill_null(False)
        & pl.col("winning_outcome").cast(pl.Utf8, strict=False).is_in(["yes", "no"])
    ).select(["market_id", "outcome_id", "venue", "winning_outcome"])

    if resolved_markets.is_empty():
        return {
            "brier_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "n_trades": 0.0,
        }

    scored = (
        fills.select(["market_id", "outcome_id", "venue", "price_fill"])
        .join(
            resolved_markets,
            on=["market_id", "outcome_id", "venue"],
            how="inner",
        )
        .with_columns(
            pl.col("outcome_id").cast(pl.Utf8, strict=False).alias("outcome_id_str"),
            pl.col("winning_outcome").cast(pl.Utf8, strict=False).alias("winning_outcome_str"),
            pl.col("price_fill").cast(pl.Float64, strict=False).alias("price"),
        )
        .with_columns(
            pl.when(
                pl.col("outcome_id_str").is_in(["yes", "no"])
                & pl.col("winning_outcome_str").is_in(["yes", "no"])
            )
            .then(pl.col("outcome_id_str") == pl.col("winning_outcome_str"))
            .otherwise(pl.lit(None, dtype=pl.Boolean))
            .alias("won")
        )
        .select(["price", "won"])
        .drop_nulls(["price", "won"])
    )

    if scored.is_empty():
        return {
            "brier_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "n_trades": 0.0,
        }

    return compute_calibration_metrics(scored)
