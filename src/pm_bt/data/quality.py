from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast

import polars as pl


@dataclass(slots=True)
class MarketDataQuality:
    venue: str
    market_id: str
    trade_count: int
    ts_null_rate: float
    price_null_rate: float
    size_null_rate: float
    price_min: float | None
    price_max: float | None
    non_finite_price_count: int
    duplicate_ts_count: int
    out_of_order_count: int
    largest_gap_minutes: float
    passes: bool
    failed_checks: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "venue": self.venue,
            "market_id": self.market_id,
            "trade_count": self.trade_count,
            "ts_null_rate": self.ts_null_rate,
            "price_null_rate": self.price_null_rate,
            "size_null_rate": self.size_null_rate,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "non_finite_price_count": self.non_finite_price_count,
            "duplicate_ts_count": self.duplicate_ts_count,
            "out_of_order_count": self.out_of_order_count,
            "largest_gap_minutes": self.largest_gap_minutes,
            "passes": self.passes,
            "failed_checks": self.failed_checks,
        }


def _float_or_zero(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def evaluate_market_trade_quality(
    trades: pl.DataFrame,
    *,
    venue: str,
    market_id: str,
    min_trade_count: int,
    max_null_rate: float,
    max_gap_minutes: float,
) -> MarketDataQuality:
    """Evaluate data quality checks on canonical trade rows for one market."""
    nulls = trades.select(
        pl.col("ts").is_null().mean().alias("ts_null_rate"),
        pl.col("price").is_null().mean().alias("price_null_rate"),
        pl.col("size").is_null().mean().alias("size_null_rate"),
    )
    ts_null_rate = _float_or_zero(cast(float | None, nulls["ts_null_rate"][0]))
    price_null_rate = _float_or_zero(cast(float | None, nulls["price_null_rate"][0]))
    size_null_rate = _float_or_zero(cast(float | None, nulls["size_null_rate"][0]))

    price_stats = trades.select(
        pl.col("price").min().alias("price_min"),
        pl.col("price").max().alias("price_max"),
        (~pl.col("price").cast(pl.Float64, strict=False).is_finite().fill_null(False))
        .sum()
        .alias("non_finite_price_count"),
    )
    price_min = cast(float | None, price_stats["price_min"][0])
    price_max = cast(float | None, price_stats["price_max"][0])
    non_finite_price_count = int(cast(int | None, price_stats["non_finite_price_count"][0]) or 0)

    ts_series = (
        trades.select(pl.col("ts").cast(pl.Datetime(time_zone="UTC"), strict=False).alias("ts"))
        .drop_nulls(["ts"])
        .get_column("ts")
    )
    duplicate_ts_count = int(ts_series.is_duplicated().sum() or 0)

    out_of_order_count = int(
        cast(
            int | None,
            trades.select(
                (
                    pl.col("ts")
                    .cast(pl.Datetime(time_zone="UTC"), strict=False)
                    .diff()
                    .dt.total_milliseconds()
                    .lt(0)
                    .fill_null(False)
                )
                .sum()
                .alias("out_of_order")
            )["out_of_order"][0],
        )
        or 0
    )

    ts_sorted = trades.select(
        pl.col("ts").cast(pl.Datetime(time_zone="UTC"), strict=False).alias("ts")
    ).drop_nulls(["ts"])
    largest_gap_minutes = 0.0
    if ts_sorted.height >= 2:
        largest_gap_minutes = _float_or_zero(
            cast(
                float | None,
                ts_sorted.sort("ts").select(
                    (pl.col("ts").diff().dt.total_seconds() / 60.0).max().alias("max_gap")
                )["max_gap"][0],
            )
        )

    trade_count = trades.height
    failed_checks: list[str] = []
    if trade_count < min_trade_count:
        failed_checks.append("trade_count_below_threshold")
    if ts_null_rate > max_null_rate:
        failed_checks.append("ts_null_rate_above_threshold")
    if price_null_rate > max_null_rate:
        failed_checks.append("price_null_rate_above_threshold")
    if size_null_rate > max_null_rate:
        failed_checks.append("size_null_rate_above_threshold")
    if non_finite_price_count > 0:
        failed_checks.append("non_finite_price_values")
    if price_min is not None and price_min < 0.0:
        failed_checks.append("price_min_below_zero")
    if price_max is not None and price_max > 1.0:
        failed_checks.append("price_max_above_one")
    if out_of_order_count > 0:
        failed_checks.append("timestamps_out_of_order")
    if largest_gap_minutes > max_gap_minutes:
        failed_checks.append("timestamp_gap_too_large")

    return MarketDataQuality(
        venue=venue,
        market_id=market_id,
        trade_count=trade_count,
        ts_null_rate=ts_null_rate,
        price_null_rate=price_null_rate,
        size_null_rate=size_null_rate,
        price_min=price_min,
        price_max=price_max,
        non_finite_price_count=non_finite_price_count,
        duplicate_ts_count=duplicate_ts_count,
        out_of_order_count=out_of_order_count,
        largest_gap_minutes=largest_gap_minutes,
        passes=len(failed_checks) == 0,
        failed_checks=failed_checks,
    )


def compute_tradability_metrics(trades: pl.DataFrame) -> dict[str, float]:
    """Compute simple tradability proxies from canonical trade rows."""
    if trades.is_empty():
        return {
            "trade_count": 0.0,
            "volume_total": 0.0,
            "avg_trade_size": 0.0,
            "trade_frequency_per_hour": 0.0,
            "spread_proxy": 0.0,
            "volatility_proxy": 0.0,
            "slippage_proxy_bps": 0.0,
        }

    trade_count = float(trades.height)
    volume_total = float(cast(float | None, trades.select(pl.col("size").sum()).item()) or 0.0)
    avg_trade_size = volume_total / trade_count if trade_count > 0 else 0.0

    ts_stats = trades.select(
        pl.col("ts").cast(pl.Datetime(time_zone="UTC"), strict=False).min().alias("ts_min"),
        pl.col("ts").cast(pl.Datetime(time_zone="UTC"), strict=False).max().alias("ts_max"),
    )
    ts_min = cast(datetime | None, ts_stats["ts_min"][0])
    ts_max = cast(datetime | None, ts_stats["ts_max"][0])
    duration_hours = 0.0
    if ts_min is not None and ts_max is not None:
        duration_hours = max((ts_max - ts_min).total_seconds() / 3600.0, 0.0)
    trade_frequency_per_hour = trade_count / duration_hours if duration_hours > 0 else trade_count

    spread_proxy = float(
        cast(
            float | None,
            trades.select(
                pl.col("price")
                .cast(pl.Float64, strict=False)
                .diff()
                .abs()
                .median()
                .alias("spread_proxy")
            ).item(),
        )
        or 0.0
    )
    volatility_proxy = float(
        cast(
            float | None,
            trades.select(
                pl.col("price").cast(pl.Float64, strict=False).pct_change().std().alias("vol_proxy")
            ).item(),
        )
        or 0.0
    )

    return {
        "trade_count": trade_count,
        "volume_total": volume_total,
        "avg_trade_size": avg_trade_size,
        "trade_frequency_per_hour": float(trade_frequency_per_hour),
        "spread_proxy": spread_proxy,
        "volatility_proxy": volatility_proxy,
        "slippage_proxy_bps": spread_proxy * 10_000.0,
    }
