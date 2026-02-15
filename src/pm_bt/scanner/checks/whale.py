from __future__ import annotations

import logging
from datetime import datetime
from typing import cast

import polars as pl

from pm_bt.common.types import AlertSeverity, Venue
from pm_bt.scanner.models import Alert, make_alert_id

logger = logging.getLogger(__name__)


def _as_str(value: object, *, field: str) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected str for {field}, got {type(value)!r}")


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected float for {field}, got {type(value)!r}")


def _as_datetime(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"Expected datetime for {field}, got {type(value)!r}")


def check_whale_trades(
    trades: pl.LazyFrame,
    *,
    rolling_window: str,
    size_multiplier: float,
) -> list[Alert]:
    """Detect trades whose size exceeds *size_multiplier* × rolling average.

    The rolling baseline is computed per ``(market_id, venue)`` over
    *rolling_window* (a Polars duration string such as ``"1h"``).
    """
    sorted_trades = trades.sort(["market_id", "venue", "ts"])

    with_rolling = sorted_trades.with_columns(
        pl.col("size")
        .rolling_mean_by("ts", window_size=rolling_window, closed="left")
        .over(["market_id", "venue"])
        .alias("rolling_avg_size"),
    )

    whales = (
        with_rolling.filter(
            pl.col("rolling_avg_size").is_not_null()
            & (pl.col("rolling_avg_size") > 0)
            & (pl.col("size") > size_multiplier * pl.col("rolling_avg_size"))
        )
        .with_columns(
            (pl.col("size") / pl.col("rolling_avg_size")).alias("size_ratio"),
        )
        .collect()
    )

    alerts: list[Alert] = []
    market_ids = cast(list[object], whales.get_column("market_id").to_list())
    venues = cast(list[object], whales.get_column("venue").to_list())
    tss = cast(list[object], whales.get_column("ts").to_list())
    sizes = cast(list[object], whales.get_column("size").to_list())
    rolling_avgs = cast(list[object], whales.get_column("rolling_avg_size").to_list())
    ratios = cast(list[object], whales.get_column("size_ratio").to_list())
    prices = cast(list[object], whales.get_column("price").to_list())

    for idx in range(whales.height):
        market_id = _as_str(market_ids[idx], field="market_id")
        venue_raw = _as_str(venues[idx], field="venue")
        ts = _as_datetime(tss[idx], field="ts")
        trade_size = _as_float(sizes[idx], field="size")
        rolling_avg = _as_float(rolling_avgs[idx], field="rolling_avg_size")
        ratio = _as_float(ratios[idx], field="size_ratio")
        price = _as_float(prices[idx], field="price")
        if ratio > 10.0:
            severity = AlertSeverity.HIGH
        elif ratio > 5.0:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        venue = Venue(venue_raw)
        alerts.append(
            Alert(
                alert_id=make_alert_id("whale_trade", market_id, ts),
                market_id=market_id,
                ts=ts,
                venue=venue,
                reason="whale_trade",
                severity=severity,
                supporting_stats={
                    "trade_size": trade_size,
                    "rolling_avg_size": rolling_avg,
                    "size_ratio": ratio,
                    "price": price,
                },
            )
        )
    return alerts


def check_price_impact(
    trades: pl.LazyFrame,
    *,
    impact_threshold: float,
) -> list[Alert]:
    """Detect trades with abnormally high price impact per unit size.

    ``impact_score = |price - prev_price| / size``

    Only consecutive trades within the same ``(market_id, venue)`` are
    compared.  The first trade in each group is skipped (no previous price).
    """
    sorted_trades = trades.sort(["market_id", "venue", "ts"])

    with_impact = sorted_trades.with_columns(
        pl.col("price").diff().over(["market_id", "venue"]).alias("price_diff"),
    ).with_columns(
        (pl.col("price_diff").abs() / pl.col("size")).alias("impact_score"),
    )

    impacts = with_impact.filter(
        pl.col("impact_score").is_not_null()
        & pl.col("impact_score").is_finite()
        & (pl.col("impact_score") > impact_threshold)
        & (pl.col("size") > 0)
    ).collect()

    alerts: list[Alert] = []
    market_ids = cast(list[object], impacts.get_column("market_id").to_list())
    venues = cast(list[object], impacts.get_column("venue").to_list())
    tss = cast(list[object], impacts.get_column("ts").to_list())
    impact_scores = cast(list[object], impacts.get_column("impact_score").to_list())
    price_diffs = cast(list[object], impacts.get_column("price_diff").to_list())
    sizes = cast(list[object], impacts.get_column("size").to_list())
    prices = cast(list[object], impacts.get_column("price").to_list())

    for idx in range(impacts.height):
        market_id = _as_str(market_ids[idx], field="market_id")
        venue_raw = _as_str(venues[idx], field="venue")
        ts = _as_datetime(tss[idx], field="ts")
        score = _as_float(impact_scores[idx], field="impact_score")
        price_diff = _as_float(price_diffs[idx], field="price_diff")
        size = _as_float(sizes[idx], field="size")
        price = _as_float(prices[idx], field="price")
        severity = AlertSeverity.HIGH if score > 3 * impact_threshold else AlertSeverity.MEDIUM

        venue = Venue(venue_raw)
        alerts.append(
            Alert(
                alert_id=make_alert_id("price_impact", market_id, ts),
                market_id=market_id,
                ts=ts,
                venue=venue,
                reason="price_impact",
                severity=severity,
                supporting_stats={
                    "price_diff": abs(price_diff),
                    "size": size,
                    "impact_score": score,
                    "price": price,
                },
            )
        )
    return alerts
