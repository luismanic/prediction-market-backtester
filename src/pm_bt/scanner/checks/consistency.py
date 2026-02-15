from __future__ import annotations

import logging
from datetime import UTC, datetime
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


def check_complement_sum(
    trades: pl.LazyFrame,
    *,
    tolerance: float,
    window: str = "1h",
) -> list[Alert]:
    """Check that yes_vwap + no_vwap ≈ 1 for binary markets.

    Only fires when *both* ``outcome_id == "yes"`` and ``outcome_id == "no"``
    are present for a given market in the same time window.  Kalshi markets
    (single ``outcome_id="yes"``) are therefore silently skipped.
    """
    binary_trades = trades.filter(pl.col("outcome_id").is_in(["yes", "no"]))

    windowed = (
        binary_trades.sort("ts")
        .group_by_dynamic(
            "ts",
            every=window,
            group_by=["market_id", "venue", "outcome_id"],
            closed="left",
            label="left",
        )
        .agg(
            ((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias("vwap"),
        )
        .rename({"ts": "window_ts"})
    )

    # Pivot: one row per (market_id, venue, window_ts) with yes/no columns.
    pivoted = windowed.collect().pivot(
        on="outcome_id",
        index=["market_id", "venue", "window_ts"],
        values="vwap",
    )

    if "yes" not in pivoted.columns or "no" not in pivoted.columns:
        return []

    deviations = (
        pivoted.lazy()
        .filter(pl.col("yes").is_not_null() & pl.col("no").is_not_null())
        .with_columns(
            (pl.col("yes") + pl.col("no") - 1.0).abs().alias("deviation"),
        )
        .filter(pl.col("deviation") > tolerance)
        .collect()
    )

    alerts: list[Alert] = []
    market_ids = cast(list[object], deviations.get_column("market_id").to_list())
    venues = cast(list[object], deviations.get_column("venue").to_list())
    window_tss = cast(list[object], deviations.get_column("window_ts").to_list())
    yes_vwaps = cast(list[object], deviations.get_column("yes").to_list())
    no_vwaps = cast(list[object], deviations.get_column("no").to_list())
    deviation_values = cast(list[object], deviations.get_column("deviation").to_list())
    for idx in range(deviations.height):
        market_id = _as_str(market_ids[idx], field="market_id")
        venue_raw = _as_str(venues[idx], field="venue")
        ts = _as_datetime(window_tss[idx], field="window_ts")
        yes_vwap = _as_float(yes_vwaps[idx], field="yes")
        no_vwap = _as_float(no_vwaps[idx], field="no")
        deviation = _as_float(deviation_values[idx], field="deviation")
        severity = AlertSeverity.HIGH if deviation > 2 * tolerance else AlertSeverity.MEDIUM
        venue = Venue(venue_raw)
        alerts.append(
            Alert(
                alert_id=make_alert_id("complement_sum_deviation", market_id, ts),
                market_id=market_id,
                ts=ts,
                venue=venue,
                reason="complement_sum_deviation",
                severity=severity,
                supporting_stats={
                    "yes_vwap": yes_vwap,
                    "no_vwap": no_vwap,
                    "deviation": deviation,
                },
            )
        )
    return alerts


def check_mutually_exclusive(
    markets: pl.LazyFrame,
    trades: pl.LazyFrame,
    *,
    tolerance: float,
    event_group_expr: pl.Expr | None = None,
) -> list[Alert]:
    """Check that outcome prices within an event group sum ≤ 1.

    The event group is derived from ``market_id`` by stripping the last ``-``
    segment (e.g. ``KXHIGH-23-T3`` → ``KXHIGH-23``).  Override with
    *event_group_expr* if a different grouping is needed.
    """
    if event_group_expr is None:
        event_group_expr = pl.col("market_id").str.replace(r"-[^-]+$", "")

    market_keys = markets.select(["market_id", "venue"]).unique()

    # Per-market VWAP from trades
    market_vwap = (
        trades.group_by(["market_id", "venue"])
        .agg(
            ((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias("vwap"),
        )
        .join(market_keys, on=["market_id", "venue"], how="inner")
    )

    # Add event_group column
    grouped = market_vwap.with_columns(event_group_expr.alias("event_group"))

    # Count outcomes per event group, keep only groups with 2+ members
    group_stats = (
        grouped.group_by(["event_group", "venue"])
        .agg(
            pl.col("vwap").sum().alias("outcome_sum"),
            pl.len().alias("n_outcomes"),
        )
        .filter(pl.col("n_outcomes") >= 2)
        .filter(pl.col("outcome_sum") > 1.0 + tolerance)
        .collect()
    )

    alerts: list[Alert] = []
    outcome_sums = cast(list[object], group_stats.get_column("outcome_sum").to_list())
    venues = cast(list[object], group_stats.get_column("venue").to_list())
    event_groups = cast(list[object], group_stats.get_column("event_group").to_list())
    n_outcomes_values = cast(list[object], group_stats.get_column("n_outcomes").to_list())
    for idx in range(group_stats.height):
        outcome_sum = _as_float(outcome_sums[idx], field="outcome_sum")
        venue_raw = _as_str(venues[idx], field="venue")
        event_group = _as_str(event_groups[idx], field="event_group")
        n_outcomes = _as_float(n_outcomes_values[idx], field="n_outcomes")
        severity = AlertSeverity.HIGH if outcome_sum > 1.0 + 2 * tolerance else AlertSeverity.MEDIUM
        venue = Venue(venue_raw)
        alerts.append(
            Alert(
                alert_id=make_alert_id(
                    "mutually_exclusive_sum",
                    event_group,
                    # No per-row timestamp; use epoch as deterministic placeholder.
                    _EPOCH,
                ),
                market_id=event_group,
                ts=_EPOCH,
                venue=venue,
                reason="mutually_exclusive_sum",
                severity=severity,
                supporting_stats={
                    "outcome_sum": outcome_sum,
                    "n_outcomes": n_outcomes,
                },
            )
        )
    return alerts


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
