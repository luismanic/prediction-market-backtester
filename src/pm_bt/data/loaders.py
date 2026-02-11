from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from pm_bt.common.types import Venue

logger = logging.getLogger(__name__)

CANONICAL_MARKET_COLUMNS = [
    "market_id",
    "venue",
    "outcome_id",
    "question",
    "category",
    "close_ts",
    "resolved",
    "winning_outcome",
    "resolved_ts",
    "market_structure",
]

CANONICAL_TRADE_COLUMNS = [
    "ts",
    "market_id",
    "outcome_id",
    "venue",
    "price",
    "size",
    "side",
    "trade_id",
    "fee_paid",
]


def _scan_parquet_glob(path_glob: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(path_glob))


def _parse_datetime_expr(column: str) -> pl.Expr:
    """Parse a column to UTC datetime. Uses strict parsing; unparseable values become null."""
    return (
        pl.when(pl.col(column).is_null())
        .then(pl.lit(None))
        .otherwise(
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(strict=False, time_zone="UTC")
        )
    )


def _to_float(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False)


def _to_str(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False)


def _normalize_markets(lf: pl.LazyFrame, venue: Venue) -> pl.LazyFrame:
    schema = set(lf.collect_schema().names())

    if set(CANONICAL_MARKET_COLUMNS).issubset(schema):
        return lf.select(CANONICAL_MARKET_COLUMNS)

    if venue == Venue.KALSHI and {"ticker", "title", "close_time", "result"}.issubset(schema):
        return lf.select(
            [
                _to_str("ticker").alias("market_id"),
                pl.lit(venue.value).alias("venue"),
                pl.lit("yes").alias("outcome_id"),
                _to_str("title").alias("question"),
                _to_str("event_ticker").alias("category"),
                _parse_datetime_expr("close_time").alias("close_ts"),
                pl.col("result").is_in(["yes", "no"]).alias("resolved"),
                pl.when(pl.col("result").is_in(["yes", "no"]))
                .then(_to_str("result"))
                .otherwise(pl.lit(None))
                .alias("winning_outcome"),
                pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("resolved_ts"),
                pl.lit("clob").alias("market_structure"),
            ]
        )

    if venue == Venue.POLYMARKET and {"id", "question", "end_date"}.issubset(schema):
        # NOTE: Polymarket markets carry multiple outcomes via `clob_token_ids` (JSON array).
        # At market-level normalization we default to "yes" as primary outcome.
        # Per-outcome expansion requires unpacking clob_token_ids + outcomes arrays.
        return lf.select(
            [
                _to_str("id").alias("market_id"),
                pl.lit(venue.value).alias("venue"),
                pl.lit("yes").alias("outcome_id"),
                _to_str("question").alias("question"),
                pl.lit(None, dtype=pl.Utf8).alias("category"),
                _parse_datetime_expr("end_date").alias("close_ts"),
                pl.lit(False).alias("resolved"),
                pl.lit(None, dtype=pl.Utf8).alias("winning_outcome"),
                pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("resolved_ts"),
                pl.when(pl.col("market_maker_address").is_not_null())
                .then(pl.lit("amm"))
                .otherwise(pl.lit("clob"))
                .alias("market_structure"),
            ]
        )

    raise ValueError(f"Unsupported market schema for venue={venue.value}: {sorted(schema)}")


def _normalize_trades(lf: pl.LazyFrame, venue: Venue) -> pl.LazyFrame:
    schema = set(lf.collect_schema().names())

    if set(CANONICAL_TRADE_COLUMNS).issubset(schema):
        return lf.select(CANONICAL_TRADE_COLUMNS).with_columns(
            [
                _parse_datetime_expr("ts").alias("ts"),
                _to_float("price").alias("price"),
                _to_float("size").alias("size"),
                _to_float("fee_paid").alias("fee_paid"),
            ]
        )

    if venue == Venue.KALSHI and {"ticker", "created_time", "yes_price", "count"}.issubset(schema):
        return lf.select(
            [
                _parse_datetime_expr("created_time").alias("ts"),
                _to_str("ticker").alias("market_id"),
                pl.lit("yes").alias("outcome_id"),
                pl.lit(venue.value).alias("venue"),
                (_to_float("yes_price") / pl.lit(100.0)).alias("price"),
                _to_float("count").alias("size"),
                _to_str("taker_side").fill_null("unknown").alias("side"),
                _to_str("trade_id").alias("trade_id"),
                pl.lit(0.0).alias("fee_paid"),
            ]
        )

    if venue == Venue.POLYMARKET and {"transaction_hash", "log_index"}.issubset(schema):
        maker_asset = _to_str("maker_asset_id")
        taker_asset = _to_str("taker_asset_id")
        maker_amount = _to_float("maker_amount")
        taker_amount = _to_float("taker_amount")

        is_buy = maker_asset == pl.lit("0")

        # Guard against zero-amount trades that would produce inf/null prices.
        safe_taker = pl.when(taker_amount > 0).then(taker_amount).otherwise(pl.lit(None))
        safe_maker = pl.when(maker_amount > 0).then(maker_amount).otherwise(pl.lit(None))

        price_expr = (
            pl.when(is_buy).then(maker_amount / safe_taker).otherwise(taker_amount / safe_maker)
        )
        size_expr = (
            pl.when(is_buy).then(taker_amount / pl.lit(1e6)).otherwise(maker_amount / pl.lit(1e6))
        )
        # Use the non-zero asset_id as outcome_id (each token is a distinct outcome).
        outcome_id_expr = pl.when(is_buy).then(taker_asset).otherwise(maker_asset)
        market_id_expr = outcome_id_expr

        ts_expr = (
            pl.when(pl.col("_fetched_at").is_not_null())
            .then(_parse_datetime_expr("_fetched_at"))
            .otherwise(pl.lit(None, dtype=pl.Datetime(time_zone="UTC")))
        )

        fee_expr = (
            pl.when(pl.col("fee").is_not_null())
            .then(_to_float("fee") / pl.lit(1e6))
            .otherwise(pl.lit(0.0))
        )

        return lf.select(
            [
                ts_expr.alias("ts"),
                market_id_expr.alias("market_id"),
                outcome_id_expr.alias("outcome_id"),
                pl.lit(venue.value).alias("venue"),
                price_expr.alias("price"),
                size_expr.alias("size"),
                pl.when(is_buy).then(pl.lit("buy")).otherwise(pl.lit("sell")).alias("side"),
                pl.concat_str(
                    [_to_str("transaction_hash"), pl.lit(":"), _to_str("log_index")],
                    ignore_nulls=False,
                ).alias("trade_id"),
                fee_expr.alias("fee_paid"),
            ]
        )

    raise ValueError(f"Unsupported trade schema for venue={venue.value}: {sorted(schema)}")


def _warn_null_timestamps(lf: pl.LazyFrame, ts_column: str, context: str) -> pl.LazyFrame:
    """Log a warning if null timestamps are detected after normalization.

    Returns the input LazyFrame unchanged (side-effect: logs on collect).
    To actually strip nulls, callers should filter explicitly.
    """
    # Materialize a count of nulls efficiently via a single-column scan.
    null_count = int(
        lf.select(pl.col(ts_column).is_null().sum().cast(pl.Int64)).collect().item(0, 0)  # pyright: ignore[reportAny]
    )
    if null_count > 0:
        logger.warning(
            "Null timestamps detected after normalization: %d rows with null '%s' (%s)",
            null_count,
            ts_column,
            context,
        )
    return lf


def _apply_market_filters(
    lf: pl.LazyFrame,
    market_id: str | None,
    start_ts: datetime | None,
    end_ts: datetime | None,
) -> pl.LazyFrame:
    if market_id:
        lf = lf.filter(pl.col("market_id") == market_id)
    # NOTE: date-range filters use close_ts. Markets with null close_ts are excluded
    # when a date range is specified. This is intentional: markets without a known
    # close date cannot be meaningfully placed in a time window.
    if start_ts:
        lf = lf.filter(pl.col("close_ts") >= start_ts)
    if end_ts:
        lf = lf.filter(pl.col("close_ts") <= end_ts)
    return lf


def _apply_trade_filters(
    lf: pl.LazyFrame,
    market_id: str | None,
    start_ts: datetime | None,
    end_ts: datetime | None,
) -> pl.LazyFrame:
    if market_id:
        lf = lf.filter(pl.col("market_id") == market_id)
    if start_ts:
        lf = lf.filter(pl.col("ts") >= start_ts)
    if end_ts:
        lf = lf.filter(pl.col("ts") <= end_ts)
    return lf


def load_markets(
    venue: Venue | str,
    *,
    data_root: Path | str = "data",
    market_id: str | None = None,
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
) -> pl.LazyFrame:
    venue_enum = Venue(venue)
    base = Path(data_root) / venue_enum.value / "markets"
    lf = _scan_parquet_glob(base / "*.parquet")
    normalized = _normalize_markets(lf, venue_enum)
    return _apply_market_filters(normalized, market_id=market_id, start_ts=start_ts, end_ts=end_ts)


def load_trades(
    venue: Venue | str,
    *,
    data_root: Path | str = "data",
    market_id: str | None = None,
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
) -> pl.LazyFrame:
    venue_enum = Venue(venue)
    base = Path(data_root) / venue_enum.value / "trades"
    lf = _scan_parquet_glob(base / "*.parquet")
    normalized = _normalize_trades(lf, venue_enum)
    _ = _warn_null_timestamps(normalized, "ts", context=f"load_trades(venue={venue_enum.value})")
    return _apply_trade_filters(normalized, market_id=market_id, start_ts=start_ts, end_ts=end_ts)
