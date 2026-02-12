from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from pm_bt.common.categories import kalshi_category_expr
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
    """Scan Parquet files matching a glob pattern.

    Excludes macOS resource fork files (``._*.parquet``) which are invalid
    Parquet and would cause ``File out of specification`` errors.
    """
    parent = path_glob.parent
    pattern = path_glob.name
    valid_files = sorted(f for f in parent.glob(pattern) if not f.name.startswith("._"))
    if not valid_files:
        raise FileNotFoundError(f"No valid Parquet files found in {parent}/{pattern}")
    return pl.scan_parquet(valid_files)


def _coerce_utc_datetime(column: str, schema: set[str], dtypes: dict[str, pl.DataType]) -> pl.Expr:
    """Coerce a column to UTC datetime, handling native datetimes and strings.

    - If the column is already a Datetime, cast its time zone to UTC.
    - If the column is a string, parse it to datetime with UTC.
    - If the column is null-typed or missing, return a null literal.
    """
    if column not in schema:
        return pl.lit(None, dtype=pl.Datetime(time_zone="UTC"))

    dtype = dtypes.get(column)

    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            return pl.col(column).dt.convert_time_zone("UTC")
        return pl.col(column).dt.replace_time_zone("UTC")

    if dtype == pl.Null:
        return pl.lit(None, dtype=pl.Datetime(time_zone="UTC"))

    # Fallback: cast to string and parse.
    return (
        pl.when(pl.col(column).is_null())
        .then(pl.lit(None, dtype=pl.Datetime(time_zone="UTC")))
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
    collected_schema = lf.collect_schema()
    schema = set(collected_schema.names())
    dtypes = dict(collected_schema)

    if set(CANONICAL_MARKET_COLUMNS).issubset(schema):
        return lf.select(CANONICAL_MARKET_COLUMNS)

    if venue == Venue.KALSHI and {"ticker", "title", "close_time", "result"}.issubset(schema):
        return lf.select(
            [
                _to_str("ticker").alias("market_id"),
                pl.lit(venue.value).alias("venue"),
                pl.lit("yes").alias("outcome_id"),
                _to_str("title").alias("question"),
                kalshi_category_expr("event_ticker").alias("category"),
                _coerce_utc_datetime("close_time", schema, dtypes).alias("close_ts"),
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
        #
        # Resolution heuristic: for closed binary markets the `outcome_prices` JSON array
        # contains final settlement prices. When one outcome is > 0.99 and the other < 0.01
        # the market is considered resolved. This covers ~45% of closed markets; the rest
        # have both prices at 0 (legacy/AMM-era markets) and remain unresolved.
        #
        # The heuristic is ported from Jon Becker's prediction-market-analysis project.
        has_outcome_prices = "outcome_prices" in schema

        is_closed = (
            pl.col("closed").cast(pl.Boolean, strict=False).fill_null(False)
            if "closed" in schema
            else pl.lit(False)
        )

        if has_outcome_prices:
            # Extract the two outcome prices from JSON array string '["0.99", "0.01"]'.
            p0 = (
                _to_str("outcome_prices").str.json_path_match("$[0]").cast(pl.Float64, strict=False)
            )
            p1 = (
                _to_str("outcome_prices").str.json_path_match("$[1]").cast(pl.Float64, strict=False)
            )
            # A market is resolved when one outcome settles near 1 and the other near 0.
            outcome_0_won = (p0 > 0.99) & (p1 < 0.01)
            outcome_1_won = (p1 > 0.99) & (p0 < 0.01)
            is_resolved = is_closed & (outcome_0_won | outcome_1_won)
            winning_outcome_expr = (
                pl.when(is_resolved & outcome_0_won)
                .then(pl.lit("yes"))
                .when(is_resolved & outcome_1_won)
                .then(pl.lit("no"))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
            )
        else:
            is_resolved = pl.lit(False)
            winning_outcome_expr = pl.lit(None, dtype=pl.Utf8)

        return lf.select(
            [
                _to_str("id").alias("market_id"),
                pl.lit(venue.value).alias("venue"),
                pl.lit("yes").alias("outcome_id"),
                _to_str("question").alias("question"),
                pl.lit(None, dtype=pl.Utf8).alias("category"),
                _coerce_utc_datetime("end_date", schema, dtypes).alias("close_ts"),
                is_resolved.alias("resolved"),
                winning_outcome_expr.alias("winning_outcome"),
                pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("resolved_ts"),
                pl.when(pl.col("market_maker_address").is_not_null())
                .then(pl.lit("amm"))
                .otherwise(pl.lit("clob"))
                .alias("market_structure"),
            ]
        )

    raise ValueError(f"Unsupported market schema for venue={venue.value}: {sorted(schema)}")


_TRADE_REQUIRED_COLUMNS = ["ts", "price", "size"]


def _drop_null_trades_required_fields(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop trade rows where required columns (ts, price, size) are null.

    Nulls in these columns are produced by strict=False casts on corrupt data.
    Filtering is expressed as a lazy predicate so Polars can push it down.
    This helper is intentionally silent to avoid eager counting/materialization.
    """
    has_null = pl.any_horizontal([pl.col(c).is_null() for c in _TRADE_REQUIRED_COLUMNS])
    return lf.filter(~has_null)


def _normalize_trades(lf: pl.LazyFrame, venue: Venue) -> pl.LazyFrame:
    collected_schema = lf.collect_schema()
    schema = set(collected_schema.names())
    dtypes = dict(collected_schema)

    if set(CANONICAL_TRADE_COLUMNS).issubset(schema):
        # NOTE: strict=False casts below can produce nulls from unparseable values.
        # We drop rows with null ts/price/size since they cannot be used in bar building
        # or execution. fee_paid nulls are filled with 0.0.
        casted = lf.select(CANONICAL_TRADE_COLUMNS).with_columns(
            [
                _coerce_utc_datetime("ts", schema, dtypes).alias("ts"),
                _to_float("price").alias("price"),
                _to_float("size").alias("size"),
                _to_float("fee_paid").fill_null(0.0).alias("fee_paid"),
            ]
        )
        return _drop_null_trades_required_fields(casted)

    if venue == Venue.KALSHI and {"ticker", "created_time", "yes_price", "count"}.issubset(schema):
        # Kalshi taker_side uses "yes"/"no" to indicate which side the taker bought.
        # We map: "yes" → "buy" (taker bought yes contracts), "no" → "sell" (taker sold yes).
        side_expr = (
            pl.when(_to_str("taker_side") == pl.lit("yes"))
            .then(pl.lit("buy"))
            .when(_to_str("taker_side") == pl.lit("no"))
            .then(pl.lit("sell"))
            .otherwise(pl.lit("unknown"))
        )
        return _drop_null_trades_required_fields(
            lf.select(
                [
                    _coerce_utc_datetime("created_time", schema, dtypes).alias("ts"),
                    _to_str("ticker").alias("market_id"),
                    pl.lit("yes").alias("outcome_id"),
                    pl.lit(venue.value).alias("venue"),
                    (_to_float("yes_price") / pl.lit(100.0)).alias("price"),
                    _to_float("count").alias("size"),
                    side_expr.alias("side"),
                    _to_str("trade_id").alias("trade_id"),
                    pl.lit(0.0).alias("fee_paid"),
                ]
            )
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
        # Prefer condition_id as market identifier when present.
        if "condition_id" in schema:
            condition_id = _to_str("condition_id")
            market_id_expr = (
                pl.when(condition_id.str.len_chars() > 0)
                .then(condition_id)
                .otherwise(outcome_id_expr)
            )
        else:
            market_id_expr = outcome_id_expr

        # LIMITATION: _fetched_at is the indexer fetch timestamp, not the on-chain
        # transaction timestamp.  The `timestamp` column exists but is null-typed in
        # the current dataset.  This introduces timing inaccuracy (typically seconds
        # to minutes) that affects latency simulation and bar alignment.  A future
        # improvement should derive ts from block_timestamp when available.
        #
        # Priority: timestamp (if non-null) > _fetched_at > null.
        if "timestamp" in schema and dtypes.get("timestamp") != pl.Null:
            ts_expr = _coerce_utc_datetime("timestamp", schema, dtypes)
        elif "_fetched_at" in schema:
            ts_expr = _coerce_utc_datetime("_fetched_at", schema, dtypes)
        else:
            ts_expr = pl.lit(None, dtype=pl.Datetime(time_zone="UTC"))

        if "fee" in schema:
            fee_expr = (
                pl.when(pl.col("fee").is_not_null())
                .then(_to_float("fee") / pl.lit(1e6))
                .otherwise(pl.lit(0.0))
            )
        else:
            fee_expr = pl.lit(0.0)

        return _drop_null_trades_required_fields(
            lf.select(
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
        )

    raise ValueError(f"Unsupported trade schema for venue={venue.value}: {sorted(schema)}")


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
    return _apply_trade_filters(normalized, market_id=market_id, start_ts=start_ts, end_ts=end_ts)
