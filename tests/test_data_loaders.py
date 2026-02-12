from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pm_bt.data import load_markets, load_trades


def _write_fixture_parquet(data_root: Path) -> None:
    (data_root / "kalshi" / "markets").mkdir(parents=True, exist_ok=True)
    (data_root / "kalshi" / "trades").mkdir(parents=True, exist_ok=True)

    markets_df = pl.DataFrame(
        {
            "market_id": ["KX-RAIN-2026-01-01", "KX-TEMP-2026-01-02"],
            "venue": ["kalshi", "kalshi"],
            "outcome_id": ["yes", "yes"],
            "question": ["Will NYC rain?", "Will NYC be above 5C?"],
            "category": ["weather", "weather"],
            "close_ts": ["2026-01-01T23:59:59Z", "2026-01-02T23:59:59Z"],
            "resolved": [True, False],
            "winning_outcome": ["yes", None],
            "resolved_ts": ["2026-01-02T12:00:00Z", None],
            "market_structure": ["clob", "clob"],
        }
    )
    markets_df.write_parquet(data_root / "kalshi" / "markets" / "markets_0_2.parquet")

    trades_df = pl.DataFrame(
        {
            "ts": [
                "2026-01-01T09:00:00Z",
                "2026-01-01T09:01:00Z",
                "2026-01-01T09:02:00Z",
                "2026-01-02T09:00:00Z",
            ],
            "market_id": [
                "KX-RAIN-2026-01-01",
                "KX-RAIN-2026-01-01",
                "KX-RAIN-2026-01-01",
                "KX-TEMP-2026-01-02",
            ],
            "outcome_id": ["yes", "yes", "yes", "yes"],
            "venue": ["kalshi", "kalshi", "kalshi", "kalshi"],
            "price": [0.44, 0.46, 0.45, 0.52],
            "size": [100.0, 90.0, 80.0, 70.0],
            "side": ["buy", "buy", "sell", "buy"],
            "trade_id": ["t1", "t2", "t3", "t4"],
            "fee_paid": [0.0, 0.0, 0.0, 0.0],
        }
    )
    trades_df.write_parquet(data_root / "kalshi" / "trades" / "trades_0_4.parquet")


def _write_polymarket_trades_fixture(data_root: Path) -> None:
    (data_root / "polymarket" / "trades").mkdir(parents=True, exist_ok=True)
    trades_df = pl.DataFrame(
        {
            "condition_id": ["0xmarket1", "0xmarket1"],
            "transaction_hash": ["0xabc", "0xdef"],
            "log_index": [0, 1],
            "maker_asset_id": ["0", "555"],
            "taker_asset_id": ["555", "0"],
            "maker_amount": [500_000, 200_000],
            "taker_amount": [1_000_000, 400_000],
            "_fetched_at": ["2026-01-01T09:00:00Z", "2026-01-01T09:01:00Z"],
        }
    )
    trades_df.write_parquet(data_root / "polymarket" / "trades" / "trades_0_2.parquet")


def _write_kalshi_raw_markets_fixture(data_root: Path) -> None:
    (data_root / "kalshi" / "markets").mkdir(parents=True, exist_ok=True)
    markets_df = pl.DataFrame(
        {
            "ticker": ["KX-PRES-2024-DJT", "KX-NFL-2026-SB", "KX-UNKNOWN-2026"],
            "event_ticker": ["PRES-2024-DJT", "NFLGAME-2026-SB", "NO_MATCH-2026"],
            "title": ["Will DJT win?", "Who wins the Super Bowl?", "Unknown market"],
            "status": ["closed", "open", "open"],
            "result": ["yes", "", ""],
            "close_time": [
                "2024-11-06T00:00:00Z",
                "2026-02-10T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
        }
    )
    markets_df.write_parquet(data_root / "kalshi" / "markets" / "markets_raw_0_3.parquet")


def test_load_markets_filters_market_id(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_fixture_parquet(data_root)

    result = load_markets(
        "kalshi",
        data_root=data_root,
        market_id="KX-RAIN-2026-01-01",
    ).collect()

    assert result.height == 1
    assert result["market_id"][0] == "KX-RAIN-2026-01-01"
    assert result["resolved"][0] is True


def test_load_trades_filters_market_and_time_range(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_fixture_parquet(data_root)

    result = load_trades(
        "kalshi",
        data_root=data_root,
        market_id="KX-RAIN-2026-01-01",
        start_ts=datetime(2026, 1, 1, 9, 1, 0, tzinfo=UTC),
        end_ts=datetime(2026, 1, 1, 9, 2, 0, tzinfo=UTC),
    ).collect()

    assert result.height == 2
    assert set(result["trade_id"].to_list()) == {"t2", "t3"}


def test_load_trades_polymarket_uses_condition_id_for_market_id(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_polymarket_trades_fixture(data_root)

    result = load_trades("polymarket", data_root=data_root).collect()

    assert result.height == 2
    assert set(result["market_id"].to_list()) == {"0xmarket1"}
    assert set(result["outcome_id"].to_list()) == {"555"}


def test_load_markets_kalshi_raw_schema_maps_category_groups(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_kalshi_raw_markets_fixture(data_root)

    result = load_markets("kalshi", data_root=data_root).collect()
    categories = dict(zip(result["market_id"].to_list(), result["category"].to_list(), strict=True))

    assert categories["KX-PRES-2024-DJT"] == "politics"
    assert categories["KX-NFL-2026-SB"] == "sports"
    assert categories["KX-UNKNOWN-2026"] == "other"
