from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from pm_bt.data.quality import compute_tradability_metrics, evaluate_market_trade_quality


def _sample_trades() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
            ],
            "market_id": ["M1", "M1", "M1"],
            "outcome_id": ["yes", "yes", "yes"],
            "venue": ["kalshi", "kalshi", "kalshi"],
            "price": [0.45, 0.50, 0.55],
            "size": [10.0, 20.0, 30.0],
            "side": ["buy", "sell", "buy"],
            "trade_id": ["t1", "t2", "t3"],
            "fee_paid": [0.0, 0.0, 0.0],
        }
    )


def test_evaluate_market_trade_quality_passes_for_clean_data() -> None:
    quality = evaluate_market_trade_quality(
        _sample_trades(),
        venue="kalshi",
        market_id="M1",
        min_trade_count=3,
        max_null_rate=0.0,
        max_gap_minutes=120.0,
    )

    assert quality.passes is True
    assert quality.failed_checks == []
    assert quality.trade_count == 3


def test_evaluate_market_trade_quality_fails_for_low_trade_count() -> None:
    quality = evaluate_market_trade_quality(
        _sample_trades().head(1),
        venue="kalshi",
        market_id="M1",
        min_trade_count=2,
        max_null_rate=0.0,
        max_gap_minutes=120.0,
    )

    assert quality.passes is False
    assert "trade_count_below_threshold" in quality.failed_checks


def test_compute_tradability_metrics_returns_expected_columns() -> None:
    metrics = compute_tradability_metrics(_sample_trades())

    assert metrics["trade_count"] == 3.0
    assert metrics["volume_total"] == 60.0
    assert "spread_proxy" in metrics
    assert "volatility_proxy" in metrics
    assert "slippage_proxy_bps" in metrics
