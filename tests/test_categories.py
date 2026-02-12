from __future__ import annotations

import polars as pl

from pm_bt.common.categories import classify_kalshi_event_ticker, kalshi_category_expr


def test_classify_kalshi_event_ticker_maps_known_groups() -> None:
    assert classify_kalshi_event_ticker("PRES-2024-DJT") == "politics"
    assert classify_kalshi_event_ticker("NFLGAME-2026-SEA-SF") == "sports"
    assert classify_kalshi_event_ticker("BTCD-2026-12-31") == "crypto"
    assert classify_kalshi_event_ticker("FEDDECISION-2026-06") == "finance"
    assert classify_kalshi_event_ticker("RAINNYC-2026-01-01") == "weather"


def test_classify_kalshi_event_ticker_handles_null_and_unknown() -> None:
    assert classify_kalshi_event_ticker(None) is None
    assert classify_kalshi_event_ticker("SOMETHING-ELSE") == "other"


def test_kalshi_category_expr_matches_python_classifier() -> None:
    df = pl.DataFrame(
        {
            "event_ticker": [
                "PRES-2024-DJT",
                "NFLGAME-2026-SEA-SF",
                "BTCD-2026-12-31",
                "SOMETHING-ELSE",
                None,
            ]
        }
    )

    out = df.select(kalshi_category_expr("event_ticker").alias("category"))
    assert out["category"].to_list() == ["politics", "sports", "crypto", "other", None]
