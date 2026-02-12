# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math

import polars as pl
import pytest

from pm_bt.reporting.calibration import (
    compute_brier_score,
    compute_calibration_metrics,
    compute_calibration_metrics_from_fills_and_markets,
    compute_ece,
    compute_log_loss,
)


def _sample_trades() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "price": [0.8, 0.2, 0.7, 0.1],
            "won": [True, False, True, False],
        }
    )


def test_compute_brier_score() -> None:
    score = compute_brier_score(_sample_trades())
    expected = ((0.8 - 1.0) ** 2 + (0.2 - 0.0) ** 2 + (0.7 - 1.0) ** 2 + (0.1 - 0.0) ** 2) / 4.0
    assert score == pytest.approx(expected)


def test_compute_log_loss() -> None:
    loss = compute_log_loss(_sample_trades())
    expected = -(math.log(0.8) + math.log(0.8) + math.log(0.7) + math.log(0.9)) / 4.0
    assert loss == pytest.approx(expected)


def test_compute_ece_with_10_bins() -> None:
    ece = compute_ece(_sample_trades(), n_bins=10)
    # Bins: 0.1 -> win 0.0, 0.2 -> win 0.0, 0.7 -> win 1.0, 0.8 -> win 1.0.
    # Midpoints: 0.15, 0.25, 0.75, 0.85 => abs diff each = 0.15, 0.25, 0.25, 0.15.
    assert ece == pytest.approx((0.15 + 0.25 + 0.25 + 0.15) / 4.0)


def test_compute_calibration_metrics() -> None:
    metrics = compute_calibration_metrics(_sample_trades())
    assert metrics["n_trades"] == 4.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["ece"] >= 0.0


def test_empty_input_returns_zero_metrics() -> None:
    empty = pl.DataFrame(
        {"price": pl.Series([], dtype=pl.Float64), "won": pl.Series([], dtype=pl.Boolean)}
    )
    metrics = compute_calibration_metrics(empty)
    assert metrics == {
        "brier_score": 0.0,
        "log_loss": 0.0,
        "ece": 0.0,
        "n_trades": 0.0,
    }


def test_compute_calibration_metrics_from_fills_and_markets() -> None:
    fills = pl.DataFrame(
        {
            "market_id": ["M1", "M1", "M2"],
            "outcome_id": ["yes", "yes", "yes"],
            "venue": ["kalshi", "kalshi", "kalshi"],
            "price_fill": [0.8, 0.3, 0.6],
        }
    )
    markets = pl.DataFrame(
        {
            "market_id": ["M1", "M2"],
            "outcome_id": ["yes", "yes"],
            "venue": ["kalshi", "kalshi"],
            "resolved": [True, True],
            "winning_outcome": ["yes", "no"],
        }
    )

    metrics = compute_calibration_metrics_from_fills_and_markets(fills, markets)

    assert metrics["n_trades"] == 3.0
    assert metrics["brier_score"] == pytest.approx(
        ((0.8 - 1.0) ** 2 + (0.3 - 1.0) ** 2 + (0.6 - 0.0) ** 2) / 3.0
    )
