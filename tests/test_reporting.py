# pyright: reportUnknownMemberType=false

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from pm_bt.reporting.metrics import (
    compute_max_exposure_by_market,
    compute_sharpe_ratio,
    compute_win_rate,
)
from pm_bt.reporting.plots import plot_drawdown, plot_equity_curve, plot_returns_distribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_equity_df(equities: list[float]) -> pl.DataFrame:
    """Build a minimal equity DataFrame with monotonic timestamps."""
    return pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1, 0, i, tzinfo=UTC) for i in range(len(equities))],
            "equity": equities,
            "cash": [e * 0.9 for e in equities],
            "realized_pnl": [0.0] * len(equities),
            "unrealized_pnl": [0.0] * len(equities),
            "gross_notional_exposure": [0.0] * len(equities),
            "cash_at_risk_gross": [0.0] * len(equities),
        },
        schema={
            "ts": pl.Datetime(time_zone="UTC"),
            "equity": pl.Float64,
            "cash": pl.Float64,
            "realized_pnl": pl.Float64,
            "unrealized_pnl": pl.Float64,
            "gross_notional_exposure": pl.Float64,
            "cash_at_risk_gross": pl.Float64,
        },
    )


def _make_exposure_df() -> pl.DataFrame:
    """Build a minimal exposure DataFrame with two markets."""
    rows = [
        (datetime(2024, 1, 1, 0, 0, tzinfo=UTC), "MKT_A", "YES", 10.0, 5.0, 5.0),
        (datetime(2024, 1, 1, 0, 1, tzinfo=UTC), "MKT_A", "YES", 10.0, 5.0, 8.0),
        (datetime(2024, 1, 1, 0, 0, tzinfo=UTC), "MKT_B", "YES", 5.0, 3.0, 3.0),
        (datetime(2024, 1, 1, 0, 1, tzinfo=UTC), "MKT_B", "YES", 5.0, 3.0, 2.0),
    ]
    return pl.DataFrame(
        rows,
        schema=[
            ("ts", pl.Datetime(time_zone="UTC")),
            ("market_id", pl.Utf8),
            ("outcome_id", pl.Utf8),
            ("position", pl.Float64),
            ("notional_exposure", pl.Float64),
            ("cash_at_risk", pl.Float64),
        ],
        orient="row",
    )


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestComputeSharpeRatio:
    def test_returns_zero_for_flat_equity(self) -> None:
        df = _make_equity_df([100.0, 100.0, 100.0, 100.0])
        assert compute_sharpe_ratio(df, "1m") == 0.0

    def test_positive_for_growing_equity(self) -> None:
        df = _make_equity_df([100.0, 101.0, 102.0, 103.0, 104.0])
        sharpe = compute_sharpe_ratio(df, "1d")
        assert sharpe > 0.0

    def test_returns_zero_for_single_bar(self) -> None:
        df = _make_equity_df([100.0])
        assert compute_sharpe_ratio(df, "1m") == 0.0

    def test_uses_default_timeframe_for_unknown(self) -> None:
        df = _make_equity_df([100.0, 101.0, 102.0, 103.0])
        # Unknown timeframe should still return a value (uses 525_600 as default).
        sharpe = compute_sharpe_ratio(df, "7m")
        assert isinstance(sharpe, float)


class TestComputeWinRate:
    def test_on_fixture_equity(self) -> None:
        # 3 up bars, 2 down bars → win rate = 3/5 = 0.6
        df = _make_equity_df([100.0, 101.0, 100.5, 102.0, 101.0, 103.0])
        assert compute_win_rate(df) == pytest.approx(3.0 / 5.0)

    def test_all_up(self) -> None:
        df = _make_equity_df([100.0, 101.0, 102.0, 103.0])
        assert compute_win_rate(df) == pytest.approx(1.0)

    def test_all_flat(self) -> None:
        df = _make_equity_df([100.0, 100.0, 100.0])
        assert compute_win_rate(df) == pytest.approx(0.0)

    def test_single_bar_returns_zero(self) -> None:
        df = _make_equity_df([100.0])
        assert compute_win_rate(df) == 0.0


class TestComputeMaxExposureByMarket:
    def test_two_markets(self) -> None:
        df = _make_exposure_df()
        result = compute_max_exposure_by_market(df)
        assert result["MKT_A"] == pytest.approx(8.0)
        assert result["MKT_B"] == pytest.approx(3.0)

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame(
            schema=[
                ("ts", pl.Datetime(time_zone="UTC")),
                ("market_id", pl.Utf8),
                ("outcome_id", pl.Utf8),
                ("position", pl.Float64),
                ("notional_exposure", pl.Float64),
                ("cash_at_risk", pl.Float64),
            ]
        )
        assert compute_max_exposure_by_market(df) == {}


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------


class TestPlotEquityCurve:
    def test_creates_png(self, tmp_path: Path) -> None:
        df = _make_equity_df([100.0, 101.0, 102.0, 103.0])
        out = tmp_path / "equity.png"
        plot_equity_curve(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotDrawdown:
    def test_creates_png(self, tmp_path: Path) -> None:
        df = _make_equity_df([100.0, 99.0, 98.0, 100.0])
        out = tmp_path / "drawdown.png"
        plot_drawdown(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotReturnsDistribution:
    def test_creates_png(self, tmp_path: Path) -> None:
        df = _make_equity_df([100.0, 101.0, 99.0, 102.0, 100.0])
        out = tmp_path / "returns.png"
        plot_returns_distribution(df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_insufficient_data_creates_png(self, tmp_path: Path) -> None:
        df = _make_equity_df([100.0])
        out = tmp_path / "returns.png"
        plot_returns_distribution(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# E2E generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_produces_all_artifacts(self, tmp_path: Path) -> None:
        from pm_bt.backtest.engine import BacktestArtifacts
        from pm_bt.common.models import (
            BacktestConfig,
            DatasetSlice,
            RunResult,
            RunTimings,
        )
        from pm_bt.reporting import generate_report

        config = BacktestConfig(
            name="test-report",
            bar_timeframe="1m",
            initial_cash=10_000.0,
        )
        run_result = RunResult(
            run_id="test-run-001",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            config=config,
            dataset_slice=DatasetSlice(),
            trading_metrics={},
            forecasting_metrics={},
            artifacts={},
            timings=RunTimings(),
        )
        equity_df = _make_equity_df([10_000.0, 10_050.0, 10_030.0, 10_100.0])
        fills_df = pl.DataFrame(
            schema=[
                ("ts_fill", pl.Datetime(time_zone="UTC")),
                ("market_id", pl.Utf8),
                ("outcome_id", pl.Utf8),
                ("venue", pl.Utf8),
                ("side", pl.Utf8),
                ("qty_filled", pl.Float64),
                ("price_fill", pl.Float64),
                ("fees", pl.Float64),
                ("slippage_cost", pl.Float64),
                ("latency_ms", pl.Int64),
                ("notional", pl.Float64),
            ]
        )
        exposure_df = _make_exposure_df()

        artifacts = BacktestArtifacts(
            run_result=run_result,
            equity_curve=equity_df,
            fills=fills_df,
            exposure_curve=exposure_df,
        )

        extra_metrics, artifact_paths = generate_report(artifacts, tmp_path, "1m")

        # Metrics
        assert "sharpe_ratio" in extra_metrics
        assert "win_rate" in extra_metrics
        assert "exposure_MKT_A" in extra_metrics
        assert "exposure_MKT_B" in extra_metrics
        assert extra_metrics["win_rate"] == pytest.approx(2.0 / 3.0)

        # Artifact files
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "equity_curve.png").exists()
        assert (tmp_path / "drawdown.png").exists()
        assert (tmp_path / "returns_distribution.png").exists()

        # Artifact paths dict
        assert "config_json" in artifact_paths
        assert "equity_curve_png" in artifact_paths
        assert "drawdown_png" in artifact_paths
        assert "returns_distribution_png" in artifact_paths
