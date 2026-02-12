from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from pm_bt.cli import run_cli


def _write_trades_fixture(data_root: Path) -> None:
    trades_dir = data_root / "kalshi" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    trades = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "market_id": ["KX-RAIN-2026-01-01", "KX-RAIN-2026-01-01"],
            "outcome_id": ["yes", "yes"],
            "venue": ["kalshi", "kalshi"],
            "price": [0.50, 0.60],
            "size": [100.0, 100.0],
            "side": ["buy", "buy"],
            "trade_id": ["t1", "t2"],
            "fee_paid": [0.0, 0.0],
        }
    )
    trades.write_parquet(trades_dir / "trades_0_2.parquet")


def _write_strategy_config(path: Path) -> None:
    _ = path.write_text(
        "strategy_name: momentum\nstrategy_params:\n  threshold: 0.03\n  qty: 5.0\n",
        encoding="utf-8",
    )


def _write_markets_fixture(data_root: Path) -> None:
    markets_dir = data_root / "kalshi" / "markets"
    markets_dir.mkdir(parents=True, exist_ok=True)
    markets = pl.DataFrame(
        {
            "market_id": ["KX-RAIN-2026-01-01"],
            "venue": ["kalshi"],
            "outcome_id": ["yes"],
            "question": ["Will it rain?"],
            "category": ["weather"],
            "close_ts": [datetime(2026, 1, 1, 23, 59, tzinfo=UTC)],
            "resolved": [True],
            "winning_outcome": ["yes"],
            "resolved_ts": [datetime(2026, 1, 2, 12, 0, tzinfo=UTC)],
            "market_structure": ["clob"],
        }
    )
    markets.write_parquet(markets_dir / "markets_0_1.parquet")


def _write_batch_trades_fixture(data_root: Path) -> None:
    (data_root / "kalshi" / "trades").mkdir(parents=True, exist_ok=True)
    (data_root / "polymarket" / "trades").mkdir(parents=True, exist_ok=True)

    kalshi_trades = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
            ],
            "market_id": ["KX-A", "KX-A", "KX-B"],
            "outcome_id": ["yes", "yes", "yes"],
            "venue": ["kalshi", "kalshi", "kalshi"],
            "price": [0.40, 0.55, 0.50],
            "size": [100.0, 90.0, 10.0],
            "side": ["buy", "buy", "buy"],
            "trade_id": ["ka1", "ka2", "kb1"],
            "fee_paid": [0.0, 0.0, 0.0],
        }
    )
    kalshi_trades.write_parquet(data_root / "kalshi" / "trades" / "trades_0_3.parquet")

    polymarket_trades = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 10, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 10, 2, tzinfo=UTC),
            ],
            "market_id": ["PM-A", "PM-A", "PM-B"],
            "outcome_id": ["yes", "yes", "yes"],
            "venue": ["polymarket", "polymarket", "polymarket"],
            "price": [0.30, 0.60, 0.50],
            "size": [80.0, 70.0, 5.0],
            "side": ["buy", "buy", "buy"],
            "trade_id": ["pa1", "pa2", "pb1"],
            "fee_paid": [0.0, 0.0, 0.0],
        }
    )
    polymarket_trades.write_parquet(data_root / "polymarket" / "trades" / "trades_0_3.parquet")


def _first_run_dir(output_root: Path) -> Path:
    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert run_dirs
    return run_dirs[0]


def test_cli_backtest_writes_results_equity_and_trades(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    _write_trades_fixture(data_root)
    _write_strategy_config(config_path)

    exit_code = run_cli(
        [
            "backtest",
            "--venue",
            "kalshi",
            "--market",
            "KX-RAIN-2026-01-01",
            "--strategy",
            "momentum",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    run_dir = _first_run_dir(output_root)

    results_path = run_dir / "results.json"
    equity_path = run_dir / "equity.csv"
    trades_path = run_dir / "trades.csv"
    assert results_path.exists()
    assert equity_path.exists()
    assert trades_path.exists()

    payload = cast(dict[str, object], json.loads(results_path.read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    trading_metrics = cast(dict[str, float], payload["trading_metrics"])
    artifacts = cast(dict[str, str], payload["artifacts"])
    assert config["strategy_name"] == "momentum"
    assert trading_metrics["bars_processed"] > 0
    assert artifacts["equity_csv"].endswith("equity.csv")
    assert artifacts["trades_csv"].endswith("trades.csv")


def test_cli_backtest_returns_nonzero_for_unknown_strategy(tmp_path: Path) -> None:
    exit_code = run_cli(
        [
            "backtest",
            "--venue",
            "kalshi",
            "--market",
            "KX-RAIN-2026-01-01",
            "--strategy",
            "does_not_exist",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 1


def test_cli_backtest_returns_nonzero_for_invalid_config(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    config_path = tmp_path / "config.yaml"
    _write_trades_fixture(data_root)
    _write_strategy_config(config_path)

    exit_code = run_cli(
        [
            "backtest",
            "--venue",
            "kalshi",
            "--market",
            "KX-RAIN-2026-01-01",
            "--strategy",
            "momentum",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--output-root",
            str(tmp_path / "output"),
            "--start-ts",
            "2026-01-02T00:00:00Z",
            "--end-ts",
            "2026-01-01T00:00:00Z",
        ]
    )

    assert exit_code == 1


def test_cli_backtest_returns_nonzero_when_data_is_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_strategy_config(config_path)

    exit_code = run_cli(
        [
            "backtest",
            "--venue",
            "kalshi",
            "--market",
            "KX-RAIN-2026-01-01",
            "--strategy",
            "momentum",
            "--config",
            str(config_path),
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 1


def test_cli_backtest_writes_forecasting_metrics_when_markets_exist(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    _write_trades_fixture(data_root)
    _write_markets_fixture(data_root)
    _write_strategy_config(config_path)

    exit_code = run_cli(
        [
            "backtest",
            "--venue",
            "kalshi",
            "--market",
            "KX-RAIN-2026-01-01",
            "--strategy",
            "momentum",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
        ]
    )
    assert exit_code == 0

    run_dir = _first_run_dir(output_root)
    payload = cast(dict[str, object], json.loads((run_dir / "results.json").read_text("utf-8")))
    forecasting_metrics = cast(dict[str, float], payload["forecasting_metrics"])
    assert "n_trades" in forecasting_metrics
    assert forecasting_metrics["n_trades"] >= 0.0


def test_cli_batch_writes_summary_and_data_quality(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    _write_batch_trades_fixture(data_root)
    _write_strategy_config(config_path)

    exit_code = run_cli(
        [
            "batch",
            "--strategy",
            "momentum",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--venues",
            "kalshi",
            "polymarket",
            "--top-n",
            "2",
            "--min-trades",
            "2",
            "--max-gap-minutes",
            "1440",
        ]
    )

    assert exit_code == 0

    batch_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert batch_dirs
    batch_dir = batch_dirs[0]

    summary_path = batch_dir / "summary.csv"
    quality_path = batch_dir / "data_quality.json"
    assert summary_path.exists()
    assert quality_path.exists()

    summary = pl.read_csv(summary_path)
    assert summary.height >= 2
    assert set(summary["venue"].to_list()) == {"kalshi", "polymarket"}


def test_cli_batch_resume_reuses_existing_batch_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    _write_batch_trades_fixture(data_root)
    _write_strategy_config(config_path)

    first_exit = run_cli(
        [
            "batch",
            "--strategy",
            "momentum",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--name",
            "resume-test",
            "--venues",
            "kalshi",
            "polymarket",
            "--top-n",
            "2",
            "--min-trades",
            "2",
            "--max-gap-minutes",
            "1440",
        ]
    )
    assert first_exit == 0

    second_exit = run_cli(
        [
            "batch",
            "--strategy",
            "momentum",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--name",
            "resume-test",
            "--resume",
        ]
    )
    assert second_exit == 0

    batch_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert len(batch_dirs) == 1
    assert (batch_dirs[0] / "checkpoint.json").exists()
