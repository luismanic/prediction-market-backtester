"""Reporting module: metrics computation, plot generation, artifact export."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pm_bt.backtest.engine import BacktestArtifacts
from pm_bt.reporting.calibration import (
    compute_brier_score,
    compute_calibration_metrics,
    compute_calibration_metrics_from_fills_and_markets,
    compute_ece,
    compute_log_loss,
)
from pm_bt.reporting.metrics import (
    compute_max_exposure_by_market,
    compute_sharpe_ratio,
    compute_win_rate,
)
from pm_bt.reporting.plots import plot_drawdown, plot_equity_curve, plot_returns_distribution

logger = logging.getLogger(__name__)

__all__ = [
    "compute_brier_score",
    "compute_calibration_metrics_from_fills_and_markets",
    "compute_calibration_metrics",
    "compute_ece",
    "compute_log_loss",
    "compute_max_exposure_by_market",
    "compute_sharpe_ratio",
    "compute_win_rate",
    "generate_report",
    "plot_drawdown",
    "plot_equity_curve",
    "plot_returns_distribution",
]


def generate_report(
    artifacts: BacktestArtifacts,
    output_dir: Path,
    bar_timeframe: str,
    skip_plots: bool = False,
) -> tuple[dict[str, float], dict[str, str]]:
    """Generate all reporting artifacts and return extra metrics + artifact paths.

    Returns:
        A tuple of (extra_trading_metrics, artifact_paths) to be merged into RunResult
        by the caller.
    """
    extra_metrics: dict[str, float] = {}
    artifact_paths: dict[str, str] = {}

    # -- Metrics --
    extra_metrics["sharpe_ratio"] = compute_sharpe_ratio(artifacts.equity_curve, bar_timeframe)
    extra_metrics["win_rate"] = compute_win_rate(artifacts.equity_curve)

    exposure_by_market = compute_max_exposure_by_market(artifacts.exposure_curve)
    for market_id, max_risk in exposure_by_market.items():
        extra_metrics[f"exposure_{market_id}"] = max_risk

    # -- Config export --
    config_path = output_dir / "config.json"
    config_payload = artifacts.run_result.config.model_dump(mode="json")
    _ = config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths["config_json"] = str(config_path)

    # -- Plots --
    if not skip_plots:
        equity_plot_path = output_dir / "equity_curve.png"
        plot_equity_curve(artifacts.equity_curve, equity_plot_path)
        artifact_paths["equity_curve_png"] = str(equity_plot_path)

        drawdown_plot_path = output_dir / "drawdown.png"
        plot_drawdown(artifacts.equity_curve, drawdown_plot_path)
        artifact_paths["drawdown_png"] = str(drawdown_plot_path)

        returns_plot_path = output_dir / "returns_distribution.png"
        plot_returns_distribution(artifacts.equity_curve, returns_plot_path)
        artifact_paths["returns_distribution_png"] = str(returns_plot_path)

    logger.info(
        "Report generated: %d extra metrics, %d artifacts", len(extra_metrics), len(artifact_paths)
    )
    return extra_metrics, artifact_paths
