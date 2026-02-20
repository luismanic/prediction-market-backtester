from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TypedDict, cast

import polars as pl
import yaml  # type: ignore[import-untyped]

from pm_bt.backtest import BacktestEngine
from pm_bt.common.models import BacktestConfig
from pm_bt.common.types import Venue
from pm_bt.common.utils import make_run_id, parse_ts_utc, safe_mkdir
from pm_bt.data import (
    compute_tradability_metrics,
    evaluate_market_trade_quality,
    load_markets,
    load_trades,
)
from pm_bt.reporting import (
    compute_calibration_metrics_from_fills_and_markets,
    generate_report,
)
from pm_bt.scanner import (
    ScannerConfig,
    run_scanner,
    write_alerts_csv,
    write_alerts_html,
    write_alerts_json,
)
from pm_bt.strategies import EventThresholdStrategy, MeanReversionStrategy, MomentumStrategy
from pm_bt.strategies.favorite_longshot import FavoriteLongshotStrategy
from pm_bt.strategies.base import Strategy

logger = logging.getLogger(__name__)

StrategyFactory = Callable[..., Strategy]


class RunPersistencePayload(TypedDict):
    run_id: str
    run_dir: str
    results_path: str
    trading_metrics: dict[str, float]
    forecasting_metrics: dict[str, float]


_STRATEGY_REGISTRY: dict[str, StrategyFactory] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "event_threshold": EventThresholdStrategy,
    "favorite_longshot": FavoriteLongshotStrategy,
}

_DEFAULT_CONFIG_PATHS: dict[str, Path] = {
    "momentum": Path("configs/momentum/default.yaml"),
    "mean_reversion": Path("configs/mean_reversion/default.yaml"),
    "event_threshold": Path("configs/event_threshold/default.yaml"),
    "favorite_longshot": Path("configs/favorite_longshot/default.yaml"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_str_object_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    mapping = cast(Mapping[object, object], value)
    output: dict[str, object] = {}
    for key, mapped_value in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        output[key] = mapped_value
    return output


def _load_yaml_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    return _as_str_object_mapping(payload, field_name="config")


def _resolve_config_path(args: argparse.Namespace) -> Path:
    explicit = cast(str | None, getattr(args, "config", None))
    if explicit:
        return Path(explicit)
    strategy_name = cast(str, args.strategy)
    default_path = _DEFAULT_CONFIG_PATHS.get(strategy_name)
    if default_path is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return default_path


def _build_strategy(strategy_name: str, strategy_params: Mapping[str, object]) -> Strategy:
    factory = _STRATEGY_REGISTRY.get(strategy_name)
    if factory is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    try:
        return factory(**strategy_params)
    except TypeError as exc:
        raise ValueError(f"Invalid strategy parameters for '{strategy_name}': {exc}") from exc


def _apply_cli_overrides(
    *,
    config_data: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    name = cast(str | None, getattr(args, "name", None))
    start_ts = cast(str | None, getattr(args, "start_ts", None))
    end_ts = cast(str | None, getattr(args, "end_ts", None))
    bar_timeframe = cast(str | None, getattr(args, "bar_timeframe", None))

    config_data["strategy_name"] = cast(str, args.strategy)
    config_data["data_root"] = Path(cast(str, args.data_root))
    config_data["output_root"] = Path(cast(str, args.output_root))
    if name:
        config_data["name"] = name
    if start_ts:
        config_data["start_ts"] = parse_ts_utc(start_ts)
    if end_ts:
        config_data["end_ts"] = parse_ts_utc(end_ts)
    if bar_timeframe:
        config_data["bar_timeframe"] = bar_timeframe
    config_data["strategy_params"] = _as_str_object_mapping(
        config_data.get("strategy_params"),
        field_name="strategy_params",
    )
    return config_data


def _build_backtest_config(
    args: argparse.Namespace,
    yaml_config: Mapping[str, object],
) -> BacktestConfig:
    config_data = _apply_cli_overrides(config_data=dict(yaml_config), args=args)
    config_data["venue"] = Venue(cast(str, args.venue))
    config_data["market_id"] = cast(str, args.market)
    return BacktestConfig.model_validate(config_data)


def _build_market_batch_config(
    *,
    args: argparse.Namespace,
    yaml_config: Mapping[str, object],
    venue: Venue,
    market_id: str,
) -> BacktestConfig:
    config_data = _apply_cli_overrides(config_data=dict(yaml_config), args=args)
    config_data["venue"] = venue
    config_data["market_id"] = market_id
    return BacktestConfig.model_validate(config_data)


def _compute_forecasting_metrics(config: BacktestConfig, fills: pl.DataFrame) -> dict[str, float]:
    if config.venue is None:
        return {}
    try:
        markets = load_markets(
            config.venue,
            data_root=config.data_root,
            market_id=config.market_id,
            start_ts=config.start_ts,
            end_ts=config.end_ts,
        ).collect()
    except FileNotFoundError:
        logger.warning("Market metadata not found; forecasting metrics skipped")
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load markets for forecasting metrics: %s", exc)
        return {}
    return compute_calibration_metrics_from_fills_and_markets(fills, markets)


def _persist_single_run(
    config: BacktestConfig,
    run_dir: Path | None,
    *,
    argv_run: bool,
    skip_plots: bool = False,
) -> RunPersistencePayload:
    strategy = _build_strategy(config.strategy_name, config.strategy_params)
    artifacts = BacktestEngine(config=config, strategy=strategy).run()

    resolved_run_dir = run_dir or (config.output_root / artifacts.run_result.run_id)
    run_dir = resolved_run_dir
    safe_mkdir(run_dir)
    results_path = run_dir / "results.json"
    equity_path = run_dir / "equity.csv"
    trades_path = run_dir / "trades.csv"

    artifacts.equity_curve.write_csv(equity_path)
    artifacts.fills.write_csv(trades_path)
    artifacts.run_result.artifacts = {
        "results_json": str(results_path),
        "equity_csv": str(equity_path),
        "trades_csv": str(trades_path),
    }

    reporting_t0 = perf_counter()
    extra_metrics, report_artifacts = generate_report(
        artifacts, run_dir, config.bar_timeframe, skip_plots=skip_plots
    )
    forecasting_metrics = _compute_forecasting_metrics(config, artifacts.fills)
    artifacts.run_result.timings.reporting_s = perf_counter() - reporting_t0

    artifacts.run_result.trading_metrics.update(extra_metrics)
    artifacts.run_result.forecasting_metrics.update(forecasting_metrics)
    artifacts.run_result.artifacts.update(report_artifacts)

    results_payload = artifacts.run_result.model_dump(mode="json")
    _ = results_path.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if argv_run:
        logger.info("Backtest completed. Artifacts written to %s", run_dir)
    return {
        "run_id": artifacts.run_result.run_id,
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "trading_metrics": artifacts.run_result.trading_metrics,
        "forecasting_metrics": artifacts.run_result.forecasting_metrics,
    }


# ---------------------------------------------------------------------------
# Top-level worker function for ProcessPoolExecutor (must NOT be nested)
# ---------------------------------------------------------------------------

def _process_one_market(
    venue_str: str,
    market_id: str,
    data_root_str: str,
    runs_dir_str: str,
    checkpoint_path_str: str,
    yaml_config: dict[str, object],
    min_trades: int,
    max_null_rate: float,
    max_gap_minutes: float,
    start_ts: datetime | None,
    end_ts: datetime | None,
    strategy_name: str,
    bar_timeframe: str,
    skip_plots: bool,
    output_root_str: str,
    run_name: str,
) -> dict[str, object]:
    """
    Worker function — runs in a separate process for each market.
    Returns a checkpoint_entry dict. All Path objects are passed as strings
    because Path objects are not always picklable across processes.
    """
    venue = Venue(venue_str)
    data_root = Path(data_root_str)
    runs_dir = Path(runs_dir_str)

    quality_payload: dict[str, object] = {}

    # Load trades
    try:
        trades_df = load_trades(
            venue,
            data_root=data_root,
            market_id=market_id,
            start_ts=start_ts,
            end_ts=end_ts,
        ).collect()
    except Exception as exc:  # noqa: BLE001
        return {
            "venue": venue_str,
            "market_id": market_id,
            "status": "failed",
            "error": str(exc),
        }

    # Quality gate
    quality = evaluate_market_trade_quality(
        trades_df,
        venue=venue_str,
        market_id=market_id,
        min_trade_count=min_trades,
        max_null_rate=max_null_rate,
        max_gap_minutes=max_gap_minutes,
    )
    quality_payload = quality.as_dict()

    if not quality.passes:
        return {
            "venue": venue_str,
            "market_id": market_id,
            "status": "skipped_quality",
            "quality": quality_payload,
        }

    # Build config and run backtest
    try:
        config = BacktestConfig.model_validate({
            "strategy_name": strategy_name,
            "strategy_params": _as_str_object_mapping(
                yaml_config.get("strategy_params"), field_name="strategy_params"
            ),
            "venue": venue,
            "market_id": market_id,
            "data_root": data_root,
            "output_root": Path(output_root_str),
            "name": run_name,
            "bar_timeframe": bar_timeframe,
            **{
                k: v for k, v in yaml_config.items()
                if k not in ("strategy_name", "strategy_params")
            },
        })

        run_dir = runs_dir / make_run_id()
        run_payload = _persist_single_run(
            config, run_dir, argv_run=False, skip_plots=skip_plots
        )

        summary_row: dict[str, object] = {
            "venue": venue_str,
            "market_id": market_id,
            "status": "completed",
            "run_id": run_payload["run_id"],
            "results_json": run_payload["results_path"],
        }
        summary_row.update(run_payload["trading_metrics"])
        summary_row.update(run_payload["forecasting_metrics"])
        summary_row.update(compute_tradability_metrics(trades_df))

        return {
            "venue": venue_str,
            "market_id": market_id,
            "status": "completed",
            "quality": quality_payload,
            "summary": summary_row,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "venue": venue_str,
            "market_id": market_id,
            "status": "failed",
            "quality": quality_payload,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _run_backtest(args: argparse.Namespace) -> int:
    try:
        config_path = _resolve_config_path(args)
        yaml_config = _load_yaml_config(config_path)
        config = _build_backtest_config(args, yaml_config)
        _ = _persist_single_run(config, None, argv_run=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backtest failed: %s", exc)
        return 1


def _resolve_batch_venues(args: argparse.Namespace) -> list[Venue]:
    raw = cast(list[str] | None, getattr(args, "venues", None))
    if raw is None or len(raw) == 0:
        return [Venue.KALSHI, Venue.POLYMARKET]
    return [Venue(v) for v in raw]


def _top_markets_by_volume(
    *,
    venue: Venue,
    data_root: Path,
    top_n: int,
    start_ts: datetime | None,
    end_ts: datetime | None,
) -> list[str]:
    ranks = (
        load_trades(
            venue,
            data_root=data_root,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        .group_by("market_id")
        .agg(
            pl.col("size").sum().alias("volume_total"),
            pl.len().alias("trade_count"),
        )
        .sort(["volume_total", "trade_count"], descending=[True, True])
        .head(top_n)
        .collect()
    )
    return cast(list[str], ranks["market_id"].to_list())


def _load_batch_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    raw_entries = payload.get("entries", [])
    if isinstance(raw_entries, list):
        entries: list[dict[str, object]] = []
        for item in cast(list[object], raw_entries):
            if isinstance(item, dict):
                entries.append(cast(dict[str, object], item))
        return entries
    return []


def _write_batch_checkpoint(path: Path, entries: list[dict[str, object]]) -> None:
    payload = {"entries": entries}
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_batch_dir(*, output_root: Path, batch_name: str, resume: bool) -> Path:
    if resume:
        existing = sorted(path for path in output_root.glob(f"{batch_name}_*") if path.is_dir())
        if existing:
            return existing[-1]
    return output_root / f"{batch_name}_{make_run_id()}"


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def _run_batch(args: argparse.Namespace) -> int:
    try:
        config_path = _resolve_config_path(args)
        yaml_config = _load_yaml_config(config_path)
        venues = _resolve_batch_venues(args)
        data_root = Path(cast(str, args.data_root))
        output_root = Path(cast(str, args.output_root))
        safe_mkdir(output_root)
        batch_name = cast(str | None, args.name) or "batch"
        batch_dir = _resolve_batch_dir(
            output_root=output_root,
            batch_name=batch_name,
            resume=cast(bool, args.resume),
        )
        safe_mkdir(batch_dir)
        runs_dir = batch_dir / "runs"
        safe_mkdir(runs_dir)

        checkpoint_path = batch_dir / "checkpoint.json"
        checkpoint_entries = (
            _load_batch_checkpoint(checkpoint_path) if cast(bool, args.resume) else []
        )
        processed = {
            (
                cast(str, entry.get("venue", "")),
                cast(str, entry.get("market_id", "")),
            )
            for entry in checkpoint_entries
        }

        all_entries = list(checkpoint_entries)
        min_trades = int(cast(int, args.min_trades))
        max_null_rate = float(cast(float, args.max_null_rate))
        max_gap_minutes = float(cast(float, args.max_gap_minutes))
        start_ts = (
            parse_ts_utc(cast(str, args.start_ts)) if getattr(args, "start_ts", None) else None
        )
        end_ts = parse_ts_utc(cast(str, args.end_ts)) if getattr(args, "end_ts", None) else None
        workers = int(cast(int, getattr(args, "workers", 1)))
        skip_plots = cast(bool, getattr(args, "no_plots", False))
        bar_timeframe = cast(str | None, getattr(args, "bar_timeframe", None)) or "1h"
        strategy_name = cast(str, args.strategy)
        run_name = cast(str | None, args.name) or "batch"

        # Build list of markets to process
        markets_to_run: list[tuple[str, str]] = []
        for venue in venues:
            try:
                markets = _top_markets_by_volume(
                    venue=venue,
                    data_root=data_root,
                    top_n=int(cast(int, args.top_n)),
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping venue %s: unable to rank markets (%s)", venue.value, exc)
                continue
            for market_id in markets:
                key = (venue.value, market_id)
                if key not in processed:
                    markets_to_run.append((venue.value, market_id))

        logger.info("Markets to process: %d (workers=%d)", len(markets_to_run), workers)

        # Shared kwargs passed to every worker
        worker_kwargs = dict(
            data_root_str=str(data_root),
            runs_dir_str=str(runs_dir),
            checkpoint_path_str=str(checkpoint_path),
            yaml_config=dict(yaml_config),
            min_trades=min_trades,
            max_null_rate=max_null_rate,
            max_gap_minutes=max_gap_minutes,
            start_ts=start_ts,
            end_ts=end_ts,
            strategy_name=strategy_name,
            bar_timeframe=bar_timeframe,
            skip_plots=skip_plots,
            output_root_str=str(output_root),
            run_name=run_name,
        )

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_market,
                    venue_str,
                    market_id,
                    **worker_kwargs,
                ): (venue_str, market_id)
                for venue_str, market_id in markets_to_run
            }

            for future in as_completed(futures):
                venue_str, market_id = futures[future]
                try:
                    entry = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Worker crashed for %s/%s", venue_str, market_id)
                    entry = {
                        "venue": venue_str,
                        "market_id": market_id,
                        "status": "failed",
                        "error": str(exc),
                    }

                status = cast(str, entry.get("status", "unknown"))
                if status == "skipped_quality":
                    logger.warning(
                        "Skipping %s/%s due to quality gate: %s",
                        venue_str,
                        market_id,
                        cast(dict, entry.get("quality", {})).get("failed_checks", ""),
                    )
                elif status == "completed":
                    logger.info("Batch completed market %s/%s", venue_str, market_id)
                else:
                    logger.error(
                        "Batch market failed: %s/%s — %s",
                        venue_str,
                        market_id,
                        entry.get("error", ""),
                    )

                all_entries.append(entry)
                _write_batch_checkpoint(checkpoint_path, all_entries)
                processed.add((venue_str, market_id))

        # Write summary.csv
        summary_rows = [
            cast(dict[str, object], entry["summary"])
            for entry in all_entries
            if cast(str, entry.get("status")) == "completed" and "summary" in entry
        ]
        summary_path = batch_dir / "summary.csv"
        if summary_rows:
            summary_df = pl.DataFrame(summary_rows).sort(
                ["total_pnl", "volume_total"], descending=[True, True]
            )
            summary_df.write_csv(summary_path)
        else:
            pl.DataFrame(schema=[("venue", pl.Utf8), ("market_id", pl.Utf8)]).write_csv(
                summary_path
            )

        quality_rows = [
            cast(dict[str, object], entry["quality"])
            for entry in all_entries
            if "quality" in entry
        ]
        quality_path = batch_dir / "data_quality.json"
        _ = quality_path.write_text(
            json.dumps({"entries": quality_rows}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        logger.info("Batch complete. Summary: %s", summary_path)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch failed: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _build_scan_config(args: argparse.Namespace) -> ScannerConfig:
    config_path = cast(str | None, getattr(args, "config", None))
    config_data: dict[str, object] = {}
    if config_path:
        config_data = _load_yaml_config(Path(config_path))

    config_data["data_root"] = Path(cast(str, args.data_root))
    config_data["output_dir"] = Path(cast(str, args.output_dir))

    raw_venues = cast(list[str] | None, getattr(args, "venues", None))
    if raw_venues:
        config_data["venues"] = [Venue(v) for v in raw_venues]

    raw_market_ids = cast(list[str] | None, getattr(args, "market_ids", None))
    if raw_market_ids:
        config_data["market_ids"] = raw_market_ids

    start_ts = cast(str | None, getattr(args, "start_ts", None))
    end_ts = cast(str | None, getattr(args, "end_ts", None))
    if start_ts:
        config_data["start_ts"] = parse_ts_utc(start_ts)
    if end_ts:
        config_data["end_ts"] = parse_ts_utc(end_ts)

    for cli_key in (
        "top_n",
        "complement_sum_tolerance",
        "mutually_exclusive_tolerance",
        "whale_rolling_window",
        "whale_size_multiplier",
        "impact_score_threshold",
    ):
        val = getattr(args, cli_key, None)
        if val is not None:
            config_data[cli_key] = val

    if getattr(args, "emit_html", False):
        config_data["emit_html"] = True

    return ScannerConfig.model_validate(config_data)


def _run_scan(args: argparse.Namespace) -> int:
    try:
        config = _build_scan_config(args)
        safe_mkdir(config.output_dir)
        alerts = run_scanner(config)
        write_alerts_json(alerts, config.output_dir / "alerts.json")
        write_alerts_csv(alerts, config.output_dir / "alerts.csv")
        if config.emit_html:
            write_alerts_html(alerts, config.output_dir / "alerts.html")
        logger.info("Scan complete. %d alert(s) written to %s", len(alerts), config.output_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scan failed: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pm-bt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- backtest --
    backtest = subparsers.add_parser("backtest", help="Run a backtest and persist run artifacts")
    _ = backtest.add_argument("--venue", choices=[venue.value for venue in Venue], required=True)
    _ = backtest.add_argument("--market", required=True)
    _ = backtest.add_argument("--strategy", required=True)
    _ = backtest.add_argument("--config", required=False)
    _ = backtest.add_argument("--name", required=False)
    _ = backtest.add_argument("--data-root", default="data")
    _ = backtest.add_argument("--output-root", default="output/runs")
    _ = backtest.add_argument("--start-ts", required=False)
    _ = backtest.add_argument("--end-ts", required=False)
    _ = backtest.add_argument("--bar-timeframe", required=False)
    _ = backtest.set_defaults(handler=_run_backtest)

    # -- batch --
    batch = subparsers.add_parser("batch", help="Run strategy over top-N markets per venue")
    _ = batch.add_argument("--strategy", required=True)
    _ = batch.add_argument("--config", required=False)
    _ = batch.add_argument("--name", required=False)
    _ = batch.add_argument("--data-root", default="data")
    _ = batch.add_argument("--output-root", default="output/runs")
    _ = batch.add_argument("--start-ts", required=False)
    _ = batch.add_argument("--end-ts", required=False)
    _ = batch.add_argument("--bar-timeframe", required=False)
    _ = batch.add_argument("--top-n", type=int, default=50)
    _ = batch.add_argument("--venues", nargs="+", choices=[venue.value for venue in Venue])
    _ = batch.add_argument("--min-trades", type=int, default=20)
    _ = batch.add_argument("--max-null-rate", type=float, default=0.01)
    _ = batch.add_argument("--max-gap-minutes", type=float, default=720.0)
    _ = batch.add_argument("--resume", action="store_true")
    _ = batch.add_argument("--no-plots", action="store_true", default=False,
                           help="Skip per-market plot generation")
    _ = batch.add_argument("--workers", type=int, default=1,
                           help="Number of parallel worker processes (default 1)")
    _ = batch.set_defaults(handler=_run_batch)

    # -- scan --
    scan = subparsers.add_parser("scan", help="Run alpha scanner and produce alerts")
    _ = scan.add_argument("--config", required=False)
    _ = scan.add_argument("--data-root", default="data")
    _ = scan.add_argument("--output-dir", default="output/scans")
    _ = scan.add_argument("--venues", nargs="+", choices=[v.value for v in Venue])
    _ = scan.add_argument("--market-ids", nargs="+")
    _ = scan.add_argument("--top-n", type=int, dest="top_n")
    _ = scan.add_argument("--start-ts", required=False)
    _ = scan.add_argument("--end-ts", required=False)
    _ = scan.add_argument("--complement-sum-tolerance", type=float, dest="complement_sum_tolerance")
    _ = scan.add_argument(
        "--mutually-exclusive-tolerance", type=float, dest="mutually_exclusive_tolerance"
    )
    _ = scan.add_argument("--whale-rolling-window", type=str, dest="whale_rolling_window")
    _ = scan.add_argument("--whale-size-multiplier", type=float, dest="whale_size_multiplier")
    _ = scan.add_argument("--impact-score-threshold", type=float, dest="impact_score_threshold")
    _ = scan.add_argument("--emit-html", action="store_true")
    _ = scan.set_defaults(handler=_run_scan)

    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(run_cli())
    