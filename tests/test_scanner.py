from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from pydantic import TypeAdapter, ValidationError

from pm_bt.cli import run_cli
from pm_bt.common.types import AlertSeverity, Venue
from pm_bt.scanner.checks.consistency import check_complement_sum, check_mutually_exclusive
from pm_bt.scanner.checks.whale import check_price_impact, check_whale_trades
from pm_bt.scanner.models import Alert, ScannerConfig, make_alert_id
from pm_bt.scanner.output import write_alerts_csv, write_alerts_html, write_alerts_json
from pm_bt.scanner.runner import run_scanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANONICAL_TRADE_SCHEMA = {
    "ts": pl.Datetime("us", "UTC"),
    "market_id": pl.Utf8,
    "outcome_id": pl.Utf8,
    "venue": pl.Utf8,
    "price": pl.Float64,
    "size": pl.Float64,
    "side": pl.Utf8,
    "trade_id": pl.Utf8,
    "fee_paid": pl.Float64,
}

_CANONICAL_MARKET_SCHEMA = {
    "market_id": pl.Utf8,
    "venue": pl.Utf8,
    "outcome_id": pl.Utf8,
    "question": pl.Utf8,
    "category": pl.Utf8,
    "close_ts": pl.Datetime("us", "UTC"),
    "resolved": pl.Boolean,
    "winning_outcome": pl.Utf8,
    "resolved_ts": pl.Datetime("us", "UTC"),
    "market_structure": pl.Utf8,
}

_JSON_ROWS_ADAPTER = TypeAdapter(list[dict[str, object]])


def _make_trades(rows: list[dict[str, object]]) -> pl.LazyFrame:
    return pl.DataFrame(rows, schema=_CANONICAL_TRADE_SCHEMA).lazy()


def _make_markets(rows: list[dict[str, object]]) -> pl.LazyFrame:
    return pl.DataFrame(rows, schema=_CANONICAL_MARKET_SCHEMA).lazy()


def _load_json_rows(path: Path) -> list[dict[str, object]]:
    return _JSON_ROWS_ADAPTER.validate_json(path.read_text("utf-8"))


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_alert_id_is_deterministic(self) -> None:
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        id1 = make_alert_id("whale_trade", "KX-A", ts)
        id2 = make_alert_id("whale_trade", "KX-A", ts)
        assert id1 == id2

    def test_scanner_config_defaults(self) -> None:
        cfg = ScannerConfig()
        assert cfg.complement_sum_tolerance == 0.05
        assert cfg.whale_size_multiplier == 20.0
        assert cfg.impact_score_threshold == 0.05
        assert cfg.emit_html is False

    def test_scanner_config_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _ = ScannerConfig.model_validate({"unknown_field": 1})

    def test_scanner_config_tolerance_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _ = ScannerConfig(complement_sum_tolerance=0.0)
        with pytest.raises(ValidationError):
            _ = ScannerConfig(complement_sum_tolerance=1.5)


# ---------------------------------------------------------------------------
# Complement sum check
# ---------------------------------------------------------------------------

_T = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
_T1 = datetime(2026, 1, 1, 9, 1, tzinfo=UTC)


def _complement_violation_trades() -> pl.LazyFrame:
    """Market M1: yes_vwap = 0.70, no_vwap = 0.70 → sum = 1.40, deviation = 0.40."""
    return _make_trades(
        [
            {
                "ts": _T,
                "market_id": "M1",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.70,
                "size": 100.0,
                "side": "buy",
                "trade_id": "t1",
                "fee_paid": 0.0,
            },
            {
                "ts": _T1,
                "market_id": "M1",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.70,
                "size": 100.0,
                "side": "buy",
                "trade_id": "t2",
                "fee_paid": 0.0,
            },
            {
                "ts": _T,
                "market_id": "M1",
                "outcome_id": "no",
                "venue": "kalshi",
                "price": 0.70,
                "size": 100.0,
                "side": "sell",
                "trade_id": "t3",
                "fee_paid": 0.0,
            },
            {
                "ts": _T1,
                "market_id": "M1",
                "outcome_id": "no",
                "venue": "kalshi",
                "price": 0.70,
                "size": 100.0,
                "side": "sell",
                "trade_id": "t4",
                "fee_paid": 0.0,
            },
        ]
    )


def _clean_complement_trades() -> pl.LazyFrame:
    """Market M2: yes = 0.60, no = 0.40 → sum = 1.00, no alert."""
    return _make_trades(
        [
            {
                "ts": _T,
                "market_id": "M2",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.60,
                "size": 100.0,
                "side": "buy",
                "trade_id": "t5",
                "fee_paid": 0.0,
            },
            {
                "ts": _T,
                "market_id": "M2",
                "outcome_id": "no",
                "venue": "kalshi",
                "price": 0.40,
                "size": 100.0,
                "side": "sell",
                "trade_id": "t6",
                "fee_paid": 0.0,
            },
        ]
    )


class TestComplementSum:
    def test_fires_when_deviation_exceeds_tolerance(self) -> None:
        alerts = check_complement_sum(_complement_violation_trades(), tolerance=0.05)
        assert len(alerts) >= 1
        assert alerts[0].reason == "complement_sum_deviation"

    def test_silent_when_only_yes_outcome(self) -> None:
        """Single-outcome markets (e.g. Kalshi with outcome_id='yes' only)."""
        trades = _make_trades(
            [
                {
                    "ts": _T,
                    "market_id": "M1",
                    "outcome_id": "yes",
                    "venue": "kalshi",
                    "price": 0.70,
                    "size": 100.0,
                    "side": "buy",
                    "trade_id": "t1",
                    "fee_paid": 0.0,
                },
            ]
        )
        alerts = check_complement_sum(trades, tolerance=0.05)
        assert alerts == []

    def test_silent_for_clean_data(self) -> None:
        alerts = check_complement_sum(_clean_complement_trades(), tolerance=0.05)
        assert alerts == []

    def test_severity_high_when_deviation_exceeds_2x(self) -> None:
        alerts = check_complement_sum(_complement_violation_trades(), tolerance=0.05)
        # deviation is 0.40, 2*0.05=0.10 → HIGH
        assert any(a.severity == AlertSeverity.HIGH for a in alerts)

    def test_alert_fields_complete(self) -> None:
        alerts = check_complement_sum(_complement_violation_trades(), tolerance=0.05)
        a = alerts[0]
        assert a.market_id == "M1"
        assert a.venue == Venue.KALSHI
        assert "yes_vwap" in a.supporting_stats
        assert "no_vwap" in a.supporting_stats
        assert "deviation" in a.supporting_stats


# ---------------------------------------------------------------------------
# Mutually exclusive check
# ---------------------------------------------------------------------------


def _mutually_exclusive_trades() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Event group KX-EVENT: 3 outcomes each with vwap ≈ 0.50 → sum = 1.50."""
    markets = _make_markets(
        [
            {
                "market_id": "KX-EVENT-A",
                "venue": "kalshi",
                "outcome_id": "yes",
                "question": "A?",
                "category": "test",
                "close_ts": None,
                "resolved": False,
                "winning_outcome": None,
                "resolved_ts": None,
                "market_structure": "clob",
            },
            {
                "market_id": "KX-EVENT-B",
                "venue": "kalshi",
                "outcome_id": "yes",
                "question": "B?",
                "category": "test",
                "close_ts": None,
                "resolved": False,
                "winning_outcome": None,
                "resolved_ts": None,
                "market_structure": "clob",
            },
            {
                "market_id": "KX-EVENT-C",
                "venue": "kalshi",
                "outcome_id": "yes",
                "question": "C?",
                "category": "test",
                "close_ts": None,
                "resolved": False,
                "winning_outcome": None,
                "resolved_ts": None,
                "market_structure": "clob",
            },
        ]
    )
    trades = _make_trades(
        [
            {
                "ts": _T,
                "market_id": "KX-EVENT-A",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.50,
                "size": 100.0,
                "side": "buy",
                "trade_id": "t1",
                "fee_paid": 0.0,
            },
            {
                "ts": _T,
                "market_id": "KX-EVENT-B",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.50,
                "size": 100.0,
                "side": "buy",
                "trade_id": "t2",
                "fee_paid": 0.0,
            },
            {
                "ts": _T,
                "market_id": "KX-EVENT-C",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.50,
                "size": 100.0,
                "side": "buy",
                "trade_id": "t3",
                "fee_paid": 0.0,
            },
        ]
    )
    return markets, trades


class TestMutuallyExclusive:
    def test_fires_when_sum_exceeds_one_plus_tolerance(self) -> None:
        markets, trades = _mutually_exclusive_trades()
        alerts = check_mutually_exclusive(markets, trades, tolerance=0.10)
        assert len(alerts) == 1
        assert alerts[0].reason == "mutually_exclusive_sum"
        assert abs(alerts[0].supporting_stats["outcome_sum"] - 1.50) < 1e-12

    def test_silent_for_single_outcome_groups(self) -> None:
        """A group with only 1 market should never fire."""
        markets = _make_markets(
            [
                {
                    "market_id": "KX-SOLO-A",
                    "venue": "kalshi",
                    "outcome_id": "yes",
                    "question": "?",
                    "category": "test",
                    "close_ts": None,
                    "resolved": False,
                    "winning_outcome": None,
                    "resolved_ts": None,
                    "market_structure": "clob",
                },
            ]
        )
        trades = _make_trades(
            [
                {
                    "ts": _T,
                    "market_id": "KX-SOLO-A",
                    "outcome_id": "yes",
                    "venue": "kalshi",
                    "price": 0.90,
                    "size": 100.0,
                    "side": "buy",
                    "trade_id": "t1",
                    "fee_paid": 0.0,
                },
            ]
        )
        alerts = check_mutually_exclusive(markets, trades, tolerance=0.10)
        assert alerts == []


# ---------------------------------------------------------------------------
# Whale check
# ---------------------------------------------------------------------------


def _whale_trades() -> pl.LazyFrame:
    """5 small trades (size=10), then 1 whale (size=600)."""
    rows: list[dict[str, object]] = []
    for i in range(5):
        rows.append(
            {
                "ts": datetime(2026, 1, 1, 9, i, tzinfo=UTC),
                "market_id": "M1",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.50,
                "size": 10.0,
                "side": "buy",
                "trade_id": f"t{i}",
                "fee_paid": 0.0,
            }
        )
    rows.append(
        {
            "ts": datetime(2026, 1, 1, 9, 10, tzinfo=UTC),
            "market_id": "M1",
            "outcome_id": "yes",
            "venue": "kalshi",
            "price": 0.55,
            "size": 600.0,
            "side": "buy",
            "trade_id": "whale",
            "fee_paid": 0.0,
        }
    )
    return _make_trades(rows)


class TestWhale:
    def test_fires_for_oversized_trade(self) -> None:
        alerts = check_whale_trades(_whale_trades(), rolling_window="1h", size_multiplier=5.0)
        assert len(alerts) >= 1
        assert alerts[0].reason == "whale_trade"

    def test_silent_when_no_trade_exceeds_multiplier(self) -> None:
        """All trades same size → no whale."""
        trades = _make_trades(
            [
                {
                    "ts": datetime(2026, 1, 1, 9, i, tzinfo=UTC),
                    "market_id": "M1",
                    "outcome_id": "yes",
                    "venue": "kalshi",
                    "price": 0.50,
                    "size": 10.0,
                    "side": "buy",
                    "trade_id": f"t{i}",
                    "fee_paid": 0.0,
                }
                for i in range(5)
            ]
        )
        alerts = check_whale_trades(trades, rolling_window="1h", size_multiplier=5.0)
        assert alerts == []

    def test_severity_high_when_ratio_exceeds_10x(self) -> None:
        alerts = check_whale_trades(_whale_trades(), rolling_window="1h", size_multiplier=5.0)
        whale_alert = [a for a in alerts if a.supporting_stats.get("size_ratio", 0) > 10]
        assert whale_alert
        assert whale_alert[0].severity == AlertSeverity.HIGH


# ---------------------------------------------------------------------------
# Price impact check
# ---------------------------------------------------------------------------


def _impact_trades() -> pl.LazyFrame:
    """Trade 2 has large price move (0.05) on small size (1.0) → impact = 0.05."""
    return _make_trades(
        [
            {
                "ts": datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                "market_id": "M1",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.50,
                "size": 10.0,
                "side": "buy",
                "trade_id": "t1",
                "fee_paid": 0.0,
            },
            {
                "ts": datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                "market_id": "M1",
                "outcome_id": "yes",
                "venue": "kalshi",
                "price": 0.55,
                "size": 1.0,
                "side": "buy",
                "trade_id": "t2",
                "fee_paid": 0.0,
            },
        ]
    )


class TestPriceImpact:
    def test_fires_when_impact_exceeds_threshold(self) -> None:
        alerts = check_price_impact(_impact_trades(), impact_threshold=0.01)
        assert len(alerts) == 1
        assert alerts[0].reason == "price_impact"
        assert abs(alerts[0].supporting_stats["impact_score"] - 0.05) < 1e-12

    def test_silent_for_normal_trades(self) -> None:
        trades = _make_trades(
            [
                {
                    "ts": datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                    "market_id": "M1",
                    "outcome_id": "yes",
                    "venue": "kalshi",
                    "price": 0.50,
                    "size": 100.0,
                    "side": "buy",
                    "trade_id": "t1",
                    "fee_paid": 0.0,
                },
                {
                    "ts": datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                    "market_id": "M1",
                    "outcome_id": "yes",
                    "venue": "kalshi",
                    "price": 0.50,
                    "size": 100.0,
                    "side": "buy",
                    "trade_id": "t2",
                    "fee_paid": 0.0,
                },
            ]
        )
        alerts = check_price_impact(trades, impact_threshold=0.01)
        assert alerts == []

    def test_skips_first_trade_per_market(self) -> None:
        """Single trade → no diff → no alert."""
        trades = _make_trades(
            [
                {
                    "ts": _T,
                    "market_id": "M1",
                    "outcome_id": "yes",
                    "venue": "kalshi",
                    "price": 0.50,
                    "size": 1.0,
                    "side": "buy",
                    "trade_id": "t1",
                    "fee_paid": 0.0,
                },
            ]
        )
        alerts = check_price_impact(trades, impact_threshold=0.01)
        assert alerts == []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    def test_deterministic_output(self) -> None:
        trades = _complement_violation_trades()
        markets = _make_markets([])
        cfg = ScannerConfig(complement_sum_tolerance=0.05, whale_size_multiplier=999.0)
        r1 = run_scanner(cfg, trades=trades, markets=markets)
        r2 = run_scanner(cfg, trades=trades, markets=markets)
        assert [a.alert_id for a in r1] == [a.alert_id for a in r2]

    def test_deduplicates_by_alert_id(self) -> None:
        trades = _complement_violation_trades()
        markets = _make_markets([])
        cfg = ScannerConfig(complement_sum_tolerance=0.05, whale_size_multiplier=999.0)
        alerts = run_scanner(cfg, trades=trades, markets=markets)
        ids = [a.alert_id for a in alerts]
        assert len(ids) == len(set(ids))

    def test_empty_for_clean_data(self) -> None:
        trades = _clean_complement_trades()
        markets = _make_markets([])
        cfg = ScannerConfig(
            complement_sum_tolerance=0.50,
            whale_size_multiplier=999.0,
            impact_score_threshold=999.0,
        )
        alerts = run_scanner(cfg, trades=trades, markets=markets)
        assert alerts == []


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _sample_alerts() -> list[Alert]:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return [
        Alert(
            alert_id=make_alert_id("whale_trade", "M1", ts),
            market_id="M1",
            ts=ts,
            venue=Venue.KALSHI,
            reason="whale_trade",
            severity=AlertSeverity.HIGH,
            supporting_stats={"size_ratio": 12.0},
        ),
    ]


class TestOutput:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "alerts.json"
        write_alerts_json(_sample_alerts(), path)
        payload = _load_json_rows(path)
        assert len(payload) == 1
        reason = cast(str, payload[0]["reason"])
        assert reason == "whale_trade"

    def test_json_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "alerts.json"
        write_alerts_json([], path)
        payload = _load_json_rows(path)
        assert payload == []

    def test_csv_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "alerts.csv"
        write_alerts_csv(_sample_alerts(), path)
        df = pl.read_csv(path)
        assert "alert_id" in df.columns
        assert "supporting_stats_json" in df.columns
        assert df.height == 1

    def test_csv_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "alerts.csv"
        write_alerts_csv([], path)
        df = pl.read_csv(path)
        assert df.height == 0
        assert "alert_id" in df.columns

    def test_html_produces_file(self, tmp_path: Path) -> None:
        path = tmp_path / "alerts.html"
        write_alerts_html(_sample_alerts(), path)
        content = path.read_text("utf-8")
        assert "<table>" in content
        assert "whale_trade" in content


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _write_scan_fixtures(data_root: Path) -> None:
    """Write trades with both yes/no outcomes to trigger complement check."""
    trades_dir = data_root / "kalshi" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    markets_dir = data_root / "kalshi" / "markets"
    markets_dir.mkdir(parents=True, exist_ok=True)

    trades = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 3, tzinfo=UTC),
            ],
            "market_id": ["KX-A", "KX-A", "KX-A", "KX-A"],
            "outcome_id": ["yes", "yes", "no", "no"],
            "venue": ["kalshi", "kalshi", "kalshi", "kalshi"],
            "price": [0.70, 0.70, 0.70, 0.70],
            "size": [100.0, 100.0, 100.0, 100.0],
            "side": ["buy", "buy", "sell", "sell"],
            "trade_id": ["t1", "t2", "t3", "t4"],
            "fee_paid": [0.0, 0.0, 0.0, 0.0],
        }
    )
    trades.write_parquet(trades_dir / "trades_0_4.parquet")

    markets = pl.DataFrame(
        {
            "market_id": ["KX-A"],
            "venue": ["kalshi"],
            "outcome_id": ["yes"],
            "question": ["Test?"],
            "category": ["test"],
            "close_ts": [datetime(2026, 1, 2, tzinfo=UTC)],
            "resolved": [False],
            "winning_outcome": [None],
            "resolved_ts": [None],
            "market_structure": ["clob"],
        }
    )
    markets.write_parquet(markets_dir / "markets_0_1.parquet")


class TestCLIScan:
    def test_produces_alerts_json_and_csv(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        output_dir = tmp_path / "scans"
        _write_scan_fixtures(data_root)

        exit_code = run_cli(
            [
                "scan",
                "--venues",
                "kalshi",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
            ]
        )

        assert exit_code == 0
        assert (output_dir / "alerts.json").exists()
        assert (output_dir / "alerts.csv").exists()

    def test_returns_nonzero_when_data_missing(self, tmp_path: Path) -> None:
        exit_code = run_cli(
            [
                "scan",
                "--venues",
                "kalshi",
                "--data-root",
                str(tmp_path / "nodata"),
                "--output-dir",
                str(tmp_path / "scans"),
            ]
        )
        assert exit_code == 1

    def test_emit_html_flag(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        output_dir = tmp_path / "scans"
        _write_scan_fixtures(data_root)

        exit_code = run_cli(
            [
                "scan",
                "--venues",
                "kalshi",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
                "--emit-html",
            ]
        )
        assert exit_code == 0
        assert (output_dir / "alerts.html").exists()
