"""Shared domain models, types, and utility helpers."""

from pm_bt.common.categories import classify_kalshi_event_ticker, kalshi_category_expr
from pm_bt.common.models import (
    BacktestConfig,
    Bar,
    DatasetSlice,
    Fill,
    Market,
    OrderIntent,
    RunResult,
    RunTimings,
    TradeTick,
)
from pm_bt.common.types import (
    MarketId,
    MarketStructure,
    OrderSide,
    OutcomeId,
    Price,
    Probability,
    Quantity,
    ResolutionOutcome,
    RunId,
    Timestamp,
    TradeSide,
    Venue,
)
from pm_bt.common.utils import (
    ensure_utc,
    get_git_commit_hash,
    make_run_id,
    parse_ts_utc,
    safe_mkdir,
)

__all__ = [
    # Categories
    "classify_kalshi_event_ticker",
    "kalshi_category_expr",
    # Models
    "BacktestConfig",
    "Bar",
    "DatasetSlice",
    "Fill",
    "Market",
    "OrderIntent",
    "RunResult",
    "RunTimings",
    "TradeTick",
    # Types / Enums
    "MarketId",
    "MarketStructure",
    "OrderSide",
    "OutcomeId",
    "Price",
    "Probability",
    "Quantity",
    "ResolutionOutcome",
    "RunId",
    "Timestamp",
    "TradeSide",
    "Venue",
    # Utils
    "ensure_utc",
    "get_git_commit_hash",
    "make_run_id",
    "parse_ts_utc",
    "safe_mkdir",
]
