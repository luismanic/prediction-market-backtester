from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field

from pm_bt.common.models import BacktestConfig, DomainModel, Fill, OrderIntent
from pm_bt.common.types import OrderSide

PositionKey = tuple[str, str]


class ExecutionConfig(DomainModel):
    fee_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    slippage_volume_k: float | None = Field(default=None, ge=0.0)
    default_spread: float = Field(default=0.02, ge=0.0, le=1.0)
    latency_bars: int = Field(default=0, ge=0)
    max_position_size: float = Field(default=100.0, gt=0.0)
    max_gross_exposure: float = Field(default=10_000.0, gt=0.0)

    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> ExecutionConfig:
        return cls(
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
            latency_bars=config.latency_bars,
            max_position_size=config.max_position_size,
            max_gross_exposure=config.max_gross_exposure,
        )


class MarketSnapshot(DomainModel):
    """Point-in-time market state used by the execution simulator.

    mid_price is clamped to [0, 1] as prediction market prices represent
    implied probabilities.  spread is intentionally nullable: when absent the
    simulator estimates it from recent_prices or falls back to a default.
    recent_volume is nullable for the same reason.
    """

    ts: datetime
    market_id: str
    outcome_id: str
    mid_price: float = Field(ge=0.0, le=1.0)
    spread: float | None = Field(default=None, ge=0.0, le=1.0)
    recent_volume: float | None = Field(default=None, ge=0.0)
    recent_prices: Sequence[float] | None = None


class AccountState(DomainModel):
    """Mutable account state tracked by the simulator."""

    cash: float = 10_000.0
    positions: dict[PositionKey, float] = Field(default_factory=dict)


@dataclass(slots=True)
class _PendingOrder:
    ready_bar_index: int
    order: OrderIntent


def estimate_spread_from_prices(recent_prices: Sequence[float], *, default_spread: float) -> float:
    """Estimate spread from recent trade prices; fallback to default when not enough data."""
    if len(recent_prices) < 2:
        return default_spread

    diffs = [abs(b - a) for a, b in zip(recent_prices[:-1], recent_prices[1:], strict=False)]
    if not diffs:
        return default_spread

    mean_diff = sum(diffs) / len(diffs)
    # A conservative proxy: twice mean absolute tick-to-tick move.
    estimated = 2.0 * mean_diff
    return min(max(estimated, 0.0), 1.0)


def _clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


class ExecutionSimulator:
    def __init__(
        self,
        config: ExecutionConfig,
        *,
        initial_cash: float = 10_000.0,
    ) -> None:
        self.config: ExecutionConfig = config
        self.state: AccountState = AccountState(cash=initial_cash)
        self._last_prices: dict[PositionKey, float] = {}
        self._pending: list[_PendingOrder] = []

    def current_position(self, market_id: str, outcome_id: str) -> float:
        return self.state.positions.get((market_id, outcome_id), 0.0)

    def current_gross_exposure(self) -> float:
        gross = 0.0
        for key, position in self.state.positions.items():
            reference_price = self._last_prices.get(key, 0.0)
            gross += abs(position) * reference_price
        return gross

    def execute_bar(
        self,
        *,
        bar_index: int,
        snapshot: MarketSnapshot,
        incoming_orders: Sequence[OrderIntent],
    ) -> list[Fill]:
        key = (snapshot.market_id, snapshot.outcome_id)
        self._last_prices[key] = snapshot.mid_price

        for order in incoming_orders:
            self._pending.append(
                _PendingOrder(ready_bar_index=bar_index + self.config.latency_bars, order=order)
            )

        fills: list[Fill] = []
        still_pending: list[_PendingOrder] = []
        for pending in self._pending:
            if pending.ready_bar_index > bar_index:
                still_pending.append(pending)
                continue

            # Keep ready orders for other markets/outcomes queued until their snapshot is processed.
            if (
                pending.order.market_id != snapshot.market_id
                or pending.order.outcome_id != snapshot.outcome_id
            ):
                still_pending.append(pending)
                continue

            fill = self._execute_order(snapshot=snapshot, order=pending.order)
            if fill is not None:
                fills.append(fill)

        self._pending = still_pending
        return fills

    def _execute_order(self, *, snapshot: MarketSnapshot, order: OrderIntent) -> Fill | None:
        if order.market_id != snapshot.market_id or order.outcome_id != snapshot.outcome_id:
            return None

        spread = self._resolve_spread(snapshot)
        base_fill_price = self._base_fill_price(
            mid_price=snapshot.mid_price,
            spread=spread,
            side=order.side,
        )

        allowed_qty = self._allowed_qty(
            key=(order.market_id, order.outcome_id),
            requested_qty=order.qty,
            side=order.side,
            reference_price=base_fill_price,
        )
        if allowed_qty <= 0.0:
            return None

        slippage_per_unit = self._slippage_per_unit(
            qty=allowed_qty,
            reference_price=base_fill_price,
            recent_volume=snapshot.recent_volume,
        )
        fill_price = self._apply_slippage(
            reference_price=base_fill_price,
            side=order.side,
            slippage_per_unit=slippage_per_unit,
        )

        notional = allowed_qty * fill_price
        fees = notional * (self.config.fee_bps / 10_000.0)
        slippage_cost = allowed_qty * abs(fill_price - base_fill_price)

        key = (order.market_id, order.outcome_id)
        previous_position = self.state.positions.get(key, 0.0)

        if order.side == OrderSide.BUY:
            new_position = previous_position + allowed_qty
            self.state.cash -= notional + fees
        else:
            new_position = previous_position - allowed_qty
            self.state.cash += notional - fees

        self.state.positions[key] = new_position
        self._last_prices[key] = snapshot.mid_price

        latency_ms = max(0, int((snapshot.ts - order.ts).total_seconds() * 1000.0))

        return Fill(
            ts_fill=snapshot.ts,
            market_id=order.market_id,
            outcome_id=order.outcome_id,
            venue=order.venue,
            side=order.side,
            qty_filled=allowed_qty,
            price_fill=fill_price,
            fees=fees,
            slippage_cost=slippage_cost,
            latency_ms=latency_ms,
        )

    def _resolve_spread(self, snapshot: MarketSnapshot) -> float:
        if snapshot.spread is not None:
            return max(snapshot.spread, 0.0)
        if snapshot.recent_prices is not None:
            return estimate_spread_from_prices(
                snapshot.recent_prices,
                default_spread=self.config.default_spread,
            )
        return self.config.default_spread

    @staticmethod
    def _base_fill_price(*, mid_price: float, spread: float, side: OrderSide) -> float:
        half_spread = spread / 2.0
        if side == OrderSide.BUY:
            return _clamp_probability(mid_price + half_spread)
        return _clamp_probability(mid_price - half_spread)

    @staticmethod
    def _apply_slippage(
        *,
        reference_price: float,
        side: OrderSide,
        slippage_per_unit: float,
    ) -> float:
        if side == OrderSide.BUY:
            return _clamp_probability(reference_price + slippage_per_unit)
        return _clamp_probability(reference_price - slippage_per_unit)

    def _slippage_per_unit(
        self,
        *,
        qty: float,
        reference_price: float,
        recent_volume: float | None,
    ) -> float:
        if self.config.slippage_volume_k is not None:
            if recent_volume is None or recent_volume <= 0.0:
                return 0.0
            return self.config.slippage_volume_k * (qty / recent_volume)
        if self.config.slippage_bps <= 0.0:
            return 0.0
        return reference_price * (self.config.slippage_bps / 10_000.0)

    def _allowed_qty(
        self,
        *,
        key: PositionKey,
        requested_qty: float,
        side: OrderSide,
        reference_price: float,
    ) -> float:
        current_position = self.state.positions.get(key, 0.0)

        if side == OrderSide.BUY:
            by_position = max(0.0, self.config.max_position_size - current_position)
        else:
            by_position = max(0.0, self.config.max_position_size + current_position)

        max_qty = min(requested_qty, by_position)
        if max_qty <= 0.0:
            return 0.0

        if self.current_gross_exposure() >= self.config.max_gross_exposure:
            return 0.0

        return self._max_qty_under_gross(
            key=key,
            side=side,
            candidate_qty=max_qty,
            reference_price=reference_price,
        )

    def _max_qty_under_gross(
        self,
        *,
        key: PositionKey,
        side: OrderSide,
        candidate_qty: float,
        reference_price: float,
    ) -> float:
        current_position = self.state.positions.get(key, 0.0)

        def gross_with(qty: float) -> float:
            current_abs = abs(current_position)
            if side == OrderSide.BUY:
                new_position = current_position + qty
            else:
                new_position = current_position - qty
            new_abs = abs(new_position)

            base_gross = self.current_gross_exposure()
            return base_gross - (current_abs * reference_price) + (new_abs * reference_price)

        if gross_with(candidate_qty) <= self.config.max_gross_exposure:
            return candidate_qty

        left = 0.0
        right = candidate_qty
        for _ in range(32):
            mid = (left + right) / 2.0
            if gross_with(mid) <= self.config.max_gross_exposure:
                left = mid
            else:
                right = mid

        return max(0.0, left)
