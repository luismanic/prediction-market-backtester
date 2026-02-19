"""
favorite_longshot.py

Favorite-Longshot Bias (FLB) exploitation strategy for Kalshi prediction markets.

Theory
------
Retail traders systematically OVERprice low-probability outcomes and UNDERprice
high-probability outcomes (Snowberg & Wolfers 2010; Whelan 2023).

On Kalshi:
  • A 5¢ YES contract resolves YES only ~2% of the time   → short longshots
  • A 95¢ YES contract resolves YES ~98% of the time      → buy favorites

Entry logic
-----------
  SELL YES (fade the longshot)  when close < longshot_threshold  (default 0.15)
  BUY  YES (back the favorite)  when close > favorite_threshold  (default 0.80)

Position management
-------------------
  • Stateful: enters once per (market_id, outcome_id) — no pyramiding.
  • Once entered, holds to resolution (no stop/limit exit bars).
  • qty is controlled by the config; Kelly sizing is applied at the batch level.

Usage
-----
  pm-bt backtest --venue kalshi --market <TICKER> --strategy favorite_longshot

  pm-bt batch --strategy favorite_longshot \\
              --config configs/favorite_longshot/default.yaml \\
              --venues kalshi --top-n 10000 --output-root output/flb_run
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pm_bt.common.models import Bar, OrderIntent
from pm_bt.common.types import OrderSide
from pm_bt.strategies.base import FeatureMap


@dataclass(slots=True)
class FavoriteLongshotStrategy:
    """
    Exploit the Favorite-Longshot Bias on Kalshi recurring index markets.

    Parameters
    ----------
    longshot_threshold : float
        Sell YES (fade) when the contract's close price is BELOW this value.
        Default 0.15 — anything priced at 15¢ or cheaper is considered a
        systematically overpriced longshot.
    favorite_threshold : float
        Buy YES (back) when the contract's close price is ABOVE this value.
        Default 0.80 — contracts priced at 80¢ or higher are underpriced
        favorites.
    qty : float
        Number of contracts per entry. Use fractional Kelly at the batch level
        (recommended: size each position at ≤ 2-3 % of bankroll).
    """

    longshot_threshold: float = 0.15
    favorite_threshold: float = 0.80
    qty: float = 5.0

    # internal state — markets already entered (not serialised to YAML)
    _entered: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.longshot_threshold < 1.0):
            raise ValueError("longshot_threshold must be in (0, 1)")
        if not (0.0 < self.favorite_threshold < 1.0):
            raise ValueError("favorite_threshold must be in (0, 1)")
        if self.longshot_threshold >= self.favorite_threshold:
            raise ValueError("longshot_threshold must be < favorite_threshold")
        if self.qty <= 0.0:
            raise ValueError("qty must be > 0")

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:  # noqa: ARG002
        key = (bar.market_id, bar.outcome_id)

        # Only enter once per market — hold to resolution
        if key in self._entered:
            return []

        price = bar.close

        if price < self.longshot_threshold:
            # Longshot: the YES contract is overpriced → SELL YES (go short YES / long NO)
            self._entered.add(key)
            return [
                OrderIntent(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    venue=bar.venue,
                    side=OrderSide.SELL,
                    qty=self.qty,
                    reason="flb_sell_longshot_yes",
                )
            ]

        if price > self.favorite_threshold:
            # Favorite: the YES contract is underpriced → BUY YES
            self._entered.add(key)
            return [
                OrderIntent(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    venue=bar.venue,
                    side=OrderSide.BUY,
                    qty=self.qty,
                    reason="flb_buy_favorite_yes",
                )
            ]

        return []