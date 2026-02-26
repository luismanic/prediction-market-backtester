"""
favorite_longshot.py  (Backtest v9 — Per-Category Thresholds + Final Exclusions)

Strategy: Fade Overpriced Favorites — Timeframe-Aware, Category-Filtered

═══════════════════════════════════════════════════════════════════════════
BACKTEST #8 FINDINGS
═══════════════════════════════════════════════════════════════════════════

Total PnL: $5,779 across 8,809 markets (avg $0.66/market, 5.0% win rate)

H1 — KXBTCD Daily Isolation: PARTIALLY FAILED
  The hourly filter worked as intended, but the expected improvement in
  win rate and avg/market did not materialize. Root cause: the 100k market
  universe inflates KXBTCD market counts with dozens of un-triggered
  strike prices per day. The 3.0% overall win rate is heavily diluted by
  markets that never reached 85¢. The critical unknown is whether the
  per-filled-trade win rate is still in the 18–22% range seen in late 2024.

  Monthly detail exposed a regime shift:
    Nov–Dec 2024: 22.2–22.4% win rate (clean structural edge)
    Jan–Jun 2025: 0% win rate (dead zone — edge absent)
    Jul–Dec 2025: 1–4% win rate (recovering but weak)
    Jan–Feb 2026: 2.0–2.3% win rate but $1,629 PnL (volume-driven)

  The 18.9% win rate from the #7 daily split may have been a late-2024
  regime artifact. Requires filled-trade-only analysis to confirm.

H2 — KXBTC15M Edge: DEFINITIVELY FAILED
  -$26.30, 1,086 markets, 0% win rate. Zero profitable markets across
  four weeks of data. These ultra-short markets are well-calibrated —
  FLB does not manifest at 15-minute resolution. Permanently excluded.

H3 — Expanded Exclusions: CONFIRMED
  All 13 excluded categories produced exactly $0. Exclusion list locked in.

New losers identified in #8:
  - KXBTC15M:   -$26.30, 0% win rate, 1,086 markets — permanently excluded
  - KXHIGHMIA:   -$4.70, 4.6% win rate, 219 markets — weather markets are
                  well-calibrated, same dynamic as KXRAINNYC

═══════════════════════════════════════════════════════════════════════════
BACKTEST #9 HYPOTHESIS
═══════════════════════════════════════════════════════════════════════════

One primary hypothesis, one diagnostic test:

  H1 — KXBTCD 90¢ threshold recovers the structural edge:
    The 22% win rate in late 2024 required markets to be priced at extreme
    certainty. As BTC markets have matured in 2025–2026, the FLB may only
    manifest at higher price extremes. Raising KXBTCD's threshold from
    85¢ to 90¢ filters to only the most overpriced favorites, potentially
    restoring the high win rate even if fewer trades are taken.

    Prediction: KXBTCD win rate recovers to >10% at 90¢+, with higher
    avg PnL per market. Total KXBTCD PnL may be lower (fewer triggers)
    but risk-adjusted quality improves.

  DIAGNOSTIC — Filled-trades-only analysis:
    Run scripts/filled_trades_analysis.py after the batch to compute win
    rates only for markets where fills_count > 0. This separates the
    true edge signal from universe dilution noise. If KXBTCD filled-trade
    win rate is still 15–22%, the edge is intact and the issue is purely
    analytical. If it is 3–5%, the edge has structurally weakened.

  NOTE: Non-crypto categories remain at 0.85 threshold (unchanged).
  Avoid touching what's working.

═══════════════════════════════════════════════════════════════════════════
CHANGES FROM V8
═══════════════════════════════════════════════════════════════════════════

  1. KXBTC15M added to EXCLUDED_PREFIXES — H2 definitively failed (#8)
  2. KXHIGHMIA added to EXCLUDED_PREFIXES — well-calibrated weather market
  3. category_thresholds dict added — KXBTCD overridden to 0.90
     All other categories continue using favorite_threshold = 0.85
  4. favorite_threshold: 0.85 (unchanged — applies to all non-overridden)
  5. Side: SELL YES (unchanged)
  6. qty: 5.0 (unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pm_bt.common.models import Bar, OrderIntent
from pm_bt.common.types import OrderSide
from pm_bt.strategies.base import FeatureMap

# Settlement hour for KXBTCD daily markets (UTC)
KXBTCD_DAILY_HOUR = "17"


@dataclass(slots=True)
class FavoriteLongshotStrategy:
    """
    Reverse Favorite-Longshot Bias v9 — Per-Category Thresholds.

    Core logic:
      - SELL YES when bar.close >= threshold for that category
      - KXBTCD uses a higher threshold (0.90) to filter for extreme pricing
      - All other categories use the default favorite_threshold (0.85)
      - Skip hourly KXBTCD markets (only trade daily settlement at 17:00)
      - Skip all confirmed losing categories (15 total)
    """

    favorite_threshold: float = 0.85
    qty:                float = 5.0

    # ── Per-category threshold overrides ──────────────────────────────────
    # Only specify categories where you want a DIFFERENT threshold than
    # favorite_threshold. Everything else uses favorite_threshold.
    #
    # KXBTCD @ 0.90: Testing whether the FLB in Bitcoin daily markets only
    # manifests at extreme pricing (≥90¢). Late-2024 produced 22% win rates;
    # 2025-2026 at 85¢ produced near-zero win rates. This tests whether
    # tighter filtering restores quality while reducing quantity.
    category_thresholds: dict = field(default_factory=lambda: {
        "KXBTCD": 0.90,
    })

    # ── Confirmed losers: permanently excluded ─────────────────────────────
    # v6 originals (4):
    #   KXEOWEEK      — crude oil weekly, well-calibrated
    #   KXRAINNYC     — NYC rain, model-accurate weather
    #   KXNETFLIXRANKMOVIE — streaming rankings, efficient
    #   KXSPOTIFYW    — Spotify weekly, efficient
    #
    # v8 additions from Backtest #7 losers (9):
    #   KXNFLMENTION, KXSNFMENTION, INX, INXD, INXU, INXDU,
    #   KXSNLMENTION, KXWTIW, NASDAQ100D
    #
    # v9 additions from Backtest #8 losers (2):
    #   KXBTC15M    — 0% win rate, 1,086 markets, FLB absent at 15m
    #   KXHIGHMIA   — 4.6% win rate, 219 markets, well-calibrated weather
    EXCLUDED_PREFIXES: frozenset = frozenset({
        # v6 exclusions
        "KXEOWEEK",
        "KXRAINNYC",
        "KXNETFLIXRANKMOVIE",
        "KXSPOTIFYW",
        # v8 exclusions
        "KXNFLMENTION",
        "KXSNFMENTION",
        "INX",
        "INXD",
        "INXU",
        "INXDU",
        "KXSNLMENTION",
        "KXWTIW",
        "NASDAQ100D",
        # v9 exclusions
        "KXBTC15M",
        "KXHIGHMIA",
    })

    def _is_kxbtcd_hourly(self, market_id: str) -> bool:
        """
        Returns True if this is a KXBTCD hourly market (should be skipped).

        KXBTCD daily markets settle at hour 17 UTC.
        Format: KXBTCD-26FEB0617-T70749.99
                         [7:9] = settlement hour

        Any settlement hour != 17 is an intraday/hourly market.
        """
        if not market_id.startswith("KXBTCD-"):
            return False
        parts = market_id.split("-")
        if len(parts) < 2:
            return False
        date_hour_segment = parts[1]          # e.g. "26FEB0617"
        if len(date_hour_segment) < 9:
            return False
        settlement_hour = date_hour_segment[7:9]
        return settlement_hour != KXBTCD_DAILY_HOUR

    def _get_threshold(self, market_prefix: str) -> float:
        """
        Returns the entry threshold for a given category prefix.

        Checks category_thresholds first; falls back to favorite_threshold.
        This allows per-category tuning without touching the default.
        """
        return self.category_thresholds.get(market_prefix, self.favorite_threshold)

    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        price = bar.close
        market_prefix = bar.market_id.split("-")[0].upper()

        # ── Filter 1: Skip confirmed losing categories ──
        if market_prefix in self.EXCLUDED_PREFIXES:
            return []

        # ── Filter 2: Skip KXBTCD hourly markets ──
        if self._is_kxbtcd_hourly(bar.market_id):
            return []

        # ── Entry: Sell overpriced favorites (category-aware threshold) ──
        threshold = self._get_threshold(market_prefix)
        if price >= threshold:
            return [
                OrderIntent(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    venue=bar.venue,
                    side=OrderSide.SELL,
                    qty=self.qty,
                    limit_price=price,
                    target_position=-self.qty,
                    reason=f"flb_favorite_reverse_v9",
                )
            ]

        return []