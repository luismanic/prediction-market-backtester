"""
favorite_longshot.py  (Backtest v8 — Hourly Filter + 15M + Expanded Exclusions)

Strategy: Fade Overpriced Favorites — Timeframe-Aware, Category-Filtered

═══════════════════════════════════════════════════════════════════════════
BACKTEST #7 FINDINGS
═══════════════════════════════════════════════════════════════════════════

Total PnL: $5,586 across 8,603 markets (avg $0.65/market, 5.5% win rate)

KXBTCD deep-dive revealed two completely different products mixed together:

  KXBTCD Daily (settlement hour = 17):
    - 753 markets | $2,545 PnL | $3.38/market | 18.9% win rate
    - Consistent 17-23% win rate every single month Nov 2024 → Feb 2026
    - No dead zones — edge is structural, not regime-dependent
    - The real driver of BTC profits

  KXBTCD Hourly (settlement hour ≠ 17):
    - 4,155 markets | -$97 PnL | -$0.02/market | 0.0% win rate
    - Zero profitable markets across 15 months of data
    - Pure noise — retail traders in hourly markets price correctly
      (or the FLB bias doesn't manifest in short-resolution windows)
    - Was contaminating aggregate stats and masking the clean daily edge

  KXBTC15M (15-minute markets, launched Jan 23 2026):
    - NOT in Backtest #7 — trade data not indexed at time of run
    - 6,383 markets in market catalog, volumes up to 291k
    - Hypothesis: ultra-short timeframes may have STRONGER FLB
      (retail traders making rapid binary decisions under time pressure
      are more susceptible to behavioral bias at price extremes)
    - Treat as isolated exploratory test — only ~4 weeks of data

  Non-crypto universe:
    - $3,138 PnL across 3,695 markets | $0.85/market | 8.8% win rate
    - More capital-efficient per market than BTC daily
    - Stable, consistent categories (political mentions, gas prices,
      weather, approval ratings) — not regime-dependent

  Confirmed losers from #7 (expanding exclusion list):
    - KXNFLMENTION:  -$32.90, 0% win rate (7 markets)
    - KXSNFMENTION:  -$27.35, 10% win rate (10 markets, deep losses)
    - INXU:          -$3.70,  2.9% win rate (34 markets)
    - INX:           -$1.90,  0% win rate (119 markets)
    - INXD:          -$1.25,  0.8% win rate (119 markets)
    - INXDU:         -$0.10,  0% win rate (3 markets)
    - KXSNLMENTION:  -$2.20,  0% win rate (7 markets)
    - KXWTIW:        -$4.70,  16.7% win rate but erratic (6 markets)

═══════════════════════════════════════════════════════════════════════════
BACKTEST #8 HYPOTHESIS
═══════════════════════════════════════════════════════════════════════════

Three separate hypotheses being tested simultaneously:

  H1 — KXBTCD Daily isolation:
    The 18.9% win rate and $3.38/market average seen in the daily-only
    split will hold when run cleanly without hourly contamination.
    Removing 4,155 worthless hourly markets should improve aggregate
    stats significantly (avg PnL/market, win rate %) while total PnL
    remains similar to #7.

  H2 — KXBTC15M edge:
    The favorite-longshot bias is amplified in ultra-short markets.
    Retail traders placing 15-minute binary bets under time pressure
    exhibit stronger anchoring and overconfidence at price extremes
    (≥85¢) than in daily markets. If the bias holds, we should see
    a positive win rate despite only 4 weeks of data.
    IMPORTANT: This is exploratory only — insufficient data to draw
    conclusions. Any result here requires confirmation in Backtest #9
    after more history accumulates.

  H3 — Expanded exclusions improve risk-adjusted returns:
    Removing 9 confirmed losing categories (adding INX series, NFL/SNF
    mentions, SNL mentions, crude oil weekly to the existing 4 exclusions)
    will reduce total losses, improve the Sharpe-equivalent metric, and
    prove these categories have genuinely different dynamics where the
    FLB doesn't apply (well-calibrated markets, sophisticated participants,
    or mean-reverting underlyings).

═══════════════════════════════════════════════════════════════════════════
CHANGES FROM V6/V7
═══════════════════════════════════════════════════════════════════════════

  1. Added KXBTCD hourly filter — skips any KXBTCD market where the
     settlement hour (characters 7-8 of the date segment) is not "17".
     This is the single most important change.

  2. Added KXBTC15M explicit pass-through — no filtering applied,
     treated as a standard series for the strategy to evaluate.

  3. Expanded EXCLUDED_PREFIXES from 4 to 13 categories based on
     confirmed multi-backtest losing patterns from #7.

  4. favorite_threshold: 0.85 (unchanged)
  5. Side: SELL YES (unchanged)
  6. qty: 5.0 (unchanged)

═══════════════════════════════════════════════════════════════════════════
EXPECTED OUTCOMES
═══════════════════════════════════════════════════════════════════════════

  - Total PnL:       ~$5,500-6,000 (similar to #7, noise removed)
  - Avg PnL/market:  >$1.00 (up from $0.65 — hourly drag removed)
  - Overall win rate: >8% (up from 5.5%)
  - KXBTCD daily:    ~$2,500+ | ~18-19% win rate | $3.00+/market
  - KXBTCD hourly:   $0 (filtered out entirely)
  - KXBTC15M:        Unknown — first look, any positive result is signal
  - Non-crypto:      ~$3,100+ (unchanged, exclusions add marginal cleanup)

  If H1 confirmed → daily KXBTCD is a validated structural edge, ready
  to inform live trading sizing decisions.

  If H2 shows positive KXBTC15M signal → schedule Backtest #9 after
  2-3 more months of 15M data to confirm.

  If H3 confirmed → permanently lock in 13-category exclusion list.
"""

from __future__ import annotations

from dataclasses import dataclass

from pm_bt.common.models import Bar, OrderIntent
from pm_bt.common.types import OrderSide
from pm_bt.strategies.base import FeatureMap

# Settlement hour for KXBTCD daily markets (UTC)
# All other hours are hourly markets — excluded from strategy
KXBTCD_DAILY_HOUR = "17"


@dataclass(slots=True)
class FavoriteLongshotStrategy:
    """
    Reverse Favorite-Longshot Bias v8 — Timeframe-Aware, Category-Filtered.

    Core logic:
      - SELL YES when bar.close >= favorite_threshold
      - Skip hourly KXBTCD markets (only trade daily settlement at 17:00)
      - Skip all confirmed losing categories

    KXBTCD market ID format:
      KXBTCD-26FEB0617-T70749.99
             ^^^^^^^^ date+hour segment
               26FEB06 = date, 17 = settlement hour (UTC)
      Characters [7:9] of the date segment give the 2-digit hour.

    KXBTC15M market ID format:
      KXBTC15M-26FEB181715-15
      No filtering applied — all 15M markets are evaluated.
    """

    favorite_threshold: float = 0.85
    qty:                float = 5.0

    # ── Confirmed losers: 0% or near-0% win rate across multiple backtests ──
    # Original 4 from v6:
    #   KXEOWEEK      — crude oil weekly, well-calibrated, 0% win rate
    #   KXRAINNYC     — NYC rain, model-accurate, 0% win rate
    #   KXNETFLIXRANKMOVIE — streaming rankings, efficient, 0% win rate
    #   KXSPOTIFYW    — Spotify weekly, efficient, 0% win rate
    #
    # New additions from Backtest #7:
    #   KXNFLMENTION  — NFL mentions: -$32.90, 0% win rate
    #   KXSNFMENTION  — SNF mentions: -$27.35, erratic losses
    #   INX           — S&P 500:      -$1.90,  0% win rate
    #   INXD          — S&P 500 down: -$1.25,  0.8% win rate
    #   INXU          — S&P 500 up:   -$3.70,  2.9% win rate
    #   INXDU         — S&P 500 du:   -$0.10,  0% win rate
    #   KXSNLMENTION  — SNL mentions: -$2.20,  0% win rate
    #   KXWTIW        — crude oil wk: -$4.70,  inconsistent
    #   NASDAQ100D    — Nasdaq down:  +$1.20,  0.7% win rate (near zero)
    EXCLUDED_PREFIXES: frozenset = frozenset({
        # Original exclusions (v6)
        "KXEOWEEK",
        "KXRAINNYC",
        "KXNETFLIXRANKMOVIE",
        "KXSPOTIFYW",
        # New exclusions (v8) — confirmed losers from Backtest #7
        "KXNFLMENTION",
        "KXSNFMENTION",
        "INX",
        "INXD",
        "INXU",
        "INXDU",
        "KXSNLMENTION",
        "KXWTIW",
        "NASDAQ100D",
    })

    def _is_kxbtcd_hourly(self, market_id: str) -> bool:
        """
        Returns True if this is a KXBTCD hourly market (should be skipped).

        KXBTCD daily markets settle at hour 17 UTC.
        Format: KXBTCD-26FEB0617-T70749.99
                         ^^^^^^^
                         [0:7] = date (26FEB06)
                         [7:9] = hour (17)

        Any KXBTCD market with settlement hour != 17 is an hourly market.
        """
        if not market_id.startswith("KXBTCD-"):
            return False
        parts = market_id.split("-")
        if len(parts) < 2:
            return False
        date_hour_segment = parts[1]          # e.g. "26FEB0617"
        if len(date_hour_segment) < 9:
            return False
        settlement_hour = date_hour_segment[7:9]  # e.g. "17"
        return settlement_hour != KXBTCD_DAILY_HOUR

    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        price = bar.close
        market_prefix = bar.market_id.split("-")[0].upper()

        # ── Filter 1: Skip confirmed losing categories ──
        if market_prefix in self.EXCLUDED_PREFIXES:
            return []

        # ── Filter 2: Skip KXBTCD hourly markets ──
        # Only trade KXBTCD when settlement hour == 17 (daily markets)
        if self._is_kxbtcd_hourly(bar.market_id):
            return []

        # ── Entry: Sell overpriced favorites ──
        if price >= self.favorite_threshold:
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
                    reason="flb_favorite_reverse_v8",
                )
            ]

        return []