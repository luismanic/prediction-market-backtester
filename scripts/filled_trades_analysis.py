"""
scripts/filled_trades_analysis.py

Filled-Trades-Only Analysis for Favorite-Longshot Backtest Results.

PURPOSE:
  The aggregate summary.csv includes every market processed, even those
  where the strategy never triggered (fills_count = 0). For KXBTCD this
  creates severe universe dilution — dozens of un-triggered strike prices
  per day drag the overall win rate far below the true signal rate.

  This script filters to fills_count > 0 (markets where a trade actually
  executed) and recomputes all key metrics, with special focus on KXBTCD
  monthly detail to assess whether the structural edge remains intact.

USAGE:
  python scripts/filled_trades_analysis.py <path_to_summary_csv>

  Example:
  python scripts/filled_trades_analysis.py \
    "D:/prediction-market-backtester/output/runs/kalshi_flb_10k_v9a_TIMESTAMP/summary.csv"

OUTPUT:
  Prints filled-trades-only stats to console.
  Saves filled_trades_analysis.csv to the same directory as summary.csv.
"""

import sys
import pandas as pd
from pathlib import Path


def extract_category(market_id: str) -> str:
    return market_id.split("-")[0].upper()


def extract_year_month(market_id: str) -> str:
    """
    Extract YYYY-MM from KXBTCD market IDs.
    Format: KXBTCD-26FEB0617-T70749.99
    Date segment: 26FEB06  → year=2026, month=FEB
    """
    try:
        parts = market_id.split("-")
        if len(parts) < 2:
            return "unknown"
        seg = parts[1]  # e.g. "26FEB0617"
        year_2d = seg[0:2]   # "26"
        month_str = seg[2:5] # "FEB"
        year = f"20{year_2d}"
        month_map = {
            "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
            "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
            "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
        }
        month = month_map.get(month_str.upper(), "??")
        return f"{year}-{month}"
    except Exception:
        return "unknown"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/filled_trades_analysis.py <path_to_summary_csv>")
        sys.exit(1)

    summary_path = Path(sys.argv[1])
    if not summary_path.exists():
        print(f"ERROR: File not found: {summary_path}")
        sys.exit(1)

    print(f"\nLoading: {summary_path}")
    df = pd.read_csv(summary_path)

    print(f"Total markets in summary : {len(df):,}")

    # ── Validate required columns ──────────────────────────────────────────
    required = {"market_id", "total_pnl", "fills_count"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # ── Filter to filled trades only ───────────────────────────────────────
    filled = df[df["fills_count"] > 0].copy()
    filled["category"] = filled["market_id"].apply(extract_category)
    filled["profitable"] = filled["total_pnl"] > 0

    print(f"Markets with fills > 0  : {len(filled):,}")
    print(f"Markets with fills = 0  : {len(df) - len(filled):,}")

    # ── OVERALL filled-trades stats ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL FILLED-TRADES STATS")
    print("=" * 70)

    total_pnl    = filled["total_pnl"].sum()
    profitable   = filled["profitable"].sum()
    win_rate     = 100 * profitable / len(filled) if len(filled) > 0 else 0
    avg_pnl      = filled["total_pnl"].mean()

    print(f"Total PnL          : ${total_pnl:,.2f}")
    print(f"Avg PnL / market   : ${avg_pnl:.4f}")
    print(f"Profitable markets : {profitable} / {len(filled)} ({win_rate:.1f}%)")

    # ── PER-CATEGORY filled-trades breakdown ───────────────────────────────
    print("\n" + "=" * 70)
    print("  PER-CATEGORY FILLED-TRADES BREAKDOWN")
    print("=" * 70)

    cat_stats = (
        filled.groupby("category")
        .agg(
            markets=("total_pnl", "count"),
            total_pnl=("total_pnl", "sum"),
            avg_pnl=("total_pnl", "mean"),
            profitable=("profitable", "sum"),
            max_win=("total_pnl", "max"),
            max_loss=("total_pnl", "min"),
        )
        .assign(
            win_rate=lambda x: 100 * x["profitable"] / x["markets"],
            pct_of_total=lambda x: 100 * x["total_pnl"] / total_pnl,
        )
        .sort_values("total_pnl", ascending=False)
    )

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(cat_stats.to_string())

    # ── KXBTCD MONTHLY DETAIL (filled only) ───────────────────────────────
    print("\n" + "=" * 70)
    print("  KXBTCD MONTHLY DETAIL — FILLED TRADES ONLY")
    print("=" * 70)

    kxbtcd_filled = filled[filled["category"] == "KXBTCD"].copy()

    if kxbtcd_filled.empty:
        print("No filled KXBTCD markets found.")
    else:
        kxbtcd_filled["month"] = kxbtcd_filled["market_id"].apply(extract_year_month)

        monthly = (
            kxbtcd_filled.groupby("month")
            .agg(
                markets=("total_pnl", "count"),
                total_pnl=("total_pnl", "sum"),
                avg_pnl=("total_pnl", "mean"),
                profitable=("profitable", "sum"),
            )
            .assign(win_rate=lambda x: 100 * x["profitable"] / x["markets"])
            .sort_index()
        )
        print(monthly.to_string())

        print(f"\nKXBTCD filled totals:")
        print(f"  Markets  : {len(kxbtcd_filled):,}")
        print(f"  Total PnL: ${kxbtcd_filled['total_pnl'].sum():,.2f}")
        print(f"  Win Rate : {100 * kxbtcd_filled['profitable'].sum() / len(kxbtcd_filled):.1f}%")
        print(f"  Avg/mkt  : ${kxbtcd_filled['total_pnl'].mean():.4f}")

    # ── NON-CRYPTO filled-trades stats ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  NON-CRYPTO FILLED-TRADES STATS")
    print("=" * 70)

    crypto_cats = {"KXBTCD", "KXBTC", "KXBTC15M", "KXETH", "KXETHD", "KXETH15M"}
    non_crypto = filled[~filled["category"].isin(crypto_cats)]

    nc_pnl      = non_crypto["total_pnl"].sum()
    nc_prof     = non_crypto["profitable"].sum()
    nc_wr       = 100 * nc_prof / len(non_crypto) if len(non_crypto) > 0 else 0
    nc_avg      = non_crypto["total_pnl"].mean()

    print(f"Total PnL          : ${nc_pnl:,.2f}")
    print(f"Avg PnL / market   : ${nc_avg:.4f}")
    print(f"Profitable markets : {nc_prof} / {len(non_crypto)} ({nc_wr:.1f}%)")

    # ── Save output ────────────────────────────────────────────────────────
    output_path = summary_path.parent / "filled_trades_analysis.csv"
    filled.to_csv(output_path, index=False)
    print(f"\nSaved filled-trades CSV to: {output_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()