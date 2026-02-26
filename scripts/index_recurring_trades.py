"""
index_recurring_trades.py

Indexes trades for the top N recurring Kalshi markets.

Kalshi hierarchy:  Series -> Event -> Market
  series_ticker : KXHIGHNY           (template: "Daily NYC High Temp")
  event_ticker  : KXHIGHNY-25JAN15   (one specific day)
  ticker        : KXHIGHNY-25JAN15-T45  (one tradeable threshold market)

series_ticker is NOT stored in the parquet (the indexer never fetched it),
but it is derivable: strip the date suffix from event_ticker.

Supported date suffix formats (applied in order):
  1. -DDMMMDDDDDD  e.g. KXBTC15M-26FEB181715 -> KXBTC15M  (15-min intraday crypto)
     Pattern: -[0-9]{2}[A-Z]{3}[0-9]{6}$

  2. -DDMMMDDHH    e.g. KXBTCD-26FEB0617  -> KXBTCD  (daily/hourly crypto with hour)
     Pattern: -[0-9]{2}[A-Z]{3}[0-9]{4}$

  3. -DDMMMYY      e.g. KXHIGHNY-25JAN15  -> KXHIGHNY  (weather / standard daily)
     Pattern: -[0-9]{2}[A-Z]{3}[0-9]{2}$

IMPORTANT: Pass 1 must run before Pass 2. The 6-digit 15M suffix (181715)
would be partially matched by the 4-digit pattern (1817), producing a wrong
series key like KXBTC15M-26FEB15 instead of KXBTC15M.

NOTE: The regex is inlined directly into each f-string using {{2}} escaping.
Do NOT extract it into a module-level string constant. A pre-built string
embedded in an f-string does not have {{}} unescaped, so DuckDB receives
literal {{2}} and throws an invalid regex error.

We group by derived series_ticker and require COUNT(DISTINCT event_ticker)
>= MIN_EVENT_INSTANCES, meaning the template recurred on at least N days.

Crypto series included:
  KXBTCD    - Daily/hourly BTC price markets (e.g. KXBTCD-26FEB0617)
  KXBTC15M  - BTC 15-minute markets (e.g. KXBTC15M-26FEB181715)
  KXBTC     - BTC weekly/monthly markets
  KXETHD    - Ethereum daily markets
  KXSOL*    - Solana equivalent series
  etc.
"""

import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import duckdb
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
vendor_root = Path("vendor/prediction-market-analysis")
sys.path.insert(0, str(vendor_root))

from src.indexers.kalshi.client import KalshiClient  # noqa: E402

MARKETS_DIR   = Path("data/kalshi/markets")
TRADES_DIR    = Path("data/kalshi/trades")
MANIFEST      = Path("data/kalshi/.indexed_tickers.csv")

TRADES_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE          = 10_000
MAX_WORKERS         = 2         # kalshi rate-limits aggressively; keep this low
MIN_VOLUME          = 10       # skip very thin markets
MIN_EVENT_INSTANCES = 5        # series must have recurred at least 5 times
LIMIT               = 500_000   # total markets to index

# ---------------------------------------------------------------------------
# Thread-local KalshiClient -- one persistent connection per worker thread
# ---------------------------------------------------------------------------
_thread_local = threading.local()

def get_client() -> KalshiClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = KalshiClient()
    return _thread_local.client

# ---------------------------------------------------------------------------
# Manifest -- only records confirmed successes
# ---------------------------------------------------------------------------
def load_manifest() -> set[str]:
    """Return set of tickers already successfully indexed."""
    if not MANIFEST.exists():
        return set()
    with open(MANIFEST, newline="") as f:
        return {row["ticker"] for row in csv.DictReader(f)}

def append_manifest(tickers: list[str]) -> None:
    """Append successfully indexed tickers to the manifest."""
    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "indexed_at"])
        if write_header:
            writer.writeheader()
        for t in tickers:
            writer.writerow({"ticker": t, "indexed_at": datetime.utcnow().isoformat()})

# ---------------------------------------------------------------------------
# Universe selection
# ---------------------------------------------------------------------------
def select_recurring_markets(limit: int) -> list[str]:
    """
    Find the top `limit` markets from recurring series, sorted by volume.

    Three-pass regex derives series keys from event_ticker (see module docstring).
    The regex is inlined in the f-string -- do not extract to a variable.
    """
    print("Scanning markets catalog for recurring series...")

    result = duckdb.sql(f"""
        WITH all_markets AS (
            SELECT
                ticker,
                event_ticker,
                volume,
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            event_ticker,
                            '-[0-9]{{2}}[A-Z]{{3}}[0-9]{{6}}$',
                            ''
                        ),
                        '-[0-9]{{2}}[A-Z]{{3}}[0-9]{{4}}$',
                        ''
                    ),
                    '-[0-9]{{2}}[A-Z]{{3}}[0-9]{{2}}$',
                    ''
                ) AS series_key
            FROM read_parquet('{MARKETS_DIR}/markets_*.parquet')
            WHERE volume >= {MIN_VOLUME}
              AND event_ticker IS NOT NULL
              AND event_ticker != ''
        ),
        recurring_series AS (
            SELECT
                series_key,
                COUNT(DISTINCT event_ticker) AS event_instances,
                SUM(volume)                  AS total_volume
            FROM all_markets
            WHERE series_key != event_ticker
            GROUP BY series_key
            HAVING COUNT(DISTINCT event_ticker) >= {MIN_EVENT_INSTANCES}
        ),
        candidate_markets AS (
            SELECT
                m.ticker,
                m.series_key,
                m.event_ticker,
                m.volume,
                s.event_instances,
                s.total_volume
            FROM all_markets AS m
            INNER JOIN recurring_series AS s
                ON m.series_key = s.series_key
        )
        SELECT ticker
        FROM candidate_markets
        ORDER BY volume DESC
        LIMIT {limit}
    """).fetchall()

    tickers = [row[0] for row in result]
    print(f"Selected {len(tickers):,} markets from recurring series.")
    return tickers


def print_series_breakdown(tickers: list[str]) -> None:
    """Print a preview of top series in the selected universe."""
    sample = tickers[:500]
    ticker_list = ", ".join(f"'{t}'" for t in sample)

    rows = duckdb.sql(f"""
        SELECT
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        event_ticker,
                        '-[0-9]{{2}}[A-Z]{{3}}[0-9]{{6}}$',
                        ''
                    ),
                    '-[0-9]{{2}}[A-Z]{{3}}[0-9]{{4}}$',
                    ''
                ),
                '-[0-9]{{2}}[A-Z]{{3}}[0-9]{{2}}$',
                ''
            ) AS series_key,
            COUNT(DISTINCT event_ticker) AS events,
            COUNT(*)                     AS markets,
            SUM(volume)                  AS total_vol
        FROM read_parquet('{MARKETS_DIR}/markets_*.parquet')
        WHERE ticker IN ({ticker_list})
        GROUP BY series_key
        ORDER BY total_vol DESC
        LIMIT 20
    """).fetchall()

    print("\nTop recurring series in your selection (sample of first 500 markets):")
    print(f"  {'Series Key':<30} {'Events':>6}  {'Markets':>7}  {'Total Vol':>12}")
    print(f"  {'-'*30} {'-'*6}  {'-'*7}  {'-'*12}")
    for row in rows:
        print(f"  {str(row[0]):<30} {row[1]:>6,}  {row[2]:>7,}  {row[3]:>12,}")
    print()

# ---------------------------------------------------------------------------
# Fetch result -- distinguishes success, empty, and error
# ---------------------------------------------------------------------------
class FetchResult(NamedTuple):
    ticker: str
    trades: list[dict]
    success: bool   # False = API error (do NOT checkpoint)
    error: str      # empty string on success

def fetch_ticker(ticker: str) -> FetchResult:
    time.sleep(0.35)
    client = get_client()
    try:
        trades = client.get_market_trades(ticker, verbose=False)
        fetched_at = datetime.utcnow()
        trade_dicts = (
            [{**asdict(t), "_fetched_at": fetched_at} for t in trades]
            if trades else []
        )
        return FetchResult(ticker=ticker, trades=trade_dicts, success=True, error="")
    except Exception as exc:
        return FetchResult(ticker=ticker, trades=[], success=False, error=str(exc))

# ---------------------------------------------------------------------------
# Chunk file naming -- sequential, thread-safe
# ---------------------------------------------------------------------------
_chunk_counter = 0
_chunk_lock    = threading.Lock()

def init_chunk_counter() -> None:
    global _chunk_counter
    import re as _re
    nums = []
    for p in TRADES_DIR.glob("trades_*.parquet"):
        m = _re.search(r"trades_(\d+)\.parquet$", p.name)
        if m:
            nums.append(int(m.group(1)))
    _chunk_counter = (max(nums) + 1) if nums else 0

def next_chunk_path() -> Path:
    global _chunk_counter
    with _chunk_lock:
        path = TRADES_DIR / f"trades_{_chunk_counter:06d}.parquet"
        _chunk_counter += 1
    return path

def save_batch(trades_batch: list[dict]) -> None:
    if not trades_batch:
        return
    pd.DataFrame(trades_batch).to_parquet(next_chunk_path(), index=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_chunk_counter()

    tickers = select_recurring_markets(LIMIT)
    if not tickers:
        print("No recurring markets found. Has the markets index finished running?")
        return

    print_series_breakdown(tickers)

    done     = load_manifest()
    to_fetch = [t for t in tickers if t not in done]
    skipped  = len(tickers) - len(to_fetch)

    if skipped:
        print(f"Skipping {skipped:,} already indexed. {len(to_fetch):,} remaining.\n")
    else:
        print(f"Nothing previously indexed. Fetching all {len(to_fetch):,} markets.\n")

    if not to_fetch:
        print("All done -- nothing to fetch.")
        return

    trades_batch: list[dict] = []
    success_tickers: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_ticker, t): t for t in to_fetch}
        with tqdm(total=len(to_fetch), unit="market") as bar:
            for future in as_completed(futures):
                result: FetchResult = future.result()
                bar.update(1)

                if not result.success:
                    bar.write(f"  ERROR {result.ticker}: {result.error}")
                    continue

                trades_batch.extend(result.trades)
                success_tickers.append(result.ticker)

                if len(trades_batch) >= BATCH_SIZE:
                    save_batch(trades_batch)
                    append_manifest(success_tickers)
                    trades_batch = []
                    success_tickers = []

    # Flush remainder
    if trades_batch:
        save_batch(trades_batch)
    if success_tickers:
        append_manifest(success_tickers)

    print("\nDone.")


if __name__ == "__main__":
    main()