from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendored prediction-market indexers")
    parser.add_argument(
        "--source",
        choices=["all", "kalshi", "polymarket"],
        default="all",
        help="Which venue indexers to run",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "markets", "trades"],
        default="all",
        help="Which indexer type to run",
    )
    parser.add_argument(
        "--kalshi-max-workers",
        type=int,
        default=10,
        help="Max workers for Kalshi trades indexer",
    )
    return parser.parse_args()


def _bootstrap_vendor_path(root_dir: Path) -> None:
    vendor_root = root_dir / "vendor" / "prediction-market-analysis"
    if not vendor_root.exists():
        raise RuntimeError(
            "Missing vendored source at vendor/prediction-market-analysis. "
            "Import it first before running make index."
        )

    sys.path.insert(0, str(vendor_root))


def _run_kalshi(mode: str, max_workers: int) -> None:
    from src.indexers.kalshi.markets import KalshiMarketsIndexer
    from src.indexers.kalshi.trades import KalshiTradesIndexer

    if mode in {"all", "markets"}:
        print("[index] kalshi markets")
        KalshiMarketsIndexer().run()

    if mode in {"all", "trades"}:
        print("[index] kalshi trades")
        KalshiTradesIndexer(max_workers=max_workers).run()


def _run_polymarket(mode: str) -> None:
    from src.indexers.polymarket.markets import PolymarketMarketsIndexer
    from src.indexers.polymarket.trades import PolymarketTradesIndexer

    if mode in {"all", "markets"}:
        print("[index] polymarket markets")
        PolymarketMarketsIndexer().run()

    if mode in {"all", "trades"}:
        print("[index] polymarket trades")
        PolymarketTradesIndexer().run()


def main() -> int:
    args = _parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    os.chdir(root_dir)

    _bootstrap_vendor_path(root_dir)

    try:
        if args.source in {"all", "kalshi"}:
            _run_kalshi(args.mode, args.kalshi_max_workers)

        if args.source in {"all", "polymarket"}:
            _run_polymarket(args.mode)

    except ModuleNotFoundError as exc:
        print(f"[index] missing dependency: {exc}")
        print("[index] run: uv sync --dev --group index")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
