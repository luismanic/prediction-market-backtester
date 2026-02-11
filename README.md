# Prediction Market Backtester

Quant-style backtesting engine for prediction markets (Polymarket + Kalshi), focused on correctness, reproducibility, and performance.

## Quickstart

```bash
uv sync --dev
make lint
make typecheck
make test
```

## Data Setup

```bash
cp .env.example .env
# set DATA_URL (and optionally DATA_SHA256)
make setup
```

`make setup` is idempotent:
- downloads `data.tar.zst` only if missing
- optionally verifies `DATA_SHA256`
- extracts to `data/`

## Data Indexing

```bash
# all venues, markets + trades
make index

# examples
make index SOURCE=kalshi MODE=markets
make index SOURCE=polymarket MODE=trades
```

Notes:
- `make index` installs the extra `index` dependency group and runs vendored indexers.
- For Polymarket trades indexing, set `POLYGON_RPC`.

## Structure

- `src/pm_bt/common/`: shared models/types/utils
- `src/pm_bt/data/`: data loading
- `src/pm_bt/features/`: bars and indicators
- `src/pm_bt/execution/`: execution simulation
- `src/pm_bt/strategies/`: strategy implementations
- `src/pm_bt/backtest/`: engine and metrics
- `src/pm_bt/reporting/`: artifacts and plots
- `vendor/prediction-market-analysis/`: vendored data indexers/schemas reference (MIT)
