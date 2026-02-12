# Prediction Market Backtester

Quant-style backtesting engine for prediction markets (Polymarket + Kalshi), focused on correctness, reproducibility, and performance.

## Project Intent

- Build an engine-first backtesting and analytics system, with UI as an optional control/exploration layer.
- Reuse robust historical ingestion patterns from `prediction-market-analysis`.
- Prioritize correctness, reproducibility, and explicit execution assumptions.
- Keep strategy/execution/accounting logic in the engine (CLI/API), not in UI code.

## Quickstart

```bash
uv sync --dev
make lint
make typecheck
make test
```

## Running a Backtest

Use the `pm-bt backtest` command to run a single-market backtest:

```bash
pm-bt backtest \
  --venue kalshi \
  --market KXPGATOUR-APIPBM25-CMOR \
  --strategy momentum \
  --config configs/momentum/default.yaml \
  --start-ts 2025-03-03T00:00:00Z \
  --end-ts 2025-03-10T00:00:00Z \
  --bar-timeframe 5m
```

### Required arguments

| Argument | Description |
|---|---|
| `--venue` | `kalshi` or `polymarket` |
| `--market` | Market identifier (e.g., `KXPGATOUR-APIPBM25-CMOR` for Kalshi) |
| `--strategy` | Strategy name: `momentum`, `mean_reversion`, or `event_threshold` |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/<strategy>/default.yaml` | Path to a YAML config file. CLI arguments override YAML values. |
| `--start-ts` | None (all data) | Start of the backtest window (ISO 8601) |
| `--end-ts` | None (all data) | End of the backtest window (ISO 8601) |
| `--bar-timeframe` | `1m` | Bar aggregation period (`1m`, `5m`, `1h`, etc.) |
| `--data-root` | `data` | Path to the data directory |
| `--output-root` | `output/runs` | Path where run artifacts are written |
| `--name` | `default` | Human-readable run name |

### Output artifacts

Each run produces a directory under `<output-root>/<run-id>/` containing:

- **`results.json`** — full run metadata: config, git commit hash, timings, and trading metrics (total PnL, max drawdown, realized/unrealized PnL, turnover, fill count)
- **`equity.csv`** — per-bar equity curve with cash, realized PnL, unrealized PnL, gross notional exposure, and cash-at-risk
- **`trades.csv`** — every fill with timestamp, side, quantity, price, fees, slippage cost, and latency

### Strategy configs

Default YAML configs are provided under `configs/`:

```
configs/
├── momentum/default.yaml         # threshold: 0.03, qty: 5.0
├── mean_reversion/default.yaml   # move_threshold: 0.02, qty: 5.0
└── event_threshold/default.yaml  # price_jump_threshold: 0.05, min_volume: 100.0, qty: 5.0
```

### Examples

```bash
# Momentum strategy with default config
pm-bt backtest --venue kalshi --market PRES-2024-DJT --strategy momentum

# Mean reversion on a 1-hour timeframe with custom time range
pm-bt backtest --venue kalshi --market PRES-2024-DJT --strategy mean_reversion \
  --bar-timeframe 1h --start-ts 2024-06-01T00:00:00Z --end-ts 2024-11-06T00:00:00Z

# Event threshold with a custom config file
pm-bt backtest --venue kalshi --market PRES-2024-DJT --strategy event_threshold \
  --config my_custom_config.yaml --output-root /tmp/my-runs
```

### Error handling

The CLI returns exit code `0` on success and `1` on failure. Error messages include the full traceback for debugging. Common failures:
- Unknown strategy name
- Missing or invalid config file
- No trade data found for the specified market/time range
- Invalid date range (`start_ts >= end_ts`)

## Running Batch Backtests

Use `pm-bt batch` to run one strategy across top-N markets per venue:

```bash
pm-bt batch \
  --strategy momentum \
  --config configs/momentum/default.yaml \
  --venues kalshi polymarket \
  --top-n 50 \
  --min-trades 20 \
  --output-root output/runs
```

Batch outputs are written under `<output-root>/<name>_<batch-id>/`:
- `summary.csv` - per-market performance + tradability metrics
- `data_quality.json` - per-market quality checks and gate status
- `checkpoint.json` - progress checkpoint for resumable runs (`--resume`)

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
