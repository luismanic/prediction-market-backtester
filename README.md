# Prediction Market Backtester

Quant-style backtesting engine for prediction markets (Polymarket + Kalshi), focused on correctness, reproducibility, and performance.

## Quickstart

```bash
uv sync --dev
make lint
make typecheck
make test
```

## Structure

- `src/pm_bt/common/`: shared models/types/utils
- `src/pm_bt/data/`: data loading
- `src/pm_bt/features/`: bars and indicators
- `src/pm_bt/execution/`: execution simulation
- `src/pm_bt/strategies/`: strategy implementations
- `src/pm_bt/backtest/`: engine and metrics
- `src/pm_bt/reporting/`: artifacts and plots
