import pandas as pd

RUNS = {
    "Backtest #10a ": r"D:\prediction-market-backtester\output\runs\kalshi_flb_10k_v10a_20260225_024635_nogit\summary.csv",
    "Backtest #10b ": r"D:\prediction-market-backtester\output\runs\kalshi_flb_10k_v10b_20260226_004247_nogit\summary.csv",
}

for run_name, path in RUNS.items():
    print(f"\n{'='*70}")
    print(f"  {run_name}")
    print(f"{'='*70}")

    df = pd.read_csv(path)

    df['category'] = df['market_id'].str.split('-').str[0].str.upper()

    breakdown = df.groupby('category').agg(
        markets=('total_pnl', 'size'),
        total_pnl=('total_pnl', 'sum'),
        avg_pnl=('total_pnl', 'mean'),
        profitable=('total_pnl', lambda x: (x > 0).sum()),
        max_win=('total_pnl', 'max'),
        max_loss=('total_pnl', 'min'),
    ).sort_values('total_pnl', ascending=False)

    breakdown['win_rate_%'] = (breakdown['profitable'] / breakdown['markets'] * 100).round(1)
    breakdown['pct_of_total'] = (breakdown['total_pnl'] / breakdown['total_pnl'].sum() * 100).round(1)

    print("\n--- PnL BY CATEGORY ---")
    print(breakdown.to_string())

    print(f"\n--- TOP 5 INDIVIDUAL WINS ---")
    print(df.nlargest(5, 'total_pnl')[['market_id', 'total_pnl', 'fills_count']].to_string(index=False))

    print(f"\n--- TOP 5 INDIVIDUAL LOSSES ---")
    print(df.nsmallest(5, 'total_pnl')[['market_id', 'total_pnl', 'fills_count']].to_string(index=False))

    top3_pnl = breakdown['total_pnl'].nlargest(3).sum()
    total_pnl = breakdown['total_pnl'].sum()
    print(f"\n--- CONCENTRATION CHECK ---")
    print(f"Top 3 categories: {top3_pnl/total_pnl*100:.1f}% of total PnL")
    print(f"Total PnL: ${total_pnl:,.2f}")