import pandas as pd

RUNS = {
    "Backtest #9a": r"D:\prediction-market-backtester\output\runs\flb_v9_90c_btcd_filled_analysis_20260222_194112_nogit\summary.csv",
    "Backtest #9b (Mention 90c)": r"D:\prediction-market-backtester\output\runs\flb_v10b_mention_90c_20260223_025359_nogit\summary.csv",
}

for run_name, path in RUNS.items():
    print(f"\n{'='*70}")
    print(f"  {run_name}")
    print(f"{'='*70}")

    df = pd.read_csv(path)

    btcd = df[df['market_id'].str.startswith('KXBTCD')].copy()

    def extract_date(market_id):
        try:
            parts = market_id.split('-')
            for part in parts[1:]:
                if len(part) == 6 and part.isdigit():
                    return pd.to_datetime(part, format='%y%m%d')
                try:
                    return pd.to_datetime(part[:7], format='%y%b%d')
                except:
                    pass
        except:
            pass
        return pd.NaT

    btcd['date'] = btcd['market_id'].apply(extract_date)
    btcd['month'] = btcd['date'].dt.to_period('M')

    monthly = btcd.groupby('month').agg(
        markets=('total_pnl', 'size'),
        total_pnl=('total_pnl', 'sum'),
        avg_pnl=('total_pnl', 'mean'),
        profitable=('total_pnl', lambda x: (x > 0).sum()),
        win_rate=('total_pnl', lambda x: (x > 0).mean() * 100),
    ).dropna()

    print("=== KXBTCD MONTHLY DETAIL ===")
    print(monthly.to_string())

    non_btcd = df[~df['market_id'].str.startswith('KXBTCD')]
    print(f"\n=== WITHOUT KXBTCD ===")
    print(f"Total PnL:  ${non_btcd['total_pnl'].sum():,.2f}")
    print(f"Markets:    {len(non_btcd)}")
    print(f"Avg/market: ${non_btcd['total_pnl'].mean():.4f}")
    print(f"Profitable: {(non_btcd['total_pnl'] > 0).sum()} / {len(non_btcd)} ({(non_btcd['total_pnl'] > 0).mean()*100:.1f}%)")