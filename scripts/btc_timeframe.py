import pandas as pd

df = pd.read_csv(r"D:\prediction-market-backtester\output\runs\kalshi_flb_10k_v7_20260221_054024_nogit\summary.csv")
df['category'] = df['market_id'].str.split('-').str[0]

kxbtcd = df[df['category'] == 'KXBTCD'].copy()

parts = kxbtcd['market_id'].str.split('-')
kxbtcd['date_part'] = parts.str[1].str[:7]
kxbtcd['hour'] = parts.str[1].str[7:].astype(int)
kxbtcd['price_tier'] = parts.str[2].str[1:].astype(float)
kxbtcd['date'] = pd.to_datetime(kxbtcd['date_part'], format='%y%b%d')
kxbtcd['month'] = kxbtcd['date'].dt.to_period('M')

# Correct classifier: hour 17 = daily settlement, all others = hourly
kxbtcd['market_type'] = kxbtcd['hour'].apply(
    lambda h: 'daily' if h == 17 else 'hourly'
)

print("=== MARKET TYPE SPLIT ===")
print(kxbtcd['market_type'].value_counts())

print("\n=== PnL BY MARKET TYPE ===")
print(kxbtcd.groupby('market_type').agg(
    markets=('market_id','count'),
    total_pnl=('total_pnl','sum'),
    avg_pnl=('total_pnl','mean'),
    profitable=('total_pnl', lambda x: (x > 0).sum()),
    win_rate=('total_pnl', lambda x: f"{(x > 0).mean()*100:.1f}%")
).to_string())

print("\n=== MONTHLY PnL: DAILY ONLY (hour=17) ===")
daily_only = kxbtcd[kxbtcd['market_type'] == 'daily']
print(daily_only.groupby('month').agg(
    markets=('market_id','count'),
    total_pnl=('total_pnl','sum'),
    profitable=('total_pnl', lambda x: (x > 0).sum()),
    win_rate=('total_pnl', lambda x: f"{(x > 0).mean()*100:.1f}%")
).to_string())

print("\n=== MONTHLY PnL: HOURLY ONLY ===")
hourly_only = kxbtcd[kxbtcd['market_type'] == 'hourly']
print(hourly_only.groupby('month').agg(
    markets=('market_id','count'),
    total_pnl=('total_pnl','sum'),
    profitable=('total_pnl', lambda x: (x > 0).sum()),
    win_rate=('total_pnl', lambda x: f"{(x > 0).mean()*100:.1f}%")
).to_string())