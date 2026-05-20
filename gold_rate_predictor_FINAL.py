# ============================================================
# 🥇 GOLD RATE PREDICTOR — Copy & Paste Ready!
# Just paste this entire code in ONE Jupyter cell & Run!
# ============================================================

# STEP 0: Install Libraries
import subprocess
subprocess.run(["pip", "install", "yfinance", "scikit-learn", "seaborn", "matplotlib", "pandas", "numpy", "-q"])

# STEP 1: Import All Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("🥇  GOLD RATE PREDICTOR — Karthikeyan M")
print("=" * 55)

# STEP 2: Download Live Data (Last 2 Years)
print("\n📥 Downloading live data from Yahoo Finance...")
try:
    gold_usd   = yf.download("GC=F",  period="2y", interval="1d", progress=False)
    usd_inr    = yf.download("INR=X", period="2y", interval="1d", progress=False)
    silver_usd = yf.download("SI=F",  period="2y", interval="1d", progress=False)

    df = pd.DataFrame()
    df['gold_usd']   = gold_usd['Close'].squeeze()
    df['usd_inr']    = usd_inr['Close'].squeeze()
    df['silver_usd'] = silver_usd['Close'].squeeze()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    print("✅ Live data downloaded successfully!")

except Exception as e:
    print(f"⚠️  Live download failed ({e})")
    print("📦  Using built-in sample dataset instead...")

    # Built-in Sample Dataset (2 years approximate)
    dates = pd.date_range(start='2023-05-01', end='2025-05-01', freq='B')
    n = len(dates)
    np.random.seed(42)
    base_gold = 1900
    gold_prices = [base_gold]
    for _ in range(n - 1):
        change = np.random.normal(0.3, 8)
        gold_prices.append(gold_prices[-1] + change)

    usd_inr_rates = np.random.uniform(82, 84, n)

    df = pd.DataFrame({
        'gold_usd'  : gold_prices,
        'usd_inr'   : usd_inr_rates,
        'silver_usd': np.random.uniform(22, 30, n)
    }, index=dates)

# STEP 3: Calculate INR Rates
df['gold_inr_per_gram'] = (df['gold_usd'] * df['usd_inr']) / 31.1035
df['gold_24k']          = df['gold_inr_per_gram'].round(2)
df['gold_22k']          = (df['gold_inr_per_gram'] * 22 / 24).round(2)
df['gold_18k']          = (df['gold_inr_per_gram'] * 18 / 24).round(2)

print(f"\n📅 Data Range  : {df.index.min().date()} → {df.index.max().date()}")
print(f"📊 Total Days  : {len(df)} trading days")
print(f"💰 24K Rate    : ₹{df['gold_24k'].iloc[-1]:,.2f} /gram")
print(f"💰 22K Rate    : ₹{df['gold_22k'].iloc[-1]:,.2f} /gram")
print(f"💰 18K Rate    : ₹{df['gold_18k'].iloc[-1]:,.2f} /gram")

# STEP 4: Save Dataset
df[['gold_24k','gold_22k','gold_18k','usd_inr','gold_usd']].to_csv("gold_dataset_2years.csv")
print("\n✅ Dataset saved → gold_dataset_2years.csv")

# STEP 5: Feature Engineering
df['month']       = df.index.month
df['day_of_week'] = df.index.dayofweek
df['quarter']     = df.index.quarter
df['day_of_year'] = df.index.dayofyear

df['lag_1']  = df['gold_24k'].shift(1)
df['lag_3']  = df['gold_24k'].shift(3)
df['lag_7']  = df['gold_24k'].shift(7)
df['lag_14'] = df['gold_24k'].shift(14)
df['ma_7']   = df['gold_24k'].rolling(7).mean()
df['ma_21']  = df['gold_24k'].rolling(21).mean()
df['std_7']  = df['gold_24k'].rolling(7).std()
df['price_change']   = df['gold_24k'].pct_change() * 100
df['usd_inr_change'] = df['usd_inr'].pct_change() * 100

df.dropna(inplace=True)

# STEP 6: Train ML Models
feature_cols = ['month','day_of_week','quarter','day_of_year',
                'lag_1','lag_3','lag_7','lag_14',
                'ma_7','ma_21','std_7',
                'usd_inr','usd_inr_change','price_change']

X = df[feature_cols]
y = df['gold_24k']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False
)

models = {
    'Linear Regression' : LinearRegression(),
    'Random Forest'     : RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
print("\n─── Model Results ───")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        'MAE'  : mean_absolute_error(y_test, preds),
        'RMSE' : np.sqrt(mean_squared_error(y_test, preds)),
        'R2'   : r2_score(y_test, preds),
        'preds': preds,
        'model': model
    }
    print(f"  {name:25s} → MAE: ₹{results[name]['MAE']:6.2f} | R²: {results[name]['R2']:.4f}")

best_name  = max(results, key=lambda x: results[x]['R2'])
best_preds = results[best_name]['preds']
best_model = results[best_name]['model']
print(f"\n🏆 Best Model: {best_name}")

# STEP 7: Predict Next Day
last_row      = df[feature_cols].iloc[[-1]]
next_pred     = best_model.predict(last_row)[0]
today_price   = df['gold_24k'].iloc[-1]
change        = next_pred - today_price
arrow         = "📈 UP" if change > 0 else "📉 DOWN"

print("\n" + "="*50)
print("🔮  NEXT TRADING DAY PREDICTION")
print("="*50)
print(f"  Today  24K : ₹{today_price:>10,.2f} /gram")
print(f"  Today  22K : ₹{today_price*22/24:>10,.2f} /gram")
print(f"  Today  18K : ₹{today_price*18/24:>10,.2f} /gram")
print(f"  ─────────────────────────────")
print(f"  Pred   24K : ₹{next_pred:>10,.2f} /gram  {arrow}")
print(f"  Pred   22K : ₹{next_pred*22/24:>10,.2f} /gram")
print(f"  Change     : ₹{change:>+10.2f} /gram")
print("="*50)

# STEP 8: Dashboard Visualization
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 14))
fig.suptitle('🥇 Gold Rate Analysis & Prediction Dashboard — Karthikeyan M',
             fontsize=15, fontweight='bold', y=0.99)

test_dates = df.index[len(X_train):]

# Chart 1: 2-Year Price Trend
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(df.index, df['gold_24k'], color='#FFD700', lw=2, label='24K Gold')
ax1.plot(df.index, df['gold_22k'], color='#FFA500', lw=1.5, ls='--', label='22K Gold')
ax1.plot(df.index, df['gold_18k'], color='#CD853F', lw=1.2, ls=':', label='18K Gold')
ax1.fill_between(df.index, df['gold_18k'], df['gold_24k'], alpha=0.1, color='gold')
ax1.set_title('Gold Rate — Last 2 Years (₹ per gram)', fontsize=12)
ax1.set_ylabel('Price (₹)')
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

# Chart 2: Actual vs Predicted
ax2 = fig.add_subplot(3, 2, 3)
ax2.plot(test_dates, y_test.values, color='steelblue', lw=1.8, label='Actual')
ax2.plot(test_dates, best_preds,    color='tomato',    lw=1.8, ls='--', label=f'Predicted')
ax2.set_title(f'Actual vs Predicted ({best_name})', fontsize=11)
ax2.set_ylabel('₹ per gram')
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

# Chart 3: Feature Importance
ax3 = fig.add_subplot(3, 2, 4)
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values().tail(8)
colors = ['#FFD700' if v > 0.08 else '#4682B4' for v in fi.values]
ax3.barh(fi.index, fi.values, color=colors, edgecolor='white')
ax3.set_title('Top Feature Importances', fontsize=11)
ax3.set_xlabel('Importance Score')

# Chart 4: Monthly Average
ax4 = fig.add_subplot(3, 2, 5)
monthly = df.groupby('month')['gold_24k'].mean()
mlabels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
bars = ax4.bar([mlabels[i-1] for i in monthly.index], monthly.values,
               color='#FFD700', edgecolor='white')
ax4.set_title('Avg 24K Gold Rate by Month', fontsize=11)
ax4.set_ylabel('Avg ₹/gram')
for bar, val in zip(bars, monthly.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
             f'₹{val:,.0f}', ha='center', fontsize=7)

# Chart 5: Model Comparison
ax5 = fig.add_subplot(3, 2, 6)
names = list(results.keys())
maes  = [results[m]['MAE'] for m in names]
bcolors = ['#FFD700' if m == best_name else '#4682B4' for m in names]
bars2 = ax5.bar(names, maes, color=bcolors, edgecolor='white')
ax5.set_title('Model Comparison (MAE ↓ Better)', fontsize=11)
ax5.set_ylabel('MAE (₹/gram)')
for bar, val in zip(bars2, maes):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'₹{val:.1f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('gold_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# STEP 9: Final Summary
print("\n📊 SUMMARY STATS (Last 2 Years)")
print("-" * 40)
print(f"  Lowest  24K : ₹{df['gold_24k'].min():>10,.2f} /gram")
print(f"  Highest 24K : ₹{df['gold_24k'].max():>10,.2f} /gram")
print(f"  Average 24K : ₹{df['gold_24k'].mean():>10,.2f} /gram")
total_return = ((df['gold_24k'].iloc[-1]/df['gold_24k'].iloc[0])-1)*100
print(f"  2Y Return   : {total_return:>+9.2f}%")
print("\n✅ Files Saved:")
print("   → gold_dataset_2years.csv")
print("   → gold_dashboard.png")
print("\n🎉 Project Complete! Ready for Resume & GitHub!")
