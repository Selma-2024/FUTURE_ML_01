# 📈 Sales & Demand Forecasting — Machine Learning Internship
### Selma Anna Saju | CIN: FIT/MAY26/ML8505 | Future Interns

---

## 📌 Project Overview

A weekly sales forecasting model built using the Superstore retail dataset.
Starting from 9,994 raw transactions, the data was cleaned, aggregated,
and used to train three ML models — with Gradient Boosting selected as
the best performer based on R² score.

**Environment:** Google Colab
**Notebook file:** `FUTURE_ML_01.ipynb`

---

## 🗂️ Notebook Structure

The notebook is organized into 6 cells, each building on the previous one:

```
Cell 1 — Library imports
Cell 2 — Dataset upload & loading
Cell 3 — Data cleaning (missing value handling)
Cell 4 — Feature engineering (time-based features + lags)
Cell 5 — Model training v1 (transaction-level → identified problem)
Cell 6 — Model training v2 (weekly aggregation → final model + charts)
```

---

## ⚙️ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
```

All libraries come pre-installed in Google Colab — no pip install needed.

---

## ▶️ How to Run

### Step 1 — Open in Google Colab
Upload `FUTURE_ML_01.ipynb` to [colab.research.google.com](https://colab.research.google.com)

### Step 2 — Upload the dataset
When Cell 2 runs, a file picker will appear:
```python
from google.colab import files
uploaded = files.upload()   # select sample.csv from your computer
```

### Step 3 — Run all cells in order
`Runtime → Run all`  or press `Shift + Enter` cell by cell.

### Step 4 — Download your charts
```python
from google.colab import files
files.download("chart1_forecast.png")
files.download("chart2_importance.png")
files.download("chart3_seasonality.png")
```

---

## 🔬 What Each Cell Does

### Cell 1 — Imports
Loads all required libraries.

---

### Cell 2 — Load & Explore Data
```python
df = pd.read_csv("sample.csv", parse_dates=["Order Date"], encoding='latin1')
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
```
- Loads the Superstore CSV with proper date parsing
- Uses `encoding='latin1'` to fix the UnicodeDecodeError
- Runs `.head()`, `.shape`, `.info()`, `.describe()`, `.isnull().sum()`

**Output:** Shape (9994, 21) — 0 missing values across all 21 columns

---

### Cell 3 — Data Cleaning
Demonstrates and compares all three missing value strategies:

| Method | Code | Best for |
|--------|------|----------|
| Drop rows | `df.dropna()` | When missing rows are few and random |
| Mean fill | `fillna(df["Sales"].mean())` | Quick fix, ignores trend |
| Interpolation ✅ | `interpolate(method="linear")` | Time series — respects trend |

Since this dataset had no missing values, the cell confirms that
and moves on — demonstrating the methods for learning purposes.

---

### Cell 4 — Feature Engineering
Extracts time-based signals from `Order Date`:

```python
df["month"]        = df["Order Date"].dt.month
df["day_of_week"]  = df["Order Date"].dt.dayofweek
df["day_of_year"]  = df["Order Date"].dt.dayofyear
df["quarter"]      = df["Order Date"].dt.quarter
df["week_of_year"] = df["Order Date"].dt.isocalendar().week.astype(int)
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

df["lag_1"]  = df["Sales"].shift(1)    # previous transaction
df["lag_7"]  = df["Sales"].shift(7)    # 7 transactions ago
df["lag_30"] = df["Sales"].shift(30)   # 30 transactions ago

df["rolling_7d"]  = df["Sales"].rolling(window=7).mean()
df["rolling_30d"] = df["Sales"].rolling(window=30).mean()
```

**Output:** Shape (9964, 32) — 9,964 rows × 32 columns

---

### Cell 5 — First Model (Transaction-level) + Problem Identified

Trained LinearRegression on transaction-level data:

```
Train size: 9043 | Test size: 921
MAE  = 250.54
R²   = 0.0918   ← too low
```

**Root cause identified:** Lag features on individual transactions
are meaningless — a $3 order followed by an $86 order has no
learnable pattern. The model needs aggregated data.

---

### Cell 6 — Fixed Model (Weekly Aggregation) + Final Charts

**Fix applied:** Aggregate 9,994 transactions → 209 weekly totals

```python
df["week"]    = df["Order Date"].dt.to_period("W")
weekly_df     = df.groupby("week")["Sales"].sum().reset_index()
```

**New features on weekly data:**

| Feature | What it captures |
|---------|-----------------|
| `month`, `quarter` | Seasonal patterns |
| `week_of_year` | Weekly cycle |
| `week_number` | Overall business growth trend |
| `lag_1` | Last week's sales |
| `lag_4` | 4 weeks ago |
| `lag_52` | Same week last year (strongest predictor) |
| `rolling_4w` | 1-month moving average |
| `rolling_12w` | 3-month moving average |

**Train/test split:** Last 12 weeks = test set (145 train, 12 test)

**Model comparison:**

| Model | MAE | R² |
|-------|-----|----|
| Linear Regression | $6,566/week | 0.28 |
| Random Forest | $6,120/week | 0.30 |
| Gradient Boosting ✅ | $6,235/week | 0.35 |

**Best model: Gradient Boosting** — highest R²

---

## 📊 Output Charts

### Chart 1 — Actual vs Predicted (`chart1_forecast.png`)
Full sales history (grey) with the 12-week test window showing
actual (blue) vs predicted (red dashed) weekly sales.
<img width="1678" height="592" alt="image" src="https://github.com/user-attachments/assets/8df57250-45a8-40f4-a23b-bfd28469c589" />


### Chart 2 — Feature Importance (`chart2_importance.png`)
Horizontal bar chart showing which features the model relied on most.
`lag_52` and `rolling_4w` were the top predictors.
<img width="1013" height="601" alt="image" src="https://github.com/user-attachments/assets/d609ba61-ce5c-41f2-be91-6e6633397e29" />


### Chart 3 — Seasonality (`chart3_seasonality.png`)
Average weekly sales by month. **November** is the clear peak month
— holiday season effect visible in the data.
<img width="1242" height="616" alt="image" src="https://github.com/user-attachments/assets/e23e6cd5-5190-4638-8055-d825dbd1d383" />


---

## 📋 Final Results

| Item | Value |
|------|-------|
| Dataset | Superstore (9,994 transactions) |
| Aggregation | Weekly totals → 209 weeks |
| Features engineered | 9 time-based features |
| Best model | Gradient Boosting |
| MAE | $6,235 per week |
| R² score | 0.35 |
| Avg weekly sales forecast | $19,325 |
| Peak sales month | November |
| Error margin | ±$6,235 per week |

---

## 💼 Business Insights

- 📦 **Inventory:** Stock up before November (peak season every year)
- 💰 **Cash flow:** Budget for ~$19,325 baseline weekly revenue
- 👥 **Staffing:** Schedule more staff in Q4 (Oct–Dec)
- 📉 **Risk planning:** Keep buffer for ±$6,235 weekly variance
- 📅 **Same week last year** is the strongest sales predictor —
  yearly patterns repeat reliably in this business

---

## 🐛 Real Errors Encountered & Fixed

| Error | Cause | Fix Applied |
|-------|-------|-------------|
| `UnicodeDecodeError: 'utf-8'` | CSV saved in Windows encoding | Added `encoding='latin-1'` |
| R² = 0.09 on transaction data | Lag on random transactions = noise | Aggregated to weekly totals |
| Only 59 test rows (daily split) | Too small to evaluate reliably | Switched to weekly + last 12 weeks |

---

## 🚀 Skills Demonstrated

`Data Cleaning` · `Feature Engineering` · `Time-Series Aggregation` ·
`Linear Regression` · `Random Forest` · `Gradient Boosting` ·
`Model Evaluation (MAE, R²)` · `Business Visualization` · `Google Colab`

---

*Internship Task 1 — Future Interns ML Fellowship Program*
*Duration: 30/05/2026 – 30/06/2026*
