"""Builder for Module 18: Time Series Analysis & Forecasting."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 18 — Time Series Analysis & Forecasting (Phase 4: T34–35)

Time series is its own discipline because **order matters** and observations are
**correlated with their past**. This is the backbone of your Kenya Food Inflation
project (Prophet + SARIMA + XGBoost, 6% MAPE). Here we build the fundamentals so
you can defend every modeling choice.

Goals:
- What makes time-series data special (autocorrelation, no shuffling!).
- **Decomposition**: trend + seasonality + residual.
- **Stationarity**, why models need it, the **ADF test**, and **differencing**.
- **ACF/PACF** and how they guide **ARIMA/SARIMA**.
- Forecast, evaluate with a **time-based split** and **MAPE**, avoid leakage.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)

# Build a realistic monthly series: upward trend + yearly seasonality + noise
n_months = 120                                  # 10 years
t = np.arange(n_months)
trend = 50 + 0.8 * t                            # steady rise (like food prices)
seasonality = 10 * np.sin(2 * np.pi * t / 12)   # 12-month cycle
noise = rng.normal(0, 3, n_months)
price = trend + seasonality + noise

dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")
ts = pd.Series(price, index=dates, name="price")
print(ts.head())
ts.plot(figsize=(10,4), title="Simulated monthly commodity price (trend + seasonality)")
plt.ylabel("price"); plt.show()
""")

nb.md(r"""
## 18.1 What makes time series different

- **Temporal order is information** — you cannot shuffle rows.
- **Autocorrelation**: today depends on recent past (yesterday's price predicts
  today's). Classic ML assumes independent rows; time series does not.
- **No random train/test split!** You must split by **time** (train on the past,
  test on the future). Random splitting leaks the future into training — a fatal,
  common mistake.
""")

nb.md(r"""
## 18.2 Decomposition — separate the signal into parts

A series can be seen as **Trend** (long-term direction) + **Seasonality**
(repeating cycle) + **Residual** (what's left). Additive when the seasonal swing is
roughly constant; multiplicative when it grows with the level.
""")

nb.code(r"""
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(ts, model="additive", period=12)
fig = decomp.plot()
fig.set_size_inches(10, 7)
plt.tight_layout(); plt.show()
print("Decomposition splits the series so you can SEE trend vs seasonality vs noise.")
""")

nb.md(r"""
## 18.3 Stationarity — the key assumption of ARIMA

A series is **stationary** if its statistical properties (mean, variance,
autocorrelation) **don't change over time**. ARIMA-type models require it. Our
series is clearly *non*-stationary (the mean rises with the trend).

Test it with the **Augmented Dickey-Fuller (ADF)** test:
- H0: the series is **non-stationary** (has a unit root).
- If **p < 0.05** → reject H0 → treat as **stationary**.
""")

nb.code(r"""
from statsmodels.tsa.stattools import adfuller

def adf_report(series, label):
    stat, p, *_ = adfuller(series.dropna())
    verdict = "STATIONARY" if p < 0.05 else "non-stationary"
    print(f"{label:20s} ADF p-value = {p:.4f}  -> {verdict}")

adf_report(ts, "original")
""")

nb.md(r"""
### Differencing — how we *achieve* stationarity

**Differencing** replaces each value with the *change* from the previous step
(`yₜ − yₜ₋₁`). This removes trend. A **seasonal difference** (`yₜ − yₜ₋₁₂`) removes
yearly seasonality. The number of differences is the **`d`** (and seasonal `D`) in
ARIMA/SARIMA.
""")

nb.code(r"""
diff1 = ts.diff()                     # first difference removes the trend
seasonal_diff = ts.diff().diff(12)    # + seasonal difference removes the cycle
adf_report(ts, "original")
adf_report(diff1, "1st difference")
adf_report(seasonal_diff, "1st + seasonal diff")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
diff1.plot(ax=ax[0], title="1st difference (trend removed)")
seasonal_diff.plot(ax=ax[1], title="1st + seasonal difference (now stationary)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
## 18.4 ACF & PACF — choosing ARIMA orders

- **ACF** (autocorrelation): correlation of the series with its own lags — includes
  indirect effects. Guides the **MA(q)** order.
- **PACF** (partial autocorrelation): correlation at a lag after removing shorter
  lags' effects. Guides the **AR(p)** order.

You read where they "cut off" or "tail off" to pick p and q. In practice, tools
like `pmdarima.auto_arima` search orders for you — but you must understand what
they're choosing.
""")

nb.code(r"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
plot_acf(seasonal_diff.dropna(), lags=24, ax=ax[0])
plot_pacf(seasonal_diff.dropna(), lags=24, ax=ax[1], method="ywm")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
## 18.5 Forecasting with SARIMA (+ a correct time split)

**ARIMA(p,d,q)** = AutoRegressive (p past values) + Integrated (d differences) +
Moving Average (q past errors). **SARIMA** adds seasonal terms `(P,D,Q,s)` — here
`s=12` for monthly yearly seasonality. We train on the past, forecast the future,
and compare to held-out actuals.
""")

nb.code(r"""
from statsmodels.tsa.statespace.sarimax import SARIMAX

# TIME-BASED split: last 12 months are the test set (the future)
train, test = ts.iloc[:-12], ts.iloc[-12:]

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False)
fit = model.fit(disp=False)
forecast = fit.forecast(steps=12)

plt.figure(figsize=(10,4))
train.plot(label="train")
test.plot(label="actual (future)", color="black")
forecast.plot(label="SARIMA forecast", color="red", ls="--")
plt.legend(); plt.title("SARIMA forecast vs actual"); plt.show()
""")

nb.code(r"""
# Evaluate with MAPE (Mean Absolute Percentage Error) — interpretable as a %
def mape(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

rmse = np.sqrt(np.mean((test.values - forecast.values)**2))
print(f"MAPE = {mape(test, forecast):.2f}%   (your food-inflation project hit ~6%)")
print(f"RMSE = {rmse:.2f}")
""")

nb.md(r"""
**Why MAPE?** It's a percentage, so stakeholders understand it instantly ("we're
off by 6% on average"). Caveat: it blows up if actuals are near zero — then use
RMSE or MAE. Reporting the *right* metric for the audience is a senior skill.
""")

nb.md(r"""
## 18.6 The modern approaches (name these in interviews)

- **Classical**: ARIMA / SARIMA (great for clear trend+seasonality, interpretable).
- **Prophet** (Meta): handles multiple seasonalities + holidays, robust to missing
  data, easy — a workhorse for business forecasting (you used it).
- **ML on lag features**: engineer lags/rolling means and feed **XGBoost/LightGBM**
  — flexible, captures non-linearities, wins many competitions. Your ensemble
  (Prophet + SARIMA + XGBoost) blends their strengths.
- **Deep learning**: LSTMs / temporal models for long, complex, high-volume series.

**Cross-validation for time series** uses expanding/rolling windows
(`TimeSeriesSplit`) — never `KFold`, which would train on the future.
""")

nb.code(r"""
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=4)
print("TimeSeriesSplit — train indices always precede test indices:")
idx = np.arange(len(ts))
for i, (tr, te) in enumerate(tscv.split(idx)):
    print(f" fold {i}: train [{tr[0]:>3}..{tr[-1]:>3}]  ->  test [{te[0]:>3}..{te[-1]:>3}]")
""")

nb.md(r"""
## 18.7 Mini-exercises

1. Fit SARIMA with `order=(2,1,2)`; does MAPE improve? Beware overfitting.
2. Remove the seasonal term (`seasonal_order=(0,0,0,0)`) — how much worse is the
   forecast, and why?
3. Build 3 lag features (`t-1, t-2, t-12`) and fit an XGBoost regressor; compare
   MAPE to SARIMA.
4. Explain why you must never use `train_test_split(shuffle=True)` on time series.
""")

nb.md(r"""
## Summary

- Time series has **order + autocorrelation** → **never shuffle**; split by time.
- **Decompose** into trend + seasonality + residual to understand it.
- ARIMA needs **stationarity**: test with **ADF**, achieve via **differencing**
  (the `d`/`D` orders); **ACF/PACF** guide `p`/`q`.
- **SARIMA** models seasonal series; evaluate with **MAPE** (interpretable %) /
  RMSE on a **future** hold-out.
- Toolbox: ARIMA/SARIMA, **Prophet**, **boosting on lag features**, deep learning —
  and ensembles of them.

Next: **Module 19 — NLP & Recommendation Systems**.
""")

out = nb.save("notebooks/18_time_series.ipynb")
print("saved", out)
