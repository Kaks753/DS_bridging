"""Builder for Module 19: Time Series Analysis & Forecasting (4-layer)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 19 — Time Series Analysis & Forecasting

Time series is its own discipline because **order matters** and each observation is
**correlated with its own past**. This is the backbone of forecasting work like a
Kenya Food Inflation project (Prophet + SARIMA + XGBoost, ~6% MAPE). Here we build the
fundamentals so you can defend every modelling choice.

**What you'll be able to do by the end:**
- Say what makes time-series data special (autocorrelation, never shuffle).
- **Decompose** a series into trend + seasonality + residual.
- Test for **stationarity** (ADF) and fix it with **differencing**.
- Read **ACF/PACF** to pick ARIMA orders.
- Forecast with **SARIMA**, evaluate with a **time-based split** and **MAPE**.
""")

nb.plain(r"""
A time series is just measurements stamped with *when* they happened — a monthly price,
a daily temperature, hourly website visits. The twist: today is usually a lot like
yesterday, so the *order* carries real information. That one fact changes everything —
most importantly, you can **never shuffle the rows** or randomly split them, because
that would let the model peek at the future.
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

nb.readcode(r"""
- We *manufacture* a price series so we know the truth: a rising `trend`, a repeating
  yearly `seasonality` (a sine wave with a 12-month period), and a little random
  `noise`.
- `pd.date_range(..., freq="MS")` stamps each point with a month start, giving us a
  proper time index.
- The plot shows the two ingredients at once: a general climb (trend) with regular
  up-and-down waves (seasonality).
""")

# ---------------------------------------------------------------------------
# 19.1 What's different
# ---------------------------------------------------------------------------
nb.md(r"""
## 19.1 What makes time series different
""")

nb.analogy(r"""
Predicting tomorrow's temperature? Your single best clue is *today's* temperature. That
"today looks like yesterday" property is **autocorrelation** — a variable correlated
with its own recent past. Ordinary ML assumes rows are independent strangers; time
series rows are a family holding hands in order.
""")

nb.jargon("Autocorrelation", "a series being correlated with its own past values (today resembles yesterday)")
nb.jargon("Stationary", "a series whose mean, variance and correlations stay constant over time")

nb.warn(r"""
**Never use a random train/test split on time series.** You must split by **time** —
train on the past, test on the future. A random split scatters future points into the
training set, leaking tomorrow's answer into today's model. It's the single most
common (and most fatal) time-series mistake.
""")

nb.takeaway("Time series has order + autocorrelation, so never shuffle: always split train=past, test=future.")

# ---------------------------------------------------------------------------
# 19.2 Decomposition
# ---------------------------------------------------------------------------
nb.md(r"""
## 19.2 Decomposition — split the signal into parts

Any series can be viewed as **Trend** (long-term direction) + **Seasonality**
(repeating cycle) + **Residual** (leftover noise).
""")

nb.plain(r"""
Decomposition is like separating a song into vocals, bass, and drums so you can hear
each on its own. Here we pull the price apart into: the slow climb (trend), the
regular yearly wave (seasonality), and the random static that's left (residual). Seeing
them separately tells you what to model and what to ignore.
""")

nb.jargon("Trend", "the slow long-term direction of the series (up, down, flat)")
nb.jargon("Seasonality", "a pattern that repeats on a fixed cycle (every 12 months, every 7 days)")
nb.jargon("Residual", "what's left after removing trend and seasonality -- ideally just noise")

nb.code(r"""
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(ts, model="additive", period=12)
fig = decomp.plot()
fig.set_size_inches(10, 7)
plt.tight_layout(); plt.show()
print("Decomposition splits the series so you can SEE trend vs seasonality vs noise.")
""")

nb.deeper(r"""
`model="additive"` assumes the pieces **add up** (`series = trend + seasonal +
residual`) — right when the seasonal swing stays about the same size over time. If the
swing *grows* as the level grows (say holiday sales that get bigger every year), use
`model="multiplicative"` instead, where the pieces **multiply**. Pick additive vs
multiplicative by eyeballing whether the wave amplitude is constant or expanding.
""")

nb.takeaway("Decompose into trend + seasonality + residual to understand a series before modelling it.")

# ---------------------------------------------------------------------------
# 19.3 Stationarity & differencing
# ---------------------------------------------------------------------------
nb.md(r"""
## 19.3 Stationarity — the key ARIMA assumption

A series is **stationary** if its statistical properties (mean, variance, correlations)
don't drift over time. ARIMA-type models require it. Our series is clearly *non*-
stationary — the mean climbs with the trend.
""")

nb.analogy(r"""
Stationary is a calm, steady heartbeat: same average, same rhythm, minute after minute.
Our price is more like someone jogging uphill — the average keeps rising. Many forecast
models can only handle the steady heartbeat, so first we have to "flatten the hill".
""")

nb.md(r"""
We test with the **Augmented Dickey-Fuller (ADF)** test:
- H0: the series is **non-stationary**.
- If **p < 0.05** -> reject H0 -> treat as **stationary**.
""")

nb.code(r"""
from statsmodels.tsa.stattools import adfuller

def adf_report(series, label):
    stat, p, *_ = adfuller(series.dropna())
    verdict = "STATIONARY" if p < 0.05 else "non-stationary"
    print(f"{label:22s} ADF p-value = {p:.4f}  -> {verdict}")

adf_report(ts, "original")
""")

nb.md(r"""
### Differencing — how we *achieve* stationarity

**Differencing** replaces each value with the *change* from the previous step
($y_t - y_{t-1}$) — this flattens the trend. A **seasonal difference**
($y_t - y_{t-12}$) removes the yearly cycle. The number of differences is the **`d`**
(and seasonal **`D`**) in ARIMA/SARIMA.
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

nb.readcode(r"""
- `.diff()` = "how much did it change since last month?" — subtracts the trend so the
  line hovers around a flat zero.
- `.diff(12)` = "how much vs the same month last year?" — cancels the yearly wave.
- Watch the ADF p-values fall below 0.05 as we difference: the series becomes
  stationary, ready for ARIMA.
""")

nb.takeaway("ADF tests stationarity (p<0.05 = stationary); differencing (the d/D orders) removes trend and seasonality to get there.")

# ---------------------------------------------------------------------------
# 19.4 ACF/PACF
# ---------------------------------------------------------------------------
nb.md(r"""
## 19.4 ACF & PACF — choosing ARIMA orders
""")

nb.plain(r"""
Two diagnostic plots help pick how many past values/errors the model should use:
- **ACF** answers "how related is today to N steps ago, *including* ripple effects?" —
  it guides the **MA (q)** term.
- **PACF** answers the same but *strips out* the in-between steps, isolating the direct
  link — it guides the **AR (p)** term.
You read where each plot "drops off" to choose the orders. In practice `auto_arima`
searches for you, but you should know what it's choosing and why.
""")

nb.jargon("ACF (autocorrelation)", "correlation of the series with its own lags, including indirect effects -> guides q")
nb.jargon("PACF (partial autocorrelation)", "correlation at a lag after removing shorter lags' effects -> guides p")

nb.code(r"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
plot_acf(seasonal_diff.dropna(), lags=24, ax=ax[0])
plot_pacf(seasonal_diff.dropna(), lags=24, ax=ax[1], method="ywm")
plt.tight_layout(); plt.show()
""")

nb.takeaway("ACF guides the MA(q) order, PACF guides the AR(p) order; read where each 'cuts off'.")

# ---------------------------------------------------------------------------
# 19.5 SARIMA forecast
# ---------------------------------------------------------------------------
nb.md(r"""
## 19.5 Forecasting with SARIMA (+ a correct time split)

**ARIMA(p,d,q)** = AutoRegressive (p past values) + Integrated (d differences) + Moving
Average (q past errors). **SARIMA** adds seasonal terms `(P,D,Q,s)` — here `s=12` for
monthly data with a yearly cycle. We train on the past, forecast the future, and
compare to held-out actuals.
""")

nb.jargon("ARIMA", "AutoRegressive Integrated Moving Average — a classic forecast model with orders (p,d,q)")
nb.jargon("SARIMA", "ARIMA plus seasonal terms (P,D,Q,s) for repeating cycles")

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

nb.readcode(r"""
- `train, test = ts.iloc[:-12], ts.iloc[-12:]` — the LAST 12 months are the future we
  hold out. Split by position in time, not randomly.
- `order=(1,1,1)` and `seasonal_order=(1,1,1,12)` set the non-seasonal and seasonal
  (p,d,q) with a 12-month cycle.
- `fit.forecast(steps=12)` predicts the next year; the plot overlays it (red dashed) on
  the true held-out values (black). Close overlap = good model.
""")

nb.code(r"""
# Evaluate with MAPE (Mean Absolute Percentage Error) -- interpretable as a %
def mape(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

rmse = np.sqrt(np.mean((test.values - forecast.values)**2))
print(f"MAPE = {mape(test, forecast):.2f}%")
print(f"RMSE = {rmse:.2f}")
""")

nb.deeper(r"""
**Why MAPE?** It's a percentage, so a stakeholder instantly gets "we're off by ~6% on
average" without knowing the units. The catch: MAPE explodes when actual values are
near zero (dividing by ~0), and it punishes over- and under-forecasts asymmetrically.
For near-zero or can-be-negative data, prefer **RMSE** or **MAE**. Choosing the right
metric for your audience is a senior-level habit.
""")

nb.takeaway("Split by time, forecast the held-out future, and report MAPE (a friendly %) with RMSE/MAE as backup.")

# ---------------------------------------------------------------------------
# 19.6 Modern toolbox + TimeSeriesSplit
# ---------------------------------------------------------------------------
nb.md(r"""
## 19.6 The modern toolbox (name these in interviews)
""")

nb.plain(r"""
- **ARIMA / SARIMA** — great for clear trend + seasonality, and interpretable.
- **Prophet** (Meta) — handles multiple seasonalities and holidays, robust to gaps,
  very easy; a business-forecasting workhorse.
- **ML on lag features** — engineer "value last month / last year" columns and feed
  **XGBoost/LightGBM**; flexible and competition-winning.
- **Deep learning** (LSTMs) — for long, complex, high-volume series.
An ensemble (Prophet + SARIMA + XGBoost) blends their strengths and often beats any one
alone.
""")

nb.warn(r"""
Cross-validation for time series must use **expanding/rolling windows**
(`TimeSeriesSplit`), where train indices always come *before* test indices. Plain
`KFold` would train on the future — leakage again.
""")

nb.code(r"""
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=4)
print("TimeSeriesSplit -- train indices always precede test indices:")
idx = np.arange(len(ts))
for i, (tr, te) in enumerate(tscv.split(idx)):
    print(f" fold {i}: train [{tr[0]:>3}..{tr[-1]:>3}]  ->  test [{te[0]:>3}..{te[-1]:>3}]")
""")

nb.interview("\"For time-series validation I use TimeSeriesSplit, never KFold -- the training window always precedes the test window, so the model is never allowed to see the future.\"")

# ---------------------------------------------------------------------------
# Practice + summary
# ---------------------------------------------------------------------------
nb.md(r"""
## 19.7 Practice
""")

nb.try_this(r"""
1. Fit SARIMA with `order=(2,1,2)`; does MAPE improve, or does it overfit?
2. Remove the seasonal term (`seasonal_order=(0,0,0,0)`) — how much worse is the
   forecast, and why?
3. Build 3 lag features (t-1, t-2, t-12) and fit an XGBoost regressor; compare MAPE to
   SARIMA.
4. Explain in one sentence why `train_test_split(shuffle=True)` is forbidden here.
""")

nb.md(r"""
## Summary

- Time series has **order + autocorrelation** -> **never shuffle**; split by time.
- **Decompose** into trend + seasonality + residual to understand it.
- ARIMA needs **stationarity**: test with **ADF**, achieve via **differencing** (the
  `d`/`D` orders); **ACF/PACF** guide `p`/`q`.
- **SARIMA** models seasonal series; evaluate with **MAPE** (interpretable %) / RMSE on
  a **future** hold-out.
- Toolbox: ARIMA/SARIMA, **Prophet**, **boosting on lag features**, deep learning — and
  ensembles of them. Validate with **TimeSeriesSplit**.

Next: **Module 20 — NLP & Recommendation Systems**.
""")

out = nb.save("notebooks/19_time_series.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
