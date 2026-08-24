"""Builder for Module 05: Feature Engineering, Scaling, Skewness, Encoding."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 05 — Feature Engineering, Scaling, Skewness & Encoding

> "Applied machine learning is basically feature engineering." — Andrew Ng

Raw columns are rarely the best inputs. This module turns them into *signal*:
handling **skew**, choosing the right **scaler** (the exact question you asked!),
**encoding** categoricals, and — most importantly — **avoiding data leakage**.

We'll make the scaler comparison concrete and visual, so you can *defend* your
choice in an interview instead of reciting it.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                    OneHotEncoder, OrdinalEncoder)

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
df = pd.read_csv("../data/customers_clean.csv")
""")

nb.md(r"""
## 5.1 Skewness — why it hurts and how to fix it

Many models (linear, KNN, neural nets, PCA) behave better when features are
roughly symmetric. A long right tail (positive skew) lets extreme values dominate
distances and coefficients. Common fixes for **positive** skew:

- `log1p(x)` = `log(1 + x)` — the workhorse (handles zeros; needs x ≥ 0).
- `sqrt(x)` — milder.
- Box-Cox / Yeo-Johnson — principled, automatic (Yeo-Johnson allows negatives).
""")

nb.code(r"""
from sklearn.preprocessing import PowerTransformer

income = df["income"].values.reshape(-1, 1)
log_income = np.log1p(df["income"].values)
yj = PowerTransformer(method="yeo-johnson").fit_transform(income).ravel()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, data, name in zip(ax, [df["income"], pd.Series(log_income), pd.Series(yj)],
                         ["raw", "log1p", "yeo-johnson"]):
    sns.histplot(data, kde=True, ax=a); a.set_title(f"{name}  (skew={data.skew():.2f})")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Takeaway:** the transform that pushes skew closest to 0 usually helps
linear/distance models most. Tree models (Module 09) are **immune to monotonic
transforms**, so you can skip this for them — knowing *when a step is unnecessary*
is itself senior-level judgment.
""")

nb.md(r"""
## 5.2 Scaling — putting features on comparable ranges

**Why scale?** Distance- and gradient-based models (KNN, SVM, K-Means, linear
models with regularization, neural nets) are dominated by large-magnitude features
if you don't. Example: `income` (tens of thousands) vs `support_calls` (0–6) —
without scaling, income drowns out calls in any distance calculation.

**Three scalers, one table you must know:**

| Scaler | Centers by | Scales by | Output range | Outlier behavior |
|---|---|---|---|---|
| **StandardScaler** | mean | std | ~unbounded, mean 0, std 1 | **distorted** (mean & std move) |
| **MinMaxScaler** | min | max−min | [0, 1] | **destroyed** (squashes normal data) |
| **RobustScaler** | median | IQR | ~unbounded | **protected** (median/IQR ignore extremes) |
""")

nb.code(r"""
# Build a feature with normal values + ONE crash outlier (like a stock return)
normal = rng.normal(0, 1, 200)
data = np.append(normal, [15.0])            # the outlier
X = data.reshape(-1, 1)

scaled = pd.DataFrame({
    "raw": data,
    "Standard": StandardScaler().fit_transform(X).ravel(),
    "MinMax": MinMaxScaler().fit_transform(X).ravel(),
    "Robust": RobustScaler().fit_transform(X).ravel(),
})

print("How the NORMAL points (excluding outlier) get spread out:")
print(scaled.iloc[:-1].describe().loc[["min", "max", "std"]].round(3))
print("\nThe OUTLIER's scaled value under each scaler:")
print(scaled.iloc[-1].round(3))
""")

nb.md(r"""
Read the numbers:
- **MinMax** crushes the 200 normal points into a tiny sliver near 0 (their max is
  small), because the single outlier defines `max=1`. Normal variation is lost.
- **Standard** inflates the std due to the outlier, so normal points shrink toward
  0 and the outlier is muted to a moderate value.
- **Robust** keeps normal points spread across a sensible range **and** leaves the
  outlier sticking out as a large value — exactly what you want for anomaly-heavy
  data (finance, fraud).
""")

nb.code(r"""
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, col, color in zip(axes, ["raw", "Standard", "MinMax", "Robust"],
                          ["gray", "steelblue", "indianred", "seagreen"]):
    sns.histplot(scaled[col], ax=ax, bins=30, color=color)
    ax.set_title(col)
plt.suptitle("Same data, four scalings — watch where normal points land", y=1.05)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Your finance question, answered crisply:** financial returns are fat-tailed and
full of shocks (crashes, earnings surprises). StandardScaler/MinMax let those
extremes distort the scale for *everything*, masking normal-day variation.
**RobustScaler** uses median + IQR — statistics that ignore the tails — so ordinary
days stay informative and true anomalies remain visibly extreme. That's why it's a
strong default for financial and fraud features.
""")

nb.md(r"""
## 5.3 Encoding categoricals — models need numbers

Two main strategies:
- **One-Hot Encoding**: one 0/1 column per category. Use for **nominal** (no
  order) features like `city`. Safe for linear/distance models. Beware very
  high-cardinality (too many columns).
- **Ordinal Encoding**: map ordered categories to integers. Use only when order is
  **real** (e.g. Basic < Standard < Premium). Never ordinal-encode nominal data
  for linear models — it invents a fake ordering.
""")

nb.code(r"""
# One-hot for nominal 'city'
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
city_ohe = ohe.fit_transform(df[["city"]])
city_cols = ohe.get_feature_names_out(["city"])
print("one-hot columns:", list(city_cols))
print(pd.DataFrame(city_ohe, columns=city_cols).head(3).astype(int))

# Ordinal for the genuinely ordered 'plan'
ord_enc = OrdinalEncoder(categories=[["Basic", "Standard", "Premium"]])
df["plan_ord"] = ord_enc.fit_transform(df[["plan"]]).astype(int)
print("\nplan -> ordinal:")
print(df[["plan", "plan_ord"]].drop_duplicates().sort_values("plan_ord"))
""")

nb.md(r"""
**High-cardinality tip:** for features with hundreds of categories (e.g. zip
code), one-hot explodes. Alternatives: **target/mean encoding** (encode by the
mean target per category — but do it *inside cross-validation* to avoid leakage),
frequency encoding, or embeddings. Mentioning this shows range.
""")

nb.md(r"""
## 5.4 Creating new features — where domain knowledge pays

The best features come from *understanding the problem*. Examples on our data:
""")

nb.code(r"""
df["spend_per_tenure"] = df["monthly_spend"] / (df["tenure_months"] + 1)  # ratio
df["calls_per_tenure"] = df["support_calls"] / (df["tenure_months"] + 1)  # friction
df["is_new"] = (df["tenure_months"] < 6).astype(int)                      # binning
df["age_bucket"] = pd.cut(df["age"], bins=[0, 25, 40, 60, 100],
                          labels=["<25", "25-40", "40-60", "60+"])
print(df[["spend_per_tenure", "calls_per_tenure", "is_new", "age_bucket"]].head())
print("\nchurn rate by age bucket:")
print(df.groupby("age_bucket", observed=True)["churn"].mean().round(3))
""")

nb.md(r"""
## 5.5 DATA LEAKAGE — the mistake that silently ruins projects

**Leakage** = information from outside the training set (especially the test set or
the target) sneaks into training. The model looks great in validation, then fails
in production. Classic causes:

1. **Scaling/imputing on the full dataset before splitting** → test stats leak into
   training. *Fix:* split first, then `fit` transforms on **train only**.
2. **Using the target to build a feature** (e.g. mean-encoding without CV).
3. **Features that won't exist at prediction time** (e.g. "date account closed"
   to predict churn).

The right defense is the **Pipeline** (Module 08), which `fit`s every step on train
folds only.
""")

nb.code(r"""
from sklearn.model_selection import train_test_split

X = df[["income", "monthly_spend"]].copy()
y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)   # FIT on train only
X_te_s = scaler.transform(X_te)       # only TRANSFORM test (no peeking)

print("train mean after scaling (~0):", X_tr_s.mean(axis=0).round(3))
print("test  mean after scaling (not forced to 0 — correct!):",
      X_te_s.mean(axis=0).round(3))
""")

nb.md(r"""
**Interview gold:** *"I fit scalers and imputers on the training fold only and
apply them to validation/test, ideally via a scikit-learn Pipeline, so no
information leaks. Test statistics being slightly off-center is expected and
proves I didn't cheat."*
""")

nb.md(r"""
## 5.6 Mini-exercises

1. Transform `monthly_spend` with `log1p` and Yeo-Johnson; which gets skew closest
   to 0?
2. Scale `[age, income, support_calls]` with all three scalers; print the ranges.
   Which feature would dominate a KNN distance *without* scaling?
3. One-hot encode `city` and ordinal-encode `plan`; explain why you would NOT swap
   the two methods.
4. Give one concrete example of leakage that could occur in the churn problem.
""")

nb.md(r"""
## Summary

- Fix **positive skew** with `log1p`/Yeo-Johnson for linear/distance models; trees
  don't need it.
- **Scale** for distance/gradient models. **Standard** (mean/std) distorts under
  outliers, **MinMax** (min/max) gets destroyed by them, **Robust** (median/IQR)
  survives — the go-to for finance/fraud.
- **One-hot** nominal, **ordinal** only truly-ordered; watch high cardinality.
- Engineer features from **domain knowledge** (ratios, bins, flags).
- **Prevent leakage**: split first, `fit` on train only, use Pipelines.

Next: **Module 06 — Regression**, our first real predictive model.
""")

out = nb.save("notebooks/05_feature_engineering_scaling.ipynb")
print("saved", out)
