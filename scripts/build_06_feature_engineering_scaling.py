"""Builder for Module 6: Feature Engineering & Scaling (4-layer rewrite of old M05)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 6 — Feature Engineering, Scaling & Encoding

> "Applied machine learning is basically feature engineering." — Andrew Ng

Here's the honest truth: the raw columns in your data are rarely the best food for a
model. **Feature engineering** is the craft of reshaping them into *signal* the
model can actually digest — fixing lopsided distributions, putting numbers on a fair
scale, and turning words into numbers.

This module also directly answers a question every interviewer asks: *"which scaler
do you use, and why?"* We'll make the answer visual and concrete, so you can defend
your choice instead of reciting it.
""")

nb.analogy("Feature engineering is like prepping ingredients for a recipe. Whole raw "
           "onions and unpeeled potatoes technically ARE food, but the dish turns out far "
           "better if you chop, peel, and measure first. Same data, better form.")

nb.md("## 6.1 Setup")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                    MaxAbsScaler, Normalizer, QuantileTransformer,
                                    OneHotEncoder, OrdinalEncoder)

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
df = pd.read_csv("../data/customers_clean.csv")
print("loaded:", df.shape)
""")

nb.jargon("feature", "an input column the model learns from (age, income, ...)")
nb.jargon("scikit-learn (sklearn)", "Python's main classic-ML library — models, scalers, encoders, and more")

nb.md("## 6.2 Skewness — why lopsided data hurts, and how to fix it")

nb.plain("""
A **skewed** column has a long tail on one side (income: most people modest, a few
enormous). Many models — linear regression, KNN, neural nets, PCA — get bullied by
those few giant values. The cure is a **transform** that squashes the tail toward
symmetry. The workhorse is `log1p` (log of 1+x), which shrinks big numbers far more
than small ones.
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

nb.readcode("""
- `np.log1p(x)` → log(1+x); the +1 lets it handle zeros safely.
- `PowerTransformer(method="yeo-johnson")` → an automatic transform that finds the
  best power to symmetrize (and, unlike log, tolerates negatives/zeros).
- The three panels show the same income with its skew number shrinking toward 0.
""")

nb.deeper("""
The transform that pushes skew closest to 0 usually helps linear/distance models the
most. But here's senior-level judgment: **tree models are immune to any monotonic
transform** (they only care about order, not magnitude), so you can skip skew-fixing
for them entirely. Knowing when a step is UNNECESSARY is as valuable as knowing when
it's needed.
""")

nb.jargon("log transform", "replacing x with log(x); compresses long right tails toward symmetry")
nb.jargon("Yeo-Johnson", "an automatic power transform that symmetrizes data, allowing zeros/negatives")

nb.md("## 6.3 Scaling — putting features on a fair playing field")

nb.plain("""
Imagine measuring 'distance' between two customers using income (tens of thousands)
and support_calls (0–6). Income's sheer size drowns out calls completely — the model
basically ignores calls. **Scaling** rescales every feature to a comparable range so
each gets a fair say. It matters for distance/gradient models (KNN, SVM, K-Means,
regularized linear models, neural nets).
""")

nb.analogy("Scaling is like converting everyone's height and weight into the same unit "
           "of 'how unusual is this?' Otherwise weight-in-grams (big numbers) would swamp "
           "height-in-metres (small numbers) in any comparison.")

nb.md(r"""
The three scalers you must know cold:

| Scaler | Centers by | Scales by | Output | Outliers |
|---|---|---|---|---|
| **StandardScaler** | mean | std | mean 0, std 1 | **distorted** |
| **MinMaxScaler** | min | max−min | [0, 1] | **destroyed** |
| **RobustScaler** | median | IQR | ~centered | **survives** |
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

nb.readcode("""
- We make 200 normal points and tack on one extreme value (15).
- `.fit_transform(X)` for each scaler → learn the scaling from the data and apply it.
- We print how the NORMAL points spread out, and where the single OUTLIER lands under
  each scaler.
""")

nb.deeper("""
Read the numbers:
- MinMax crushes the 200 normal points into a tiny sliver near 0, because the lone
  outlier defines max=1. Normal variation is lost.
- Standard inflates the std because of the outlier, so normal points shrink and the
  outlier is muted.
- Robust uses median + IQR (which ignore the tails), so normal points keep a sensible
  spread AND the outlier stays visibly extreme — ideal for anomaly-heavy data.
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

nb.interview("\"For fat-tailed data like financial returns I default to RobustScaler: it "
             "centers on the median and scales by the IQR, so a crash doesn't distort the "
             "scale for every normal day. StandardScaler and MinMax both let outliers "
             "hijack the scaling.\"")

nb.md("## 6.4 Three more scalers you should recognize")

nb.plain("""
The big three cover ~90% of cases. These three complete the picture — and the last
one is a favorite interview trap.
""")

nb.md(r"""
| Scaler | What it does | Reach for it when |
|---|---|---|
| **MaxAbsScaler** | divides by max absolute value → [−1, 1], keeps zeros as zeros | **sparse** data (TF-IDF, one-hot) where zeros must stay zero |
| **Normalizer** | scales each **ROW** to length 1 | **direction matters more than size** — text/cosine similarity |
| **QuantileTransformer** | remaps values by rank to a target shape | you want to **force** normal/uniform; crushes outliers hard |
""")

nb.warn("The classic mix-up: Normalizer works PER ROW (per sample), while every other "
        "scaler works PER COLUMN (per feature). Say it out loud: 'Normalizer is per-row, "
        "the rest are per-column.'")

nb.code(r"""
# MaxAbs vs the rest on a SPARSE-style feature (many zeros + a few large values)
sparse_feat = np.array([0, 0, 0, 2, 0, 0, 8, 0, 0, 20]).reshape(-1, 1).astype(float)
comp = pd.DataFrame({
    "raw":      sparse_feat.ravel(),
    "MaxAbs":   MaxAbsScaler().fit_transform(sparse_feat).ravel(),   # zeros stay 0
    "MinMax":   MinMaxScaler().fit_transform(sparse_feat).ravel(),
    "Standard": StandardScaler().fit_transform(sparse_feat).ravel(), # zeros move!
})
print(comp.round(3))
print("\nMaxAbs keeps zeros EXACTLY 0 (sparsity preserved);")
print("Standard turns zeros into non-zeros (sparsity destroyed).")
""")

nb.code(r"""
# Normalizer works per-ROW: scale each sample's [x, y] to unit length.
rows = np.array([[3.0, 4.0],     # length 5  -> becomes (0.6, 0.8)
                 [1.0, 0.0],     # length 1  -> stays (1, 0)
                 [10.0, 10.0]])  # length ~14.1 -> (0.707, 0.707)
normed = Normalizer().fit_transform(rows)
print("row lengths BEFORE:", np.linalg.norm(rows, axis=1).round(3))
print("row lengths AFTER :", np.linalg.norm(normed, axis=1).round(3), "(all 1.0)")

# QuantileTransformer forces income to a normal shape:
qt = QuantileTransformer(output_distribution="normal", n_quantiles=200,
                         random_state=0)
income_q = qt.fit_transform(df[["income"]]).ravel()
print("\nincome skew  raw: {:.2f}  ->  quantile-normal: {:.2f}".format(
    df["income"].skew(), pd.Series(income_q).skew()))
""")

nb.readcode("""
- MaxAbs block: notice the zeros stay 0 under MaxAbs but move under Standard.
- Normalizer block: `np.linalg.norm(rows, axis=1)` = each row's length; after
  Normalizer every row has length 1.0.
- QuantileTransformer: forces income's skew from big → ~0 by remapping ranks to a
  normal shape.
""")

nb.md(r"""
### The scaler decision ladder (memorize)
1. Outliers / fat tails / finance → **RobustScaler**
2. Sparse data (TF-IDF, one-hot) → **MaxAbsScaler** (keeps zeros)
3. Row-direction matters (text similarity) → **Normalizer**
4. Must be Gaussian & shape is ugly → **PowerTransformer** or **QuantileTransformer**
5. Neat-ish data, no strong reason → **StandardScaler** (sane default)
6. Trees / Random Forest / boosting → **don't scale at all**
""")

nb.takeaway("Pick the scaler from the data's shape: Robust for outliers, MaxAbs for "
            "sparse, Normalizer for direction, Standard as the default, and skip scaling "
            "entirely for tree models.")

nb.md("## 6.5 Encoding categoricals — models need numbers")

nb.plain("""
Models do math, and you can't multiply the word "Nairobi". **Encoding** turns
categories into numbers. Two main styles:
- **One-Hot**: one 0/1 column per category. Use for **nominal** (unordered) things
  like city.
- **Ordinal**: map ordered categories to integers (0,1,2). Use ONLY when the order
  is real, like Basic < Standard < Premium.
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

nb.readcode("""
- `OneHotEncoder(handle_unknown="ignore")` → new column per city; unseen cities at
  predict-time become all-zeros instead of crashing.
- `get_feature_names_out(...)` → the readable names of the new 0/1 columns.
- `OrdinalEncoder(categories=[[...]])` → map the plan tiers to 0/1/2 IN that order.
""")

nb.warn("Never ordinal-encode an UNORDERED category (like city) for a linear/distance "
        "model — it invents a fake ranking (Nairobi=0, Mombasa=1...) the model will wrongly "
        "treat as meaningful. One-hot avoids that.")

nb.deeper("""
For very high-cardinality features (e.g. zip code with hundreds of values) one-hot
explodes into hundreds of columns. Alternatives: target/mean encoding (encode by the
average target per category — but compute it INSIDE cross-validation to avoid
leakage), frequency encoding, or learned embeddings. Naming these shows range.
""")

nb.jargon("one-hot encoding", "a 0/1 column per category — for unordered (nominal) features")
nb.jargon("ordinal encoding", "mapping ordered categories to integers 0,1,2 — only when order is real")
nb.jargon("cardinality", "how many distinct values a categorical column has")

nb.md("## 6.6 Creating new features — where domain knowledge pays")

nb.plain("""
The most valuable features often don't exist in the raw data — you build them from
understanding the problem. Ratios, flags, and buckets frequently beat raw columns.
""")

nb.code(r"""
df["spend_per_tenure"] = df["monthly_spend"] / (df["tenure_months"] + 1)  # ratio
df["calls_per_tenure"] = df["support_calls"] / (df["tenure_months"] + 1)  # friction
df["is_new"] = (df["tenure_months"] < 6).astype(int)                      # flag
df["age_bucket"] = pd.cut(df["age"], bins=[0, 25, 40, 60, 100],
                          labels=["<25", "25-40", "40-60", "60+"])
print(df[["spend_per_tenure", "calls_per_tenure", "is_new", "age_bucket"]].head())
print("\nchurn rate by age bucket:")
print(df.groupby("age_bucket", observed=True)["churn"].mean().round(3))
""")

nb.readcode("""
- `monthly_spend / (tenure_months + 1)` → a ratio (the +1 avoids divide-by-zero).
- `(tenure_months < 6).astype(int)` → a 0/1 flag for brand-new customers.
- `pd.cut(age, bins=[...], labels=[...])` → bucket a numeric column into named ranges.
- The groupby shows churn rate per age bucket — a quick check that the feature carries signal.
""")

nb.md("## 6.7 DATA LEAKAGE — the silent project-killer")

nb.plain("""
**Leakage** is when information the model shouldn't have at prediction time sneaks
into training. The model looks amazing in testing, then flops in the real world. The
most common cause: scaling or imputing on the WHOLE dataset before splitting — so the
test set's secrets leak into training. The fix: **split first, then fit transforms on
the training data only.**
""")

nb.analogy("Leakage is like practicing for an exam using the actual exam answer key. "
           "You'll ace the practice and bomb the real thing — because you never learned, "
           "you memorized answers you won't have on the day.")

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

nb.readcode("""
- `train_test_split(...)` → split into training and test sets BEFORE any scaling.
- `scaler.fit_transform(X_tr)` → LEARN the mean/std from training AND apply it.
- `scaler.transform(X_te)` → only APPLY the training-learned scaling to test (never
  `fit` on test). The test mean NOT being exactly 0 is proof we didn't cheat.
""")

nb.interview("\"I fit scalers and imputers on the training fold only and apply them to "
             "validation/test — ideally via a scikit-learn Pipeline — so no information "
             "leaks. Test statistics being slightly off-center is expected and proves I "
             "didn't peek.\"")

nb.jargon("data leakage", "when info from test/target sneaks into training, inflating scores unrealistically")

nb.md("## 6.8 Try it yourself")

nb.try_this("""
1. Transform `monthly_spend` with `log1p` and Yeo-Johnson; which gets skew closest to 0?
2. Scale `[age, income, support_calls]` with all three scalers; which feature would
   dominate a KNN distance WITHOUT scaling?
3. One-hot encode `city` and ordinal-encode `plan`; explain why swapping the two
   methods would be wrong.
4. Give one concrete example of leakage that could occur in the churn problem.
""")

nb.md(r"""
## Summary

- Fix **positive skew** with `log1p`/Yeo-Johnson for linear/distance models; trees don't need it.
- **Scale** for distance/gradient models: Standard distorts under outliers, MinMax
  gets destroyed by them, **Robust** survives (finance/fraud default). MaxAbs for
  sparse, Normalizer for row-direction, Quantile to force a shape.
- **One-hot** unordered, **ordinal** only truly-ordered; watch high cardinality.
- Engineer features from **domain knowledge** (ratios, flags, buckets).
- **Prevent leakage**: split first, `fit` on train only, use Pipelines.

Next: **Regression**, our first real predictive model.
""")

out = nb.save("notebooks/06_feature_engineering_scaling.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
