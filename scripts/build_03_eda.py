"""Builder for Module 03: EDA & Visualization."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 03 — Exploratory Data Analysis & Visualization

EDA is a **conversation with your data**. You ask questions; plots answer. Good
EDA is what lets you say something true and useful *before* any model exists —
and it's what makes a portfolio project sound insightful instead of mechanical.

We'll build the habit: **question → chart → interpretation**. Every plot here
ends in a sentence of *meaning*, because a chart with no takeaway is decoration.

Tools: `matplotlib` (the engine) and `seaborn` (statistical charts on top).
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")   # clean, consistent look
plt.rcParams["figure.figsize"] = (7, 4)

df = pd.read_csv("../data/customers_clean.csv")   # the cleaned artifact
print("shape:", df.shape)
df.head()
""")

nb.md(r"""
## 3.1 The EDA framework (memorize this)

1. **Shape & types** — how big, what columns (`.shape`, `.info()`).
2. **Univariate** — one variable at a time: distribution, center, spread, skew.
3. **Bivariate** — relationships: numeric-vs-numeric, category-vs-numeric,
   category-vs-category.
4. **Target-focused** — how does each feature relate to what we predict (`churn`)?
5. **Multivariate** — correlations, interactions.

Always narrate: *what do I see, and so what?*
""")

nb.md(r"""
## 3.2 Univariate — numeric distributions (histogram + KDE)

A histogram bins values and counts them; the shape tells you center, spread, and
**skewness**. We already know `income` is right-skewed — let's *see* it.
""")

nb.code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["income"], kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Income — right-skewed (long right tail)")
sns.histplot(df["age"], kde=True, ax=axes[1], color="seagreen")
axes[1].set_title("Age — roughly symmetric / bell-ish")
plt.tight_layout()
plt.show()

print("income skew:", round(df["income"].skew(), 2),
      "| age skew:", round(df["age"].skew(), 2))
""")

nb.md(r"""
**Interpretation:** income's positive skew (>0) means a few high earners stretch
the right tail — the mean sits above the median. This is exactly why we imputed
with the median and why we'll consider a **log transform** in Module 05.
""")

nb.md(r"""
## 3.3 Univariate — the boxplot (five-number summary + outliers)

A boxplot shows median (line), IQR (box = middle 50%), whiskers (≈ typical range),
and points beyond as **outliers**. It's the fastest outlier scan you have.
""")

nb.code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(y=df["monthly_spend"], ax=ax[0], color="salmon")
ax[0].set_title("Monthly spend — see the upper outliers")
sns.boxplot(x="plan", y="monthly_spend", data=df, ax=ax[1],
            order=["Basic", "Standard", "Premium"])
ax[1].set_title("Spend rises with plan tier")
plt.tight_layout()
plt.show()
""")

nb.md(r"""
**Interpretation:** the box climbs from Basic → Premium, confirming higher tiers
spend more (a sanity check that the data behaves sensibly). Boxplots-by-category
are one of the most useful EDA charts you'll draw.
""")

nb.md(r"""
## 3.4 Categorical distributions — counts & proportions
""")

nb.code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x="city", data=df, ax=ax[0],
              order=df["city"].value_counts().index)
ax[0].set_title("Customers per city")
ax[0].tick_params(axis="x", rotation=30)

churn_rate = df.groupby("plan", observed=True)["churn"].mean() \
               .reindex(["Basic", "Standard", "Premium"])
sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=ax[1], color="indianred")
ax[1].set_title("Churn rate by plan")
ax[1].set_ylabel("churn rate")
plt.tight_layout()
plt.show()
print(churn_rate.round(3))
""")

nb.md(r"""
**Interpretation:** if Basic churns most, that's an actionable business insight —
retention effort should target Basic customers. *This* is the kind of sentence
that turns a chart into value.
""")

nb.md(r"""
## 3.5 Bivariate — scatter for numeric vs numeric
""")

nb.code(r"""
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=df, x="tenure_months", y="monthly_spend",
                hue="churn", alpha=0.7, palette={0: "steelblue", 1: "crimson"})
ax.set_title("Tenure vs spend, colored by churn")
plt.tight_layout()
plt.show()
""")

nb.md(r"""
**Interpretation:** look at where the crimson (churned) points cluster. If churn
concentrates at *low tenure*, tenure is a strong predictor — which matches how we
designed the data. You're now *reading* signal a model will later exploit.
""")

nb.md(r"""
## 3.6 Multivariate — the correlation heatmap

Correlation measures **linear** association in `[-1, 1]`. A heatmap scans all
numeric pairs at once. Caveats you must state: correlation is *linear only*, and
**correlation ≠ causation**.
""")

nb.code(r"""
num_cols = ["age", "income", "tenure_months", "monthly_spend",
            "support_calls", "churn"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, ax=ax)
ax.set_title("Correlation matrix (numeric features)")
plt.tight_layout()
plt.show()
""")

nb.md(r"""
**How to read the churn row:** the sign tells direction, the magnitude tells
strength. Expect `support_calls` positive (more calls → more churn) and
`tenure_months` negative (longer tenure → less churn). If two *features* are very
highly correlated with each other, that's **multicollinearity** — a heads-up for
linear models (Module 06).
""")

nb.md(r"""
## 3.7 The pairplot — a fast multi-view (use on a few columns)
""")

nb.code(r"""
sns.pairplot(df[["age", "monthly_spend", "tenure_months", "churn"]],
             hue="churn", corner=True, plot_kws={"alpha": 0.6})
plt.suptitle("Pairwise relationships (colored by churn)", y=1.02)
plt.show()
""")

nb.md(r"""
## 3.8 Principles of an honest chart (interview-worthy)

- **Label everything**: title, axes, units. An unlabeled axis is a red flag.
- **Pick the right chart**: distribution→hist/box; relationship→scatter;
  category comparison→bar; part-of-whole→(sparingly) bar, not pie.
- **Don't distort**: avoid truncated y-axes that exaggerate differences.
- **One message per chart**: if it needs a paragraph to explain, split it.
- **End with a takeaway**: "so what?" beats "here's a plot".
""")

nb.md(r"""
## 3.9 Mini-exercises

1. Plot the distribution of `income` **before vs after** a `log1p` transform.
   Which looks more symmetric? (Foreshadows Module 05.)
2. Make a boxplot of `monthly_spend` split by `churn`. Do churners spend
   differently?
3. Build a churn-rate bar chart by `city`. Is any city notably worse?
4. Write three insight sentences you could put on a portfolio project page.
""")

nb.code(r"""
# Scratch: log transform preview
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["income"], kde=True, ax=ax[0]); ax[0].set_title("income (raw)")
sns.histplot(np.log1p(df["income"]), kde=True, ax=ax[1], color="darkorange")
ax[1].set_title("log1p(income) — more symmetric")
plt.tight_layout(); plt.show()
print("skew raw:", round(df['income'].skew(),2),
      "| skew log:", round(np.log1p(df['income']).skew(),2))
""")

nb.md(r"""
## Summary

- EDA = **question → chart → interpretation**, always ending in a takeaway.
- **Univariate**: histogram (shape/skew), boxplot (spread/outliers).
- **Bivariate**: scatter (num–num), box/bar (cat–num), grouped rates.
- **Multivariate**: correlation heatmap (linear only; ≠ causation), pairplot.
- Charts must be **labeled, honest, and single-message**.

Next: **Module 04 — Statistics & Probability**, the reasoning behind the charts.
""")

out = nb.save("notebooks/03_eda_visualization.ipynb")
print("saved", out)
