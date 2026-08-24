"""Builder for Module 01: Pandas deep dive."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 01 — Pandas Deep Dive (the language of tabular data)

Pandas is where 70–80% of a data scientist's day actually happens. If you're
fluent here, cleaning and EDA feel effortless; if not, everything is a struggle.

We build a precise mental model, then drill the operations you'll use every day:
**selecting, filtering, grouping, joining, reshaping**, and the crucial
`apply`-vs-vectorized decision.

Interview goals for this module:
- Explain **Series vs DataFrame** and what the **index** really is.
- Know **`.loc` vs `.iloc`** cold (and why chained indexing is dangerous).
- Do a **groupby** and explain the *split–apply–combine* pattern.
- Choose the right **merge/join** and predict its row count.
- Reshape with **pivot / melt** and say when each is needed.
""")

nb.md(r"""
## 1.1 The two core objects

- **Series** = one column: a 1-D array **plus a labeled index**.
- **DataFrame** = a dict of Series sharing one index: a 2-D labeled table.

The **index** is not just row numbers — it's a set of *labels* pandas uses to
align data automatically. This alignment is a superpower (and occasionally a
surprise).
""")

nb.code(r"""
import numpy as np
import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="score")
print(s)
print("\nindex:", list(s.index))
print("values:", s.values, "  <- underlying NumPy array")
""")

nb.code(r"""
# Automatic alignment by index label (not by position!)
s1 = pd.Series({"a": 1, "b": 2, "c": 3})
s2 = pd.Series({"b": 10, "c": 20, "d": 30})
print(s1 + s2)   # 'a' and 'd' are NaN because they don't exist in both
""")

nb.md(r"""
**Takeaway:** arithmetic aligns on the index. Mismatched labels produce `NaN`.
This is why a merge/concat gone wrong often shows up as unexpected `NaN`s.
""")

nb.md(r"""
## 1.2 Load the real (messy) dataset

We use `data/customers.csv`, which we deliberately made messy. Here we only
*explore* it — cleaning gets its own module.
""")

nb.code(r"""
df = pd.read_csv("../data/customers.csv")
print("shape:", df.shape)
df.head()
""")

nb.code(r"""
# The 3 commands you run on EVERY new dataset:
df.info()          # dtypes + non-null counts -> spot missing & wrong types
""")

nb.code(r"""
df.describe()      # summary stats for numeric columns -> spot outliers/scale
""")

nb.code(r"""
df.describe(include="object")   # for text/categorical columns
""")

nb.md(r"""
Already we can *read the story*: `income` has fewer non-null values (missing
data), and its `max` in `describe()` is wildly larger than the 75% quantile
(that planted outlier). This is the habit — **let the summaries talk to you.**
""")

nb.md(r"""
## 1.3 Selecting columns and rows: `.loc` vs `.iloc`

- `df["col"]` → a Series (one column).
- `df[["a", "b"]]` → a DataFrame (list of columns).
- **`.loc[row_label, col_label]`** → selection by **label**.
- **`.iloc[row_pos, col_pos]`** → selection by **integer position**.
""")

nb.code(r"""
print(type(df["age"]))          # Series
print(type(df[["age", "city"]]))# DataFrame

# label-based
print("\n.loc first row, chosen cols:")
print(df.loc[0, ["age", "plan", "churn"]])

# position-based
print("\n.iloc rows 0-2, first 3 cols:")
print(df.iloc[0:3, 0:3])
""")

nb.md(r"""
**Gotcha — chained indexing:** `df[df.age > 30]["income"] = 0` may fail silently
(`SettingWithCopyWarning`) because you're assigning into a *temporary copy*.
Always assign through a single `.loc`:

```python
df.loc[df.age > 30, "income"] = 0   # correct, unambiguous
```
""")

nb.md(r"""
## 1.4 Filtering with boolean masks (same idea as NumPy)
""")

nb.code(r"""
premium = df[df["plan"] == "Premium"]
print("Premium customers:", len(premium))

# multiple conditions -> parentheses + & | ~
high_risk = df[(df["support_calls"] >= 3) & (df["tenure_months"] < 12)]
print("High-risk (>=3 calls, <12 months):", len(high_risk))

# isin for membership; ~ to negate
coastal = df[df["city"].str.strip().str.title().isin(["Mombasa"])]
print("Mombasa (after cleaning text):", len(coastal))
""")

nb.md(r"""
## 1.5 Creating & transforming columns — vectorized first

Prefer **vectorized** operations and `np.where` / `.map` over `.apply` with a
Python function, because vectorized runs in C (fast) while `.apply` loops in
Python (slow). Use `.apply` only when there's no vectorized alternative.
""")

nb.code(r"""
# vectorized arithmetic
df["spend_per_month_ratio"] = df["monthly_spend"] / (df["tenure_months"] + 1)

# conditional column WITHOUT a loop
df["risk_flag"] = np.where(df["support_calls"] >= 3, "high", "normal")

# map categories to numbers
plan_rank = {"Basic": 0, "Standard": 1, "Premium": 2}
df["plan_rank"] = df["plan"].map(plan_rank)

df[["monthly_spend", "tenure_months", "spend_per_month_ratio",
    "support_calls", "risk_flag", "plan", "plan_rank"]].head()
""")

nb.md(r"""
### When `.apply` is justified
""")

nb.code(r"""
# A rule with no clean vectorized form -> apply on a row (axis=1).
def segment(row):
    if row["plan"] == "Premium" and row["support_calls"] == 0:
        return "loyal_premium"
    if row["support_calls"] >= 3:
        return "at_risk"
    return "standard"

df["segment"] = df.apply(segment, axis=1)
print(df["segment"].value_counts())
""")

nb.md(r"""
## 1.6 GroupBy — the *split–apply–combine* pattern

This is the beating heart of analysis. Pandas **splits** rows into groups,
**applies** a function to each group, then **combines** the results.
""")

nb.code(r"""
# Average monthly spend and churn rate per plan
summary = (
    df.groupby("plan")
      .agg(customers=("customer_id", "count"),
           avg_spend=("monthly_spend", "mean"),
           churn_rate=("churn", "mean"))
      .sort_values("churn_rate", ascending=False)
)
summary.round(3)
""")

nb.md(r"""
Read it like a sentence: *"Group by plan; for each plan count customers, average
the spend, and average churn (which, since churn is 0/1, IS the churn rate)."*
The trick `mean of a 0/1 column = proportion of 1s` is worth memorizing.
""")

nb.code(r"""
# Group by two keys -> a nice cross-table of churn rate
two_way = df.groupby(["plan", "risk_flag"])["churn"].mean().unstack().round(3)
two_way
""")

nb.md(r"""
## 1.7 Joining tables: merge

Real projects have many tables. `merge` is the SQL `JOIN` of pandas. The two
things to always ask: **on which key?** and **which join type?**

- `inner` — only matching keys (default). Rows can *shrink*.
- `left` — keep all left rows; unmatched right side = NaN.
- `right`, `outer` — the mirror / union.

**Predict the row count before you run it** — a good habit that catches bugs.
""")

nb.code(r"""
# Build a small lookup table: a discount per city
discounts = pd.DataFrame({
    "city": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"],
    "discount_pct": [5, 8, 6, 4, 7],
})

# Clean the city text first so keys match (we'll formalize this in Module 02)
df["city"] = df["city"].str.strip().str.title()

merged = df.merge(discounts, on="city", how="left")
print("rows before:", len(df), " rows after left-merge:", len(merged))
merged[["city", "discount_pct"]].drop_duplicates().sort_values("city")
""")

nb.md(r"""
**Takeaway:** a `left` merge should not change your row count *unless* the right
table has duplicate keys (then rows multiply). Checking `len()` before/after is a
cheap safeguard interviewers love to hear you mention.
""")

nb.md(r"""
## 1.8 Reshaping: wide ↔ long (pivot / melt)

- **Wide**: one row per entity, many columns (human-friendly, good for reports).
- **Long / tidy**: one row per observation (model- and plot-friendly).

`pivot_table` goes long→wide (and can aggregate). `melt` goes wide→long.
""")

nb.code(r"""
wide = df.pivot_table(index="plan", columns="risk_flag",
                      values="monthly_spend", aggfunc="mean").round(1)
print("WIDE (avg spend by plan x risk):")
print(wide)

long = wide.reset_index().melt(id_vars="plan",
                               var_name="risk_flag", value_name="avg_spend")
print("\nLONG (tidy) form:")
print(long)
""")

nb.md(r"""
## 1.9 Sorting, ranking, and quick value counts
""")

nb.code(r"""
print("Top 5 spenders:")
print(df.nlargest(5, "monthly_spend")[["customer_id", "plan", "monthly_spend"]])

print("\nCity distribution:")
print(df["city"].value_counts(normalize=True).round(3))  # proportions
""")

nb.md(r"""
## 1.10 Mini-exercises

1. Compute the **average income per city**, ignoring missing values. Which city
   is highest? (Hint: `groupby(...).income.mean()`.)
2. Create a column `is_loyal` = 1 if `tenure_months > 36` else 0, **vectorized**.
3. Make a pivot table of **churn rate by city × plan**. Which cell is worst?
4. Explain, in your own words, the difference between `.loc` and `.iloc`, and
   why chained assignment is risky.
""")

nb.code(r"""
# Scratch space
print(df.groupby("city")["income"].mean().round(0).sort_values(ascending=False))
df["is_loyal"] = (df["tenure_months"] > 36).astype(int)
print("\nloyal customers:", df["is_loyal"].sum())
""")

nb.md(r"""
## Summary — pandas fluency checklist

- Series/DataFrame are **labeled** arrays; the **index aligns** data automatically.
- `.info()`, `.describe()`, `.head()` are your first three moves on any dataset.
- `.loc` = labels, `.iloc` = positions; assign via a **single `.loc`**.
- Filter with boolean masks (`&`, `|`, `~`, parentheses).
- Prefer **vectorized / `np.where` / `.map`** over `.apply`; use `.apply` only when forced.
- **GroupBy = split–apply–combine**; `mean` of a 0/1 column is a rate.
- Choose the right **merge** type and **check row counts**.
- **pivot** = long→wide, **melt** = wide→long (tidy).

Next: **Module 02 — Data Cleaning**, where we fix everything wrong with this data.
""")

out = nb.save("notebooks/01_pandas_deep_dive.ipynb")
print("saved", out)
