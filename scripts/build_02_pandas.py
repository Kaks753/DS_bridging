"""Builder for Module 2: Pandas — spreadsheets in Python (4-layer rewrite)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 2 — Pandas: Spreadsheets in Python

**Pandas** is where a data scientist spends most of the day. If NumPy (Module 1) is
fast maths on numbers, **pandas is a smart spreadsheet you control with code** — with
named columns, mixed types (text *and* numbers), and one-line tools for filtering,
grouping, joining, and reshaping tables. Get fluent here and cleaning + exploring
data feels easy; stay shaky and everything is a fight. So we go slow and clear.

By the end you'll comfortably do, and *explain*:
- the two objects: **Series** (one column) and **DataFrame** (a whole table)
- the **index** (row labels) and why it matters
- **selecting** columns/rows with `.loc` and `.iloc`
- **filtering** rows with conditions
- **creating** new columns (the fast, vectorized way)
- **groupby** (the split–apply–combine pattern — the heart of analysis)
- **merge** (joining tables) and **pivot/melt** (reshaping)

> Rhythm as always: 🌱 plain English → 🔤 code line-by-line → 🎓 go deeper → ✅ takeaway.
""")

# ---------------------------------------------------------------------------
nb.md("## 2.1 The two objects: Series and DataFrame")

nb.plain(r"""
Two words you'll hear forever:
- A **Series** is **one column** of data — a list of values, each with a **label**
  (the index) down the side.
- A **DataFrame** is a **whole table** — many Series (columns) stacked side by side,
  sharing the same row labels.

Analogy: a Series is a **single column** in Excel; a DataFrame is the **entire
sheet**. The **index** is like the row numbers on the left of Excel — but in pandas
those labels can be anything (names, dates), and pandas uses them to *line data up
automatically*.

We import pandas as `pd` (the universal nickname), and NumPy as `np`.
""")

nb.code(r"""
import numpy as np
import pandas as pd

# A Series: three scores, labelled a/b/c
s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="score")
print(s)
print("\nthe labels (index):", list(s.index))
print("the raw values    :", s.values, "  <- a NumPy array underneath")
""")

nb.readcode(r"""
- `pd.Series([10,20,30], index=[...], name="score")` → make one column: values
  `10,20,30`, row-labels `a,b,c`, and a column name `score`.
- `s.index` → the row labels. `s.values` → the plain numbers (a NumPy array — pandas
  is built *on* NumPy, so Module 1 wasn't wasted!).
""")

nb.code(r"""
# 'Automatic alignment': maths lines up by LABEL, not by position.
s1 = pd.Series({"a": 1, "b": 2, "c": 3})
s2 = pd.Series({"b": 10, "c": 20, "d": 30})
print(s1 + s2)     # only b and c exist in BOTH -> a and d become NaN
""")

nb.readcode(r"""
- `s1 + s2` → pandas matches labels: `b` (2+10=12) and `c` (3+20=23) add up. But `a`
  is only in `s1` and `d` only in `s2`, so there's nothing to add → the result is
  `NaN` ("Not a Number" = missing).
""")

nb.warn("Unexpected `NaN`s after combining data usually mean the **labels didn't "
        "match**. This is the #1 surprise from pandas' auto-alignment — helpful when "
        "you want it, confusing when you don't.")
nb.jargon("Series", "one labelled column of data")
nb.jargon("DataFrame", "a table: many columns (Series) sharing one row index")
nb.jargon("index", "the row labels of a Series/DataFrame")
nb.jargon("NaN", "'Not a Number' — pandas' marker for a missing value")
nb.takeaway("A Series is one labelled column; a DataFrame is a full table; pandas lines data up by index label.")

# ---------------------------------------------------------------------------
nb.md("## 2.2 Loading a real (messy) dataset & first look")

nb.plain(r"""
Almost always your data lives in a **CSV file** (comma-separated values — a plain-text
spreadsheet). `pd.read_csv` loads it into a DataFrame in one line.

The **first thing** any pro does with a new dataset is *look at it* three ways:
`.head()` (peek at rows), `.info()` (columns, types, missing counts), and
`.describe()` (number summaries). We planted mess in this file on purpose — let's see
if the summaries reveal it.
""")

nb.code(r"""
df = pd.read_csv("../data/customers.csv")   # load the CSV into a DataFrame
print("shape (rows, columns):", df.shape)
df.head()                                    # show the first 5 rows
""")

nb.code(r"""
df.info()      # each column's type + how many non-missing values it has
""")

nb.code(r"""
df.describe()  # count/mean/std/min/quartiles/max for NUMERIC columns
""")

nb.readcode(r"""
- `pd.read_csv("../data/customers.csv")` → read the file (the `../` means "go up one
  folder from `notebooks/` to find the `data/` folder").
- `df.shape` → `(rows, columns)`.
- `df.head()` → first 5 rows — a quick eyeball.
- `df.info()` → lists every column, its type, and its non-missing count. If a column
  shows fewer non-null values than the row count, it has **missing data**.
- `df.describe()` → summary numbers. Compare the `max` to the `75%` value: if `max`
  is wildly bigger, you likely have an **outlier**.
""")

nb.deeper(r"""
Reading these summaries is a *skill*, not a formality. Here `income` will show fewer
non-null values than the total rows (missing data), and its `max` will dwarf the 75%
quartile (that planted outlier of 5,000,000). You just *diagnosed two data problems
without any charts* — purely by letting the summaries talk. That habit is what
separates people who "know pandas" from people who *understand data*.
""")

nb.takeaway("On any new dataset, always run `.head()`, `.info()`, `.describe()` first — they reveal missingness, wrong types, and outliers instantly.")

# ---------------------------------------------------------------------------
nb.md("## 2.3 Selecting columns and rows — `.loc` vs `.iloc`")

nb.plain(r"""
Two ways to grab a slice of a table, and beginners mix them up:
- **`.loc`** selects by **label** (the *name* of a row/column).
- **`.iloc`** selects by **integer position** (the *number*, starting at 0 — like
  Module 0's list indexing).

Memory hook: **`.loc` = Label**, **`.iloc` = Integer** (the `i` is for integer). Also:
`df["age"]` gives one column (a Series); `df[["age","city"]]` (a *list* inside) gives
a smaller DataFrame.
""")

nb.code(r"""
print(type(df["age"]))            # one column -> a Series
print(type(df[["age", "city"]]))  # a list of columns -> a DataFrame

# .loc = by label:
print("\n.loc  row label 0, chosen columns:")
print(df.loc[0, ["age", "plan", "churn"]])

# .iloc = by position:
print("\n.iloc  first 3 rows, first 3 columns:")
print(df.iloc[0:3, 0:3])
""")

nb.readcode(r"""
- `df["age"]` → the `age` column as a Series.
- `df[["age","city"]]` → the double brackets pass a *list* of column names → a
  2-column DataFrame.
- `df.loc[0, ["age","plan","churn"]]` → by **label**: row labelled `0`, and those
  three named columns.
- `df.iloc[0:3, 0:3]` → by **position**: rows 0,1,2 and columns 0,1,2 (the `3` stop
  is excluded, as with NumPy/lists).
""")

nb.warn("Avoid **chained indexing** like `df[df.age > 30][\"income\"] = 0`. It quietly "
        "edits a *temporary copy* (the `SettingWithCopyWarning`) and your change may "
        "vanish. Always write it as ONE `.loc`: `df.loc[df.age > 30, \"income\"] = 0`.")
nb.jargon(".loc", "select rows/columns by their label (name)")
nb.jargon(".iloc", "select rows/columns by integer position (starting at 0)")
nb.takeaway("`.loc` = by label, `.iloc` = by position; make assignments through a single `.loc[...]`.")

# ---------------------------------------------------------------------------
nb.md("## 2.4 Filtering rows with conditions")

nb.plain(r"""
Filtering = "keep only the rows that match a rule". It's the same **boolean mask**
idea from NumPy (Module 1.5): write a condition, pandas checks it for every row and
returns True/False, then you keep the True rows.

Example rule: *"only Premium customers"* → `df[df["plan"] == "Premium"]`.
""")

nb.code(r"""
premium = df[df["plan"] == "Premium"]        # keep rows where plan is Premium
print("Premium customers:", len(premium))

# Two rules at once: use & (and), | (or), ~ (not), each in parentheses
high_risk = df[(df["support_calls"] >= 3) & (df["tenure_months"] < 12)]
print("High-risk (>=3 calls AND <12 months):", len(high_risk))

# 'isin' checks membership in a list of allowed values
coastal = df[df["city"].str.strip().str.title().isin(["Mombasa"])]
print("Mombasa (after tidying the text):", len(coastal))
""")

nb.readcode(r"""
- `df["plan"] == "Premium"` → a True/False for every row; `df[...]` keeps the Trues.
- `(A) & (B)` → both rules must hold; note the parentheses (required!) and `&` for
  "and" (not the word `and`), exactly like NumPy.
- `.str.strip().str.title()` → clean the text first: `.str.strip()` removes stray
  spaces, `.str.title()` fixes capitalisation, so `" mombasa "` matches `"Mombasa"`.
- `.isin([...])` → keep rows whose value is in the given list.
""")

nb.takeaway("Filter with a boolean condition inside `df[...]`; combine rules with `&`, `|`, `~` and parentheses.")
nb.try_this("Keep customers who are on `Basic` **or** have `support_calls == 0`, and print how many.")

# ---------------------------------------------------------------------------
nb.md("## 2.5 Creating new columns — the fast (vectorized) way")

nb.plain(r"""
You'll constantly build new columns from existing ones (e.g. spend-per-month). Do it
the **vectorized** way — an expression on whole columns — which is fast and clean.
Only fall back to a slow row-by-row `.apply` when a rule truly can't be written as a
column expression.

Three everyday tools:
- plain arithmetic on columns → `df["a"] / df["b"]`
- `np.where(condition, x, y)` → "if condition then x else y", for the whole column
- `.map(dictionary)` → translate categories into numbers/labels
""")

nb.code(r"""
# 1) arithmetic on whole columns (vectorized)
df["spend_per_month_ratio"] = df["monthly_spend"] / (df["tenure_months"] + 1)

# 2) a conditional column with NO loop: if calls>=3 -> 'high' else 'normal'
df["risk_flag"] = np.where(df["support_calls"] >= 3, "high", "normal")

# 3) translate plan text -> a rank number with a dictionary
plan_rank = {"Basic": 0, "Standard": 1, "Premium": 2}
df["plan_rank"] = df["plan"].map(plan_rank)

df[["monthly_spend", "tenure_months", "spend_per_month_ratio",
    "support_calls", "risk_flag", "plan", "plan_rank"]].head()
""")

nb.readcode(r"""
- `df["monthly_spend"] / (df["tenure_months"] + 1)` → divides the two columns
  element-by-element (the `+1` avoids dividing by zero for brand-new customers).
- `np.where(df["support_calls"] >= 3, "high", "normal")` → for each row: if calls ≥ 3
  put "high", else "normal" — all at once, no loop.
- `df["plan"].map(plan_rank)` → look each plan up in the dictionary and swap in its
  number (Basic→0, etc.).
""")

nb.deeper(r"""
Why avoid `.apply` with a Python function when you can? Because `.apply` runs your
function **row by row in slow Python**, while vectorized ops and `np.where`/`.map`
run in fast compiled code — often 10–100x quicker on big data. Reserve `.apply` for
genuinely awkward logic that spans several columns with no clean vectorized form:
""")

nb.code(r"""
# A rule that mixes columns awkwardly -> .apply on each row (axis=1) is justified.
def segment(row):
    if row["plan"] == "Premium" and row["support_calls"] == 0:
        return "loyal_premium"
    if row["support_calls"] >= 3:
        return "at_risk"
    return "standard"

df["segment"] = df.apply(segment, axis=1)   # axis=1 => feed one ROW at a time
print(df["segment"].value_counts())          # count how many in each segment
""")

nb.jargon("vectorized (pandas)", "operating on whole columns at once (fast) instead of looping rows with .apply (slow)")
nb.jargon("np.where", "a vectorized if/else: np.where(condition, value_if_true, value_if_false)")
nb.takeaway("Build columns with vectorized maths, `np.where`, and `.map`; use `.apply(axis=1)` only for awkward multi-column rules.")

# ---------------------------------------------------------------------------
nb.md("## 2.6 GroupBy — split, apply, combine (the heart of analysis)")

nb.plain(r"""
**GroupBy** answers questions like *"what's the average spend **per plan**?"* or
*"the churn rate **per city**?"*. It works in three steps, called
**split–apply–combine**:
1. **Split** the rows into groups (one per plan).
2. **Apply** a calculation to each group (e.g. average).
3. **Combine** the answers into a small result table.

Analogy: sort a pile of receipts into stacks by shop (split), total each stack
(apply), then write the totals in a summary (combine).
""")

nb.code(r"""
summary = (
    df.groupby("plan")                              # 1) split rows by plan
      .agg(customers=("customer_id", "count"),      # 2) apply: count customers,
           avg_spend=("monthly_spend", "mean"),     #           average spend,
           churn_rate=("churn", "mean"))            #           average churn
      .sort_values("churn_rate", ascending=False)   # 3) combine + sort
)
summary.round(3)
""")

nb.readcode(r"""
- `df.groupby("plan")` → make one group per distinct plan.
- `.agg(name=(column, function))` → for each group compute: `count` of customer_id,
  `mean` of monthly_spend, `mean` of churn. Each line names the output column.
- `.sort_values("churn_rate", ascending=False)` → show worst-churn plan first.
""")

nb.deeper(r"""
A trick worth memorising: **the mean of a 0/1 column is a rate/proportion.** `churn`
is 1 for churned, 0 for stayed, so `mean(churn)` = the fraction who churned = the
**churn rate**. This lets you compute "% who did X" with a plain `mean`. You can also
group by *two* keys for a cross-table:
""")

nb.code(r"""
# churn rate broken down by plan AND risk_flag -> a tidy cross-table
two_way = df.groupby(["plan", "risk_flag"])["churn"].mean().unstack().round(3)
two_way
""")

nb.jargon("groupby", "split rows into groups, apply a calculation to each, combine the results")
nb.takeaway("GroupBy = split–apply–combine; and the mean of a 0/1 column is that event's rate.")
nb.try_this("Group by `city` and compute the **average income** per city; sort highest first.")

# ---------------------------------------------------------------------------
nb.md("## 2.7 Joining tables with `merge`")

nb.plain(r"""
Real projects have **several** tables that you stitch together on a shared column (a
"key"). `merge` is pandas' version of joining. Two questions to always ask:
1. **On which key** do the tables match? (here: `city`)
2. **Which join type**? The common ones:
   - **inner** — keep only rows whose key exists in *both* tables.
   - **left** — keep *all* left-table rows; fill missing right-side values with NaN.

Analogy: `merge` is looking up extra info in a second address book by matching names.
""")

nb.code(r"""
# a small lookup table: a discount for each city
discounts = pd.DataFrame({
    "city": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"],
    "discount_pct": [5, 8, 6, 4, 7],
})

df["city"] = df["city"].str.strip().str.title()   # tidy keys so they match

merged = df.merge(discounts, on="city", how="left")
print("rows before:", len(df), " | rows after left-merge:", len(merged))
merged[["city", "discount_pct"]].drop_duplicates().sort_values("city")
""")

nb.readcode(r"""
- `df.merge(discounts, on="city", how="left")` → attach `discount_pct` to each
  customer by matching on `city`; `how="left"` keeps every customer even if a city
  had no discount listed (those get NaN).
- We tidy `city` text first so `" nairobi "` matches `"Nairobi"` in the lookup.
- We print `len()` before and after as a sanity check.
""")

nb.warn("A `left` merge should **not** change your row count — UNLESS the right table "
        "has duplicate keys, which makes rows *multiply*. Always compare `len()` "
        "before and after a merge; it catches a whole class of silent bugs.")
nb.jargon("merge / join", "combine two tables by matching a shared key column")
nb.takeaway("Pick the key and the join type (inner vs left), then verify the row count didn't change unexpectedly.")

# ---------------------------------------------------------------------------
nb.md("## 2.8 Reshaping: wide ↔ long (pivot / melt)")

nb.plain(r"""
The same data can be laid out two ways:
- **Wide**: one row per thing, many columns — nice for a human to read in a report.
- **Long ('tidy')**: one row per observation — what charts and models prefer.

`pivot_table` turns long → wide (and can average along the way); `melt` turns wide →
long. You'll flip between them a lot.
""")

nb.code(r"""
# WIDE: average spend, plan as rows, risk_flag as columns
wide = df.pivot_table(index="plan", columns="risk_flag",
                      values="monthly_spend", aggfunc="mean").round(1)
print("WIDE (avg spend, plan x risk):")
print(wide)

# LONG: unpivot it back to one-row-per-observation
long = wide.reset_index().melt(id_vars="plan",
                               var_name="risk_flag", value_name="avg_spend")
print("\nLONG (tidy) form:")
print(long)
""")

nb.readcode(r"""
- `pivot_table(index="plan", columns="risk_flag", values="monthly_spend",
  aggfunc="mean")` → build a grid: one row per plan, one column per risk level, each
  cell = average spend for that combo.
- `melt(id_vars="plan", ...)` → collapse those risk columns back into two tidy
  columns: `risk_flag` (which one) and `avg_spend` (the value).
""")

nb.jargon("pivot_table", "reshape long → wide (rows×columns grid), optionally aggregating")
nb.jargon("melt", "reshape wide → long (one row per observation, the 'tidy' shape)")
nb.takeaway("`pivot_table` = long→wide (report-friendly); `melt` = wide→long (chart/model-friendly).")

# ---------------------------------------------------------------------------
nb.md("## 2.9 Quick wins: sorting, top-N, value counts")

nb.plain(r"""
A few tiny commands you'll reach for daily: `nlargest` (top rows by a column),
`sort_values` (order a table), and `value_counts` (how many of each category — the
fastest way to understand a text column).
""")

nb.code(r"""
print("Top 5 spenders:")
print(df.nlargest(5, "monthly_spend")[["customer_id", "plan", "monthly_spend"]])

print("\nCity distribution (as proportions):")
print(df["city"].value_counts(normalize=True).round(3))
""")

nb.readcode(r"""
- `df.nlargest(5, "monthly_spend")` → the 5 rows with the highest spend.
- `df["city"].value_counts()` → count rows per city; `normalize=True` turns counts
  into proportions (fractions that sum to 1).
""")

nb.takeaway("`value_counts()` is the quickest way to understand a categorical column; `nlargest`/`sort_values` order your data.")

# ---------------------------------------------------------------------------
nb.md(r"""
## 2.10 Practice (do these before Module 3)

1. Average **income per city**, ignoring missing values — which city is highest?
   (Hint: `df.groupby("city")["income"].mean()`.)
2. Add a column `is_loyal` = 1 if `tenure_months > 36` else 0, the **vectorized** way.
3. Make a pivot table of **churn rate by city × plan** — which cell is worst?
4. In your own words, explain `.loc` vs `.iloc`, and why chained assignment is risky.
""")

nb.code(r"""
# Scratch space — try first, then check.
print(df.groupby("city")["income"].mean().round(0).sort_values(ascending=False))
df["is_loyal"] = (df["tenure_months"] > 36).astype(int)
print("\nloyal customers:", df["is_loyal"].sum())
""")

nb.interview(r"""
"Pandas is my daily driver. A DataFrame is labelled columns that auto-align on the
index. I select with .loc (labels) and .iloc (positions), filter with boolean masks,
build columns vectorized rather than with .apply, and I think in split–apply–combine
for groupby. On merges I always sanity-check the row count before and after."
""")

nb.md(r"""
## Summary — your pandas fluency checklist

- **Series** = one labelled column; **DataFrame** = a table; the **index aligns** data.
- First moves on any dataset: **`.head()`, `.info()`, `.describe()`**.
- **`.loc`** = labels, **`.iloc`** = positions; assign through a single `.loc`.
- **Filter** with boolean masks (`&`, `|`, `~`, parentheses).
- Build columns **vectorized** / `np.where` / `.map`; `.apply` only when forced.
- **GroupBy = split–apply–combine**; mean of a 0/1 column = a rate.
- Choose the right **merge** and **check row counts**.
- **pivot_table** = long→wide, **melt** = wide→long.

Next: **Module 3 — Data Cleaning**, where we fix every problem in this messy dataset.
""")

out = nb.save("notebooks/02_pandas.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
