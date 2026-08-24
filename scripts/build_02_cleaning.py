"""Builder for Module 02: Data Cleaning."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 02 — Data Cleaning (the unglamorous skill that pays)

Models are easy to swap. **Clean data is the moat.** Interviewers probe cleaning
hard because it separates people who ran a tutorial from people who've handled
real data. We'll clean `customers.csv` end-to-end and, crucially, **justify every
decision** — because "why did you drop those rows?" is a real interview question.

We cover, precisely:
1. Missing data — the three mechanisms (MCAR / MAR / MNAR) and what to *do*.
2. Wrong dtypes — numbers-as-strings, dates-as-strings.
3. Duplicates — exact and key-based.
4. Categorical noise — whitespace, casing, typos.
5. Outliers — detect with IQR & z-score, and decide keep/cap/drop.
""")

nb.code(r"""
import numpy as np
import pandas as pd

df = pd.read_csv("../data/customers.csv")
print("shape:", df.shape)
df.info()
""")

nb.md(r"""
## 2.1 A cleaning philosophy (say this in interviews)

> "I never mutate the raw file. I keep the original, work on a copy, and record
> every transformation so the pipeline is **reproducible** and **auditable**.
> For each issue I ask: *what caused it*, *what's the risk of each fix*, and
> *does the fix leak information from the target?*"

That mindset alone makes you sound senior.
""")

nb.code(r"""
raw = df.copy()          # keep the untouched original
clean = df.copy()        # we transform this one
""")

nb.md(r"""
## 2.2 Duplicates — remove noise before you measure it

There are two kinds:
- **Exact duplicates**: entire rows repeated (data pipeline glitch).
- **Key duplicates**: same entity id appears twice (business-logic issue).

We planted 8 exact duplicates. But note: `customer_id` should be unique, so we
dedupe on that key too.
""")

nb.code(r"""
print("exact duplicate rows:", clean.duplicated().sum())
clean = clean.drop_duplicates()
print("after dropping exact dupes:", clean.shape)

print("duplicate customer_id:", clean.duplicated(subset='customer_id').sum())
clean = clean.drop_duplicates(subset="customer_id", keep="first")
print("after dedup on key:", clean.shape)
""")

nb.md(r"""
**Takeaway:** dedupe *before* computing statistics; otherwise repeated rows bias
your means and inflate counts. Always state which key defines a unique row.
""")

nb.md(r"""
## 2.3 Categorical noise — standardize text keys

`city` has leading/trailing spaces and inconsistent casing (`NAIROBI`, ` nairobi `).
To software, `"Nairobi"` and `" NAIROBI "` are *different categories* — this
silently breaks group-bys, joins, and one-hot encoding.
""")

nb.code(r"""
print("BEFORE — raw unique cities:")
print(sorted(clean["city"].unique().tolist()))

clean["city"] = clean["city"].str.strip().str.title()

print("\nAFTER — standardized cities:")
print(sorted(clean["city"].unique().tolist()))
""")

nb.md(r"""
**Takeaway:** `.str.strip().str.title()` (or `.lower()`) is a cheap fix that
prevents a whole class of silent bugs. For real typos (e.g. "Nairoby"), you'd map
them explicitly or use fuzzy matching — but never guess silently.
""")

nb.md(r"""
## 2.4 Wrong dtypes — make types match meaning

Check `info()`: are numbers stored as numbers, categories as `category`, dates as
`datetime`? Wrong types block math and waste memory. Here `income` became float
(because of the injected `NaN`), which is *correct*. We convert `plan` to an
ordered category to encode its natural order.
""")

nb.code(r"""
plan_type = pd.CategoricalDtype(categories=["Basic", "Standard", "Premium"],
                                ordered=True)
clean["plan"] = clean["plan"].astype(plan_type)
print(clean["plan"].dtype)
# Ordered categories allow comparisons:
print("rows with plan > Basic:", (clean["plan"] > "Basic").sum())

# Safe numeric coercion pattern (turns bad strings into NaN instead of crashing):
clean["income"] = pd.to_numeric(clean["income"], errors="coerce")
print("income dtype:", clean["income"].dtype)
""")

nb.md(r"""
## 2.4b Renaming columns — clean, consistent names save hours

Messy column names (`"Customer ID"`, `"Monthly  Charges"`, mixed case, spaces)
break dot-access (`df.Customer ID` is a syntax error) and make joins fragile.
A senior habit: **snake_case everything, once, up front.**

Three tools:
- `df.rename(columns={...})` — rename *specific* columns explicitly (safest, readable).
- `df.columns = [...]` — replace *all* names at once (use when standardizing wholesale).
- A vectorized string clean — programmatic, scales to 200 columns.

`rename` also takes `index=` to rename rows, and `inplace=True` (we avoid inplace —
returning a new frame is clearer and chains better).
""")

nb.code(r"""
# Simulate messy names so you see the fix (real files look like this):
messy = clean.copy()
messy.columns = ["Customer ID", "Age ", "Plan", "Monthly  Charges",
                 "City", "Income", "Churn", "income_was_missing",
                 "income_capped"][:len(messy.columns)]
print("BEFORE:", list(messy.columns))

# (1) Explicit rename of just the columns you care about:
messy = messy.rename(columns={"Customer ID": "customer_id",
                              "Monthly  Charges": "monthly_charges"})

# (2) Programmatic standardization of ALL names -> snake_case:
messy.columns = (messy.columns
                 .str.strip()          # kill leading/trailing spaces
                 .str.lower()          # case-insensitive
                 .str.replace(r"\s+", "_", regex=True))  # collapse spaces -> _
print("AFTER: ", list(messy.columns))
""")

nb.md(r"""
**Takeaway:** do this immediately after loading. Every downstream line
(`df.customer_id`, merges, `groupby`) becomes predictable. `str.replace(regex=True)`
with `\s+` handles single *and* double spaces in one shot.
""")

nb.md(r"""
## 2.4c Datetimes — the column type that unlocks time-based analysis

Dates arrive as **strings** (`"2023-07-15"`, `"15/07/2023"`). As strings you
*cannot* subtract them, sort them chronologically, or extract the month. You must
parse them into real `datetime64` with `pd.to_datetime`. Then the `.dt` accessor
gives you year/month/day/weekday/etc. — the raw material for seasonality features.

Key arguments:
- `errors="coerce"` → unparseable values become `NaT` (missing) instead of crashing.
- `format="..."` → give the exact format when you know it (faster + avoids ambiguity
  like month-vs-day). `dayfirst=True` for `DD/MM/YYYY` data.
""")

nb.code(r"""
# customers.csv has no date, so we attach a realistic 'signup_date' to teach parsing.
rng = np.random.default_rng(0)
n = len(clean)
raw_dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
    rng.integers(0, 730, size=n), unit="D")
# Store as messy strings, exactly like a real CSV would:
clean["signup_date"] = raw_dates.strftime("%Y-%m-%d")
print("dtype as loaded (string!):", clean["signup_date"].dtype)
print(clean["signup_date"].head(3).tolist())

# Parse to real datetime:
clean["signup_date"] = pd.to_datetime(clean["signup_date"], errors="coerce")
print("dtype after to_datetime:", clean["signup_date"].dtype)
""")

nb.code(r"""
# The .dt accessor: pull calendar parts out of a datetime column (vectorized).
clean["signup_year"]    = clean["signup_date"].dt.year
clean["signup_month"]   = clean["signup_date"].dt.month
clean["signup_weekday"] = clean["signup_date"].dt.day_name()
clean["days_as_member"] = (pd.Timestamp("2024-01-01") - clean["signup_date"]).dt.days

print(clean[["signup_date", "signup_year", "signup_month",
             "signup_weekday", "days_as_member"]].head())
""")

nb.md(r"""
**Why this matters:** `days_as_member` (a *duration*) is often far more predictive
than the raw date. `signup_month`/`weekday` capture **seasonality**. You can only
compute these once the column is a true datetime — which is exactly why parsing is
step one of any time-aware cleaning. (Full time-series modeling: Module 18.)

**Interview line:** *"Strings that look like dates are a trap — I parse with
`to_datetime(errors='coerce')`, verify no unexpected `NaT`, then derive durations
and calendar features via `.dt`."*
""")

nb.md(r"""
## 2.5 Missing data — the part everyone gets wrong

First, **quantify** it. Then diagnose the **mechanism**, because the mechanism
dictates the honest fix.

- **MCAR** (Missing Completely At Random): missingness unrelated to anything.
  Dropping is safe-ish but wasteful.
- **MAR** (Missing At Random): missingness depends on *other observed* columns.
  Imputation using those columns is defensible.
- **MNAR** (Missing Not At Random): missingness depends on the *unobserved value
  itself* (e.g. high earners hide income). Dangerous — may need a "missing"
  indicator or domain handling.
""")

nb.code(r"""
miss = clean.isna().sum()
miss_pct = (clean.isna().mean() * 100).round(1)
print(pd.DataFrame({"missing": miss, "pct": miss_pct})[miss > 0])
""")

nb.code(r"""
# Diagnose: is income-missingness related to plan? (we built it as MAR on Premium)
tmp = clean.copy()
tmp["income_missing"] = tmp["income"].isna()
print("share of income missing, by plan:")
print(tmp.groupby("plan", observed=True)["income_missing"].mean().round(3))
""")

nb.md(r"""
The missing rate is far higher for `Premium` → this is **MAR** (depends on the
observed `plan`). So a **group-wise median imputation by plan** is honest: we fill
using the most similar customers. We ALSO add a `income_was_missing` flag, because
the fact that it was missing can itself be predictive.
""")

nb.code(r"""
clean["income_was_missing"] = clean["income"].isna().astype(int)

# group-wise median (robust to the income outlier) imputation
clean["income"] = clean.groupby("plan", observed=True)["income"] \
                       .transform(lambda s: s.fillna(s.median()))

# age missingness looked random (MCAR-ish) -> simple median is fine
clean["age"] = clean["age"].fillna(clean["age"].median())

print("remaining missing:\n", clean.isna().sum()[clean.isna().sum() > 0]
      if clean.isna().sum().sum() else "none — all filled")
""")

nb.md(r"""
**Why median, not mean?** `income` is right-skewed and has an outlier; the mean is
dragged upward, the **median is robust**. Choosing median here shows you
understand your data, not just `fillna`.

**Leakage warning:** never impute using the *target* column, and (strictly) fit
imputation statistics on the **training set only**, then apply to test. In a real
pipeline we'd use `SimpleImputer` inside a `Pipeline` (Module 08) so test data
can't peek at training stats.
""")

nb.md(r"""
## 2.6 Outliers — detect, then *decide* (don't auto-delete)

Two standard detectors:
- **Z-score**: how many std devs from the mean. Assumes roughly normal; itself
  sensitive to outliers (mean/std get distorted).
- **IQR rule**: flag points below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`. Robust,
  distribution-free — the safer default.
""")

nb.code(r"""
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

low, high = iqr_bounds(clean["income"])
outliers = clean[(clean["income"] < low) | (clean["income"] > high)]
print(f"IQR bounds for income: [{low:,.0f}, {high:,.0f}]")
print("income outliers flagged:", len(outliers))
print(outliers[["customer_id", "plan", "income"]].head())
""")

nb.md(r"""
Now the judgment call. Options:
1. **Keep** — if it's a real, meaningful extreme (a genuine high-net-worth client).
2. **Cap / winsorize** — clip to the bound, keeping the row but taming its leverage.
3. **Drop** — only if you're confident it's an error (e.g. impossible value).

Our planted value (5,000,000) is implausibly large vs the rest → likely a data
error. We'll **cap** it rather than drop, to keep the customer while limiting
distortion. We document the decision.
""")

nb.code(r"""
clean["income_capped"] = clean["income"].clip(lower=low, upper=high)
print("max income before cap:", clean["income"].max())
print("max income after  cap:", clean["income_capped"].max().round(0))
""")

nb.md(r"""
**Interview line:** *"I don't delete outliers reflexively. I check whether it's an
error or a real extreme, quantify its leverage, and choose keep/cap/drop with a
written rationale. For skewed money data I lean on IQR and robust statistics."*
""")

nb.md(r"""
## 2.7 Final validation — prove the data is clean
""")

nb.code(r"""
assert clean.duplicated().sum() == 0, "still have exact dupes"
assert clean["customer_id"].is_unique, "customer_id not unique"
assert clean.isna().sum().sum() == 0, "still have missing values"
assert clean["city"].str.strip().eq(clean["city"]).all(), "whitespace remains"

print("ALL CLEANING CHECKS PASSED ✅")
print("final shape:", clean.shape)
clean.to_csv("../data/customers_clean.csv", index=False)
print("saved data/customers_clean.csv")
""")

nb.md(r"""
## 2.8 Mini-exercises

1. Re-diagnose `age` missingness: is it related to `plan` or `city`? Was median
   imputation justified?
2. Try **mean** imputation for income and compare the resulting distribution to
   the median version (plot histograms in Module 03). Which shifts more?
3. Change the IQR multiplier `k` from 1.5 to 3.0. How many outliers now? What does
   `k` control conceptually?
4. Write a one-paragraph "data cleaning report" for this dataset as if handing it
   to a teammate.
""")

nb.md(r"""
## Summary — the cleaning checklist you can recite

1. **Copy** the raw; never mutate it.
2. **Duplicates**: drop exact, then dedupe on the business key.
3. **Text/categoricals**: strip + normalize case; map real typos explicitly.
4. **Dtypes**: make types match meaning (numeric, category, datetime).
5. **Missing**: quantify → diagnose MCAR/MAR/MNAR → impute honestly (+ a missing
   flag); fit stats on train only (no leakage).
6. **Outliers**: detect with IQR/z-score, then **decide** keep/cap/drop with reasons.
7. **Validate** with assertions; save a clean artifact.

Next: **Module 03 — EDA & Visualization**, where clean data starts telling stories.
""")

out = nb.save("notebooks/02_data_cleaning.ipynb")
print("saved", out)
