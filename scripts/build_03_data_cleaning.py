"""Builder for Module 3: Data Cleaning (4-layer rewrite of old M02)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 3 — Data Cleaning: the unglamorous skill that pays

Here's a secret nobody tells you on day one: a data scientist spends **most** of
their time not building fancy models — but **cleaning messy data**. Models are
easy to swap out. Clean, trustworthy data is the hard part, the *moat*.

Interviewers know this, so they probe cleaning hard — it's how they tell apart
someone who followed a tidy tutorial from someone who's wrestled real, dirty data.

In this module we take a genuinely messy file, `customers.csv`, and clean it
**end to end** — and for every fix we say *why*. Because "why did you drop those
rows?" is a real interview question, and "uh, it seemed right" is a real rejection.
""")

nb.analogy("Cleaning data is like prepping ingredients before cooking. You wash the "
           "vegetables, cut off the bad bits, and measure things out. Skip it and the "
           "fanciest recipe (model) still tastes bad.")

nb.md("## 3.1 First, look at the mess")

nb.plain("""
Before touching anything, load the file and *look at it*. We use `.info()` — it's
like turning on the lights in a messy room: how many rows, what columns, what type
each column is, and how many values are missing. You never clean blind.
""")

nb.code(r"""
import numpy as np
import pandas as pd

df = pd.read_csv("../data/customers.csv")
print("shape:", df.shape)
df.info()
""")

nb.readcode("""
- `pd.read_csv(...)` → load the spreadsheet-like file into a DataFrame `df`.
- `df.shape` → `(rows, columns)`; here it tells us how big the dataset is.
- `df.info()` → per-column report: the name, how many non-missing values, and the
  **dtype** (data type). If a column has fewer non-null values than the row count,
  it has **missing data**.
""")

nb.jargon("dtype", "the data type of a column — int (whole number), float (decimal), object (usually text), datetime, or category")
nb.jargon("missing value / NaN", "an empty cell; pandas shows it as NaN ('Not a Number')")

nb.md("## 3.2 A cleaning philosophy (say this in interviews)")

nb.plain("""
Golden rule: **never destroy your raw data.** Keep the original untouched, work on
a *copy*, and record every change. That way if you mess up, you can start over —
and anyone can audit exactly what you did. This one habit makes you sound senior.
""")

nb.code(r"""
raw = df.copy()          # keep the untouched original — our safety net
clean = df.copy()        # we transform THIS one
""")

nb.readcode("""
- `df.copy()` → make an independent duplicate (not just another name pointing to
  the same data). We keep `raw` frozen and only ever edit `clean`.
""")

nb.interview("\"I never mutate the raw file. I keep the original, work on a copy, and record "
             "every transformation so the pipeline is reproducible and auditable.\"")

nb.md("## 3.3 Duplicates — remove noise before you measure it")

nb.plain("""
A duplicate is the same information counted twice. If a customer appears twice, any
average you compute is quietly wrong and your counts are inflated. There are two
flavours: **exact duplicates** (the whole row is repeated) and **key duplicates**
(the same customer id shows up twice, even if some other cell differs).
""")

nb.analogy("Duplicates are like counting the same person twice in a headcount — your "
           "total is wrong before you even start.")

nb.code(r"""
print("exact duplicate rows:", clean.duplicated().sum())
clean = clean.drop_duplicates()
print("after dropping exact dupes:", clean.shape)

print("duplicate customer_id:", clean.duplicated(subset='customer_id').sum())
clean = clean.drop_duplicates(subset="customer_id", keep="first")
print("after dedup on key:", clean.shape)
""")

nb.readcode("""
- `clean.duplicated()` → a True/False for each row: is this row an exact copy of an
  earlier one? `.sum()` counts the Trues.
- `clean.drop_duplicates()` → remove those exact-copy rows.
- `duplicated(subset='customer_id')` → now check duplicates *based only on the id
  column* (same customer twice).
- `drop_duplicates(subset="customer_id", keep="first")` → keep the first occurrence
  of each customer, drop the rest.
""")

nb.takeaway("Dedupe BEFORE computing any statistics; repeated rows bias your means and "
            "inflate counts. Always state which key defines a unique row.")

nb.md("## 3.4 Categorical noise — standardize your text")

nb.plain("""
Computers are painfully literal. To pandas, `"Nairobi"`, `" nairobi "`, and
`"NAIROBI"` are **three different cities**. That silently breaks grouping, joining,
and counting. The fix: strip stray spaces and force a consistent capitalization.
""")

nb.code(r"""
print("BEFORE — raw unique cities:")
print(sorted(clean["city"].unique().tolist()))

clean["city"] = clean["city"].str.strip().str.title()

print("\nAFTER — standardized cities:")
print(sorted(clean["city"].unique().tolist()))
""")

nb.readcode("""
- `clean["city"].unique()` → the distinct values in the city column (look at the mess).
- `.str.strip()` → remove leading/trailing spaces from every value at once.
- `.str.title()` → capitalize each word consistently ("nairobi" → "Nairobi").
- We reassign the result back into `clean["city"]`.
""")

nb.warn("`.strip().title()` fixes spacing and casing, but NOT real typos like "
        "\"Nairoby\". For those you map them explicitly or use fuzzy matching — never "
        "guess silently.")

nb.jargon("categorical", "a column whose values are labels/categories (like city or plan) rather than measured numbers")

nb.md("## 3.5 Wrong dtypes — make the type match the meaning")

nb.plain("""
A number stored as text can't be added. A date stored as text can't be sorted by
time. Getting the *type* right unlocks what you're allowed to do with a column.
Here we also introduce an **ordered category** — because `Basic < Standard <
Premium` has a natural order we want pandas to respect.
""")

nb.code(r"""
plan_type = pd.CategoricalDtype(categories=["Basic", "Standard", "Premium"],
                                ordered=True)
clean["plan"] = clean["plan"].astype(plan_type)
print(clean["plan"].dtype)
# Ordered categories allow comparisons:
print("rows with plan > Basic:", (clean["plan"] > "Basic").sum())

# Safe numeric coercion (turns bad strings into NaN instead of crashing):
clean["income"] = pd.to_numeric(clean["income"], errors="coerce")
print("income dtype:", clean["income"].dtype)
""")

nb.readcode("""
- `pd.CategoricalDtype([...], ordered=True)` → define a category type WITH an order.
- `.astype(plan_type)` → convert the plan column to that ordered category.
- `clean["plan"] > "Basic"` → now allowed, because the order is known.
- `pd.to_numeric(..., errors="coerce")` → convert to numbers; anything unconvertible
  becomes NaN instead of throwing an error and stopping everything.
""")

nb.deeper("""
`errors="coerce"` is the professional pattern for messy numeric columns: you convert
what you can and get NaN where you can't — then you handle those NaNs deliberately
in the missing-data step, rather than the whole script crashing on one bad cell.
""")

nb.md("## 3.6 Renaming columns — consistent names save hours")

nb.plain("""
Real files have column names like `"Customer ID"`, `"Monthly  Charges"` (with a
double space!), and random capitalization. Those spaces and cases break your code
and make joins fragile. The senior habit: convert ALL names to tidy `snake_case`
**once, right after loading.**
""")

nb.jargon("snake_case", "lowercase words joined by underscores, e.g. monthly_charges — the standard for column names")

nb.code(r"""
# Simulate messy names so you see the fix (real files really look like this):
messy = clean.copy()
messy.columns = ["Customer ID", "Age ", "Plan", "Monthly  Charges",
                 "City", "Income", "Churn"][:len(messy.columns)] + \
                list(messy.columns[7:])
print("BEFORE:", list(messy.columns)[:7])

# (1) Explicit rename of just the columns you care about:
messy = messy.rename(columns={"Customer ID": "customer_id",
                              "Monthly  Charges": "monthly_charges"})

# (2) Programmatic standardization of ALL names -> snake_case:
messy.columns = (messy.columns
                 .str.strip()                       # kill stray spaces
                 .str.lower()                        # case-insensitive
                 .str.replace(r"\s+", "_", regex=True))  # spaces -> _
print("AFTER: ", list(messy.columns)[:7])
""")

nb.readcode("""
- `messy.columns = [...]` → overwrite ALL column names at once.
- `df.rename(columns={"old": "new"})` → rename SPECIFIC columns explicitly (safest,
  most readable when you only touch a few).
- The chained `.str.strip().str.lower().str.replace(r"\\s+", "_", regex=True)` →
  strip spaces, lowercase, then turn any run of whitespace into a single underscore.
""")

nb.takeaway("Do the snake_case pass immediately after loading. Then every downstream "
            "line (df.customer_id, merges, groupby) becomes predictable. The regex `\\s+` "
            "handles single AND double spaces in one shot.")

nb.md("## 3.7 Datetimes — the type that unlocks time-based analysis")

nb.plain("""
Dates almost always arrive as **strings** ("2023-07-15"). As a string you can't
subtract two dates, sort chronologically, or pull out the month. You must *parse*
them into real datetimes. Then the `.dt` accessor hands you year/month/weekday and
lets you compute durations — the raw material for spotting seasonality.
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

nb.readcode("""
- We first BUILD fake signup dates and store them as strings (to mimic a real CSV).
- `pd.to_datetime(col, errors="coerce")` → parse strings into real datetime64;
  anything unparseable becomes `NaT` (missing) instead of crashing.
- Notice the dtype change: `object` (string) → `datetime64[ns]`.
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

nb.deeper("""
`days_as_member` (a *duration*) is often far more predictive than the raw date, and
`signup_month`/`weekday` capture **seasonality**. You can only compute these once
the column is a true datetime — which is exactly why parsing is step one of any
time-aware cleaning. Full time-series modeling comes later in the Time Series module.
""")

nb.jargon("NaT", "'Not a Time' — the datetime version of a missing value")
nb.jargon(".dt accessor", "pandas' gateway to date parts: .dt.year, .dt.month, .dt.day_name(), etc.")

nb.interview("\"Strings that look like dates are a trap — I parse with "
             "to_datetime(errors='coerce'), verify no unexpected NaT, then derive "
             "durations and calendar features via .dt.\"")

nb.md("## 3.8 Missing data — the part everyone gets wrong")

nb.plain("""
Empty cells aren't all the same, and *how* a value went missing decides the honest
way to fill it. Three cases:
- **MCAR** — missing for no reason related to anything. Random gaps.
- **MAR** — missing depends on *other columns you can see* (e.g. Premium customers
  hide income more often). You can fill using those columns.
- **MNAR** — missing depends on the *hidden value itself* (high earners hide income
  *because* it's high). The dangerous one.
""")

nb.jargon("MCAR", "Missing Completely At Random — gaps unrelated to anything")
nb.jargon("MAR", "Missing At Random — gaps explained by OTHER observed columns")
nb.jargon("MNAR", "Missing Not At Random — gaps depend on the missing value itself")

nb.code(r"""
miss = clean.isna().sum()
miss_pct = (clean.isna().mean() * 100).round(1)
print(pd.DataFrame({"missing": miss, "pct": miss_pct})[miss > 0])
""")

nb.readcode("""
- `clean.isna()` → True/False table of where values are missing.
- `.sum()` → count missing per column; `.mean()*100` → percentage missing per column.
- We only print columns where something is actually missing.
""")

nb.code(r"""
# Diagnose: is income-missingness related to plan? (built as MAR on Premium)
tmp = clean.copy()
tmp["income_missing"] = tmp["income"].isna()
print("share of income missing, by plan:")
print(tmp.groupby("plan", observed=True)["income_missing"].mean().round(3))
""")

nb.plain("""
The missing rate is much higher for Premium customers → the gaps depend on the
*plan* column we can see → this is **MAR**. So filling income with the median *of
each plan group* is honest: we use the most similar customers. We also add a flag
marking which rows were originally missing — because "was missing" can itself be a
useful signal.
""")

nb.code(r"""
clean["income_was_missing"] = clean["income"].isna().astype(int)

# group-wise median (robust to the income outlier) imputation
clean["income"] = clean.groupby("plan", observed=True)["income"] \
                       .transform(lambda s: s.fillna(s.median()))

# age missingness looked random (MCAR-ish) -> simple median is fine
clean["age"] = clean["age"].fillna(clean["age"].median())

print("remaining missing:")
rem = clean.isna().sum()
print(rem[rem > 0] if rem.sum() else "none — all filled")
""")

nb.readcode("""
- `income_was_missing` → 1 where income was originally empty, else 0 (the flag).
- `groupby("plan").transform(lambda s: s.fillna(s.median()))` → within each plan
  group, fill the gaps with that group's median. `transform` returns a column the
  same length as the original so it lines up perfectly.
- `age.fillna(age.median())` → age gaps looked random, so a plain median is fine.
""")

nb.deeper("""
Why median, not mean? Income is right-skewed with an outlier; the mean gets dragged
upward, while the median is robust. Choosing median here shows you understand your
data, not just `fillna`.

Leakage warning: in a real model you'd fit imputation statistics on the TRAINING
set only (via SimpleImputer inside a Pipeline) so the test set can't peek — otherwise
you leak information and overstate performance.
""")

nb.warn("Never impute using the target column, and never let test data influence the "
        "fill values. That's data leakage — it makes your model look better than it is.")

nb.md("## 3.9 Outliers — detect, then DECIDE (don't auto-delete)")

nb.plain("""
An outlier is a value far from the rest. It might be a genuine rare case (a real
billionaire client) or a data-entry error (income = 5,000,000 by a typo). The skill
isn't deleting them — it's *deciding* what each one is. Two standard detectors:
z-score (how many std-devs from the mean) and the IQR rule (robust, our default).
""")

nb.jargon("IQR", "Inter-Quartile Range = Q3 - Q1, the spread of the middle 50% of the data")
nb.jargon("z-score", "how many standard deviations a value sits from the mean")

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

nb.readcode("""
- `s.quantile(0.25)`/`(0.75)` → the 25th and 75th percentiles (Q1, Q3).
- `iqr = q3 - q1` → the middle-50% spread.
- Anything below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR` is flagged as an outlier.
- We select those rows with a boolean condition and count/inspect them.
""")

nb.plain("""
Now the judgment call — three honest options:
1. **Keep** it (a real, meaningful extreme).
2. **Cap / winsorize** — clip it to the boundary, keeping the row but taming its pull.
3. **Drop** it — only if you're sure it's an error.

Our planted 5,000,000 is implausible vs the rest, so we **cap** it (keep the
customer, limit the distortion) and document the decision.
""")

nb.code(r"""
clean["income_capped"] = clean["income"].clip(lower=low, upper=high)
print("max income before cap:", clean["income"].max())
print("max income after  cap:", clean["income_capped"].max().round(0))
""")

nb.readcode("""
- `.clip(lower=low, upper=high)` → any value below `low` becomes `low`, any above
  `high` becomes `high`. That's winsorizing: keep the row, cap the extremity.
""")

nb.interview("\"I don't delete outliers reflexively. I check whether it's an error or a "
             "real extreme, quantify its leverage, and choose keep/cap/drop with a written "
             "rationale. For skewed money data I lean on IQR and robust statistics.\"")

nb.md("## 3.10 Final validation — PROVE the data is clean")

nb.plain("""
Don't just believe you cleaned it — *prove* it with assertions. An `assert` is a
tripwire: if a condition isn't true, the code stops loudly. This is how you catch a
cleaning mistake before it silently poisons a model.
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

nb.readcode("""
- `assert CONDITION, "message"` → if CONDITION is False, stop and print the message.
- We check: no exact dupes, unique ids, zero missing, no stray whitespace.
- `to_csv(..., index=False)` → save the cleaned data (without the row-number index).
""")

nb.takeaway("Ending a cleaning notebook with assertions + a saved clean artifact is a "
            "senior move — it makes your pipeline trustworthy and repeatable.")

nb.md("## 3.11 Try it yourself")

nb.try_this("""
1. Re-diagnose `age` missingness: is it related to `plan` or `city`? Was median
   imputation justified?
2. Try mean imputation for income and compare the distribution to the median version.
   Which shifts more, and why?
3. Change the IQR multiplier `k` from 1.5 to 3.0. How many outliers now? What does
   `k` control conceptually?
4. Write a one-paragraph "data cleaning report" for this dataset as if handing it to
   a teammate.
""")

nb.md(r"""
## Summary — the cleaning checklist you can recite

1. **Copy** the raw; never mutate it.
2. **Duplicates**: drop exact, then dedupe on the business key.
3. **Text/categoricals**: strip + normalize case; map real typos explicitly.
4. **Dtypes**: make types match meaning (numeric, category, datetime).
5. **Rename**: snake_case everything, once, up front.
6. **Missing**: quantify → diagnose MCAR/MAR/MNAR → impute honestly (+ a missing
   flag); fit stats on train only (no leakage).
7. **Outliers**: detect with IQR/z-score, then **decide** keep/cap/drop with reasons.
8. **Validate** with assertions; save a clean artifact.

Next: **EDA & Visualization**, where clean data starts telling stories.
""")

out = nb.save("notebooks/03_data_cleaning.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
