"""Builder for Module 13: SQL for Data Science (SQLite, zero-setup)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 13 — SQL for Data Science (Phase 1: Topics 6–7)

SQL is on your CV and it's the **#1 most-tested skill** in data interviews after
Python. Data lives in databases; you must fetch and shape it *before* pandas ever
sees it. We use **SQLite** (built into Python — nothing to install) and load our
customer data so every query runs live.

Goals:
- The **logical order** SQL executes in (not the order you write it).
- `WHERE` vs `HAVING`, aggregation, `GROUP BY`.
- All **JOIN** types with predictable row counts.
- **Subqueries**, **CTEs**, and **window functions** (the senior-level stuff).
- How SQL and pandas mirror each other.
""")

nb.code(r"""
import sqlite3
import pandas as pd

# Load our CSVs into an in-memory SQLite database
con = sqlite3.connect(":memory:")
customers = pd.read_csv("../data/customers_clean.csv")
customers.to_sql("customers", con, index=False, if_exists="replace")

# A tiny lookup table so we can practice JOINs
cities = pd.DataFrame({
    "city": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"],
    "region": ["Central", "Coast", "Nyanza", "Rift Valley", "Rift Valley"],
    "population_k": [4397, 1208, 610, 570, 475],
})
cities.to_sql("cities", con, index=False, if_exists="replace")

def q(sql):
    "Run SQL and return a DataFrame (our little query helper)."
    return pd.read_sql_query(sql, con)

print("tables:", q("SELECT name FROM sqlite_master WHERE type='table'")["name"].tolist())
q("SELECT customer_id, age, city, plan, monthly_spend, churn FROM customers LIMIT 5")
""")

nb.md(r"""
## 13.1 The clause order you WRITE vs the order SQL RUNS

You write: `SELECT … FROM … WHERE … GROUP BY … HAVING … ORDER BY … LIMIT`.
SQL logically **executes** in this order:

1. **FROM** (+ JOIN) — get and combine the tables
2. **WHERE** — filter individual **rows**
3. **GROUP BY** — bucket rows into groups
4. **HAVING** — filter **groups**
5. **SELECT** — choose/compute columns
6. **ORDER BY** — sort
7. **LIMIT** — cut

Knowing this explains *why* you can't use a `SELECT` alias in `WHERE` (WHERE runs
first) and why `HAVING` is for aggregated conditions.
""")

nb.code(r"""
# WHERE filters rows BEFORE grouping
q('''
SELECT city, plan, monthly_spend
FROM customers
WHERE monthly_spend > 100 AND plan = 'Premium'
ORDER BY monthly_spend DESC
LIMIT 5
''')
""")

nb.md(r"""
## 13.2 Aggregation: GROUP BY + WHERE vs HAVING

`WHERE` filters **rows** (before grouping). `HAVING` filters **groups** (after
aggregation). This distinction is a classic interview question.
""")

nb.code(r"""
# Churn rate & average spend per plan (AVG of a 0/1 column = a rate — same trick as pandas)
q('''
SELECT plan,
       COUNT(*)                AS n_customers,
       ROUND(AVG(monthly_spend), 2) AS avg_spend,
       ROUND(AVG(churn), 3)    AS churn_rate
FROM customers
GROUP BY plan
ORDER BY churn_rate DESC
''')
""")

nb.code(r"""
# HAVING: keep only cities with more than 50 customers
q('''
SELECT city, COUNT(*) AS n
FROM customers
GROUP BY city
HAVING COUNT(*) > 50
ORDER BY n DESC
''')
""")

nb.md(r"""
## 13.3 JOINs — combining tables

Join `customers` to `cities` to bring in `region`. Ask two questions every time:
**on what key?** and **which join type?**
- `INNER JOIN` — only rows with a match in both.
- `LEFT JOIN` — all left rows; unmatched right = NULL.
""")

nb.code(r"""
q('''
SELECT c.customer_id, c.city, ci.region, c.plan, c.monthly_spend
FROM customers AS c
LEFT JOIN cities AS ci
       ON c.city = ci.city
LIMIT 5
''')
""")

nb.code(r"""
# Aggregate AFTER joining: churn rate by region
q('''
SELECT ci.region,
       COUNT(*)             AS n_customers,
       ROUND(AVG(c.churn),3) AS churn_rate
FROM customers AS c
LEFT JOIN cities AS ci ON c.city = ci.city
GROUP BY ci.region
ORDER BY churn_rate DESC
''')
""")

nb.md(r"""
## 13.4 Subqueries — a query inside a query

Use a subquery to compare rows against an aggregate — e.g. customers who spend
**above the overall average**.
""")

nb.code(r"""
q('''
SELECT customer_id, plan, monthly_spend
FROM customers
WHERE monthly_spend > (SELECT AVG(monthly_spend) FROM customers)
ORDER BY monthly_spend DESC
LIMIT 5
''')
""")

nb.md(r"""
## 13.5 CTEs (`WITH`) — readable, multi-step queries

A **Common Table Expression** names an intermediate result, so complex logic reads
top-to-bottom instead of nesting subqueries. Interviewers love clean CTEs.
""")

nb.code(r"""
q('''
WITH plan_stats AS (
    SELECT plan,
           AVG(monthly_spend) AS plan_avg_spend
    FROM customers
    GROUP BY plan
)
SELECT c.customer_id, c.plan, c.monthly_spend,
       ROUND(p.plan_avg_spend, 2) AS plan_avg,
       ROUND(c.monthly_spend - p.plan_avg_spend, 2) AS vs_plan_avg
FROM customers AS c
JOIN plan_stats AS p ON c.plan = p.plan
ORDER BY vs_plan_avg DESC
LIMIT 5
''')
""")

nb.md(r"""
## 13.6 Window functions — aggregate WITHOUT collapsing rows

The senior-level tool. A window function computes across a set of rows **related to
the current row** while keeping every row. Syntax:
`FUNC(...) OVER (PARTITION BY ... ORDER BY ...)`.

- `ROW_NUMBER / RANK / DENSE_RANK` — ranking within groups.
- `SUM/AVG ... OVER (PARTITION BY ...)` — group stat attached to each row.
- `LAG / LEAD` — previous/next row (great for time series & churn diffs).
""")

nb.code(r"""
# Rank customers by spend WITHIN each plan (top spender per plan = rank 1)
q('''
SELECT customer_id, plan, monthly_spend,
       RANK() OVER (PARTITION BY plan ORDER BY monthly_spend DESC) AS spend_rank
FROM customers
ORDER BY plan, spend_rank
LIMIT 9
''')
""")

nb.code(r"""
# Each customer's spend vs their plan's average, on the SAME row (no GROUP BY collapse)
q('''
SELECT customer_id, plan, monthly_spend,
       ROUND(AVG(monthly_spend) OVER (PARTITION BY plan), 2) AS plan_avg,
       ROUND(monthly_spend - AVG(monthly_spend) OVER (PARTITION BY plan), 2) AS diff
FROM customers
ORDER BY diff DESC
LIMIT 5
''')
""")

nb.md(r"""
## 13.7 SQL ↔ pandas (the same ideas, two languages)

| Task | SQL | pandas |
|---|---|---|
| filter rows | `WHERE x > 5` | `df[df.x > 5]` |
| choose columns | `SELECT a, b` | `df[["a","b"]]` |
| group + aggregate | `GROUP BY g` + `AVG()` | `df.groupby("g").mean()` |
| join | `JOIN ... ON` | `df.merge(other, on=...)` |
| sort | `ORDER BY x DESC` | `df.sort_values("x", ascending=False)` |
| top-n | `LIMIT 5` | `df.head(5)` / `nlargest` |
| distinct | `SELECT DISTINCT` | `df.drop_duplicates()` |

Being fluent in **both**, and knowing they mirror each other, is exactly what data
roles want.
""")

nb.md(r"""
## 13.8 Mini-exercises (write the SQL, then check with `q(...)`)

1. Average `income` per `region` (join required), highest first.
2. Customers whose `tenure_months` is above their **own plan's** average tenure
   (window function or CTE).
3. For each city, the **rank** of customers by `monthly_spend`; return only rank 1.
4. Count churned vs stayed per plan as **two columns** (hint: `SUM(CASE WHEN ...)`).
""")

nb.code(r"""
# Example solution to #4 — conditional aggregation (pivot in SQL)
q('''
SELECT plan,
       SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) AS churned,
       SUM(CASE WHEN churn = 0 THEN 1 ELSE 0 END) AS stayed
FROM customers
GROUP BY plan
''')
""")

nb.code(r"""
con.close()   # always close the connection when done
print("connection closed — module complete")
""")

nb.md(r"""
## Summary

- SQL **executes** `FROM→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT`.
- **WHERE** filters rows; **HAVING** filters groups.
- Master all **JOINs** and predict row counts.
- **Subqueries** and **CTEs** structure complex logic; CTEs read cleaner.
- **Window functions** (`RANK`, `SUM() OVER`, `LAG/LEAD`) aggregate without
  collapsing rows — the skill that marks a strong candidate.
- SQL and pandas are two dialects of the same ideas.

Next: **Module 14 — APIs & Web Scraping** (getting data that isn't handed to you).
""")

out = nb.save("notebooks/13_sql_for_data_science.ipynb")
print("saved", out)
