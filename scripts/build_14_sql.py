"""Builder for Module 14: SQL for Data Science (4-layer rewrite of old M13)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 14 — SQL for Data Science

SQL is on your CV, and it's the **#1 most-tested skill** in data interviews after
Python. Here's the reality of the job: data lives in databases, and you must fetch
and shape it with SQL *before* pandas ever sees it. We'll use **SQLite** (built into
Python — nothing to install) and load the customer data so every query below runs
live and returns a real table.
""")

nb.analogy("A database is a giant, well-organised warehouse; SQL is the order form you hand "
           "the warehouse staff. You don't wander the aisles yourself (that's slow) — you "
           "write a precise request ('all Premium customers in Nairobi, sorted by spend') "
           "and they bring back exactly that.")

nb.md("## 14.1 Setup — a live database in memory")

nb.code(r"""
import sqlite3
import pandas as pd

# Load our CSVs into an in-memory SQLite database
con = sqlite3.connect(":memory:")
customers = pd.read_csv("../data/customers_clean.csv")
customers.to_sql("customers", con, index=False, if_exists="replace")

# A tiny lookup table so we can practise JOINs
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

nb.readcode("""
- `sqlite3.connect(':memory:')` spins up a throwaway database that lives in RAM — perfect
  for practice, gone when we close it.
- `df.to_sql(...)` loads a DataFrame in as a table; our `q(...)` helper runs any SQL
  string and hands back the result as a DataFrame so it prints nicely.
""")

nb.jargon("SQL", "Structured Query Language — the standard language for asking questions of databases")
nb.jargon("table", "a database's rows-and-columns dataset, like a spreadsheet tab")
nb.jargon("query", "a single SQL request for data")

nb.md("## 14.2 The order you WRITE vs the order SQL RUNS")

nb.plain("""
This one insight unlocks most SQL confusion. You WRITE a query as
SELECT … FROM … WHERE … GROUP BY … HAVING … ORDER BY … LIMIT. But SQL RUNS the
clauses in a different logical order — it grabs the tables first and chooses the
displayed columns almost last.
""")

nb.md(r"""
SQL's logical execution order:
1. **FROM** (+ JOIN) — get and combine the tables
2. **WHERE** — filter individual **rows**
3. **GROUP BY** — bucket rows into groups
4. **HAVING** — filter **groups**
5. **SELECT** — choose/compute the columns
6. **ORDER BY** — sort
7. **LIMIT** — cut
""")

nb.deeper("""
Why does this matter? Because it explains two things beginners trip on: (1) you can't
use a column ALIAS you defined in SELECT inside your WHERE clause — WHERE runs long
before SELECT exists; and (2) HAVING, not WHERE, is where you filter on an aggregate
like COUNT(*) > 50, because aggregation (GROUP BY) hasn't happened yet when WHERE runs.
Recite the execution order in an interview and a whole class of 'why doesn't this work?'
questions answer themselves.
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

nb.md("## 14.3 Aggregation: GROUP BY, and WHERE vs HAVING")

nb.plain("""
Aggregation collapses many rows into a summary: count per plan, average spend per
city. The key distinction — a classic interview trap — is WHERE filters individual
ROWS (before grouping), while HAVING filters GROUPS (after aggregation).
""")

nb.code(r"""
# Churn rate & average spend per plan.
# Neat trick: AVG of a 0/1 column IS the rate — same idea as in pandas.
q('''
SELECT plan,
       COUNT(*)                     AS n_customers,
       ROUND(AVG(monthly_spend), 2) AS avg_spend,
       ROUND(AVG(churn), 3)         AS churn_rate
FROM customers
GROUP BY plan
ORDER BY churn_rate DESC
''')
""")

nb.code(r"""
# HAVING: keep only cities with more than 50 customers (a filter on the GROUP)
q('''
SELECT city, COUNT(*) AS n
FROM customers
GROUP BY city
HAVING COUNT(*) > 50
ORDER BY n DESC
''')
""")

nb.takeaway("WHERE filters rows, HAVING filters groups. If your condition uses an aggregate "
            "(COUNT, AVG, SUM), it belongs in HAVING; if it's about a single row's raw value, "
            "it belongs in WHERE.")

nb.jargon("GROUP BY", "bucket rows that share a value so you can aggregate within each bucket")
nb.jargon("aggregate function", "COUNT/SUM/AVG/MIN/MAX — collapses many rows into one summary number")

nb.md("## 14.4 JOINs — combining tables")

nb.plain("""
Real questions span tables. To attach each customer's region, we JOIN customers to
cities. Every JOIN needs two answers: ON WHAT KEY do we match, and WHICH TYPE of
join? INNER keeps only matched rows; LEFT keeps every left-table row and fills NULL
where the right table has no match.
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
       COUNT(*)               AS n_customers,
       ROUND(AVG(c.churn), 3) AS churn_rate
FROM customers AS c
LEFT JOIN cities AS ci ON c.city = ci.city
GROUP BY ci.region
ORDER BY churn_rate DESC
''')
""")

nb.interview("""
Be ready to PREDICT ROW COUNTS: an INNER JOIN can only shrink or hold the row count
(unmatched rows drop); a LEFT JOIN keeps all left rows but can GROW the total if a
left row matches several right rows (a one-to-many join multiplies). 'How many rows
does this join return?' is a favourite screen — reason it from the key's uniqueness.
""")

nb.jargon("JOIN", "combine rows from two tables based on a matching key column")
nb.jargon("INNER JOIN", "keep only rows that have a match in BOTH tables")
nb.jargon("LEFT JOIN", "keep every row from the left table; NULL-fill where the right has no match")

nb.md("## 14.5 Subqueries and CTEs — structuring complex logic")

nb.plain("""
A subquery is a query nested inside another — handy for comparing rows against an
aggregate (e.g. 'who spends above the overall average?'). A CTE (the WITH clause)
does the same job but names the intermediate result, so complex logic reads
top-to-bottom instead of nesting inside-out.
""")

nb.code(r"""
# Subquery: customers spending above the overall average
q('''
SELECT customer_id, plan, monthly_spend
FROM customers
WHERE monthly_spend > (SELECT AVG(monthly_spend) FROM customers)
ORDER BY monthly_spend DESC
LIMIT 5
''')
""")

nb.code(r"""
# CTE: same idea per PLAN, but readable top-to-bottom
q('''
WITH plan_stats AS (
    SELECT plan, AVG(monthly_spend) AS plan_avg_spend
    FROM customers
    GROUP BY plan
)
SELECT c.customer_id, c.plan, c.monthly_spend,
       ROUND(p.plan_avg_spend, 2)              AS plan_avg,
       ROUND(c.monthly_spend - p.plan_avg_spend, 2) AS vs_plan_avg
FROM customers AS c
JOIN plan_stats AS p ON c.plan = p.plan
ORDER BY vs_plan_avg DESC
LIMIT 5
''')
""")

nb.readcode("""
- The subquery `(SELECT AVG(monthly_spend) FROM customers)` computes one number that
  the outer WHERE compares each row against.
- The CTE `plan_stats` is computed once, given a name, then JOINed like a table —
  cleaner and reusable versus nesting the same subquery repeatedly.
""")

nb.jargon("subquery", "a query nested inside another query")
nb.jargon("CTE (WITH)", "a named intermediate result that makes multi-step queries readable")

nb.md("## 14.6 Window functions — the senior-level tool")

nb.plain("""
Window functions are what separate a strong SQL candidate from an average one. They
compute across a set of rows RELATED to the current row — but, unlike GROUP BY, they
keep every row instead of collapsing them. Syntax:
FUNC(...) OVER (PARTITION BY ... ORDER BY ...).
""")

nb.code(r"""
# Rank customers by spend WITHIN each plan (top spender in a plan = rank 1)
q('''
SELECT customer_id, plan, monthly_spend,
       RANK() OVER (PARTITION BY plan ORDER BY monthly_spend DESC) AS spend_rank
FROM customers
ORDER BY plan, spend_rank
LIMIT 9
''')
""")

nb.code(r"""
# Attach each plan's average to EVERY row (no GROUP BY collapse) and compute the gap
q('''
SELECT customer_id, plan, monthly_spend,
       ROUND(AVG(monthly_spend) OVER (PARTITION BY plan), 2) AS plan_avg,
       ROUND(monthly_spend - AVG(monthly_spend) OVER (PARTITION BY plan), 2) AS diff
FROM customers
ORDER BY diff DESC
LIMIT 5
''')
""")

nb.deeper("""
PARTITION BY is 'GROUP BY that doesn't collapse' — it defines the window each row's
function looks across. The heavy hitters: ROW_NUMBER / RANK / DENSE_RANK for ranking
within groups (RANK leaves gaps after ties, DENSE_RANK doesn't); SUM/AVG OVER for a
group stat glued onto each row; and LAG / LEAD to reach the previous/next row — the
key to period-over-period changes and churn diffs in time-series data. 'Give me the
top-2 spenders per plan' is a canonical window-function question.
""")

nb.jargon("window function", "computes over rows related to the current one WITHOUT collapsing them")
nb.jargon("PARTITION BY", "defines the group of rows a window function operates within")

nb.md("## 14.7 SQL ↔ pandas — two dialects of the same ideas")

nb.md(r"""
| Task | SQL | pandas |
|---|---|---|
| filter rows | `WHERE x > 5` | `df[df.x > 5]` |
| choose columns | `SELECT a, b` | `df[["a","b"]]` |
| group + aggregate | `GROUP BY g` + `AVG()` | `df.groupby("g").mean()` |
| join | `JOIN ... ON` | `df.merge(other, on=...)` |
| sort | `ORDER BY x DESC` | `df.sort_values("x", ascending=False)` |
| top-n | `LIMIT 5` | `df.head(5)` / `nlargest` |
| distinct | `SELECT DISTINCT` | `df.drop_duplicates()` |
""")

nb.takeaway("SQL and pandas are the same ideas in two languages. Being fluent in BOTH — and "
            "able to say 'this GROUP BY is that .groupby()' — is exactly what data roles want.")

nb.md("## 14.8 Try it yourself")

nb.code(r"""
# Worked example of a common interview move: conditional aggregation (a pivot in SQL)
q('''
SELECT plan,
       SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) AS churned,
       SUM(CASE WHEN churn = 0 THEN 1 ELSE 0 END) AS stayed
FROM customers
GROUP BY plan
''')
""")

nb.try_this("""
Write the SQL, then check with q(...):
1. Average income per region (join required), highest first.
2. Customers whose tenure_months is above their OWN plan's average tenure (window or CTE).
3. For each city, rank customers by monthly_spend; return only rank 1.
4. Count churned vs stayed per plan as two columns — the SUM(CASE WHEN...) trick above.
""")

nb.code(r"""
con.close()   # always close the connection when done
print("connection closed — module complete")
""")

nb.md("## Summary")

nb.takeaway("""
- SQL **executes** FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT.
- **WHERE** filters rows; **HAVING** filters groups.
- Master all **JOINs** and be able to predict row counts.
- **Subqueries** and **CTEs** structure complex logic; CTEs read cleaner.
- **Window functions** (RANK, SUM() OVER, LAG/LEAD) aggregate WITHOUT collapsing rows — the skill that marks a strong candidate.
- SQL and pandas are two dialects of the same ideas.
""")

nb.md(r"""
Next: **Module 15 — APIs & Web Scraping** — getting data that isn't handed to you
in a tidy CSV.
""")

out = nb.save("notebooks/14_sql_for_data_science.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
