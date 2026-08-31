"""Builder for Module 22: Big Data & Apache Spark (PySpark) (4-layer)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 22 — Big Data & Apache Spark (PySpark)

When a dataset no longer fits in one machine's RAM, pandas stops working and you reach
for a **distributed** engine. **Apache Spark** is the industry standard. This module
teaches *why* big-data tools exist, the mental model (lazy, distributed, partitioned),
and the **PySpark DataFrame API** — which deliberately mirrors pandas and SQL so your
existing skills transfer.

**What you'll be able to do by the end:**
- Explain vertical vs horizontal scaling and why Spark exists.
- Use Spark's core ideas: DataFrame, partitions, lazy evaluation,
  transformations vs actions.
- Do real work in PySpark: select/filter, `groupBy`, joins, window functions, SQL.
- Judge when Spark is *overkill* (most jobs fit on one big machine!).
""")

nb.plain(r"""
"Big data" just means the data is too big for one computer's memory. Two fixes: buy a
*bigger* computer (only goes so far), or split the work across *many* computers working
at once. Spark is the manager that coordinates that team of computers — chopping the
data into pieces, handing a piece to each worker, and gluing the results back together.
The good news: the Spark code looks almost exactly like pandas and SQL you already know.
""")

nb.md(r"""
> **Honesty note:** this notebook runs Spark in **local mode** (one JVM pretending to
> be a cluster) so every cell executes and you see real output. The *code is identical*
> on a 500-node cluster — you'd only change the `master` URL and read from S3/HDFS. If
> Spark can't start here, the notebook automatically falls back to a pandas illustration
> so it still runs.
""")

# ---------------------------------------------------------------------------
# 22.1 Why big data
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.1 Why "big data" tools exist — two ways to scale
""")

nb.analogy(r"""
You need to move a mountain of sand. **Scaling up** = buy one gigantic truck — fast to
organize, but there's a limit to how big a truck you can buy, and the biggest ones cost
a fortune. **Scaling out** = hire 100 people with wheelbarrows working in parallel —
cheaper and almost unlimited, but now you need a *foreman* to divide the work and
combine it. Spark is that foreman for data.
""")

nb.jargon("Vertical scaling (scale up)", "buy a bigger single machine (more RAM/CPU); simple but has a hard ceiling")
nb.jargon("Horizontal scaling (scale out)", "split work across many machines running in parallel; cheaper and near-unlimited")
nb.jargon("Partition", "a chunk of the data that one worker processes; parallelism = many partitions at once")

nb.deeper(r"""
Spark keeps data **partitioned** across the cluster's memory and runs the same operation
on every partition in parallel, then combines the results. The core trick is *bringing
the computation to the data, in parallel*, rather than pulling all the data to one place.
That's why it can chew through terabytes that would never fit in a single machine's RAM.
""")

nb.takeaway("Scale up = bigger box (limited); scale out = many boxes in parallel (Spark). Spark partitions data and computes on each chunk simultaneously.")

# ---------------------------------------------------------------------------
# 22.2 Mental model
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.2 Spark's mental model (the concepts interviewers probe)
""")

nb.plain(r"""
- **Driver + Executors**: the *driver* is you/your program planning the work; the
  *executors* are the workers doing it in parallel.
- **DataFrame**: a distributed table with named columns — the modern API, and it looks
  just like pandas.
- **Lazy evaluation**: this is the big one. When you write `filter`, `select`,
  `groupBy`, Spark doesn't *do* anything yet — it just writes down the recipe. Only when
  you call an **action** (`show`, `count`, `collect`, `write`) does it actually cook.
""")

nb.jargon("Transformation", "a lazy operation (select/filter/groupBy) that adds a step to the plan but doesn't run yet")
nb.jargon("Action", "an eager operation (show/count/collect/write) that forces Spark to actually execute the plan")
nb.jargon("Lazy evaluation", "building up a plan without running it, so Spark can optimize the whole chain before executing")

nb.interview(r"""
"Transformations are lazy and return a new DataFrame; actions are eager and force
computation. Spark builds a DAG of transformations and only executes when an action is
called — which lets it optimize the whole chain and avoid wasted work."
""")

nb.code(r"""
# Try to start a real local Spark session. Fall back to pandas if unavailable.
SPARK_OK = False
try:
    from pyspark.sql import SparkSession, functions as F, Window
    spark = (SparkSession.builder
             .master("local[2]")                 # 2 local 'executors'
             .appName("bootcamp-m22")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    SPARK_OK = True
    print("Spark version:", spark.version)
    print("default parallelism (cores):", spark.sparkContext.defaultParallelism)
except Exception as e:
    print("Spark unavailable, using pandas fallback. Reason:", type(e).__name__)
""")

# ---------------------------------------------------------------------------
# 22.3 Create + inspect
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.3 Create a Spark DataFrame & inspect it

The API feels like pandas + SQL. Notice `show()` is an **action** — that's the line
that actually runs the job.
""")

nb.code(r"""
if SPARK_OK:
    data = [
        ("Ann",  "Nairobi",  "Premium",  92000, 3),
        ("Ben",  "Nairobi",  "Basic",    41000, 0),
        ("Cara", "Mombasa",  "Standard", 55000, 5),
        ("Dan",  "Mombasa",  "Premium",  120000, 1),
        ("Eve",  "Kisumu",   "Basic",    38000, 6),
        ("Fay",  "Nairobi",  "Standard", 60000, 2),
    ]
    cols = ["name", "city", "plan", "income", "support_calls"]
    sdf = spark.createDataFrame(data, cols)
    sdf.printSchema()
    sdf.show()
    print("num partitions:", sdf.rdd.getNumPartitions())
else:
    import pandas as pd
    sdf = pd.DataFrame([
        ("Ann","Nairobi","Premium",92000,3),("Ben","Nairobi","Basic",41000,0),
        ("Cara","Mombasa","Standard",55000,5),("Dan","Mombasa","Premium",120000,1),
        ("Eve","Kisumu","Basic",38000,6),("Fay","Nairobi","Standard",60000,2)],
        columns=["name","city","plan","income","support_calls"])
    print(sdf)
""")

nb.readcode(r"""
- `createDataFrame(data, cols)` builds a distributed table from rows + column names.
- `printSchema()` shows the column types (like `df.dtypes` in pandas).
- `show()` is the **action** that triggers the job and prints the table.
- `getNumPartitions()` reveals how many chunks the data is split into — that's how much
  parallelism is available.
""")

# ---------------------------------------------------------------------------
# 22.4 select/filter/withColumn
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.4 select / filter / withColumn — transformations (lazy)

Everything here is a **transformation** — no computation happens until the `show()`
action fires.
""")

nb.code(r"""
if SPARK_OK:
    result = (sdf
              .filter(F.col("income") > 50000)             # WHERE
              .withColumn("income_k", F.round(F.col("income")/1000, 1))  # new col
              .select("name", "city", "plan", "income_k")) # projection
    result.show()   # <-- ACTION: this is what triggers the job
else:
    r = sdf[sdf.income > 50000].copy()
    r["income_k"] = (r["income"]/1000).round(1)
    print(r[["name","city","plan","income_k"]])
""")

nb.readcode(r"""
- `filter` = SQL's WHERE (keep rows where income > 50000).
- `withColumn` adds a computed column (income in thousands).
- `select` picks the columns to keep.
- All three are lazy — Spark just records them. The final `show()` is the action that
  runs the whole chain at once.
""")

nb.takeaway("select/filter/withColumn are lazy transformations; nothing computes until an action like show() runs the recorded plan.")

# ---------------------------------------------------------------------------
# 22.5 groupBy
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.5 groupBy + aggregate — distributed split-apply-combine

`groupBy().agg()` is the distributed version of pandas `groupby`.
""")

nb.analogy(r"""
To count customers per city across 100 machines, Spark first has to make sure all the
"Nairobi" rows end up on the *same* worker before counting — that reshuffling of rows
across the network is called a **shuffle**, and it's the slow, expensive part of big-data
jobs. Half of Spark tuning is just "how do I do fewer shuffles?"
""")

nb.jargon("Shuffle", "moving rows across machines so each group's rows land together; the expensive part of a distributed job")

nb.code(r"""
if SPARK_OK:
    (sdf.groupBy("city")
        .agg(F.count("*").alias("customers"),
             F.round(F.avg("income"), 0).alias("avg_income"),
             F.max("support_calls").alias("max_calls"))
        .orderBy(F.col("customers").desc())
        .show())
else:
    g = (sdf.groupby("city")
            .agg(customers=("name","count"),
                 avg_income=("income","mean"),
                 max_calls=("support_calls","max"))
            .sort_values("customers", ascending=False))
    print(g.round(0))
""")

nb.takeaway("groupBy().agg() is distributed split-apply-combine; it triggers a shuffle, so minimizing shuffles is core to Spark performance.")

# ---------------------------------------------------------------------------
# 22.6 Joins & windows
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.6 Joins & window functions

These work just like SQL. A **window function** computes across a group *without
collapsing rows* — e.g. rank customers by income *within* each city.
""")

nb.code(r"""
if SPARK_OK:
    plans = spark.createDataFrame(
        [("Basic", 10), ("Standard", 25), ("Premium", 60)],
        ["plan", "monthly_fee"])

    joined = sdf.join(plans, on="plan", how="left")   # enrich with plan fee

    w = Window.partitionBy("city").orderBy(F.col("income").desc())
    ranked = joined.withColumn("income_rank_in_city", F.rank().over(w))
    ranked.select("name","city","plan","income","monthly_fee",
                  "income_rank_in_city").orderBy("city","income_rank_in_city").show()
else:
    plans = {"Basic":10,"Standard":25,"Premium":60}
    j = sdf.copy(); j["monthly_fee"] = j["plan"].map(plans)
    j["income_rank_in_city"] = (j.groupby("city")["income"]
                                 .rank(ascending=False, method="min").astype(int))
    print(j.sort_values(["city","income_rank_in_city"])
            [["name","city","plan","income","monthly_fee","income_rank_in_city"]])
""")

nb.readcode(r"""
- `join(plans, on="plan", how="left")` enriches each customer with their plan's fee —
  identical idea to a SQL LEFT JOIN.
- `Window.partitionBy("city").orderBy(income desc)` defines "within each city, ordered
  by income"; `F.rank().over(w)` then ranks customers *inside* their city without
  merging rows. Your SQL window-function knowledge transfers directly.
""")

# ---------------------------------------------------------------------------
# 22.7 Spark SQL
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.7 Spark SQL — write plain SQL over a DataFrame

Register the DataFrame as a temp view and query it with SQL. Many teams do 90% of their
Spark work this way — your SQL transfers directly to petabytes.
""")

nb.code(r"""
if SPARK_OK:
    sdf.createOrReplaceTempView("customers")
    spark.sql('''
        SELECT plan,
               COUNT(*)            AS n,
               ROUND(AVG(income))  AS avg_income
        FROM customers
        GROUP BY plan
        ORDER BY avg_income DESC
    ''').show()
else:
    print(sdf.groupby("plan")
             .agg(n=("name","count"), avg_income=("income","mean"))
             .sort_values("avg_income", ascending=False).round(0))
""")

nb.takeaway("createOrReplaceTempView + spark.sql lets you run ordinary SQL at petabyte scale -- your SQL skills carry straight over.")

# ---------------------------------------------------------------------------
# 22.8 explain()
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.8 Lazy evaluation, made visible with `explain()`

`explain()` prints the plan Spark will run. Because transformations are lazy, Spark sees
the *whole* chain and optimizes it (e.g. pushing filters down to read less data). That's
why you chain transformations and call an action once, rather than forcing computation
repeatedly.
""")

nb.code(r"""
if SPARK_OK:
    plan = (sdf.filter(F.col("income") > 40000)
               .groupBy("city").count())
    plan.explain()          # shows the optimized physical plan (still lazy!)
    print("\n--- now the ACTION runs it ---")
    plan.show()
else:
    print("(explain() is Spark-specific; pandas is eager so there is no plan.)")
""")

nb.warn("Calling an action (like .count()) repeatedly in a loop re-runs the whole plan each time. Chain transformations and act once, or .cache() a reused DataFrame.")

nb.code(r"""
# Always release cluster resources when done.
if SPARK_OK:
    spark.stop()
    print("Spark session stopped.")
""")

# ---------------------------------------------------------------------------
# 22.9 When NOT to use Spark
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.9 When NOT to use Spark (senior judgment)
""")

nb.plain(r"""
Spark isn't free — starting the cluster, shuffling data, and packaging it up all add
overhead. If your data fits comfortably in one machine's RAM (roughly < 10-50 GB),
plain pandas — or the faster **Polars** / **DuckDB** — will beat Spark and be simpler.
Reach for Spark only when data genuinely won't fit on one machine, you're already on a
cluster (Databricks/EMR), or you need distributed ML/streaming.
""")

nb.jargon("DuckDB / Polars", "modern single-machine engines that often outrun Spark up to ~100 GB with far less setup")

nb.interview(r"""
"Big data isn't about using Spark — it's about *when* you need to scale out. I reach for
Spark once data won't fit on one machine or I'm already on a cluster; below that,
DuckDB/Polars are faster and simpler. Picking the right tool beats picking the trendiest."
""")

nb.takeaway("Don't over-reach: below single-machine memory, pandas/DuckDB/Polars win; use Spark only when you truly must scale out.")

# ---------------------------------------------------------------------------
# Practice + summary
# ---------------------------------------------------------------------------
nb.md(r"""
## 22.10 Practice
""")

nb.try_this(r"""
1. Add a `withColumn` that flags high-risk customers (`support_calls >= 4`), then
   `groupBy` city to count them. Which parts were transformations vs the action?
2. Rewrite the 22.5 aggregation as Spark SQL.
3. Explain why calling `.count()` three times in a loop is slow, and how `.cache()` or
   restructuring helps.
4. Given a 5 GB CSV, argue for or against Spark vs DuckDB. What would you ask first?
""")

nb.md(r"""
## Summary

- **Scale up** (bigger box) vs **scale out** (many boxes); Spark is for scaling out.
- Spark data is **partitioned** and processed in parallel by **executors** under a
  **driver**.
- **DataFrame API** ~ pandas + SQL; prefer it over raw RDDs.
- **Lazy**: transformations build a plan; **actions** (`show/count/collect/write`)
  trigger execution — chain transformations, act once. Watch out for **shuffles**.
- `groupBy`, joins, window functions, and **Spark SQL** mirror what you already know —
  your skills transfer to any scale.
- **Don't over-reach**: below single-machine memory, pandas/DuckDB/Polars win.

Next: **Module 23 — AWS Deployment & MLOps** (getting a model out of the notebook and
into production).
""")

out = nb.save("notebooks/22_bigdata_spark.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
