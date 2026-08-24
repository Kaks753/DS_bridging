"""Builder for Module 21: Big Data & Apache Spark (PySpark)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 21 — Big Data & Apache Spark (PySpark)

When a dataset no longer fits in one machine's RAM, pandas stops working and you
reach for a **distributed** engine. **Apache Spark** is the industry standard. This
module teaches *why* big-data tools exist, the mental model (lazy, distributed,
partitioned), and the **PySpark DataFrame API** — which deliberately mirrors pandas
and SQL so your existing skills transfer.

Goals:
- Understand the big-data problem: vertical vs horizontal scaling.
- Spark's core ideas: **RDD → DataFrame**, **partitions**, **lazy evaluation**,
  **transformations vs actions**, the **DAG**.
- Do real work in PySpark: load, select/filter, `groupBy`, joins, `withColumn`,
  window functions, Spark SQL.
- Know when Spark is *overkill* (most jobs fit on one big machine!).

> **Honesty note:** this notebook runs Spark in **local mode** (one JVM pretending
> to be a cluster) so every cell executes and you see real output. The *code is
> identical* on a 500-node cluster — you'd only change the `master` URL and read
> from S3/HDFS instead of a local file. If Spark can't start in this environment,
> the notebook automatically falls back to a pandas illustration so it still runs.
""")

nb.md(r"""
## 21.1 Why "big data" tools exist — two ways to scale

You have 2 TB of logs; your laptop has 16 GB RAM. Two escape routes:

- **Vertical scaling (scale up):** buy a bigger machine (more RAM/CPU). Simple, but
  hits a hard ceiling and gets exponentially expensive.
- **Horizontal scaling (scale out):** split the data across **many commodity
  machines** and compute in parallel. Cheaper, near-unlimited — but now you need a
  framework to coordinate the machines, handle failures, and move data. **That
  framework is Spark.**

Spark keeps data **partitioned** across the cluster's memory and runs the same
operation on every partition in parallel, then combines results. That's the whole
trick: *bring the computation to the data, in parallel.*
""")

nb.md(r"""
## 21.2 Spark's mental model (the concepts interviewers probe)

- **Driver + Executors:** the *driver* runs your program and builds a plan; many
  *executors* (on worker nodes) do the actual parallel work on partitions.
- **RDD (Resilient Distributed Dataset):** the low-level, fault-tolerant
  distributed collection. You rarely use it directly now.
- **DataFrame:** a distributed table with a schema — the modern API. Optimized by
  Spark's **Catalyst** query optimizer, so it's both easier *and* faster than RDDs.
- **Partitions:** the data is chopped into chunks; parallelism = number of
  partitions worked on at once.
- **Lazy evaluation:** *transformations* (`select`, `filter`, `groupBy`, `withColumn`,
  `join`) build up a plan (a **DAG**) but **do nothing yet**. Only an **action**
  (`show`, `count`, `collect`, `write`) triggers execution. This lets Catalyst
  optimize the whole chain before running it.

**Transformations vs actions — say this in interviews:**
> "Transformations are lazy and return a new DataFrame; actions are eager and force
> computation. Spark builds a DAG of transformations and only executes it when an
> action is called, which lets it optimize and avoid wasted work."
""")

nb.code(r"""
# Try to start a real local Spark session. Fall back to pandas if unavailable.
SPARK_OK = False
try:
    from pyspark.sql import SparkSession, functions as F, Window
    spark = (SparkSession.builder
             .master("local[2]")                 # 2 local 'executors'
             .appName("bootcamp-m21")
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

nb.md(r"""
## 21.3 Create a Spark DataFrame & inspect it

The API feels like pandas + SQL. Notice `printSchema()` (dtypes) and that `show()`
is an **action** — that's the line that actually runs the job.
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

nb.md(r"""
## 21.4 select / filter / withColumn — transformations (lazy)

Compare the two dialects side by side. Everything here is a **transformation** — no
computation happens until the `show()` action.
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

nb.md(r"""
## 21.5 groupBy + aggregate — the distributed split-apply-combine

`groupBy().agg()` is the distributed version of pandas `groupby`. Under the hood
Spark **shuffles** rows so all rows of a group land on the same partition, then
aggregates. Shuffles are the expensive part of big-data jobs — minimizing them is
half of Spark performance tuning.
""")

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

nb.md(r"""
## 21.6 Joins & window functions

Joins and windows work like SQL (Module 13). A **window function** computes across a
group *without collapsing rows* — e.g. rank customers by income *within* each city.
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

nb.md(r"""
## 21.7 Spark SQL — write plain SQL over a DataFrame

Register the DataFrame as a temp view and query it with SQL. Many teams do 90% of
their Spark work this way — your Module 13 SQL transfers directly to petabytes.
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

nb.md(r"""
## 21.8 Lazy evaluation, made visible with `explain()`

`explain()` prints the **physical plan** Spark will run. Because transformations are
lazy, Spark sees the *whole* chain and optimizes it (e.g. pushing filters down to
read less data). This is why you should chain transformations and call an action
once, rather than forcing computation repeatedly.
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

nb.code(r"""
# Always release cluster resources when done.
if SPARK_OK:
    spark.stop()
    print("Spark session stopped.")
""")

nb.md(r"""
## 21.9 When NOT to use Spark (senior judgment)

Spark has real overhead (JVM startup, shuffles, serialization). **If your data fits
comfortably in one machine's RAM (say < 10–50 GB), pandas/polars/DuckDB will be
faster and simpler.** Reach for Spark when:

- data genuinely exceeds single-machine memory/disk, **or**
- you already run on a cluster (Databricks, EMR, Dataproc) and need to process
  TB–PB, **or**
- you need distributed ML (Spark MLlib) or streaming at scale.

**Modern alternatives to know:** **Polars** and **DuckDB** (blazing on a single
machine, often beat Spark up to ~100 GB), **Dask** (parallel pandas). Naming these
shows you pick the right tool, not the trendiest.

**Interview soundbite:**
> "Big data isn't about using Spark — it's about *when* you need to scale out. I
> reach for Spark once data won't fit on one machine or I'm already on a cluster;
> below that, DuckDB/Polars are faster and simpler."
""")

nb.md(r"""
## 21.10 Mini-exercises

1. Add a `withColumn` that flags high-risk customers (`support_calls >= 4`), then
   `groupBy` city to count them. Which transformation vs action did you use?
2. Rewrite the 21.5 aggregation as Spark SQL.
3. Explain to a teammate why calling `.count()` three times in a loop is slow, and
   how caching (`.cache()`) or restructuring helps.
4. Given a 5 GB CSV, argue for or against Spark vs DuckDB. What would you ask first?
""")

nb.md(r"""
## Summary

- **Scale up** (bigger box) vs **scale out** (many boxes); Spark is for scaling out.
- Spark data is **partitioned** and processed in parallel by **executors** under a
  **driver**.
- **DataFrame API** ≈ pandas + SQL, optimized by Catalyst; prefer it over raw RDDs.
- **Lazy**: transformations build a DAG; **actions** (`show/count/collect/write`)
  trigger execution — chain transformations, act once.
- `groupBy`, joins, window functions, and **Spark SQL** mirror what you learned in
  Modules 01 & 13 — your skills transfer to any scale.
- **Don't over-reach**: below single-machine memory, pandas/DuckDB/Polars win.

Next: **Module 22 — AWS Deployment & MLOps** (getting a model out of the notebook
and into production).
""")

out = nb.save("notebooks/21_bigdata_spark.ipynb")
print("saved", out)
