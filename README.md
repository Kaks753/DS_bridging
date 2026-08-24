# DS_bridging — A Data Science Bootcamp (Zero → Job-Ready)

> Built for **Stephen Muema** — Mathematics grad (Maasai Mara) + Moringa School DS.
> Goal: turn "I have projects but shaky foundations" into **quiet, deep confidence** —
> the kind where you can explain *why*, not just run the code.

This is not a cheat-sheet dump. Every notebook teaches **from first principles**,
shows the **math intuition**, runs on **realistic messy data**, and ends with
**interview-grade talking points** so you can *speak* like a pro, not just code like one.

---

## How to use this repo

1. Install deps: `pip install -r requirements.txt`
2. Regenerate the practice datasets (already committed, but reproducible):
   `python scripts/make_datasets.py`
3. Open notebooks **in order** in `notebooks/`. Run every cell. Then re-run from
   scratch (`Kernel → Restart & Run All`) — if it breaks, you learn why.
4. Read the matching file in `notes/` for fast revision before interviews.
5. Do the drills in `exercises/`.

**Golden rule:** don't just read. Type it, break it, fix it, explain it out loud.

---

## Curriculum

| # | Module | You will *own* |
|---|--------|----------------|
| 00 | Python + NumPy foundations | vectorization, broadcasting, why loops are slow |
| 01 | Pandas deep dive | indexing, groupby, merge, reshape, `apply` vs vectorized |
| 02 | Data cleaning | missing data (MCAR/MAR/MNAR), dtypes, duplicates, outliers |
| 03 | EDA + visualization | asking questions with data, matplotlib/seaborn, storytelling |
| 04 | Statistics & probability | distributions, CLT, hypothesis tests, p-values, confidence intervals |
| 05 | Feature engineering & scaling | skewness, encoding, Standard/MinMax/Robust, leakage |
| 06 | Regression | linear regression from scratch + sklearn, assumptions, metrics |
| 07 | Classification + KNN | logistic regression, KNN, decision boundaries, imbalance |
| 08 | Evaluation, CV & tuning | train/val/test, k-fold, GridSearch, the right metric |
| 09 | Trees, ensembles & boosting | decision trees, Random Forest, XGBoost, bias–variance |
| 10 | Unsupervised: clustering & PCA | KMeans, silhouette, PCA math + intuition |
| 11 | Neural networks intro | neurons, activations, backprop intuition, a tiny net |
| 12 | Job readiness | how to *talk* DS, project narratives, interview Q&A |
| 13 | SQL for data science | execution order, JOINs, subqueries, CTEs, window functions |
| 14 | APIs & web scraping | requests, JSON, status codes, BeautifulSoup, ethics |
| 15 | Probability, combinatorics & Bayes | counting, Bayes' theorem, Naive Bayes classifier |
| 16 | A/B testing, ANOVA & power | experiments, z/t-tests, ANOVA+Tukey, power analysis |
| 17 | OOP + linear algebra & calculus | classes, sklearn estimator pattern, eigen/PCA, gradient descent |
| 18 | Time series & forecasting | decomposition, stationarity, ARIMA/SARIMA, MAPE |
| 19 | NLP & recommendation systems | TF-IDF, **n-grams/bigrams**, cosine similarity, sentiment, **LDA topics**, content/collaborative |
| 20 | Bash & Git for data science | shell file-wrangling (`grep/cut/awk/pipes`), the git model, branching, `.gitignore` |
| 21 | Big Data & Apache Spark | scale-out, partitions, lazy eval, PySpark DataFrames, Spark SQL, when *not* to use it |
| 22 | Deployment & MLOps | serialize (joblib), serving API, Docker, the AWS map, drift monitoring |

> **Modules 00–12** are the core ML spine; **13–19** fill in the full
> Moringa/Flatiron syllabus (SQL, APIs/scraping, probability & Bayes, A/B testing,
> OOP/linear-algebra/calculus, time series, NLP & recommenders); **20–22** add the
> engineering/production track (Bash/Git, Spark/Big Data, AWS deployment & MLOps).
> Nothing crucial is left out.

**Deepened in this pass (per your checklist):** column **renaming** + **datetime
parsing** (`to_datetime`/`.dt`) in M02; **all** scalers — Standard/MinMax/Robust/
**MaxAbs**/**Normalizer**/**Quantile**/Power — in M05; KNN **distance metrics**,
**weights**, **regression**, and the **curse of dimensionality** in M07;
**hierarchical (dendrogram)**, **DBSCAN**, **GMM**, and **LDA vs PCA** in M10;
**specificity** added to the metrics vocabulary in M07; **n-grams** + **LDA topic
modeling** in M19.

### Coverage vs the official 40-topic Data Science syllabus

| Phase | Official topics | Where it's covered here |
|---|---|---|
| **1** | Python, Pandas, cleaning, **SQL**, **APIs**, **web scraping** | M00–M02, **M13**, **M14** |
| **2** | Probability, distributions, CLT/CI, hypothesis tests, **ANOVA/power**, **A/B testing**, **Bayes**, linear regression | M04, **M15**, **M16**, M06 |
| **3** | **OOP**+sklearn, **linear algebra/calculus**, ML fundamentals, logistic, class metrics, trees, KNN, **Naive Bayes**, tuning/pipelines, ensembles | **M17**, M05–M09, M15 |
| **4** | PCA, **LDA**, clustering (KMeans/**hierarchical**/**DBSCAN**/**GMM**), **time series**, **NLP**, **recommenders**, neural nets, tuning | M10, **M18**, **M19**, M11 |
| **Infra / Production** | **Bash/Git**, **Big Data/Spark**, **AWS deployment & MLOps** | **M20**, **M21**, **M22** |

*Every topic across all four phases — plus the infrastructure/production track — is
now covered with runnable, verified notebooks.*

---

## Datasets (in `data/`, generated by `scripts/make_datasets.py`)

- **customers.csv** — churn problem; deliberately messy (missing, dupes, noise, outlier).
- **house_prices.csv** — clean-ish regression target.
- **blobs_2d.csv** — 3 natural clusters for KMeans / PCA.

All synthetic and reproducible (`seed=42`) — no downloads, no licensing issues,
and you can *see* the ground truth to check whether your methods actually work.

---

## Repo layout

```
notebooks/   # the lessons (run these)
notes/       # 1-page revision notes per module
exercises/   # practice methodology + drills (deliberately no answer key)
data/        # practice datasets
scripts/     # dataset generator + notebook build tooling
```

---

*Learn it so well you could teach it. That's the bar.*
