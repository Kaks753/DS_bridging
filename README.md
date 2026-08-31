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

## Learning website (read on any device)

The notebooks are also published as a **static learning website** — every lesson
rendered as a clean, colour-coded page with the 4-layer callouts, a searchable
sidebar, and **"Open in Colab" / "View on GitHub"** run badges so you can execute
any lesson for real in one click.

- **Build it locally:** `python scripts/build_site.py` → output in `site/`
  (serve with `python -m http.server` inside `site/`).
- **Deploy:** the site is built into `public/` for **Vercel**
  (`SITE_OUT=public python scripts/build_site.py`). Deployment is left to you —
  no tokens are stored in this repo.
- **Fully mobile-responsive:** tested at 320 / 360 / 414 px with **zero horizontal
  overflow** — readable on the smallest phones. The sidebar collapses into an
  off-canvas menu (☰) with a dimming overlay; code blocks and wide tables scroll
  sideways inside their own box instead of stretching the page; headings scale
  fluidly with `clamp()`.

---

## Curriculum

**24 modules, numbered 0–23.** Every lesson is written in the **4-layer teaching
style**: 🌱 *plain English* first → 🔤 *reading the code* line-by-line → 🎓 *go deeper*
(the rigorous why) → ✅ *takeaway* + 🗣️ *say-this-in-an-interview*. Math-heavy modules
also include **by-hand worked examples** you can check against the code.

| # | Module | You will *own* |
|---|--------|----------------|
| 0  | **Absolute Basics: Python from Zero** | what code/a notebook/a "cell" is, variables, data types, dicts, `if`/`for`, functions, imports, reading errors |
| 1  | NumPy: fast arrays | vectorization, broadcasting, why loops are slow |
| 2  | Pandas: spreadsheets in Python | indexing, groupby, merge, reshape, renaming, datetime parsing, `apply` vs vectorized |
| 3  | Data cleaning | missing data (MCAR/MAR/MNAR), dtypes, duplicates, outliers |
| 4  | EDA + visualization | asking questions with data, matplotlib/seaborn, storytelling |
| 5  | Statistics & probability | distributions, CLT, hypothesis tests, p-values, confidence intervals |
| 6  | Feature engineering & scaling | skewness, encoding, Standard/MinMax/Robust/MaxAbs/Normalizer/Quantile/Power, leakage |
| 7  | Regression | linear regression from scratch + sklearn, assumptions, metrics |
| 8  | Classification + KNN | logistic regression, KNN (metrics/weights/regression), decision boundaries, imbalance |
| 9  | Evaluation, CV & tuning | train/val/test, k-fold, GridSearch, the right metric |
| 10 | Trees, ensembles & boosting | decision trees, Random Forest, XGBoost, bias–variance |
| 11 | Clustering & PCA | KMeans, silhouette, hierarchical/DBSCAN/GMM, PCA math + intuition |
| 12 | Neural networks intro | neurons, activations, backprop intuition, a tiny net |
| 13 | Job readiness | how to *talk* DS, project narratives, interview Q&A |
| 14 | SQL for data science | execution order, JOINs, subqueries, CTEs, window functions |
| 15 | APIs & web scraping | requests, JSON, status codes, BeautifulSoup, ethics |
| 16 | Probability, combinatorics & Bayes | counting, Bayes' theorem, Naive Bayes classifier (+ by-hand examples) |
| 17 | A/B testing, ANOVA & power | experiments, z/t-tests, ANOVA+Tukey, power analysis (+ by-hand t-test) |
| 18 | OOP + linear algebra & calculus | classes, sklearn estimator pattern, eigen/PCA, gradient descent (+ worked examples) |
| 19 | Time series & forecasting | decomposition, stationarity, ARIMA/SARIMA, MAPE |
| 20 | NLP & recommendation systems | TF-IDF, **n-grams/bigrams**, cosine similarity, sentiment, **LDA topics**, content/collaborative |
| 21 | Bash & Git for data science | shell file-wrangling (`grep/cut/awk/pipes`), the git model, branching, `.gitignore` |
| 22 | Big Data & Apache Spark | scale-out, partitions, lazy eval, PySpark DataFrames, Spark SQL, when *not* to use it |
| 23 | AWS Deployment & MLOps | serialize (joblib), serving API, Docker, the AWS map, drift monitoring |

> **Module 0** starts from *absolute* zero (no assumptions). **Modules 1–13** are the
> core ML spine; **14–20** fill in the full Moringa/Flatiron syllabus (SQL, APIs/scraping,
> probability & Bayes, A/B testing, OOP/linear-algebra/calculus, time series, NLP &
> recommenders); **21–23** add the engineering/production track (Bash/Git, Spark/Big
> Data, AWS deployment & MLOps). Nothing crucial is left out.

**Deepened throughout:** column **renaming** + **datetime parsing** (`to_datetime`/`.dt`)
in M2; **all** scalers — Standard/MinMax/Robust/**MaxAbs**/**Normalizer**/**Quantile**/
Power — in M6; KNN **distance metrics**, **weights**, **regression**, and the **curse of
dimensionality** in M8; **hierarchical (dendrogram)**, **DBSCAN**, **GMM**, and **LDA vs
PCA** in M11; **specificity** added to the metrics vocabulary in M8; **n-grams** + **LDA
topic modeling** in M20; **by-hand worked examples** in M16 (Bayes), M17 (t-test) and
M18 (dot product, eigenvalues, gradient step).

### Coverage vs the official 40-topic Data Science syllabus

| Phase | Official topics | Where it's covered here |
|---|---|---|
| **1** | Python, Pandas, cleaning, **SQL**, **APIs**, **web scraping** | M0–M3, **M14**, **M15** |
| **2** | Probability, distributions, CLT/CI, hypothesis tests, **ANOVA/power**, **A/B testing**, **Bayes**, linear regression | M5, **M16**, **M17**, M7 |
| **3** | **OOP**+sklearn, **linear algebra/calculus**, ML fundamentals, logistic, class metrics, trees, KNN, **Naive Bayes**, tuning/pipelines, ensembles | **M18**, M6–M10, M16 |
| **4** | PCA, **LDA**, clustering (KMeans/**hierarchical**/**DBSCAN**/**GMM**), **time series**, **NLP**, **recommenders**, neural nets, tuning | M11, **M19**, **M20**, M12 |
| **Infra / Production** | **Bash/Git**, **Big Data/Spark**, **AWS deployment & MLOps** | **M21**, **M22**, **M23** |

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
