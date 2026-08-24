# Data Science Bootcamp — One-Page Revision Cheatsheet

Fast recall before interviews. Each line is a *claim you can defend* from the notebooks.

---

## 00 — Python & NumPy
- NumPy is fast: **contiguous typed memory + compiled vectorized loops**. Avoid Python loops over arrays.
- **Broadcasting**: aligns shapes (dims equal or one is 1) to operate without copying. `X - X.mean(0)` centers columns.
- Slices are **views** (edit affects original); use `.copy()` to detach.
- Boolean masks filter; combine with `&  |  ~` and **parentheses**.
- `axis` = dimension that collapses: `axis=0` → per column, `axis=1` → per row.
- sklearn wants `X` as 2-D `(n_samples, n_features)` → `reshape(-1, 1)`.

## 01 — Pandas
- Series/DataFrame = **labeled** arrays; index **auto-aligns** (mismatch → NaN).
- First moves: `.info()`, `.describe()`, `.head()`.
- `.loc` = labels, `.iloc` = positions. Assign via a **single `.loc`** (avoid chained indexing).
- Prefer vectorized / `np.where` / `.map` over `.apply` (Python loop = slow).
- **GroupBy = split–apply–combine**; `mean` of a 0/1 column = a rate.
- Choose merge type (`inner/left/right/outer`); **check row counts** before/after.
- `pivot` = long→wide; `melt` = wide→long (tidy).

## 02 — Data Cleaning
- Never mutate raw; work on a copy; make it reproducible.
- Duplicates: drop exact, then dedupe on the **business key**.
- Text keys: `.str.strip().str.title()`; map real typos explicitly.
- **Rename cols** to snake_case: `df.rename(columns={...})` or `df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)`.
- **Datetimes**: `pd.to_datetime(col, errors="coerce")` → then `.dt.year/month/day_name()`; durations (e.g. `days_as_member`) often beat raw dates.
- Dtypes match meaning (numeric / category / datetime).
- Missing: **quantify → diagnose MCAR/MAR/MNAR → impute honestly (+ missing flag)**; fit on train only.
- Use **median** (robust) for skewed vars.
- Outliers: detect **IQR** (`Q1-1.5·IQR`, `Q3+1.5·IQR`) / z-score; then **decide** keep/cap/drop with a reason.

## 03 — EDA & Viz
- EDA = **question → chart → interpretation** (always a takeaway).
- Univariate: **histogram** (shape/skew), **boxplot** (spread/outliers).
- Bivariate: **scatter** (num–num), **box/bar** (cat–num).
- Multivariate: **correlation heatmap** (linear only; ≠ causation), pairplot.
- Charts: labeled, honest axes, one message.

## 04 — Statistics
- Skewed data → report **median + IQR**; symmetric → mean + std.
- **CLT**: sample-mean distribution → normal regardless of population shape → enables CIs & t-tests.
- **SE = std/√n**; more data → tighter estimate.
- **CI**: long-run coverage of the *procedure*, not one interval's probability.
- Hypothesis test: H0/H1 → statistic → **p-value** vs α. Type I (FP, =α), Type II (FN, =β), power = 1−β.
- **t-test** (compare means), **chi-square** (categorical association); always add **effect size** (Cohen's d).
- p-value ≠ P(H0 true). Significant ≠ important. **Correlation ≠ causation.**

## 05 — Feature Engineering & Scaling
- Fix positive skew: `log1p`, Yeo-Johnson (linear/distance models). **Trees don't need it.**
- **Scalers**: Standard (mean/std) → distorted by outliers; MinMax (min/max) → destroyed by outliers; **Robust (median/IQR)** → survives (finance/fraud default).
- More scalers: **MaxAbs** (`x/max|x|`, keeps zeros → **sparse/TF-IDF**); **Normalizer** (per-**ROW** unit length → text/cosine); **Quantile/Power** (force normal shape). *Normalizer is per-row; all others per-column.* Trees → don't scale.
- Encoding: **one-hot** nominal, **ordinal** only truly-ordered. High-cardinality → target/frequency encoding (inside CV).
- Engineer ratios/bins/flags from domain knowledge.
- **Leakage**: split first, `fit` transforms on train only, use Pipelines.

## 06 — Regression
- OLS minimizes **Σ(residual²)**; closed form = normal equation `β=(XᵀX)⁻¹Xᵀy`.
- Coefficient = effect per unit **holding others fixed**; standardize to compare.
- Metrics: **MAE** (robust), **RMSE** (penalizes big errors), **R²** (variance explained — never alone).
- Diagnose with **residual plots**; watch **multicollinearity** (VIF > 5–10).
- **Ridge (L2)** shrinks; **Lasso (L1)** shrinks + selects (zeros out). Tune α by CV.

## 07 — Classification & KNN
- Logistic regression: linear score → **sigmoid** → probability; coefficients are **log-odds**.
- KNN: majority vote of nearest neighbors; **must scale**; k = bias–variance knob. Also does **regression** (average neighbors).
- KNN knobs: `metric`/`p` (**euclidean** p=2, **manhattan** p=1, minkowski), `weights` (uniform vs **distance**). Suffers the **curse of dimensionality** → reduce dims / use manhattan.
- **Accuracy lies** under imbalance → use the **confusion matrix**.
- **Precision** = TP/(TP+FP) (of flagged, right?); **Recall/Sensitivity** = TP/(TP+FN) (of real, caught?); **Specificity** = TN/(TN+FP); **F1** = harmonic mean (high only when both high).
- Fraud/churn/disease → maximize **recall**; spam/expensive review → **precision**. Tune the **threshold**.
- **ROC/AUC**: ranking quality across thresholds (0.5 random, 1 perfect).
- Imbalance: right metric, `class_weight="balanced"`, resampling (in CV), threshold.

## 08 — Evaluation, CV, Pipelines, Tuning
- One split is noisy → **k-fold CV**; report **mean ± std**.
- **StratifiedKFold** preserves class balance.
- **Pipeline + ColumnTransformer** = leakage-proof, deployable (fit on train folds only).
- **GridSearchCV** (exhaustive) / **RandomizedSearchCV** (sampled) tune hyperparameters via CV, not the test set.
- **Parameter** = learned (coefficients); **hyperparameter** = you set (k, C, depth).
- Diagnose bias/variance with **validation** (vary a hyperparameter) and **learning** (vary data size) curves.

## 09 — Trees, Ensembles, Boosting
- Tree splits to increase purity (Gini/entropy; MSE for regression); alone → **overfits** (high variance).
- Trees are **scale-invariant** (use value *order*, not magnitude).
- **Random Forest** = bagging → averages diverse trees → cuts variance; strong low-tuning baseline.
- **Boosting/XGBoost** = sequential error-correction → cuts bias → best tabular accuracy; needs tuning.
- XGBoost knobs: `n_estimators`, `learning_rate` (low LR + many trees), `max_depth` (3–6), `subsample`/`colsample_bytree`, `reg_lambda`, `scale_pos_weight`.
- Feature importance: prefer **permutation importance** (impurity is biased).

## 10 — Clustering & PCA
- **K-Means** minimizes within-cluster variance; assumes spherical/similar-size; needs **scaling** + chosen **k**; `n_init>1`.
- Choose k: **elbow** (inertia bend) + **silhouette** ([-1,1]; higher = better separated).
- **Hierarchical/Agglomerative**: build a **dendrogram** (cut at tallest gap); linkage `ward` (default), complete, average, single. No k up front.
- **DBSCAN**: density-based → **arbitrary shapes** + **noise (−1)**; params `eps`, `min_samples`; no k. **GMM**: **soft**, **elliptical** clusters w/ probabilities (EM; pick components by **BIC**).
- Match algo to **shape**: spheres→KMeans, tree→Agglomerative, blobs+noise→DBSCAN, ellipses/probabilities→GMM.
- **PCA** (unsupervised, max variance) vs **LDA** (supervised, max class separation; ≤ n_classes−1 comps; also a classifier). PCA/LDA: **scale first**.
- **PCA**: orthogonal max-variance axes = eigenvectors of covariance; keep PCs for ~90–95% variance. Less interpretable.

## 11 — Neural Networks
- Neuron = weighted sum + **non-linear activation**; sigmoid neuron = logistic regression.
- Non-linearity is why depth matters (else it collapses to one linear map).
- Training loop = **forward → loss → backprop (chain rule) → gradient-descent update**, over epochs.
- **Always scale**; tune **learning rate** (most important), architecture, regularization; use **early stopping**. ReLU default; Adam optimizer.
- **Classical ML wins on tabular**; deep learning shines on images/text/large unstructured data.

## 12 — Job Readiness
- Project pitch: **problem → data → approach (why each step) → quantified result + meaning → next step**.
- Tie every model to **business impact**.
- "I don't know, but here's how I'd find out" is a **strong** answer.
- Keep SQL sharp: JOINs, GROUP BY/HAVING, window functions, CTEs.

## 13 — SQL
- Order of execution: `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`.
- `WHERE` filters rows **before** grouping; `HAVING` filters groups **after** aggregation.
- JOINs: `INNER` (matches only), `LEFT` (all left + NULLs), `RIGHT`, `FULL OUTER`.
- Aggregates: `COUNT/SUM/AVG/MIN/MAX` + `GROUP BY`.
- **Window functions**: `ROW_NUMBER/RANK/DENSE_RANK`, `SUM() OVER (PARTITION BY ...)`, `LAG/LEAD` — aggregate **without** collapsing rows.
- **CTE** (`WITH`) for readable multi-step queries; subqueries for filters.
- `NULL`: use `IS NULL`, `COALESCE`; `NULL` breaks `=` comparisons.

## 14 — APIs & Web Scraping
- **API**: request (GET/POST) → JSON response; check **status codes** (200 ok, 404, 429 rate-limit, 500).
- `requests.get(url, params=..., headers=...)` → `.json()`. Respect **rate limits** & **auth** (API keys).
- **Scraping**: `requests` + **BeautifulSoup** (`find/find_all`, CSS selectors). Use only when no API.
- Be ethical/legal: check `robots.txt` & ToS; throttle; identify yourself.
- Pagination + retries + caching = robust ingestion.

## 15 — Probability, Combinatorics & Bayes
- Permutations (order matters) `nPr = n!/(n-r)!`; combinations `nCr = n!/(r!(n-r)!)`.
- Rules: complement `P(A')=1-P(A)`; addition `P(A∪B)=P(A)+P(B)-P(A∩B)`; multiplication (independent) `P(A∩B)=P(A)P(B)`.
- **Conditional**: `P(A|B)=P(A∩B)/P(B)`. **Bayes**: `P(H|E)=P(E|H)P(H)/P(E)` — update prior with evidence.
- **Naive Bayes** classifier: assumes features conditionally independent given class; fast, great baseline for **text** (spam/sentiment).

## 16 — A/B Testing, ANOVA & Power
- **A/B test** = randomized experiment → supports **causal** claims (unlike correlation).
- Compare 2 groups: **t-test** (means) / **z-test for proportions**. >2 groups: **ANOVA** (F-test), then post-hoc (Tukey).
- **Power** = 1−β = P(detect a real effect). Driven by effect size, α, and **sample size** → do a **power analysis before** running.
- Watch **peeking / multiple comparisons** (inflates false positives → correct with Bonferroni/FDR).

## 17 — OOP + Linear Algebra & Calculus
- **OOP**: `class`, `__init__`, attributes/methods, `self`; sklearn estimators use `fit/transform/predict` (the pattern behind custom transformers).
- **Vectors/matrices**: dot product = similarity/projection; matrix multiply = compose linear maps.
- **Eigenvectors/eigenvalues**: directions unchanged by a transform (basis of PCA).
- **Derivative** = slope = how output changes with input; **gradient** = vector of partials → direction of steepest ascent.
- **Gradient descent**: step *against* the gradient, scaled by learning rate → minimizes loss (engine of regression & neural nets).

## 18 — Time Series
- Components: **trend + seasonality + residual** (additive vs multiplicative). Decompose to understand.
- **Stationarity** (constant mean/variance) needed for ARIMA → test with **ADF**; achieve via **differencing**.
- **ACF/PACF** guide ARIMA(p,d,q); **SARIMA** adds seasonality.
- **Never shuffle** time-series CV — split by time (`TimeSeriesSplit`). No future leakage.
- Metrics: **MAPE** (%, interpretable), RMSE. Models: ARIMA/SARIMA, **Prophet**, boosting on lag features.

## 19 — NLP & Recommenders
- NLP pipeline: clean → tokenize → remove stopwords → stem/lemmatize → vectorize.
- **Bag-of-Words / TF-IDF**: term frequency × inverse document frequency (rare-but-present words weigh more).
- **N-grams** `ngram_range=(1,2)`: unigrams+bigrams so **"not good" ≠ "good"** (vital for sentiment). Tune with `min_df`/`max_df`.
- Similarity: **cosine similarity** of TF-IDF vectors (document/semantic search).
- Classify text with **Naive Bayes / Logistic Regression** (your sentiment app).
- **Topic modeling = LDA (Latent Dirichlet Allocation)**: unsupervised topics over unlabeled docs (≠ Linear Discriminant Analysis!). Feed **counts**, not TF-IDF.
- Recommenders: **content-based** (item features) vs **collaborative filtering** (user–item matrix); cold-start problem.

## 20 — Bash & Git
- **Bash core**: `pwd/cd/ls -la/mkdir -p/cp/mv/rm -r/cat/head/tail`.
- **Peek at big files** without Python: `head -n5`, `wc -l` (row count), `cut -d, -f2 | sort | uniq -c` (value counts), `grep pattern`, `awk -F, 'NR>1{s+=$4}END{print s}'`. Pipe with `|`.
- **Git model**: working dir →`add`→ **staging** →`commit`→ local repo →`push`→ remote.
- Everyday: `git status/diff/log --oneline`, `git add -A`, `git commit -m`, `git pull` **before** push.
- **Branches**: `git checkout -b feat` → work → `git checkout main` → `git merge feat`. Conflicts marked `<<<<<<< ======= >>>>>>>` → edit, `add`, `commit`.
- **.gitignore**: exclude data, `__pycache__`, `.ipynb_checkpoints`, `.venv`, and **secrets** (`.env`, keys). Never commit credentials.
- Team loop: **pull → branch → small commits → push → Pull Request → review → merge**.

## 21 — Big Data & Spark
- Scale **up** (bigger box) vs scale **out** (many boxes). Spark = scale out.
- **Driver** plans; **executors** run work on **partitions** in parallel.
- **DataFrame** API (≈pandas+SQL, optimized by Catalyst) ≫ raw RDDs.
- **Lazy**: transformations (`select/filter/groupBy/join/withColumn`) build a DAG; **actions** (`show/count/collect/write`) trigger it → chain transforms, act once.
- `groupBy().agg()`, joins, **window functions**, and **Spark SQL** (`createOrReplaceTempView`) mirror Modules 01 & 13.
- **When NOT to**: if data fits one machine (<~50GB), **DuckDB/Polars/pandas** are faster & simpler. Use Spark for TB–PB or existing clusters.

## 22 — Deployment & MLOps
- Deploy the **whole Pipeline** (preprocessing + model) → avoids train/serve skew.
- **Serialize**: `joblib.dump/load` (better than pickle for sklearn). Save **metadata** (versions, feature baseline). Only load **trusted** artifacts.
- **Serve**: `/predict` API (Flask/FastAPI) — load model **once** at startup, accept JSON, return JSON; `/health` probe.
- **Docker**: package code+deps+runtime → runs identically everywhere. `docker build -t api . && docker run -p 8080:8080 api`.
- **AWS map**: **S3** (storage), **EC2/ECS/Fargate** (compute), **ECR** (images), **Lambda+API Gateway** (serverless), **SageMaker** (managed endpoints), **IAM** (access), **CloudWatch** (logs/metrics).
- **MLOps**: reproducibility (pin versions, DVC/MLflow) + CI/CD + monitoring. **Data drift** (inputs shift) vs **concept drift** (input→target relationship changes) → compare live stats to training baseline → retrain.
