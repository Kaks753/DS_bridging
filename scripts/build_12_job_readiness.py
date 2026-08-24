"""Builder for Module 12: Job Readiness & Interview Prep."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 12 — Job Readiness & Interview Prep (turning skill into offers)

You now *understand* the craft. This module makes you **sound** like it — because
interviews test communication as much as code. We cover: how to talk about your
own projects, a battle-tested Q&A bank (with model answers), a whiteboard/coding
checklist, and how to run the end-to-end workflow live.

> Confidence isn't pretending to know everything. It's knowing **how you think**
> so clearly that you stay calm even when you hit something you don't know.
""")

nb.md(r"""
## 12.1 The 60-second project pitch (STAR-for-DS)

For every portfolio project, prepare a crisp story:

- **Situation / problem**: what business question, whose pain?
- **Data**: source, size, messiness, target.
- **Approach**: cleaning → EDA insight → features → models tried → why the winner.
- **Result**: the metric *and* the business meaning.
- **Reflection**: what you'd improve / what you learned.

### Worked example — your NSE Stock Risk Clustering

> *"Nairobi Securities Exchange investors lacked a simple way to group stocks by
> risk profile. I engineered 19 indicators across risk, returns, technical, and
> liquidity dimensions from raw price data. Because clustering is distance-based, I
> standardized every feature, then used K-Means and chose k with the elbow and
> silhouette methods. Careful feature engineering lifted the silhouette score from
> 0.32 to 0.717 — a 124% improvement — meaning the four risk clusters went from
> overlapping to clearly separated. I shipped it as an interactive Streamlit
> dashboard so an investor can see each stock's profile. If I extended it, I'd try
> Gaussian Mixture Models for soft assignments and validate stability over time."*

Notice: it names **techniques**, **why** each was chosen, a **quantified** result,
its **meaning**, and a **next step**. Practice one of these for each project until
it's smooth.
""")

nb.md(r"""
## 12.2 Core conceptual Q&A bank (say these in your own words)

**Q: Why is StandardScaler sensitive to outliers, and when do you use RobustScaler?**
> StandardScaler centers by the mean and scales by the std — both are pulled by
> extreme values, so an outlier shifts the center and inflates the std, squashing
> normal points. RobustScaler uses the median and IQR, which ignore the tails, so
> it's my default for outlier-heavy data like financial returns or fraud features.

**Q: Bias–variance tradeoff?**
> Bias is error from wrong assumptions (too-simple model → underfits). Variance is
> sensitivity to the training sample (too-complex model → overfits). Total error
> balances both. I diagnose with learning/validation curves and control it with
> regularization, model complexity, and more data.

**Q: How do you prevent data leakage?**
> Split first; fit all preprocessing (scaling, imputation, encoding) on the
> training fold only and apply to validation/test — ideally inside a scikit-learn
> Pipeline so it happens automatically within each CV fold. I also make sure no
> feature encodes the future or the target.

**Q: Precision vs recall — which do you optimize?**
> It depends on the cost of errors. For fraud or disease or churn I favor recall
> (don't miss positives); for a spam filter or expensive manual review I favor
> precision (don't cry wolf). I set the threshold from the business cost, and
> report F1 or PR-AUC rather than accuracy on imbalanced data.

**Q: Why don't tree models need feature scaling?**
> Trees split on a threshold of one feature at a time and only use the *ordering*
> of values. Rescaling changes the threshold's number but not the order, so the
> same splits are chosen. Distance/gradient models (KNN, SVM, linear, NN) do need
> scaling.

**Q: Random Forest vs XGBoost?**
> Random Forest bags many independent deep trees and averages them to cut variance
> — a strong, low-tuning baseline. XGBoost boosts shallow trees sequentially, each
> correcting the last, cutting bias for often top accuracy — but it needs careful
> tuning (learning rate × n_estimators, depth, sampling, regularization).

**Q: Supervised vs unsupervised?**
> Supervised learns a mapping from features to a **labeled** target (regression /
> classification). Unsupervised finds structure in **unlabeled** data (clustering,
> dimensionality reduction, anomaly detection). Evaluation differs: supervised
> compares to known labels; unsupervised uses internal measures like silhouette.

**Q: What is a p-value — without lying?**
> The probability of seeing data at least as extreme as ours *if the null
> hypothesis were true*. It is **not** the probability the null is true, and a
> small p-value with a tiny effect size can be statistically significant yet
> practically meaningless — so I always report effect size too.

**Q: How do you handle missing data?**
> First quantify it, then diagnose the mechanism — MCAR, MAR, or MNAR. For MAR I
> impute using related columns (e.g. group-wise median) and add a "was-missing"
> flag. I fit imputation on training data only, and for skewed variables I use the
> median because it's robust.

**Q: How do you choose k in K-Means?**
> The elbow method (where inertia's improvement flattens) as a guide, and the
> silhouette score for rigor — higher silhouette means better-separated clusters.
> I always scale features first because K-Means is distance-based.

**Q: Explain PCA.**
> PCA finds orthogonal directions of maximum variance — the eigenvectors of the
> covariance matrix — ordered by explained variance. I keep the top components that
> capture ~90–95% of variance to compress/denoise or visualize. Scale first, and
> accept reduced interpretability since components mix original features.
""")

nb.md(r"""
## 12.3 The end-to-end workflow you should be able to run *live*

Interviewers love a take-home or live task. Have this skeleton in muscle memory:

```python
# 1. Load & peek
df = pd.read_csv(path); df.info(); df.describe()

# 2. Split FIRST (prevent leakage)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

# 3. Preprocess inside a Pipeline (impute -> scale/encode)
num = Pipeline([("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))])
prep = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])

# 4. Model in the same Pipeline
pipe = Pipeline([("prep", prep), ("model", XGBClassifier(...))])

# 5. Cross-validate (report mean ± std) and tune
scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1")
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1").fit(X_tr, y_tr)

# 6. Final, ONE-TIME test evaluation
best = grid.best_estimator_
print(classification_report(y_te, best.predict(X_te)))
```

If you can narrate *why* each line exists (and you now can), you're interviewing at
a strong level.
""")

nb.code(r"""
# Proof you can run the whole thing end-to-end on our data, in one cell.
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("../data/customers_clean.csv")
num_cols = ["age","income","tenure_months","monthly_spend","support_calls"]
cat_cols = ["city","plan"]
X, y = df[num_cols+cat_cols], df["churn"]
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

prep = ColumnTransformer([
    ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                      ("oh",OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
spw = (y_tr==0).sum()/(y_tr==1).sum()
pipe = Pipeline([("prep",prep),
                 ("model",XGBClassifier(n_estimators=200,learning_rate=0.05,
                                        max_depth=3,subsample=0.9,
                                        eval_metric="logloss",
                                        scale_pos_weight=spw,random_state=42))])
cv = cross_val_score(pipe,X_tr,y_tr,cv=5,scoring="f1")
print(f"CV F1 = {cv.mean():.3f} ± {cv.std():.3f}")
pipe.fit(X_tr,y_tr)
print("test AUC:", round(roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1]),3))
print(classification_report(y_te, pipe.predict(X_te), target_names=["stay","churn"]))
""")

nb.md(r"""
## 12.4 Coding-round checklist (Python/pandas/SQL)

- **Pandas**: filtering, `groupby`+`agg`, `merge` (and predicting row counts),
  `pivot`/`melt`, handling NaNs, `value_counts`, sorting/ranking.
- **SQL** (you list it — keep it sharp): `SELECT/WHERE/GROUP BY/HAVING`, `JOIN`
  types, window functions (`ROW_NUMBER`, `RANK`, `SUM() OVER`), subqueries/CTEs.
- **Python basics**: comprehensions, dict/set usage, functions, complexity sense.
- **Stats**: mean vs median, variance, distributions, CLT, hypothesis tests,
  p-values, correlation ≠ causation.

Practice on real prompts (LeetCode SQL/easy-medium Python, StrataScratch). Speak
your reasoning aloud while coding — interviewers score your *thought process*.
""")

nb.md(r"""
## 12.5 Behavioral & communication

- Explain a technical idea to a **non-technical** person in ≤3 sentences (practice
  with "what is overfitting?").
- Have a story for: a project that **failed / was hard**, a time you found a
  **data-quality issue**, a time you **simplified** a model for the business.
- Always tie models back to **impact**: money saved, time saved, decisions enabled.
- It's OK to say **"I don't know, but here's how I'd find out."** That's a *strong*
  answer, not a weak one.
""")

nb.md(r"""
## 12.6 A 6-week sharpening plan (concrete)

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Re-run Modules 00–03; redo every mini-exercise from scratch | clean notebook of your own |
| 2 | Modules 04–05; write the scaler & stats answers in your own words | a "concepts in my words" note |
| 3 | Modules 06–08; build one full Pipeline+GridSearch on a Kaggle set | a tidy modeling notebook |
| 4 | Module 09; enter a Kaggle/DrivenData comp; tune XGBoost | a leaderboard submission |
| 5 | Module 10–11; redo NSE clustering explanation; tiny NN | polished project pitch |
| 6 | Module 12; mock interviews (record yourself); SQL drills | 3 rehearsed project stories |

Do the *talking* out loud and record it. Hearing yourself is how the confidence
becomes real.
""")

nb.md(r"""
## 12.7 Final word

Stephen — your projects are genuinely strong. The only gap was **feeling** the
fundamentals under them. You've now derived OLS, seen backprop run, proven the CLT,
watched scalers behave under outliers, and run the full professional pipeline. When
you can *explain* these calmly, the market reads it instantly as competence.

Re-run these notebooks until each idea feels obvious. Then go interview like the
mathematician-engineer you already are. You've got this. 🚀
""")

out = nb.save("notebooks/12_job_readiness_interview_prep.ipynb")
print("saved", out)
