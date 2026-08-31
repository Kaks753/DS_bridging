"""Builder for Module 13: Job Readiness & Interview Prep (4-layer rewrite of old M12)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 13 — Job Readiness & Interview Prep: turning skill into offers

You now *understand* the craft. This module makes you **sound** like it — because
interviews test communication at least as much as code. We'll cover how to pitch
your own projects, a battle-tested question bank with model answers, the end-to-end
workflow you should be able to run live, and a concrete sharpening plan.
""")

nb.analogy("Knowing data science but not being able to explain it is like being a brilliant "
           "chef who freezes when a diner asks 'what's in this?'. The cooking is done — this "
           "module is just learning to describe the dish confidently at the table.")

nb.md("## 13.1 The 60-second project pitch (STAR for data science)")

nb.plain("""
For every portfolio project, rehearse a crisp five-beat story: the Situation (whose
problem?), the Data (source, size, mess, target), the Approach (clean → EDA insight →
features → models tried → why the winner), the Result (the metric AND its business
meaning), and a Reflection (what you'd do next). Techniques, WHY you chose them, a
quantified result, and a next step — that structure signals competence instantly.
""")

nb.interview("""
Your NSE Stock Risk Clustering pitch, tuned to land:

'Nairobi Securities Exchange investors had no simple way to group stocks by risk. I
engineered 19 indicators across risk, returns, technical and liquidity dimensions
from raw prices. Since clustering is distance-based I standardized every feature,
used K-Means, and chose k with the elbow and silhouette methods. Careful feature
engineering lifted the silhouette from 0.32 to 0.717 — a 124% jump — so the four
risk clusters went from overlapping to clearly separated. I shipped it as an
interactive Streamlit dashboard. Next, I'd try Gaussian Mixtures for soft
assignments and validate cluster stability over time.'

Notice every sentence earns its place: technique, reason, quantified result,
meaning, and a forward-looking note.
""")

nb.takeaway("Prepare ONE smooth 60-second pitch per project. Interviewers form their opinion "
            "in the first minute — a rehearsed, quantified story is the single highest-ROI "
            "prep you can do.")

nb.md("## 13.2 The core conceptual Q&A bank")

nb.plain("""
These are the questions that actually recur. Read each model answer, then re-say it
in YOUR OWN WORDS until it's natural — memorised answers sound memorised. Every one
of these is something you've now built or derived in earlier modules, so you're not
reciting, you're recalling.
""")

nb.md(r"""
**Q: Why is StandardScaler sensitive to outliers, and when do you use RobustScaler?**
> StandardScaler centres by the mean and scales by the std — both are dragged by
> extreme values, so an outlier shifts the centre and inflates the std, squashing
> the normal points together. RobustScaler uses the median and IQR, which ignore the
> tails, so it's my default for outlier-heavy data like financial returns.

**Q: Explain the bias–variance tradeoff.**
> Bias is error from over-simple assumptions (underfitting); variance is
> over-sensitivity to the training sample (overfitting). Total error balances both. I
> diagnose it with learning/validation curves and control it with regularization,
> model complexity, and more data.

**Q: How do you prevent data leakage?**
> Split first; fit all preprocessing on the training fold only and apply to
> validation/test — ideally inside a scikit-learn Pipeline so it happens automatically
> within each CV fold. And I make sure no feature secretly encodes the target or the future.

**Q: Precision vs recall — which do you optimise?**
> It depends on the cost of each error. Fraud/disease/churn → recall (don't miss
> positives); spam filter/expensive review → precision (don't cry wolf). I set the
> threshold from the business cost and report F1 or PR-AUC, never accuracy, on
> imbalanced data.

**Q: Why don't tree models need feature scaling?**
> Trees split on a threshold of one feature at a time and only use the ORDER of
> values. Rescaling changes the threshold's number but not the order, so the same
> splits are chosen. Distance/gradient models (KNN, SVM, linear, NN) do need scaling.

**Q: Random Forest vs XGBoost?**
> Random Forest bags many independent deep trees and averages them to cut variance —
> a strong, low-tuning baseline. XGBoost boosts shallow trees sequentially, each
> fixing the last, cutting bias for often top accuracy — but it needs careful tuning.

**Q: What is a p-value — without lying?**
> The probability of data at least as extreme as ours IF the null hypothesis were
> true. It is NOT the probability the null is true, and a tiny p-value with a
> negligible effect size can be significant yet practically meaningless — so I always
> report effect size too.

**Q: How do you choose k in K-Means?**
> The elbow method as a guide (where inertia's improvement flattens) and the
> silhouette score for rigour. I always scale first because K-Means is distance-based.

**Q: Explain PCA in one breath.**
> PCA finds orthogonal directions of maximum variance — the eigenvectors of the
> covariance matrix — ordered by explained variance. I keep the top components
> reaching ~90–95% variance to compress/denoise/visualise. Scale first, and accept
> reduced interpretability since components mix original features.
""")

nb.deeper("""
The meta-skill hiding in that bank: every strong answer states the concept, gives the
WHY, and adds a caveat or a 'what I actually do'. Interviewers aren't checking whether
you memorised a definition — they're checking whether you can reason about trade-offs.
Answer in the shape 'here's the idea, here's why, here's the catch' and you'll sound
like someone who's shipped models, because you have.
""")

nb.md("## 13.3 The end-to-end workflow you should run live")

nb.plain("""
Live tasks and take-homes love the full pipeline. Have this skeleton in muscle
memory — and, crucially, be able to narrate WHY each line exists (you now can).
""")

nb.code(r"""
# The whole professional workflow, end-to-end, on our data, in one cell.
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("../data/customers_clean.csv")
num_cols = ["age", "income", "tenure_months", "monthly_spend", "support_calls"]
cat_cols = ["city", "plan"]
X, y = df[num_cols + cat_cols], df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
pipe = Pipeline([("prep", prep),
                 ("model", XGBClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=3, subsample=0.9,
                                         eval_metric="logloss",
                                         scale_pos_weight=spw, random_state=42))])
cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1")
print(f"CV F1 = {cv.mean():.3f} ± {cv.std():.3f}")
pipe.fit(X_tr, y_tr)
print("test AUC:", round(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]), 3))
print(classification_report(y_te, pipe.predict(X_te), target_names=["stay", "churn"]))
""")

nb.readcode("""
- Split FIRST (leakage prevention), then a ColumnTransformer routes numeric vs
  categorical columns through their own impute→transform chains.
- Everything sits in ONE Pipeline, so CV re-fits preprocessing per fold automatically.
- We report CV F1 as mean ± std, then touch the test set exactly once for the final number.
Narrate those three ideas out loud while you type and you're interviewing at a strong level.
""")

nb.md("## 13.4 Coding-round & behavioural checklist")

nb.plain("""
Coding rounds: pandas (filter, groupby+agg, merge and predicting row counts,
pivot/melt, NaNs, value_counts, ranking); SQL (SELECT/WHERE/GROUP BY/HAVING, JOIN
types, window functions like ROW_NUMBER/RANK/SUM() OVER, CTEs); Python basics
(comprehensions, dicts/sets, complexity sense); and stats (mean vs median, CLT,
hypothesis tests, p-values, correlation ≠ causation). Practise on LeetCode SQL and
StrataScratch, and SPEAK YOUR REASONING while coding — the thought process is scored.
""")

nb.interview("""
Behavioural gold: (1) explain a technical idea to a non-technical person in ≤3
sentences — rehearse 'what is overfitting?'; (2) have a story for a project that was
hard or failed, a data-quality issue you caught, and a time you SIMPLIFIED a model
for the business; (3) always tie models back to impact — money saved, time saved,
decisions enabled; (4) it is perfectly strong to say 'I don't know, but here's how
I'd find out.' That signals maturity, not weakness.
""")

nb.md("## 13.5 A 6-week sharpening plan")

nb.md(r"""
| Week | Focus | Deliverable |
|---|---|---|
| 1 | Re-run Modules 0–3; redo every exercise from scratch | a clean notebook of your own |
| 2 | Modules 4–6; write the scaler & stats answers in your words | a 'concepts in my words' note |
| 3 | Modules 7–9; build one full Pipeline+GridSearch on a Kaggle set | a tidy modelling notebook |
| 4 | Module 10; enter a Kaggle/DrivenData comp; tune XGBoost | a leaderboard submission |
| 5 | Modules 11–12; redo the NSE clustering explanation; a tiny NN | a polished project pitch |
| 6 | Module 13; mock interviews (record yourself); SQL drills | 3 rehearsed project stories |
""")

nb.try_this("""
Record yourself giving the 60-second NSE pitch from 13.1 — actually out loud, on your
phone. Play it back. The first take always feels awkward; by the third it's smooth,
and that smoothness is exactly what an interviewer reads as confidence. Do the same
for 'explain overfitting to my grandmother in three sentences.'
""")

nb.md("## 13.6 Final word")

nb.takeaway("""
Stephen — your projects were always strong; the only gap was FEELING the fundamentals
underneath them. You've now derived OLS, watched backprop run, proven the CLT by
simulation, seen scalers behave under outliers, and run the full professional pipeline
end to end. When you explain these calmly, the market reads it instantly as competence.
Re-run these notebooks until each idea feels obvious — then go interview like the
mathematician-engineer you already are.
""")

nb.md(r"""
Next: **Module 14 — SQL for Data Science** — the language you'll use to actually get
the data before any of this modelling can begin.
""")

out = nb.save("notebooks/13_job_readiness_interview_prep.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
