"""Builder for Module 09: Trees, Ensembles & Boosting."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 09 — Decision Trees, Random Forests & Boosting (XGBoost)

Tree-based models win a huge share of real-world tabular problems (including your
DrivenData water-wells project). They're powerful, need little preprocessing, and
handle non-linearities and interactions automatically. We'll build the intuition
from a single tree up to **gradient boosting**, and cover **feature importance**
and the **bias–variance** story that ties it all together.

Goals:
- How a decision tree splits (Gini/entropy, MSE) and why it overfits alone.
- **Bagging** → **Random Forest**: variance reduction by averaging.
- **Boosting** → Gradient Boosting / **XGBoost**: bias reduction by correcting
  errors sequentially.
- Feature importance (and its caveats).
- Why trees **don't need scaling** (finally, the full explanation).
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/customers_clean.csv")

features = ["age", "income", "tenure_months", "monthly_spend",
            "support_calls", "plan_ord"] if "plan_ord" in df else \
           ["age", "income", "tenure_months", "monthly_spend", "support_calls"]
# encode plan as ordered ints (trees are fine with integer-coded categories)
df["plan_ord"] = df["plan"].map({"Basic":0, "Standard":1, "Premium":2})
features = ["age","income","tenure_months","monthly_spend","support_calls","plan_ord"]
X = df[features]; y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
""")

nb.md(r"""
## 9.1 A single decision tree — how it thinks

A tree asks a sequence of yes/no questions ("is tenure < 12?"), splitting the data
to make each resulting group **purer** (more one-class). For classification it
picks splits that reduce **Gini impurity** or **entropy**; for regression it
reduces **variance / MSE**.

- **Pro:** interpretable, handles non-linearities & interactions, no scaling
  needed.
- **Con:** a deep tree memorizes the training data → **high variance / overfits**.
""")

nb.code(r"""
stump = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
plt.figure(figsize=(14, 6))
plot_tree(stump, feature_names=features, class_names=["stay","churn"],
          filled=True, rounded=True, fontsize=8, impurity=True)
plt.title("A shallow decision tree (depth=3) — readable rules")
plt.show()
""")

nb.code(r"""
# Demonstrate overfitting: deep tree memorizes train, fails to generalize
deep = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)   # unlimited depth
print(f"Deep tree  -> train F1={f1_score(y_tr, deep.predict(X_tr)):.3f}, "
      f"test F1={f1_score(y_te, deep.predict(X_te)):.3f}   (big gap = overfit)")
print(f"Depth-3    -> train F1={f1_score(y_tr, stump.predict(X_tr)):.3f}, "
      f"test F1={f1_score(y_te, stump.predict(X_te)):.3f}")
""")

nb.md(r"""
**Why trees don't need scaling (the full reason):** a tree splits on a
**threshold** of a single feature at a time ("income < 50,000?"). Rescaling a
feature just relabels the threshold — the *ordering* of values is unchanged, so
the same splits are found. Distance/gradient models care about magnitudes; trees
care only about **order**. That's the crisp interview answer.
""")

nb.md(r"""
## 9.2 Random Forest — bagging tames the variance

**Bagging** (Bootstrap AGGregating): train many trees, each on a random bootstrap
sample of rows **and** a random subset of features per split, then **average** their
votes. Individual trees overfit in *different* ways; averaging cancels their noise,
slashing variance while keeping low bias. That's the "wisdom of a diverse crowd".
""")

nb.code(r"""
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                            min_samples_leaf=5, class_weight="balanced",
                            random_state=42, n_jobs=-1).fit(X_tr, y_tr)
rf_pred = rf.predict(X_te)
print(f"Random Forest -> train F1={f1_score(y_tr, rf.predict(X_tr)):.3f}, "
      f"test F1={f1_score(y_te, rf_pred):.3f}")
print(f"AUC = {roc_auc_score(y_te, rf.predict_proba(X_te)[:,1]):.3f}")
print("Note the train-test gap is much smaller than the single deep tree.")
""")

nb.md(r"""
## 9.3 Feature importance — what drives predictions (with caveats)

Random forests report how much each feature reduced impurity across all trees.
Useful, but two caveats to *state*:
1. Impurity importance is **biased toward high-cardinality / continuous** features.
2. It says nothing about **direction** (up or down) — only "used a lot".

The more trustworthy alternative is **permutation importance** (shuffle a feature,
measure the performance drop).
""")

nb.code(r"""
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7,4))
imp.plot(kind="barh", color="teal")
plt.title("Random Forest feature importance (impurity-based)")
plt.tight_layout(); plt.show()

from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_te, y_te, n_repeats=10,
                              random_state=42, scoring="f1")
pi = pd.Series(perm.importances_mean, index=features).sort_values()
print("Permutation importance (more reliable):")
print(pi.round(4))
""")

nb.md(r"""
## 9.4 Boosting — build experts that fix each other's mistakes

Where bagging builds trees **in parallel** and averages, **boosting** builds them
**sequentially**: each new (shallow) tree focuses on the residual errors the
previous ensemble got wrong. This drives down **bias**, often yielding the best
tabular accuracy — at the cost of more careful tuning (it *can* overfit if you let
it).

**Gradient boosting** frames this as gradient descent in function space; **XGBoost**
/ LightGBM are fast, regularized, production-grade implementations (LightGBM is
what you used on the water-wells project).
""")

nb.code(r"""
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                max_depth=3, random_state=42).fit(X_tr, y_tr)
print(f"sklearn GB -> test F1={f1_score(y_te, gb.predict(X_te)):.3f}, "
      f"AUC={roc_auc_score(y_te, gb.predict_proba(X_te)[:,1]):.3f}")

xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, eval_metric="logloss", random_state=42,
    scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),  # handle imbalance
)
xgb.fit(X_tr, y_tr)
print(f"XGBoost    -> test F1={f1_score(y_te, xgb.predict(X_te)):.3f}, "
      f"AUC={roc_auc_score(y_te, xgb.predict_proba(X_te)[:,1]):.3f}")
""")

nb.md(r"""
### The key XGBoost hyperparameters (know these cold)

- **n_estimators** — number of trees (more = higher capacity).
- **learning_rate** (η) — how much each tree contributes; smaller = need more
  trees but generalizes better. The classic tradeoff: **low LR + many trees**.
- **max_depth** — tree complexity (3–6 typical for boosting).
- **subsample / colsample_bytree** — row/column sampling → regularization + speed.
- **reg_lambda / reg_alpha** — L2/L1 penalties on leaf weights.
- **scale_pos_weight** — up-weight the positive class for imbalance.

Tune these with **RandomizedSearchCV** (Module 08); never eyeball them.
""")

nb.md(r"""
## 9.5 Head-to-head — pick with cross-validation, not vibes
""")

nb.code(r"""
models = {
    "DecisionTree(d=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200,
                                                   learning_rate=0.05,
                                                   max_depth=3, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=42),
}
rows = []
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5, scoring="f1", n_jobs=-1)
    rows.append((name, s.mean(), s.std()))
res = pd.DataFrame(rows, columns=["model", "cv_f1_mean", "cv_f1_std"]) \
        .sort_values("cv_f1_mean", ascending=False)
print(res.round(3).to_string(index=False))
""")

nb.md(r"""
## 9.6 Bias–variance, summarized on trees

| Model | Bias | Variance | Fix it provides |
|---|---|---|---|
| Single deep tree | low | **high** | (overfits) |
| Random Forest (bagging) | low | **reduced** | averages many decorrelated trees |
| Boosting (XGBoost) | **reduced** | low–moderate | sequentially corrects errors |

**When to reach for which:** start with a Random Forest as a strong, low-fuss
baseline; move to XGBoost/LightGBM when you want to squeeze out maximum accuracy
and are willing to tune. Keep a **single tree** when you need to *explain* rules.
""")

nb.md(r"""
## 9.7 Mini-exercises

1. Plot XGBoost test F1 as you increase `n_estimators` with a small
   `learning_rate`. Where does it plateau?
2. Compare impurity vs permutation importance rankings — do they disagree? Why?
3. Set `max_depth=1` ("stumps") in boosting. Can weak learners still combine into a
   strong model?
4. Explain to an interviewer why Random Forest rarely overfits badly but a single
   tree does.
""")

nb.md(r"""
## Summary

- A single tree splits to increase purity; alone it **overfits** (high variance).
- Trees are **scale-invariant** — they use value *order*, not magnitude.
- **Random Forest** = bagging → averages diverse trees → cuts variance; great
  low-effort baseline.
- **Boosting / XGBoost** = sequential error-correction → cuts bias → often
  best-in-class on tabular data; needs tuning (LR × n_estimators, depth, sampling,
  regularization).
- Judge **feature importance** with permutation importance; pick models by **CV**.

Next: **Module 10 — Unsupervised Learning: Clustering & PCA**.
""")

out = nb.save("notebooks/09_trees_ensembles_boosting.ipynb")
print("saved", out)
