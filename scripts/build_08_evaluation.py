"""Builder for Module 08: Evaluation, Cross-Validation, Pipelines & Grid Search."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 08 — Evaluation, Cross-Validation, Pipelines & Grid Search

This is the module that separates hobbyists from professionals. Anyone can call
`.fit()`. The pro question is: **"How do I know this number is real and not luck
or leakage?"** Answer: proper splitting, **cross-validation**, **Pipelines**, and
disciplined **hyperparameter tuning**.

Goals:
- Train/validation/test — the three-way split and why.
- **k-fold cross-validation** — a stable performance estimate.
- **Pipelines** — bundle preprocessing + model so leakage is impossible.
- **GridSearchCV / RandomizedSearchCV** — tune hyperparameters correctly.
- **Overfitting vs underfitting** via learning/validation curves.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV, validation_curve,
                                     learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/customers_clean.csv")
""")

nb.md(r"""
## 8.1 Why one split isn't enough

A single train/test split gives **one** noisy estimate — change the random seed
and the score wobbles. Worse, if you tune your model by repeatedly checking the
*test* set, you slowly **leak** test information and overstate performance.

The disciplined setup:
- **Train** — fit model parameters.
- **Validation** — tune hyperparameters / compare models.
- **Test** — touched **once**, at the very end, for an honest final number.

Cross-validation replaces the train/validation part with something far more stable.
""")

nb.md(r"""
## 8.2 k-Fold Cross-Validation

Split training data into **k folds**. Train on k−1, validate on the held-out fold,
rotate k times, average the scores. Every row is used for validation exactly once.
For classification use **StratifiedKFold** to preserve class balance in each fold.
""")

nb.code(r"""
features_num = ["age", "income", "tenure_months", "monthly_spend", "support_calls"]
X = df[features_num]; y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1")
print("F1 per fold:", np.round(scores, 3))
print(f"CV F1 = {scores.mean():.3f} ± {scores.std():.3f}")
print("The ± is your uncertainty — report it, don't hide it.")
""")

nb.md(r"""
**Takeaway:** report performance as **mean ± std across folds**. A big std means an
unstable model / too little data — itself a finding. This is exactly how you'd
describe your DrivenData result: "80.02% CV accuracy, ±0.29%" — that ± is the CV
std, and now you know precisely what it means and why it matters.
""")

nb.md(r"""
## 8.3 Pipelines — make leakage structurally impossible

A **Pipeline** chains preprocessing steps and a final model into one object. When
you call `fit`, each step is fit on the **training fold only**; when you call
`predict`, the *same* fitted transforms apply. Inside cross-validation this means
scalers/imputers never see validation data → **no leakage, ever**.

We use a **ColumnTransformer** to apply different preprocessing to numeric vs
categorical columns — the professional pattern for mixed data.
""")

nb.code(r"""
num_features = ["age", "income", "tenure_months", "monthly_spend", "support_calls"]
cat_features = ["city", "plan"]

X = df[num_features + cat_features]
y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric, num_features),
    ("cat", categorical, cat_features),
])

clf = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

clf.fit(X_tr, y_tr)
print("Test F1 (full pipeline):", round(f1_score(y_te, clf.predict(X_te)), 3))
print("\nOne object handles impute -> scale -> encode -> model. "
      "No manual step to forget, no leakage.")
""")

nb.md(r"""
**Interview gold:** *"I wrap preprocessing and the model in a single sklearn
Pipeline with a ColumnTransformer. That guarantees every transform is fit on
training folds only — leakage becomes structurally impossible — and it deploys as
one reusable object."*
""")

nb.md(r"""
## 8.4 GridSearchCV — tune hyperparameters the right way

**Parameters** are learned from data (e.g. regression coefficients).
**Hyperparameters** are set by *you* before training (e.g. `k` in KNN, `C` in
logistic regression, tree depth). GridSearch tries every combination on a grid,
scoring each with cross-validation, and picks the best — all **without touching
the test set**.

Note the `model__C` syntax: `<step name>__<param>` reaches into a pipeline step.
""")

nb.code(r"""
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],      # inverse regularization strength
    "model__penalty": ["l2"],
}
grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_tr, y_tr)

print("best params:", grid.best_params_)
print("best CV F1 :", round(grid.best_score_, 3))
print("test F1    :", round(f1_score(y_te, grid.predict(X_te)), 3))
""")

nb.md(r"""
### RandomizedSearchCV — when the grid is huge

Grid search cost explodes combinatorially. **RandomizedSearchCV** samples a fixed
number of random combinations — often finding a near-best config far cheaper.
Prefer it when you have many hyperparameters (e.g. tuning Random Forest / XGBoost).
""")

nb.code(r"""
from scipy.stats import randint
rf_pipe = Pipeline([("prep", preprocess),
                    ("model", RandomForestClassifier(random_state=42,
                                                     class_weight="balanced"))])
rf_dist = {
    "model__n_estimators": randint(100, 400),
    "model__max_depth": randint(3, 15),
    "model__min_samples_leaf": randint(1, 20),
}
rand = RandomizedSearchCV(rf_pipe, rf_dist, n_iter=15, cv=5,
                          scoring="f1", random_state=42, n_jobs=-1)
rand.fit(X_tr, y_tr)
print("best params:", rand.best_params_)
print("best CV F1 :", round(rand.best_score_, 3))
print("test F1    :", round(f1_score(y_te, rand.predict(X_te)), 3))
""")

nb.md(r"""
## 8.5 Overfitting vs underfitting — see it with curves

- **Underfitting** (high bias): model too simple; poor on *both* train and
  validation.
- **Overfitting** (high variance): model too complex; great on train, poor on
  validation (memorized noise).

A **validation curve** varies one hyperparameter; a **learning curve** varies the
training-set size.
""")

nb.code(r"""
# Validation curve over Random Forest depth (simple numeric-only view)
Xn = df[num_features].fillna(df[num_features].median())
depths = [1, 2, 3, 5, 8, 12, 20, None]
depth_vals = [d if d is not None else 25 for d in depths]
train_sc, val_sc = validation_curve(
    RandomForestClassifier(n_estimators=150, random_state=42),
    Xn, y, param_name="max_depth",
    param_range=[d if d is not None else None for d in depths],
    cv=5, scoring="f1", n_jobs=-1)

plt.figure(figsize=(7,4))
plt.plot(depth_vals, train_sc.mean(1), "o-", label="train F1")
plt.plot(depth_vals, val_sc.mean(1), "o-", label="validation F1")
plt.xlabel("max_depth"); plt.ylabel("F1")
plt.title("Validation curve: gap = overfitting"); plt.legend(); plt.show()
print("As depth grows, train F1 keeps rising but validation plateaus/drops -> "
      "that widening gap IS overfitting.")
""")

nb.code(r"""
# Learning curve: does more data help?
sizes, tr_sc, va_sc = learning_curve(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    Xn, y, cv=5, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
plt.figure(figsize=(7,4))
plt.plot(sizes, tr_sc.mean(1), "o-", label="train")
plt.plot(sizes, va_sc.mean(1), "o-", label="validation")
plt.xlabel("training examples"); plt.ylabel("F1")
plt.title("Learning curve: converging lines => more data won't help much")
plt.legend(); plt.show()
""")

nb.md(r"""
**How to act on the curves:**
- Big train–val gap → overfitting → simplify model, add regularization, or get
  more data.
- Both low and close → underfitting → more features / a more flexible model.
- Val curve still rising with data → collecting more data will help.
""")

nb.md(r"""
## 8.6 Mini-exercises

1. Change `scoring` to `"roc_auc"` in the CV and grid search. Do the best params
   change?
2. Add `SimpleImputer` strategy to the grid (`num__impute__strategy`:
   `["median","mean"]`). Which wins?
3. Reduce the training size to 20% and re-plot the learning curve. What happens to
   the gap?
4. Explain the difference between a parameter and a hyperparameter with one example
   each.
""")

nb.md(r"""
## Summary

- One split is noisy; **k-fold CV** gives **mean ± std** — always report both.
- **StratifiedKFold** preserves class balance for classification.
- **Pipelines + ColumnTransformer** make preprocessing part of the model →
  leakage-proof, deployable.
- **GridSearchCV** (exhaustive) / **RandomizedSearchCV** (sampled) tune
  hyperparameters using CV, never the test set.
- Diagnose **bias/variance** with **validation** and **learning** curves; act
  accordingly.

Next: **Module 09 — Trees, Ensembles & Boosting** (Random Forest, XGBoost).
""")

out = nb.save("notebooks/08_evaluation_cv_gridsearch.ipynb")
print("saved", out)
