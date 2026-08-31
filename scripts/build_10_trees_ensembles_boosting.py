"""Builder for Module 10: Trees, Ensembles & Boosting (4-layer rewrite of old M09)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 10 — Decision Trees, Random Forests & Boosting (XGBoost)

Tree-based models win a huge share of real-world **tabular** problems — including
your DrivenData water-wells project. They're powerful, need almost no
preprocessing, and capture non-linearities and feature interactions automatically.

We'll build the intuition from a single tree all the way up to **gradient
boosting**, cover **feature importance** honestly, and finally give the complete
answer to a question you've met before: *why don't trees need scaling?*
""")

nb.analogy("A decision tree is the game 'Twenty Questions'. Each question ('is tenure under "
           "a year?') splits the remaining possibilities into cleaner groups. A forest is "
           "asking a whole crowd of slightly-different players and taking a vote; boosting "
           "is a relay where each player specifically fixes the previous one's mistakes.")

nb.md("## 10.1 Setup")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/customers_clean.csv")

# encode plan as ordered ints — trees are perfectly happy with integer-coded categories
df["plan_ord"] = df["plan"].map({"Basic": 0, "Standard": 1, "Premium": 2})
features = ["age", "income", "tenure_months", "monthly_spend", "support_calls", "plan_ord"]
X = df[features]; y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
print("train:", X_tr.shape, "test:", X_te.shape)
""")

nb.jargon("decision tree", "a model that makes predictions via a sequence of yes/no threshold questions")
nb.jargon("ensemble", "many models combined so the group is better than any single member")
nb.jargon("tabular data", "classic rows-and-columns data (spreadsheets/databases) — trees' home turf")

nb.md("## 10.2 A single decision tree — how it thinks")

nb.plain("""
A tree asks a chain of yes/no questions about one feature at a time ('is tenure
< 12 months?'). Each question splits the data into groups that are **purer** — more
dominated by a single class. It keeps splitting until the groups are clean (or you
tell it to stop). The result is a set of human-readable rules.
""")

nb.code(r"""
stump = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
plt.figure(figsize=(14, 6))
plot_tree(stump, feature_names=features, class_names=["stay", "churn"],
          filled=True, rounded=True, fontsize=8, impurity=True)
plt.title("A shallow decision tree (depth=3) — you can literally read the rules")
plt.show()
""")

nb.readcode("""
- Each box is a question; follow 'True' left and 'False' right down to a leaf.
- 'filled=True' colours boxes by the majority class; darker = purer group.
- 'gini' in each box measures impurity — the split is chosen to reduce it the most.
""")

nb.code(r"""
# The dark side of a single tree: let it grow unlimited and it MEMORISES the training set
deep = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)   # unlimited depth
print(f"Deep tree -> train F1={f1_score(y_tr, deep.predict(X_tr)):.3f}, "
      f"test F1={f1_score(y_te, deep.predict(X_te)):.3f}   (big gap = overfit)")
print(f"Depth-3   -> train F1={f1_score(y_tr, stump.predict(X_tr)):.3f}, "
      f"test F1={f1_score(y_te, stump.predict(X_te)):.3f}   (smaller gap)")
""")

nb.deeper("""
A tree chooses each split to maximise the drop in **Gini impurity** or **entropy**
(for classification) or in **variance/MSE** (for regression). Both Gini and entropy
just measure 'how mixed is this group?' — 0 means one pure class. The reason a lone
deep tree overfits is that with enough splits it can carve out a tiny box around
every single training point — perfect on the exam it memorised, lost on anything new.
That's high variance, and it's exactly what the ensembles below are designed to cure.
""")

nb.warn("""
This is the full answer to 'why don't trees need scaling?' — a tree splits on a
THRESHOLD of one feature ('income < 50,000?'). Rescaling income just relabels the
threshold number; the ORDER of the values is unchanged, so the tree finds the same
splits. Distance and gradient models care about magnitudes; trees care only about
order. That crisp sentence is a reliable interview point-winner.
""")

nb.jargon("Gini impurity", "a 0–0.5 measure of how class-mixed a group is; 0 = perfectly pure")
nb.jargon("overfitting (trees)", "a deep tree memorising training rows → great train score, poor test score")

nb.md("## 10.3 Random Forest — bagging tames the variance")

nb.plain("""
One tree is jumpy. So grow a whole forest: train many trees, each on a random
bootstrap sample of the rows AND a random subset of features at each split, then
**average their votes**. Because each tree overfits in a *different* way, the errors
largely cancel out when you average — dramatically cutting variance while keeping
the low bias. This is 'bagging' (Bootstrap AGGregating), the wisdom of a diverse crowd.
""")

nb.code(r"""
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                            min_samples_leaf=5, class_weight="balanced",
                            random_state=42, n_jobs=-1).fit(X_tr, y_tr)
rf_pred = rf.predict(X_te)
print(f"Random Forest -> train F1={f1_score(y_tr, rf.predict(X_tr)):.3f}, "
      f"test F1={f1_score(y_te, rf_pred):.3f}")
print(f"AUC = {roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]):.3f}")
print("The train-vs-test gap is far smaller than the single deep tree above.")
""")

nb.takeaway("Random Forest is the strong, low-fuss baseline you reach for first on tabular "
            "data: little tuning, resistant to overfitting, and it just works. Averaging "
            "many decorrelated trees is what buys that stability.")

nb.jargon("bagging", "training many models on random resamples and averaging them to cut variance")
nb.jargon("Random Forest", "an ensemble of decision trees, each on random rows and random features, then voted")
nb.jargon("bootstrap sample", "a random sample of rows drawn WITH replacement (same size as the original)")

nb.md("## 10.4 Feature importance — what drives predictions (with caveats)")

nb.plain("""
A forest can tell you which features it leaned on most, by totting up how much each
one reduced impurity across all the trees. Handy for a first look — but treat it
with suspicion, because the default 'impurity' importance has real biases.
""")

nb.code(r"""
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="teal")
plt.title("Random Forest feature importance (impurity-based)")
plt.tight_layout(); plt.show()
""")

nb.code(r"""
# The more trustworthy method: permutation importance
# (shuffle one feature, measure how much performance drops)
perm = permutation_importance(rf, X_te, y_te, n_repeats=10,
                              random_state=42, scoring="f1")
pi = pd.Series(perm.importances_mean, index=features).sort_values()
print("Permutation importance (more reliable):")
print(pi.round(4))
""")

nb.deeper("""
Two caveats to SAY OUT LOUD about impurity importance: (1) it's biased toward
high-cardinality / continuous features (they offer more places to split), and (2) it
tells you a feature was *used a lot*, never the *direction* of its effect. Permutation
importance sidesteps bias (1) by directly measuring the performance drop when a
feature is scrambled — if breaking a feature barely hurts, the model wasn't really
relying on it, whatever the impurity chart claimed.
""")

nb.jargon("feature importance", "a score for how much each feature contributed to the model's predictions")
nb.jargon("permutation importance", "shuffle one feature and measure the score drop — a bias-resistant importance")

nb.md("## 10.5 Boosting — experts who fix each other's mistakes")

nb.plain("""
Bagging builds trees in PARALLEL and averages them. Boosting builds them in
SEQUENCE: each new (deliberately shallow) tree focuses on the rows the current
ensemble is getting WRONG. Round after round, the team's errors shrink. This attacks
**bias** and usually delivers the best tabular accuracy — but it can overfit if you
let it run unchecked, so it needs more careful tuning than a forest.
""")

nb.analogy("A forest is a hundred independent students answering separately and voting. "
           "Boosting is a study group where each student specialises in the exact questions "
           "the group keeps missing — so the group's weak spots keep shrinking.")

nb.code(r"""
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                max_depth=3, random_state=42).fit(X_tr, y_tr)
print(f"sklearn GradientBoosting -> test F1={f1_score(y_te, gb.predict(X_te)):.3f}, "
      f"AUC={roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1]):.3f}")

xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    eval_metric="logloss", random_state=42,
    scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),   # handle imbalance
)
xgb.fit(X_tr, y_tr)
print(f"XGBoost                  -> test F1={f1_score(y_te, xgb.predict(X_te)):.3f}, "
      f"AUC={roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1]):.3f}")
""")

nb.readcode("""
- `learning_rate=0.05` keeps each tree's contribution small; `n_estimators=300`
  compensates with many small steps — the classic 'low LR + many trees' recipe.
- `subsample`/`colsample_bytree` randomly drop rows/columns per tree = built-in
  regularization; `scale_pos_weight` up-weights the rare churn class for imbalance.
""")

nb.deeper("""
Gradient boosting is literally **gradient descent in function space**: each tree is a
step down the loss surface, fitting the negative gradient (the residual errors) of
what the ensemble has predicted so far. XGBoost and LightGBM are fast, regularized,
production-grade implementations of this idea — LightGBM is the one you used on the
water-wells project. Knowing 'it's gradient descent, but the thing being adjusted is
an added tree instead of a weight vector' is the senior-level one-liner.
""")

nb.interview("""
Know these XGBoost knobs cold: n_estimators (number of trees), learning_rate η (each
tree's contribution — smaller generalises better but needs more trees), max_depth
(3–6 typical for boosting), subsample / colsample_bytree (row/column sampling for
regularization + speed), reg_lambda / reg_alpha (L2/L1 penalties on leaf weights),
and scale_pos_weight (up-weight the positive class for imbalance). Tune them with
RandomizedSearchCV from the Evaluation module — never eyeball them.
""")

nb.jargon("boosting", "building models sequentially, each correcting the previous ensemble's errors")
nb.jargon("gradient boosting", "boosting framed as gradient descent, adding a tree that fits the residual errors")
nb.jargon("learning rate (η)", "how much each boosting tree contributes; smaller = slower but generalises better")

nb.md("## 10.6 Head-to-head — pick with cross-validation, not vibes")

nb.code(r"""
models = {
    "DecisionTree(d=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForest":      RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                random_state=42, n_jobs=-1),
    "GradientBoosting":  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                    max_depth=3, random_state=42),
    "XGBoost":           XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                       subsample=0.8, colsample_bytree=0.8,
                                       eval_metric="logloss", random_state=42),
}
rows = []
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5, scoring="f1", n_jobs=-1)
    rows.append((name, s.mean(), s.std()))
res = (pd.DataFrame(rows, columns=["model", "cv_f1_mean", "cv_f1_std"])
         .sort_values("cv_f1_mean", ascending=False))
print(res.round(3).to_string(index=False))
""")

nb.takeaway("Never pick a model on a single split or a hunch — rank them by cross-validated "
            "mean ± std. The winner here is the one you'd defend, with its uncertainty attached.")

nb.md("## 10.7 Bias–variance, summarised on trees")

nb.md(r"""
| Model | Bias | Variance | What it fixes |
|---|---|---|---|
| Single deep tree | low | **high** | nothing — it overfits |
| Random Forest (bagging) | low | **reduced** | averages many decorrelated trees |
| Boosting (XGBoost) | **reduced** | low–moderate | sequentially corrects errors |
""")

nb.deeper("""
The practical decision rule: start with a **Random Forest** as a strong, low-effort
baseline. Move to **XGBoost/LightGBM** when you want to squeeze out maximum accuracy
and are willing to tune. Keep a **single shallow tree** whenever you need to *explain*
the rules to a stakeholder — interpretability is sometimes worth more than a couple of
F1 points.
""")

nb.md("## 10.8 Try it yourself")

nb.try_this("""
1. Plot XGBoost test F1 as you raise n_estimators with a small learning_rate. Where
   does it plateau?
2. Compare the impurity vs permutation importance rankings — do they disagree, and why?
3. Set max_depth=1 ('stumps') in boosting. Can such weak learners still combine into a
   strong model? (Spoiler: yes — that's the whole point of boosting.)
4. Explain to an interviewer why a Random Forest rarely overfits badly but a single
   deep tree does.
""")

nb.md("## Summary")

nb.takeaway("""
- A single tree splits to increase purity; alone it **overfits** (high variance).
- Trees are **scale-invariant** — they use value *order*, not magnitude.
- **Random Forest** = bagging → averages diverse trees → cuts variance; the great low-effort baseline.
- **Boosting / XGBoost** = sequential error-correction → cuts bias → often best on tabular data, but needs tuning.
- Judge **feature importance** with permutation importance; pick models by **cross-validation**.
""")

nb.md(r"""
Next: **Module 11 — Unsupervised Learning: Clustering & PCA** — finding structure
when there are no labels at all.
""")

out = nb.save("notebooks/10_trees_ensembles_boosting.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
