"""Builder for Module 8: Classification & KNN (4-layer rewrite of old M07)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 8 — Classification & KNN: predicting categories

Regression predicted a *number*. **Classification** predicts a *category* — will
this customer churn or stay? Is this email spam or not? Is this transaction fraud
or legit? We'll master two beautifully intuitive models — **Logistic Regression**
and **K-Nearest Neighbors** — and then the part that separates juniors from
seniors: **how to evaluate a classifier honestly**, because plain accuracy will
lie to your face on real data.
""")

nb.analogy("A classifier is a bouncer at a door deciding 'in' or 'out'. The interesting "
           "question is never just 'how often is the bouncer right?' — it's 'who does he "
           "wrongly turn away, and who does he wrongly let in?' Those two mistakes have "
           "very different costs.")

nb.md("## 8.1 Setup")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve,
                             ConfusionMatrixDisplay)

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/customers_clean.csv")
print("churn balance (fraction of each class):")
print(df["churn"].value_counts(normalize=True).round(3))
""")

nb.jargon("classification", "predicting a category/label instead of a number")
nb.jargon("class", "one possible label (here: churn=1 or stay=0)")
nb.jargon("positive class", "the label you care about catching (here: churn)")

nb.md("## 8.2 Logistic Regression — a linear model that outputs probabilities")

nb.plain("""
Despite the confusing name, logistic regression is a **classifier**, not a
regressor. It does the same linear-score trick as linear regression
(z = weighted sum of features), then squashes that score through an S-shaped curve
called the **sigmoid** so the answer always lands between 0 and 1 — a probability.
Above 0.5 → predict the positive class; below → the negative class.
""")

nb.md(r"""
$$ z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p, \qquad
   \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1) $$
""")

nb.code(r"""
z = np.linspace(-8, 8, 200)
plt.figure(figsize=(6.5, 3.8))
plt.plot(z, 1/(1+np.exp(-z)), lw=2)
plt.axhline(0.5, color="red", ls="--"); plt.axvline(0, color="gray", ls=":")
plt.title("Sigmoid: turns any score z into a probability in (0,1)")
plt.xlabel("z (linear score)"); plt.ylabel("probability"); plt.show()
""")

nb.readcode("""
- The curve is flat-then-steep-then-flat: extreme scores saturate near 0 or 1, while
  scores near z=0 map to probabilities near the 0.5 decision line.
- That red line at 0.5 is the DEFAULT threshold — we'll later move it deliberately.
""")

nb.code(r"""
features = ["age", "income", "tenure_months", "monthly_spend", "support_calls"]
X = df[features]; y = df["churn"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

logit = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)

coef = pd.Series(logit.coef_[0], index=features).sort_values(key=abs, ascending=False)
print("Standardized log-odds coefficients:")
print(coef.round(3))
""")

nb.readcode("""
- `stratify=y` keeps the churn/stay ratio identical in train and test — essential
  when one class is rare.
- We scale (fit on train only), then read the coefficients: positive raises churn
  probability, negative lowers it. Expect support_calls positive, tenure negative.
""")

nb.deeper(r"""
The coefficients are in **log-odds**. A coefficient β means: a one-unit increase in
that (standardized) feature multiplies the *odds* of churn by $e^{\beta}$. So
β = 0.7 → odds multiply by e^0.7 ≈ 2 (odds double). This is why logistic regression
stays a favourite in medicine and credit scoring: the effects translate into
'the odds roughly double', which regulators and doctors can actually reason about.
It's fit by maximising likelihood, equivalently minimising **log loss**.
""")

nb.jargon("sigmoid", "the S-shaped function that squashes any score into a (0,1) probability")
nb.jargon("log-odds", "log of the odds p/(1-p); logistic-regression coefficients live on this scale")
nb.jargon("log loss", "the error logistic regression minimizes; punishes confident wrong predictions hard")

nb.md("## 8.3 KNN — you are the company you keep")

nb.plain("""
K-Nearest Neighbors is the most human algorithm there is: to classify a new point,
look at its **k closest neighbours** in the training data and take a majority vote.
There's no equation to train — it just memorises the data and measures distances at
prediction time (a 'lazy learner'). Two consequences follow directly.
""")

nb.analogy("KNN is judging someone by their five closest friends. Two catches: (1) you must "
           "measure 'closeness' fairly — if one trait is in dollars and another in "
           "years, dollars will dominate unless you scale; (2) how many friends you poll "
           "(k) changes the verdict.")

nb.md(r"""
- **Scaling is mandatory** — distance is dominated by big-range features otherwise.
  This is the concrete payoff of the scaling module.
- **k is the bias–variance knob** — small k = wiggly, overfit boundary; large k =
  smooth, possibly underfit boundary.
""")

nb.code(r"""
# Proof that scaling changes KNN's accuracy meaningfully
knn_unscaled = KNeighborsClassifier(n_neighbors=15).fit(X_tr,   y_tr)
knn_scaled   = KNeighborsClassifier(n_neighbors=15).fit(X_tr_s, y_tr)
print("KNN accuracy WITHOUT scaling:", round(knn_unscaled.score(X_te,   y_te), 3))
print("KNN accuracy WITH    scaling:", round(knn_scaled.score(X_te_s, y_te), 3))
""")

nb.code(r"""
# k sweep: the bias-variance knob, seen live
ks  = range(1, 40, 2)
acc = [KNeighborsClassifier(n_neighbors=k).fit(X_tr_s, y_tr).score(X_te_s, y_te) for k in ks]
plt.figure(figsize=(7, 4))
plt.plot(list(ks), acc, marker="o")
plt.xlabel("k (number of neighbours)"); plt.ylabel("test accuracy")
plt.title("Choosing k: too small overfits, too large underfits"); plt.show()
""")

nb.takeaway("KNN needs two decisions: scale your features (always) and pick k (by trying a "
            "range and watching validation performance). Small k memorises noise; large k "
            "blurs real structure.")

nb.md("## 8.4 KNN, deeper — distance, weighting, regression, and its Achilles heel")

nb.deeper("""
Four things that turn 'I know KNN' into 'I really know KNN':

(1) **Distance metric.** 'Nearest' depends on the ruler. Euclidean (straight-line,
p=2, the default) is standard; Manhattan (city-block, p=1) is more robust to
outliers and often better in high dimensions. Minkowski is the general family.

(2) **Weighting.** `weights="uniform"` lets all k neighbours vote equally;
`weights="distance"` gives closer neighbours a louder vote — helpful when clusters
overlap.

(3) **KNN also does regression.** Swap the majority vote for an *average* of the
neighbours' targets and you have `KNeighborsRegressor` — same geometry, continuous
output.

(4) **The curse of dimensionality — KNN's weakness.** Add enough feature columns
and every point becomes almost equally far from every other, so 'nearest' stops
meaning anything and KNN degrades.
""")

nb.code(r"""
# (1)+(2): compare metrics and weighting by cross-validated F1
for metric, p in [("euclidean", 2), ("manhattan", 1)]:
    for w in ["uniform", "distance"]:
        clf = KNeighborsClassifier(n_neighbors=15, weights=w, p=p)
        s = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring="f1").mean()
        print(f"metric={metric:9s} weights={w:8s} -> CV F1 = {s:.3f}")
""")

nb.code(r"""
# (3): KNN regression = local average of neighbours' targets
xr = np.linspace(0, 10, 60)
yr = np.sin(xr) + np.random.default_rng(0).normal(0, 0.15, xr.size)
knr = KNeighborsRegressor(n_neighbors=5).fit(xr.reshape(-1, 1), yr)
grid = np.linspace(0, 10, 300).reshape(-1, 1)
plt.figure(figsize=(7, 3.8))
plt.scatter(xr, yr, s=15, alpha=0.6, label="data")
plt.plot(grid, knr.predict(grid), "r-", lw=2, label="KNN regression (k=5)")
plt.title("KNN regression: predict a point as the average of its 5 nearest neighbours")
plt.legend(); plt.show()
""")

nb.code(r"""
# (4): the curse, quantified. As dims grow, nearest/farthest -> 1.
from scipy.spatial.distance import pdist
rng = np.random.default_rng(0)
print(f"{'dims':>5} {'min_dist':>9} {'max_dist':>9} {'min/max':>8}")
for d in [2, 5, 20, 100, 500]:
    pts = rng.random((500, d))
    dd = pdist(pts)
    print(f"{d:5d} {dd.min():9.3f} {dd.max():9.3f} {dd.min()/dd.max():8.3f}")
print("\nmin/max heading to 1 => every point looks equally far => KNN loses its edge.")
print("Fixes: cut dimensions (PCA / feature selection) or switch to Manhattan distance.")
""")

nb.jargon("lazy learner", "an algorithm that stores the data and does the work at prediction time (KNN)")
nb.jargon("curse of dimensionality", "in high dimensions points become near-equidistant, breaking distance methods")

nb.md("## 8.5 Why plain accuracy is a TRAP")

nb.warn("""
If 95% of customers don't churn, a model that blindly predicts 'no churn' for
everyone scores 95% accuracy — and is completely useless, because it never catches
a single churner. On any imbalanced problem, accuracy alone is a lie. You MUST look
at the confusion matrix.
""")

nb.code(r"""
pred = logit.predict(X_te_s)
cm = confusion_matrix(y_te, pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["stay(0)", "churn(1)"])
disp.plot(cmap="Blues", values_format="d"); plt.title("Confusion Matrix"); plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
""")

nb.plain("""
The confusion matrix is just the four ways a yes/no prediction can land:
- **TP** true positive  — predicted churn, actually churned ✅
- **TN** true negative  — predicted stay, actually stayed ✅
- **FP** false positive — predicted churn, actually stayed (false alarm, Type I)
- **FN** false negative — predicted stay, actually churned (missed it, Type II)
Everything else is built from these four counts.
""")

nb.md("## 8.6 Precision, recall, F1 — worked out by hand")

nb.plain("""
Let's compute the three headline metrics straight from TP/FP/FN/TN so the formulas
stop being abstract. Then we let scikit-learn print the same numbers.
""")

nb.code(r"""
# By hand, from the four counts we just extracted:
precision = tp / (tp + fp)          # of those we FLAGGED, how many were right?
recall    = tp / (tp + fn)          # of all REAL churners, how many did we catch?
f1        = 2 * precision * recall / (precision + recall)

print(f"precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {precision:.3f}")
print(f"recall    = TP/(TP+FN) = {tp}/({tp}+{fn}) = {recall:.3f}")
print(f"F1        = 2·P·R/(P+R)                  = {f1:.3f}")
""")

nb.code(r"""
# sklearn agrees:
print(classification_report(y_te, pred, target_names=["stay", "churn"]))
""")

nb.deeper("""
Why is F1 the *harmonic* mean, not the plain average? Because the harmonic mean is
dragged down by the smaller of the two. A model with precision 0.9 and recall 0.1
has a plain average of 0.5 (looks fine!) but an F1 of only 0.18 (honestly bad). F1
refuses to be fooled by one good number hiding one terrible one.
""")

nb.interview("""
Q: 'Precision vs recall — which matters more?' The senior answer is: it depends on
the COST of each mistake, and that's a business decision, not a maths one.
- Catching fraud / disease / churn you must not miss → maximise RECALL (missing a
  positive is expensive; a few false alarms are tolerable).
- Spam filter / flagging cases for costly manual review → maximise PRECISION (crying
  wolf wastes money and trust).
You slide between the two by moving the decision threshold — which is exactly the
next section.
""")

nb.jargon("precision", "of the cases you flagged positive, the fraction that truly were positive")
nb.jargon("recall", "of all truly positive cases, the fraction you managed to flag (a.k.a. sensitivity/TPR)")
nb.jargon("F1 score", "harmonic mean of precision and recall; high only when BOTH are high")

nb.md("## 8.7 ROC curve & AUC — evaluating across every threshold")

nb.plain("""
Every metric above assumed the 0.5 cutoff. But the model really outputs a
*probability*, and you're free to choose the cutoff. The **ROC curve** sweeps every
possible threshold and plots how many real positives you catch (TPR) against how
many false alarms you raise (FPR). **AUC** — the area under that curve — is one
number summarising the model's *ranking* skill, from 0.5 (coin flip) to 1.0 (perfect).
""")

nb.code(r"""
proba = logit.predict_proba(X_te_s)[:, 1]          # P(churn) for each test row
fpr, tpr, thr = roc_curve(y_te, proba)
auc = roc_auc_score(y_te, proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"Logistic (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random (AUC=0.5)")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (recall)")
plt.title("ROC curve"); plt.legend(); plt.show()
""")

nb.deeper("""
AUC has a lovely plain-English meaning: it's the probability that the model gives a
randomly chosen real churner a HIGHER churn-score than a randomly chosen non-churner.
Because it judges ranking rather than a single cutoff, AUC is threshold-independent —
great for comparing models. Caveat: on heavily imbalanced data the precision-recall
curve (PR-AUC) is often more informative than ROC-AUC.
""")

nb.code(r"""
# Deliberately move the threshold to hit a recall target (catch >=70% of churners)
prec, rec, thresholds = precision_recall_curve(y_te, proba)
target_recall = 0.70
idx = np.argmin(np.abs(rec[:-1] - target_recall))
chosen = thresholds[idx]
print(f"To reach ~{target_recall:.0%} recall, set the threshold to about {chosen:.2f}")
new_pred = (proba >= chosen).astype(int)
print(classification_report(y_te, new_pred, target_names=["stay", "churn"]))
""")

nb.readcode("""
- We lower the cutoff below 0.5, so more customers get flagged as churn: recall goes
  UP (we catch more real churners) but precision goes DOWN (more false alarms).
- That single knob is how you translate a business priority into model behaviour.
""")

nb.jargon("ROC curve", "TPR vs FPR traced across every threshold; shows the full precision/recall trade")
nb.jargon("AUC", "area under the ROC curve; probability the model ranks a random positive above a random negative")
nb.jargon("decision threshold", "the probability cutoff for calling something positive (default 0.5)")

nb.md("## 8.8 Handling class imbalance (the honest toolkit)")

nb.plain("""
When the positive class is rare, help the model care about it. In rough order of
preference: (1) use the right metric (F1/recall/AUC, never accuracy alone);
(2) `class_weight="balanced"` to reweight the loss toward the rare class — cheap and
effective; (3) resample (oversample the minority e.g. SMOTE, or undersample the
majority) *inside* CV folds to avoid leakage; (4) move the threshold to match the
cost of a false alarm vs a miss.
""")

nb.code(r"""
balanced = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)
bal_pred = balanced.predict(X_te_s)
print("With class_weight='balanced':")
print(classification_report(y_te, bal_pred, target_names=["stay", "churn"]))
""")

nb.takeaway("Compared with the default model, class_weight='balanced' usually lifts churn "
            "RECALL (we catch more churners) at some cost to precision. Being able to say "
            "'here's the trade I made and why' out loud is exactly what interviewers reward.")

nb.md("## 8.9 Try it yourself")

nb.try_this("""
1. Fit KNN with the best k from the sweep; compare its F1 to logistic regression.
2. Find the threshold that MAXIMISES F1 for the logistic model (hint: scan the
   precision_recall_curve output).
3. Explain precision vs recall to a manager using the churn example, in two sentences.
4. Describe a real situation where a 99%-accuracy model is WORSE than an 80%-accuracy one.
""")

nb.md("## Summary")

nb.takeaway("""
- **Logistic regression**: linear score → sigmoid → probability; coefficients are log-odds.
- **KNN**: majority vote of nearest neighbours; you MUST scale; k is the bias–variance knob; beware the curse of dimensionality.
- **Accuracy lies under imbalance** — read the **confusion matrix**.
- **Precision** (of flagged, how many right) vs **recall** (of real positives, how many caught); **F1** balances them.
- **ROC/AUC** judge ranking across all thresholds; move the **threshold** to match business cost.
- Fight imbalance with metrics, `class_weight`, resampling, and thresholding.
""")

nb.md(r"""
Next: **Module 9 — Evaluation, Cross-Validation & Grid Search**: doing all of this
without fooling yourself.
""")

out = nb.save("notebooks/08_classification_knn.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
