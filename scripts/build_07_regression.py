"""Builder for Module 7: Regression (4-layer rewrite of old M06)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 7 — Regression: your first predictive model, understood deeply

**Regression** predicts a *continuous number* — a price, a demand, a temperature —
as opposed to a category. It's the "hello world" of machine learning, and because
you have a Maths background, this is the module where the theory actually *clicks*
instead of feeling like a black box.

We'll do it in three passes:
1. Build linear regression **from the math up** (and prove scikit-learn isn't magic).
2. Use it the **professional way** (split → scale → fit → evaluate).
3. Learn to **diagnose and improve** it (assumptions, metrics, regularization).
""")

nb.analogy("Linear regression is drawing the single straight line that sits as close as "
           "possible to a cloud of dots. 'As close as possible' has a precise meaning — "
           "and that precise meaning is the whole subject.")

nb.md("## 7.1 Setup")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/house_prices.csv")
print("shape:", df.shape)
df.head()
""")

nb.jargon("regression", "predicting a continuous number (e.g. price), not a category")
nb.jargon("feature", "an input column the model learns from (size, bedrooms, ...)")
nb.jargon("target", "the thing you're trying to predict (here: price)")

nb.md("## 7.2 What the model actually optimizes")

nb.plain("""
Linear regression assumes the target is a weighted sum of the features plus some
noise. Each feature gets a **coefficient** (a weight) saying "how much price moves
when this feature goes up by one". Training = finding the weights that make the
line fit the data as tightly as possible.
""")

nb.md(r"""
Formally, the model is:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \varepsilon $$

and we pick the betas to **minimize the sum of squared residuals** — this is
**OLS (Ordinary Least Squares)**:

$$ \text{minimize} \quad \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
""")

nb.deeper(r"""
**Why *squared* residuals, not absolute?** Three reasons, and interviewers love this:
(1) squaring makes every error positive so they don't cancel out; (2) it punishes
one big miss far more than several small ones (a 10-unit error counts 100, two
5-unit errors count 50 total); (3) it's smooth and differentiable, which gives a
clean closed-form solution — the **normal equation**:

$$ \hat{\beta} = (X^\top X)^{-1} X^\top y $$

Using absolute error instead gives *quantile/median* regression, which is more
robust to outliers but has no neat closed form.
""")

nb.jargon("residual", "the miss for one row: actual minus predicted (y − ŷ)")
nb.jargon("OLS", "Ordinary Least Squares — the method that minimizes summed squared residuals")
nb.jargon("coefficient", "the weight on a feature: how much y changes per 1-unit change in that feature")

nb.md("## 7.3 From scratch — prove sklearn isn't magic")

nb.plain("""
Let's solve the normal equation ourselves with NumPy on a one-feature model
(price vs size), then check scikit-learn lands on the exact same numbers. Seeing
them match once removes the 'black box' fear forever.
""")

nb.code(r"""
x = df["size_sqft"].values
y = df["price"].values

# Design matrix: a column of 1s (for the intercept) next to the feature
X_mat = np.column_stack([np.ones_like(x), x])       # shape (n, 2)

# Normal equation:  beta = (X'X)^-1 X'y
beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y
print(f"from scratch:  intercept = {beta[0]:,.1f}, slope = {beta[1]:.2f}")

lr = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"scikit-learn:  intercept = {lr.intercept_:,.1f}, slope = {lr.coef_[0]:.2f}")
print("\nSame numbers -> OLS is just linear algebra, not magic.")
""")

nb.readcode("""
- `np.column_stack([ones, x])` builds the design matrix X with a bias column so the
  intercept β0 is learned like any other weight.
- `X.T @ X` is matrix multiply; `np.linalg.inv(...)` inverts it; the whole line is
  the normal equation typed out.
- We fit sklearn on the same data and print both — they agree to the decimal.
""")

nb.code(r"""
plt.figure(figsize=(7, 4.5))
plt.scatter(x, y, alpha=0.5, label="data")
xs = np.linspace(x.min(), x.max(), 100)
plt.plot(xs, beta[0] + beta[1]*xs, "r", lw=2, label="OLS fit")
plt.xlabel("size (sqft)"); plt.ylabel("price"); plt.legend()
plt.title(f"Each extra sqft is worth about +{beta[1]:.0f} in price"); plt.show()
""")

nb.takeaway("The slope is a price tag: a fixed amount of extra price per extra square foot. "
            "That single interpretable number is why linear regression is still the first "
            "model businesses trust.")

nb.md("## 7.4 A tiny worked example — by hand, then verified")

nb.plain("""
Numbers on paper beat a formula every time. Take four houses and fit the best line
BY HAND with the slope/intercept formulas, then let NumPy confirm we got it right.
""")

nb.md(r"""
For one feature, OLS has a friendly closed form:

$$ \text{slope} = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}, \qquad
   \text{intercept} = \bar y - \text{slope}\cdot \bar x $$
""")

nb.code(r"""
# Four toy houses: size (100s of sqft) -> price (10k)
xs = np.array([1.0, 2.0, 3.0, 4.0])
ys = np.array([2.0, 3.0, 5.0, 4.0])

xbar, ybar = xs.mean(), ys.mean()
print(f"step 1  means:      x̄ = {xbar}, ȳ = {ybar}")

num = np.sum((xs - xbar) * (ys - ybar))
den = np.sum((xs - xbar) ** 2)
print(f"step 2  numerator Σ(x-x̄)(y-ȳ) = {num}")
print(f"step 3  denominator Σ(x-x̄)²   = {den}")

slope = num / den
intercept = ybar - slope * xbar
print(f"step 4  slope     = {num}/{den} = {slope}")
print(f"step 5  intercept = {ybar} - {slope}*{xbar} = {intercept}")

# verify against numpy's polyfit
m, b = np.polyfit(xs, ys, 1)
print(f"\nnumpy check: slope = {m:.4f}, intercept = {b:.4f}")
""")

nb.takeaway("You just fit a regression with nothing but a mean, a sum, and a division. "
            "'Machine learning' is often this humble underneath.")

nb.md("## 7.5 Multiple regression — the professional workflow")

nb.plain("""
Real predictions use several features at once. The disciplined recipe never changes:
**split** the data (so we can measure honestly), **scale** the features (so their
sizes are comparable), **fit** on the training set, and **evaluate on the held-out
test set** the model has never seen.
""")

nb.code(r"""
features = ["size_sqft", "bedrooms", "age_years", "dist_center_km"]
X = df[features]
y = df["price"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_tr)              # FIT ON TRAIN ONLY (no leakage)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

model = LinearRegression().fit(X_tr_s, y_tr)
pred = model.predict(X_te_s)

coefs = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
print("Standardized coefficients (impact per 1 std of the feature):")
print(coefs.round(0))
""")

nb.readcode("""
- `train_test_split(..., test_size=0.2, random_state=42)` holds back 20% for an
  honest exam; the seed makes it reproducible.
- `StandardScaler().fit(X_tr)` learns mean/std FROM TRAIN, then we `.transform` both
  sets with those same numbers — the test set must not influence the scaling.
- Because features are standardized, coefficient *magnitude* now means importance on
  a common scale: `size_sqft` dominates, `dist_center_km` is negative (farther = cheaper).
""")

nb.deeper("""
'Holding others fixed' is the soul of multiple regression. A coefficient answers:
*if I nudge THIS feature by one unit and freeze all the rest, how does price move?*
That conditional interpretation is exactly why you can't read a coefficient in
isolation when features are correlated — a classic interview trap.
""")

nb.md("## 7.6 Metrics — measuring error like a pro")

nb.plain("""
'Is the model good?' needs a number. Four you must know cold:
- **MAE** — average size of the miss, in real units, shrugs off outliers.
- **MSE** — average squared miss; punishes big misses; units are squared (unreadable).
- **RMSE** — square root of MSE; back in real units; the most-reported one.
- **R²** — the fraction of the target's variance the model explains.
""")

nb.code(r"""
mae  = mean_absolute_error(y_te, pred)
mse  = mean_squared_error(y_te, pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_te, pred)
print(f"MAE  = {mae:,.0f}   (typical miss, in price units)")
print(f"RMSE = {rmse:,.0f}   (miss, but big errors weigh more)")
print(f"R2   = {r2:.3f}   -> explains {r2*100:.1f}% of price variance")
""")

nb.md(r"""
Reading R²: it lives in $(-\infty, 1]$. **1** = perfect, **0** = no better than
just predicting the average price every time, **negative** = worse than that.
""")

nb.interview("""
Q: 'MAE vs RMSE — which and when?' Answer: RMSE when large errors are especially
costly (pricing, safety, demand planning), because squaring makes the metric scream
about big misses. MAE when you want a robust, plain-English 'we're off by about X on
average' that outliers can't hijack. And never report R² alone — a high R² can still
hide predictions that are useless in practice.
""")

nb.md("## 7.7 Diagnostics — is a straight line even appropriate?")

nb.plain("""
Before trusting a linear model, check its assumptions. The single most useful check
is the **residual plot**: plot the misses against the predictions. You want a
shapeless, boring cloud centred on zero. Any pattern (a curve, a widening funnel)
means the straight line is the wrong shape for this data.
""")

nb.code(r"""
resid = y_te - pred
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(pred, resid, alpha=0.5); ax[0].axhline(0, color="red", ls="--")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("residual (actual - predicted)")
ax[0].set_title("Residuals vs predicted (want: shapeless cloud around 0)")
sns.histplot(resid, kde=True, ax=ax[1], color="steelblue")
ax[1].set_title("Residual distribution (want: ~normal, centred on 0)")
plt.tight_layout(); plt.show()
""")

nb.readcode("""
- Left panel: residual vs prediction. A funnel shape = non-constant spread
  (heteroscedasticity); a curve = the true relationship is non-linear.
- Right panel: the shape of the misses. Roughly bell-shaped and centred on 0 is
  what valid confidence intervals assume.
""")

nb.deeper("""
The four OLS assumptions worth naming: linearity, constant residual spread
(homoscedasticity), roughly-normal residuals (for valid inference), and low
**multicollinearity** — features that aren't near-duplicates of each other.
Multicollinearity doesn't hurt predictions much but makes individual coefficients
wobble wildly, so you can't trust their story. The formal detector is **VIF**
(Variance Inflation Factor); VIF above ~5–10 is a red flag. Fixes: drop one of the
pair, combine them, or use Ridge.
""")

nb.jargon("residual plot", "misses plotted against predictions; a pattern means the model shape is wrong")
nb.jargon("multicollinearity", "two or more features carry nearly the same information, destabilising coefficients")
nb.jargon("homoscedasticity", "residuals have roughly constant spread across all predictions")

nb.md("## 7.8 Regularization — Ridge & Lasso fight overfitting")

nb.plain("""
An unconstrained model can chase noise and memorise the training set — great on the
exam it already saw, bad on new data. **Regularization** adds a penalty for large
coefficients, gently forcing the model to stay simple. Two flavours:
- **Ridge (L2)** — shrinks all coefficients smoothly toward zero.
- **Lasso (L1)** — can push weak coefficients *exactly* to zero, doing automatic
  feature selection.
""")

nb.md(r"""
Ridge minimises $\text{RSS} + \alpha\sum\beta_j^2$; Lasso minimises
$\text{RSS} + \alpha\sum|\beta_j|$. The knob **α** sets the strength: `α = 0` is
plain OLS; larger α = simpler, more biased, less variance.
""")

nb.code(r"""
for name, mdl in [("OLS",            LinearRegression()),
                  ("Ridge(a=10)",    Ridge(alpha=10)),
                  ("Lasso(a=1000)",  Lasso(alpha=1000, max_iter=10000))]:
    mdl.fit(X_tr_s, y_tr)
    p = mdl.predict(X_te_s)
    print(f"{name:14s} R2={r2_score(y_te, p):.3f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_te, p)):,.0f}  "
          f"coefs={np.round(mdl.coef_, 0)}")
""")

nb.readcode("""
- Same features, three penalties. Watch Lasso zero-out weak coefficients while Ridge
  merely shrinks them.
- On this clean synthetic data plain OLS is already strong; the payoff from
  regularization shows up on real, noisy, correlated data.
""")

nb.analogy("α is a strictness dial on a teacher. Turn it up and the model is forced to keep "
           "its answer simple (few strong coefficients); turn it to zero and it's free to "
           "overcomplicate and memorise. The sweet spot is found by cross-validation.")

nb.deeper("""
This IS the bias–variance tradeoff made tangible: raising α adds bias (the model
can't fit as freely) but cuts variance (it stops chasing noise). The professional
move is to pick α by **cross-validation** — `RidgeCV` / `LassoCV` — which we cover
properly in the Evaluation module. Never hand-pick α from the test set: that leaks.
""")

nb.jargon("regularization", "adding a penalty on coefficient size to discourage overfitting")
nb.jargon("Ridge (L2)", "penalty on squared coefficients; shrinks all of them smoothly toward 0")
nb.jargon("Lasso (L1)", "penalty on absolute coefficients; can zero some out (feature selection)")
nb.jargon("alpha (α)", "the regularization strength knob: 0 = plain OLS, larger = simpler model")

nb.md("## 7.9 Try it yourself")

nb.try_this("""
1. Refit using only `size_sqft` and `bedrooms`. How much R² do you lose — and what
   does that tell you about the features you dropped?
2. Add a column of pure random noise to X. Watch train R² creep up while test R²
   stalls — that gap IS overfitting, seen live.
3. Sweep Ridge `alpha` over [0.1, 1, 10, 100, 1000] and plot test RMSE vs α.
4. Explain 'holding other variables constant' to a non-technical stakeholder in
   one sentence.
""")

nb.md("## Summary")

nb.takeaway("""
- Linear regression minimises **squared residuals**; the closed form is the normal equation.
- A coefficient = effect per unit **holding others fixed**; standardize to compare them.
- Metrics: **MAE** (robust), **RMSE** (penalizes big misses), **R²** (variance explained — never alone).
- Diagnose with a **residual plot**; watch for **multicollinearity** (VIF).
- **Ridge/Lasso** trade a little bias for less variance; Lasso also selects features. Tune α by cross-validation.
""")

nb.md(r"""
Next: **Module 8 — Classification & KNN**, where we stop predicting numbers and
start predicting *categories*.
""")

out = nb.save("notebooks/07_regression.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
