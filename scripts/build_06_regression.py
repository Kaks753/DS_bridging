"""Builder for Module 06: Regression."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 06 — Regression (your first predictive model, understood deeply)

Regression predicts a **continuous number** (price, demand, temperature). We'll
build linear regression **from the math up**, then use scikit-learn the
professional way, then learn to **diagnose** it (assumptions, metrics,
regularization). Your Math degree makes this the module where theory clicks.

Goals:
- Derive what linear regression *optimizes* (least squares) and why.
- Read coefficients honestly.
- Know the metrics: MAE, MSE, RMSE, R² — and when each matters.
- Diagnose assumptions (linearity, residuals, multicollinearity).
- Fight overfitting with **Ridge/Lasso** regularization.
""")

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
print("shape:", df.shape); df.head()
""")

nb.md(r"""
## 6.1 The model and what it optimizes

Linear regression assumes:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \varepsilon $$

We choose the betas to **minimize the sum of squared residuals** (OLS — Ordinary
Least Squares):

$$ \text{minimize} \quad \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

**Why squared?** Squaring (1) makes all errors positive, (2) punishes big misses
harder, and (3) yields a smooth function with a clean closed-form solution — the
**normal equation**:

$$ \hat{\beta} = (X^\top X)^{-1} X^\top y $$
""")

nb.md(r"""
## 6.2 From scratch — prove sklearn isn't magic

We solve the normal equation with NumPy on a one-feature model, then confirm
sklearn matches. Seeing the same numbers builds real trust.
""")

nb.code(r"""
x = df["size_sqft"].values
y = df["price"].values

# Design matrix with a bias (intercept) column of ones
X_mat = np.column_stack([np.ones_like(x), x])       # shape (n, 2)

# Normal equation: beta = (X'X)^-1 X'y
beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y
print(f"from scratch:  intercept = {beta[0]:,.1f}, slope = {beta[1]:.2f}")

lr = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"scikit-learn:  intercept = {lr.intercept_:,.1f}, slope = {lr.coef_[0]:.2f}")
print("\nThey match -> OLS is just linear algebra.")
""")

nb.code(r"""
plt.figure(figsize=(7,4.5))
plt.scatter(x, y, alpha=0.5, label="data")
xs = np.linspace(x.min(), x.max(), 100)
plt.plot(xs, beta[0] + beta[1]*xs, "r", lw=2, label="OLS fit")
plt.xlabel("size (sqft)"); plt.ylabel("price"); plt.legend()
plt.title(f"Each extra sqft ≈ +{beta[1]:.0f} in price"); plt.show()
""")

nb.md(r"""
**Reading a coefficient:** slope ≈ price change per **one unit** increase of the
feature, *holding others fixed*. That "holding others fixed" clause is the whole
point of multiple regression — and a favorite interview probe.
""")

nb.md(r"""
## 6.3 Multiple regression — the professional workflow

Split → (scale) → fit → evaluate on the **held-out** test set. We scale so
coefficients are comparable and regularization behaves well.
""")

nb.code(r"""
features = ["size_sqft", "bedrooms", "age_years", "dist_center_km"]
X = df[features]
y = df["price"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_tr)          # fit on train only (no leakage)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

model = LinearRegression().fit(X_tr_s, y_tr)
pred = model.predict(X_te_s)

coefs = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
print("Standardized coefficients (impact per 1 std of feature):")
print(coefs.round(0))
""")

nb.md(r"""
Because features are standardized, coefficient **magnitude** now reflects
importance on a common scale: `size_sqft` should dominate, `dist_center_km` should
be negative (farther = cheaper). This matches how we generated the data — a good
sanity check.
""")

nb.md(r"""
## 6.4 Metrics — measure error like a pro

- **MAE** (mean absolute error): average |error|, in original units, robust to
  outliers. "On average we're off by X."
- **MSE** (mean squared error): squares errors → punishes big misses; units are
  squared (hard to read).
- **RMSE** (√MSE): back to original units; still outlier-sensitive. Most common.
- **R²**: fraction of variance explained, in (−∞, 1]. 1 = perfect, 0 = no better
  than predicting the mean, negative = worse than the mean.
""")

nb.code(r"""
mae = mean_absolute_error(y_te, pred)
mse = mean_squared_error(y_te, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_te, pred)
print(f"MAE  = {mae:,.0f}")
print(f"RMSE = {rmse:,.0f}")
print(f"R2   = {r2:.3f}   -> explains {r2*100:.1f}% of price variance")
""")

nb.md(r"""
**Which metric?** If large errors are especially costly (pricing, safety), watch
RMSE. If you want a robust, business-friendly number, report MAE. R² is great for
*communication* but can look high even when predictions are practically off — so
never report R² alone.
""")

nb.md(r"""
## 6.5 Diagnostics — is linear regression even appropriate?

Key OLS assumptions and how to check them:
1. **Linearity** — relationship is actually linear.
2. **Residuals centered at 0 with constant spread** (homoscedasticity).
3. **Residuals roughly normal** (for valid inference / intervals).
4. **Low multicollinearity** — features not near-duplicates.

The **residual plot** (residual vs predicted) is the single most informative
diagnostic: you want a shapeless cloud around 0.
""")

nb.code(r"""
resid = y_te - pred
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(pred, resid, alpha=0.5); ax[0].axhline(0, color="red", ls="--")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("residual")
ax[0].set_title("Residuals vs predicted (want: shapeless cloud)")
sns.histplot(resid, kde=True, ax=ax[1], color="steelblue")
ax[1].set_title("Residual distribution (want: ~normal, centered 0)")
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Multicollinearity** inflates coefficient variance and makes them
uninterpretable/unstable. A quick check is the correlation matrix; the formal tool
is **VIF** (Variance Inflation Factor) — VIF > ~5–10 signals trouble. Fixes: drop
one of the correlated pair, combine them, or use Ridge.
""")

nb.md(r"""
## 6.6 Regularization — Ridge & Lasso (control overfitting)

Regularization adds a penalty on coefficient size to the loss, discouraging the
model from fitting noise:

- **Ridge (L2):** penalty `α · Σβ²` — shrinks coefficients smoothly toward 0;
  great against multicollinearity.
- **Lasso (L1):** penalty `α · Σ|β|` — can drive some coefficients **exactly to
  0**, doing automatic **feature selection**.

`α` (alpha) is the strength: 0 = plain OLS; larger = simpler, more biased model.
This is the **bias–variance tradeoff** in a knob.
""")

nb.code(r"""
for name, mdl in [("OLS", LinearRegression()),
                  ("Ridge(a=10)", Ridge(alpha=10)),
                  ("Lasso(a=1000)", Lasso(alpha=1000, max_iter=10000))]:
    mdl.fit(X_tr_s, y_tr)
    p = mdl.predict(X_te_s)
    print(f"{name:14s} R2={r2_score(y_te,p):.3f}  RMSE={np.sqrt(mean_squared_error(y_te,p)):,.0f}  "
          f"coefs={np.round(mdl.coef_,0)}")
""")

nb.md(r"""
Notice how Lasso may zero-out weak features while Ridge shrinks all of them. On
this clean synthetic data OLS is already good, but on real, noisy, correlated data
regularization often wins — and *choosing α by cross-validation* (Module 08) is
the professional move.
""")

nb.md(r"""
## 6.7 Mini-exercises

1. Refit using only `size_sqft` and `bedrooms`. How much R² do you lose? What does
   that say about the dropped features?
2. Add a useless random-noise column to `X`. Does OLS R² on **train** go up while
   **test** stagnates? (That's overfitting.)
3. Sweep Ridge `alpha` over `[0.1, 1, 10, 100, 1000]` and plot test RMSE.
4. Explain "holding other variables constant" to a non-technical stakeholder.
""")

nb.md(r"""
## Summary

- Linear regression minimizes **squared residuals**; closed form = normal equation.
- Coefficients = effect per unit **holding others fixed**; standardize to compare.
- Metrics: **MAE** (robust), **RMSE** (penalizes big errors), **R²** (variance
  explained — never report alone).
- Diagnose with **residual plots**; watch **multicollinearity** (VIF).
- **Ridge/Lasso** trade a little bias for less variance; Lasso also selects
  features. Tune α by CV.

Next: **Module 07 — Classification & KNN** (predicting categories).
""")

out = nb.save("notebooks/06_regression.ipynb")
print("saved", out)
