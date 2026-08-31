"""Builder for Module 18: OOP + Linear Algebra & Calculus for ML (4-layer)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 18 — OOP + Linear Algebra & Calculus for ML

Two pillars that make ML *click* instead of feeling like magic:
1. **OOP (Object-Oriented Programming)** — the code structure behind scikit-learn
   (every model is an object with `fit`/`predict`). Understanding it lets you write
   custom transformers and read library source.
2. **Linear algebra & calculus** — the math engine under the hood. Your degree gives
   you the theory; here we ground it in the *exact* operations ML uses, so you can
   explain gradient descent and PCA from the ground up.

**What you'll be able to do by the end:**
- Write a class with `__init__`, `self`, methods, and understand the sklearn
  estimator pattern.
- Read a dot product / matrix multiply as ML *meaning*, not abstract symbols.
- Explain eigenvectors as the heart of PCA.
- Implement **gradient descent** from scratch and know what the learning rate does.
""")

nb.plain(r"""
Two ideas, one plain sentence each:
- **OOP** is a way to bundle *data* and the *functions that work on it* into one tidy
  package (an "object") — like a smartphone bundling a camera, its photos, and the
  buttons to use them.
- **The math** is just two moves on tables of numbers: **multiply-and-add** (linear
  algebra) to make predictions, and **follow the slope downhill** (calculus) to
  improve them. Every model you've met is those two moves on repeat.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
print("Tools ready.")
""")

# ---------------------------------------------------------------------------
# 18.1 OOP
# ---------------------------------------------------------------------------
nb.md(r"""
## 18.1 OOP — objects with state and behaviour
""")

nb.analogy(r"""
A **class** is a *cookie cutter*; an **object** is a *cookie* stamped from it. The
cutter defines the shape (what data and actions every cookie has); each cookie is a
separate thing with its own filling. `model = RandomForest()` stamps one cookie;
`model.fit(X, y)` fills it with what it learned.
""")

nb.jargon("Class", "a blueprint describing what data and actions its objects will have")
nb.jargon("Object / instance", "one concrete thing built from a class, with its own data")
nb.jargon("self", "the word a method uses to refer to 'this particular object'")
nb.jargon("Attribute", "a piece of data stored on an object (its state)")
nb.jargon("Method", "a function that belongs to an object (its behaviour)")

nb.code(r"""
class RunningStats:
    "A small object that tracks numbers and computes stats -- demonstrates state."
    def __init__(self, name):
        self.name = name          # attribute (state)
        self.values = []          # attribute (state)

    def add(self, x):             # method (behaviour)
        self.values.append(x)
        return self               # returning self enables chaining

    def mean(self):
        return sum(self.values) / len(self.values) if self.values else float("nan")

    def __repr__(self):           # controls how the object prints
        return f"RunningStats(name={self.name!r}, n={len(self.values)})"

s = RunningStats("spend")
s.add(10).add(20).add(30)         # method chaining
print(s, "-> mean =", s.mean())
""")

nb.readcode(r"""
- `__init__` is the setup that runs when you create the object; it stores the starting
  state (`name`, empty `values`).
- Every method takes `self` first — that's how it reaches *this* object's data.
- `add()` returns `self`, which is why we can chain `.add(10).add(20).add(30)` in one
  line.
- `__repr__` just makes `print(s)` show something readable instead of a memory address.
""")

nb.md(r"""
### The scikit-learn estimator pattern (write your own transformer)

Every sklearn transformer implements `fit` (learn parameters from training data) and
`transform` (apply them). Inheriting from `BaseEstimator, TransformerMixin` makes your
class drop straight into a **Pipeline**. Here's a custom scaler — now you'll know what
`StandardScaler` actually *is* underneath.
""")

nb.code(r"""
from sklearn.base import BaseEstimator, TransformerMixin

class MyStandardScaler(BaseEstimator, TransformerMixin):
    "Reimplements StandardScaler to show the fit/transform contract."
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)   # learned state (trailing _ = 'fitted')
        self.std_  = X.std(axis=0)
        return self                   # fit ALWAYS returns self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

X = rng.normal(50, 10, size=(6, 2))
mine = MyStandardScaler().fit(X)
out = mine.transform(X)
print("learned mean:", mine.mean_.round(2), "| learned std:", mine.std_.round(2))
print("transformed mean (~0):", out.mean(0).round(6), "| std (~1):", out.std(0).round(6))

from sklearn.preprocessing import StandardScaler
ref = StandardScaler().fit_transform(X)
print("matches sklearn StandardScaler:", np.allclose(out, ref))
""")

nb.deeper(r"""
Two conventions worth internalizing: (1) **`fit` learns and stores** results as
attributes ending in `_` (like `mean_`), then returns `self`; **`transform` uses**
those stored attributes. Keeping learning and applying separate is exactly how
Pipelines avoid data leakage — `fit` sees only training data.
(2) **Inheritance**: `class MyStandardScaler(BaseEstimator, TransformerMixin)` means
it *is-a* transformer and inherits methods like `fit_transform` for free. The four
pillars of OOP to name in an interview: **encapsulation, inheritance, polymorphism,
abstraction**.
""")

nb.takeaway("A class bundles state (attributes) + behaviour (methods); sklearn's fit/transform/predict is that pattern, and fit always returns self.")
nb.interview("\"Every sklearn estimator follows the same fit/predict contract, so once you understand the pattern you can drop a custom transformer into any Pipeline.\"")

# ---------------------------------------------------------------------------
# 18.2 Linear algebra
# ---------------------------------------------------------------------------
nb.md(r"""
## 18.2 Linear algebra — the meaning behind the math

Your data is a **matrix** `X` of shape `(rows = samples, columns = features)`. Almost
every model is just matrix operations on it. Two operations dominate.
""")

nb.plain(r"""
- **Dot product**: multiply two lists element-by-element and add up the results. It
  measures how much two things "point the same way" — the weighted sum inside every
  linear model and every neuron.
- **Matrix multiplication**: doing many dot products at once. `X @ weights` computes a
  prediction for *every* row in a single step, which is why NumPy is so fast.
""")

nb.jargon("Dot product", "multiply two vectors element-wise and sum -> a similarity / weighted-sum score")
nb.jargon("Matrix multiplication", "many dot products at once; lets you predict all rows in one operation")

nb.md(r"""
### ✏️ Worked example (by hand): a dot product

Take weights $w = [180,\ 15000]$ (price per sqft, price per bedroom) and one house
$x = [1400,\ 3]$ (1400 sqft, 3 bedrooms). The prediction (before bias) is the dot
product:

$$ w \cdot x = (180)(1400) + (15000)(3) = 252{,}000 + 45{,}000 = 297{,}000 $$

That's it — a dot product is "multiply matching pieces, then add". A linear model's
prediction is exactly this, plus a bias term.
""")

nb.code(r"""
# A linear model's predictions ARE a matrix-vector product
X = np.array([[1400, 3], [1800, 4], [1200, 2]], dtype=float)  # [sqft, bedrooms]
weights = np.array([180.0, 15000.0])
bias = 50000.0
preds = X @ weights + bias           # one matmul = all predictions at once
print("predictions:", preds)

# verify the by-hand row
print("row 0 by hand: 180*1400 + 15000*3 + 50000 =", 180*1400 + 15000*3 + 50000)

# dot product as similarity (cosine) -- used in NLP & recommenders
a = np.array([1, 0, 1]); b = np.array([1, 1, 1])
cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("cosine similarity a,b:", round(cos, 3))
""")

nb.md(r"""
### Eigenvectors & eigenvalues — the engine of PCA

For a matrix `A`, an **eigenvector** `v` is a special direction that `A` only
**stretches** (never rotates): `A v = λ v`. The number `λ` is the **eigenvalue** (the
stretch factor). PCA finds the eigenvectors of the data's **covariance matrix** — they
are the **principal components**, and each eigenvalue is the variance captured along
that direction.
""")

nb.analogy(r"""
Push on a stretchy sheet in most directions and it both moves and twists. But a few
special directions only stretch straight, no twist — those are the **eigenvectors**,
and *how much* they stretch is the **eigenvalue**. PCA hunts for the directions your
data stretches out the most (the most variance).
""")

nb.jargon("Eigenvector", "a direction a matrix only stretches (doesn't rotate)")
nb.jargon("Eigenvalue", "how much that eigenvector direction gets stretched (its variance, in PCA)")

nb.md(r"""
### ✏️ Worked example (by hand): eigenvalues of a diagonal matrix

For $A = \begin{bmatrix} 2 & 0 \\ 0 & 5 \end{bmatrix}$, the axes are already the
special directions. Applying $A$ to $\begin{bmatrix}1\\0\end{bmatrix}$ gives
$\begin{bmatrix}2\\0\end{bmatrix}$ — same direction, stretched by **2**. Applying it to
$\begin{bmatrix}0\\1\end{bmatrix}$ gives $\begin{bmatrix}0\\5\end{bmatrix}$ — stretched
by **5**. So the eigenvalues are **2 and 5**, with eigenvectors along the x- and
y-axes. For a diagonal matrix, the diagonal entries *are* the eigenvalues.
""")

nb.code(r"""
A = np.array([[2, 0], [0, 5]], dtype=float)
vals, vecs = np.linalg.eig(A)
print("eigenvalues:", vals)          # expect 2 and 5
print("eigenvectors (columns):\n", vecs.round(3))

# Correlated 2-D data -> covariance eigenvectors point along the spread
data = rng.multivariate_normal([0, 0], [[3, 2], [2, 2]], size=500)
cov = np.cov(data.T)
eigvals, eigvecs = np.linalg.eigh(cov)      # eigh for symmetric matrices
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
print("\ncovariance eigenvalues (variance per component):", eigvals.round(2))

plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], alpha=0.3)
for val, vec in zip(eigvals, eigvecs.T):
    plt.arrow(0, 0, vec[0]*np.sqrt(val)*2, vec[1]*np.sqrt(val)*2, width=0.05, color="red")
plt.axis("equal"); plt.title("Eigenvectors of covariance = principal components")
plt.show()
""")

nb.takeaway("Data is a matrix; dot product = weighted-sum/similarity; matmul predicts all rows at once; eigenvectors of covariance = PCA components.")

# ---------------------------------------------------------------------------
# 18.3 Calculus
# ---------------------------------------------------------------------------
nb.md(r"""
## 18.3 Calculus — how models *learn*
""")

nb.plain(r"""
A **derivative** is just a slope: "if I nudge the input a little, how much does the
output move, and in which direction?" When the slope is zero, you're at the bottom of a
valley (or top of a hill) — that flat spot is what "best answer" looks like. The
**gradient** is the slope when there are several inputs, and it points *uphill*. To make
an error smaller, we walk the *opposite* way. That walk is **gradient descent**, and it
trains basically everything.
""")

nb.jargon("Derivative", "the slope of a function: how fast output changes as input changes")
nb.jargon("Gradient", "the vector of slopes for many inputs; points in the steepest-uphill direction")
nb.jargon("Gradient descent", "repeatedly stepping downhill (against the gradient) to minimize error")
nb.jargon("Learning rate", "how big each downhill step is; too big overshoots, too small crawls")

nb.code(r"""
# Numerically verify a derivative: f(x)=x^2 -> f'(x)=2x
def f(x): return x**2
def numerical_deriv(f, x, h=1e-6): return (f(x+h) - f(x-h)) / (2*h)
for x in [1, 3, -2]:
    print(f"f'({x}): numerical={numerical_deriv(f, x):.4f}, exact(2x)={2*x}")
""")

nb.md(r"""
### ✏️ Worked example (by hand): one gradient-descent step

Fit a line $y = wx$ (no bias) to a single point $(x, y) = (2, 10)$. Start at $w = 0$,
learning rate $lr = 0.1$. Loss = squared error $(wx - y)^2$.

**Prediction:** $\hat{y} = w x = 0 \times 2 = 0$.
**Error:** $\hat{y} - y = 0 - 10 = -10$.
**Gradient** of the loss w.r.t. $w$ (chain rule): $2(\hat{y}-y)\,x = 2(-10)(2) = -40$.
**Step downhill:** $w \leftarrow w - lr \times \text{grad} = 0 - 0.1(-40) = 4$.

One step moved $w$ from 0 toward the ideal $w = 5$ (since $5 \times 2 = 10$). Repeat
and it converges. The gradient was negative, so we *increased* $w$ — always step
opposite the slope.
""")

nb.code(r"""
x_pt, y_pt, w, lr = 2.0, 10.0, 0.0, 0.1
y_hat = w * x_pt
error = y_hat - y_pt
grad  = 2 * error * x_pt
w_new = w - lr * grad
print(f"pred={y_hat}, error={error}, grad={grad}, w after 1 step={w_new}")
print("(ideal w is 5, since 5*2=10 -- we're heading there)")
""")

nb.md(r"""
### Gradient descent from scratch — fit a whole line by *learning*

Now the full loop on noisy data. This is the same loop that trains neural nets, minus
the extra layers.
""")

nb.code(r"""
# Synthetic linear data: y = 3x + 7 + noise
x = rng.uniform(-5, 5, 200)
y = 3*x + 7 + rng.normal(0, 2, 200)

w, b = 0.0, 0.0            # start at zero
lr = 0.01
history = []
for step in range(200):
    y_hat = w*x + b
    error = y_hat - y
    grad_w = 2 * np.mean(error * x)   # dMSE/dw
    grad_b = 2 * np.mean(error)       # dMSE/db
    w -= lr * grad_w                  # step DOWNHILL
    b -= lr * grad_b
    history.append(np.mean(error**2))

print(f"learned:  w={w:.3f} (true 3), b={b:.3f} (true 7)")

plt.figure(figsize=(7,4))
plt.plot(history)
plt.xlabel("iteration"); plt.ylabel("MSE loss")
plt.title("Gradient descent: loss falls as parameters learn"); plt.show()
""")

nb.readcode(r"""
- Start with `w=b=0` (a flat, wrong line).
- Each loop: predict, measure the error, compute how the loss slopes w.r.t. `w` and
  `b`, then nudge both a little downhill.
- `history` records the loss each step — the plot shows it dropping to near zero as
  `w`->3 and `b`->7 (the true values).
- The `lr` (learning rate) sets the step size: too big and it overshoots/diverges, too
  small and it crawls.
""")

nb.warn("Learning rate is the make-or-break knob: too large -> loss explodes/oscillates; too small -> training is painfully slow. Tune it first.")
nb.takeaway("Predict -> error -> gradient -> step opposite the gradient -> repeat. That five-line loop is how almost every ML model learns.")

# ---------------------------------------------------------------------------
# Practice + summary
# ---------------------------------------------------------------------------
nb.md(r"""
## 18.4 Practice
""")

nb.try_this(r"""
1. Add a `std == 0` guard to `MyStandardScaler` (constant columns divide by zero).
2. Give `RunningStats` a `.std()` method; test it against `np.std`.
3. In gradient descent, set `lr=0.5`, then `lr=0.001`. Describe each loss curve — what
   breaks?
4. Confirm the eigenvalues of `[[2,0],[0,5]]` by hand, then check with `np.linalg.eig`.
""")

nb.md(r"""
## Summary

- **OOP**: classes bundle **state** (attributes) + **behaviour** (methods); `self` is
  the instance. sklearn's **fit/transform/predict** pattern lets you build
  Pipeline-compatible components; `fit` returns `self`. Pillars: encapsulation,
  inheritance, polymorphism, abstraction.
- **Linear algebra**: data = matrix; **dot product** = weighted sum / similarity;
  **matmul** = predict-all-at-once; **eigenvectors of covariance = PCA components**.
- **Calculus**: derivative = slope; **gradient** points uphill; **gradient descent**
  steps downhill to minimize loss — the universal learning engine, with **learning
  rate** as the key knob.

Next: **Module 19 — Time Series Analysis & Forecasting**.
""")

out = nb.save("notebooks/18_oop_linear_algebra_calculus.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
