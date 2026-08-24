"""Builder for Module 17: OOP + Linear Algebra & Calculus for ML."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 17 — OOP + Linear Algebra & Calculus for ML (Phase 3: T21–22)

Two pillars that make ML *click* instead of feeling like magic:
1. **OOP** — the code structure behind scikit-learn (every model is an object with
   `fit`/`predict`). Understanding it lets you write custom transformers and read
   library source.
2. **Linear algebra & calculus** — the math engine. Your degree gives you the
   theory; here we ground it in the *exact* operations ML uses, so you can explain
   gradient descent and PCA from the ground up.

Goals:
- Classes, `__init__`, `self`, methods, attributes; the sklearn estimator pattern.
- Vectors, dot products, matrix multiplication — as ML *meaning*, not abstract math.
- Eigenvectors/eigenvalues → the heart of PCA.
- Derivatives, gradients, and **gradient descent** implemented from scratch.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
""")

nb.md(r"""
## 17.1 OOP — objects with state and behavior

A **class** is a blueprint; an **object** is an instance of it. `__init__` sets up
initial **state** (attributes); **methods** are functions that act on that state;
`self` refers to the specific instance. This is exactly how `model = RandomForest()`
then `model.fit(X, y)` works — `fit` stores learned state on `self`.
""")

nb.code(r"""
class RunningStats:
    "A small object that tracks numbers and computes stats — demonstrates state."
    def __init__(self, name):
        self.name = name          # attribute (state)
        self.values = []          # attribute (state)

    def add(self, x):             # method (behavior)
        self.values.append(x)
        return self               # returning self enables chaining

    def mean(self):
        return sum(self.values) / len(self.values) if self.values else float("nan")

    def __repr__(self):           # nice printing
        return f"RunningStats(name={self.name!r}, n={len(self.values)})"

s = RunningStats("spend")
s.add(10).add(20).add(30)         # method chaining
print(s, "-> mean =", s.mean())
""")

nb.md(r"""
### The scikit-learn estimator pattern (write your own transformer)

Every sklearn transformer implements `fit` (learn parameters from training data)
and `transform` (apply them). Inheriting from `BaseEstimator, TransformerMixin`
makes your class drop into a **Pipeline** (Module 08). Here's a custom scaler — now
you understand what StandardScaler *is* under the hood.
""")

nb.code(r"""
from sklearn.base import BaseEstimator, TransformerMixin

class MyStandardScaler(BaseEstimator, TransformerMixin):
    "Reimplements StandardScaler to show the fit/transform contract."
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)      # learned state (note trailing underscore = fitted)
        self.std_ = X.std(axis=0)
        return self                      # fit ALWAYS returns self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

X = rng.normal(50, 10, size=(6, 2))
mine = MyStandardScaler().fit(X)
out = mine.transform(X)
print("learned mean:", mine.mean_.round(2), "| learned std:", mine.std_.round(2))
print("transformed mean (~0):", out.mean(0).round(6), "| std (~1):", out.std(0).round(6))

# Prove it matches sklearn
from sklearn.preprocessing import StandardScaler
ref = StandardScaler().fit_transform(X)
print("matches sklearn StandardScaler:", np.allclose(out, ref))
""")

nb.md(r"""
**Inheritance in one line:** `class B(A)` means B *is-a* A and gets A's methods.
That's how `MyStandardScaler` inherited `fit_transform` from `TransformerMixin`
without us writing it. Four pillars of OOP to name: **encapsulation, inheritance,
polymorphism, abstraction**.
""")

nb.md(r"""
## 17.2 Linear algebra — the meaning behind the math

Data is a **matrix** `X` of shape `(n_samples, n_features)`. Nearly every model is
matrix operations on it. Two operations dominate:

- **Dot product** `a·b = Σ aᵢbᵢ` — measures alignment/similarity; it's the weighted
  sum inside every linear model and neuron.
- **Matrix multiplication** — applies a linear transformation / composes many dot
  products at once (e.g. `X @ weights` computes all predictions in one shot).
""")

nb.code(r"""
# A linear model's predictions ARE a matrix-vector product
X = np.array([[1400, 3], [1800, 4], [1200, 2]], dtype=float)  # [sqft, bedrooms]
weights = np.array([180.0, 15000.0])                           # price per unit
bias = 50000.0
preds = X @ weights + bias           # one matmul = all predictions
print("predictions:", preds)

# dot product as similarity (cosine) — used in NLP & recommenders (Module 19)
a = np.array([1, 0, 1]); b = np.array([1, 1, 1])
cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("cosine similarity:", round(cos, 3))
""")

nb.md(r"""
### Eigenvectors & eigenvalues — the engine of PCA

For a matrix `A`, an **eigenvector** `v` is a special direction that `A` only
**stretches** (doesn't rotate): `A v = λ v`. The scalar `λ` is its **eigenvalue**
(the stretch factor). PCA finds the eigenvectors of the data's **covariance
matrix**: they are the **principal components**, and each eigenvalue is the variance
captured along that direction. This is Module 10's PCA, now from the linear-algebra
side.
""")

nb.code(r"""
# Correlated 2-D data -> its covariance eigenvectors point along the spread
data = rng.multivariate_normal([0, 0], [[3, 2], [2, 2]], size=500)
cov = np.cov(data.T)
eigvals, eigvecs = np.linalg.eigh(cov)         # eigh for symmetric matrices
order = np.argsort(eigvals)[::-1]              # largest variance first
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
print("eigenvalues (variance per component):", eigvals.round(2))

plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], alpha=0.3)
for val, vec in zip(eigvals, eigvecs.T):
    plt.arrow(0, 0, vec[0]*np.sqrt(val)*2, vec[1]*np.sqrt(val)*2,
              width=0.05, color="red")
plt.axis("equal"); plt.title("Eigenvectors of covariance = principal components")
plt.show()
""")

nb.md(r"""
## 17.3 Calculus — how models *learn*

- A **derivative** `f'(x)` is the slope: how fast the output changes as the input
  changes. Zero slope = flat = a minimum/maximum (how we optimize).
- For many inputs, the **gradient** `∇f` is the vector of partial derivatives — it
  points in the direction of **steepest ascent**. To *minimize* a loss, we step in
  the **opposite** direction.

That single idea — **gradient descent** — trains linear/logistic regression, SVMs,
and every neural network.
""")

nb.code(r"""
# Numerically verify a derivative: f(x)=x^2 -> f'(x)=2x
def f(x): return x**2
def numerical_deriv(f, x, h=1e-6): return (f(x+h) - f(x-h)) / (2*h)
for x in [1, 3, -2]:
    print(f"f'({x}): numerical={numerical_deriv(f, x):.4f}, exact(2x)={2*x}")
""")

nb.md(r"""
### Gradient descent from scratch — fit a line by *learning*

Instead of the normal equation (Module 06), we'll **learn** the slope & intercept
by repeatedly stepping downhill on the mean-squared-error surface. This is the same
loop as neural-net training (Module 11), stripped to its essence.
""")

nb.code(r"""
# Synthetic linear data: y = 3x + 7 + noise
x = rng.uniform(-5, 5, 200)
y = 3*x + 7 + rng.normal(0, 2, 200)

# Parameters to learn
w, b = 0.0, 0.0            # start at zero
lr = 0.01
history = []
for step in range(200):
    y_hat = w*x + b
    error = y_hat - y
    # gradients of MSE = mean(error^2) w.r.t. w and b (from the chain rule)
    grad_w = 2 * np.mean(error * x)
    grad_b = 2 * np.mean(error)
    w -= lr * grad_w        # step DOWNHILL (against the gradient)
    b -= lr * grad_b
    history.append(np.mean(error**2))

print(f"learned:  w={w:.3f} (true 3), b={b:.3f} (true 7)")

plt.figure(figsize=(7,4))
plt.plot(history)
plt.xlabel("iteration"); plt.ylabel("MSE loss")
plt.title("Gradient descent: loss falls as parameters learn"); plt.show()
""")

nb.md(r"""
**The whole of ML optimization in five lines:** predict → compute error → compute
gradient → step against it → repeat. The **learning rate** `lr` controls step size:
too big overshoots/diverges, too small crawls. You've now built the engine behind
regression *and* neural nets by hand.
""")

nb.md(r"""
## 17.4 Mini-exercises

1. Add a `std=0` guard to `MyStandardScaler` (constant columns cause divide-by-zero).
2. Give `RunningStats` a `.std()` method; test it against `np.std`.
3. In gradient descent, set `lr=0.5` then `lr=0.001`. Describe the loss curve for
   each — what breaks?
4. Compute the eigenvalues of `[[2,0],[0,5]]` by hand, then verify with
   `np.linalg.eig`. What are the eigenvectors?
""")

nb.md(r"""
## Summary

- **OOP**: classes bundle **state** (attributes) + **behavior** (methods); `self`
  is the instance. sklearn's **`fit`/`transform`/`predict`** pattern lets you write
  Pipeline-compatible components. Pillars: encapsulation, inheritance,
  polymorphism, abstraction.
- **Linear algebra**: data = matrix; **dot product** = similarity/weighted sum;
  **matmul** = predict-all-at-once; **eigenvectors of covariance = PCA components**.
- **Calculus**: derivative = slope; **gradient** points uphill; **gradient descent**
  steps downhill to minimize loss — the universal learning engine. **Learning rate**
  is the key knob.

Next: **Module 18 — Time Series Analysis & Forecasting**.
""")

out = nb.save("notebooks/17_oop_linear_algebra_calculus.ipynb")
print("saved", out)
