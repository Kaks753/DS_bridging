"""Builder for Module 00: Python for Data Science + NumPy foundations."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 00 — Python for Data Science + NumPy Foundations

**Why start here?** Every model, every chart, every clean-up you will ever do
sits on top of two things: (1) core Python you can *reason about*, and (2) NumPy
arrays. If these are shaky, everything above wobbles. We fix that now.

By the end you will be able to explain, out loud, in an interview:
- Why NumPy is fast and Python loops are slow (**vectorization**).
- What **broadcasting** is and when it silently saves you.
- The difference between a **view** and a **copy** (a real bug source).
- How **axis** works (the #1 confusion for beginners).

> Teaching style: we never say "just do X". We show *why*, then the *how*,
> then a *gotcha*, then a *one-line takeaway*.
""")

nb.md(r"""
## 0.1 The mental model: Python is a *dispatcher*, NumPy is the *engine*

Plain Python is flexible but slow for number crunching, because every number is a
full Python object (with type info, reference count, etc.) and every `+` goes
through the interpreter. NumPy stores numbers in one tight block of memory of a
single type (`dtype`) and runs the loop in compiled C. That's the whole trick.
""")

nb.code(r"""
import numpy as np
import time

# Same task: square 1,000,000 numbers.
n = 1_000_000
py_list = list(range(n))

t0 = time.perf_counter()
squared_py = [x * x for x in py_list]     # pure Python loop
t1 = time.perf_counter()

arr = np.arange(n)
t2 = time.perf_counter()
squared_np = arr ** 2                       # vectorized: no Python loop
t3 = time.perf_counter()

print(f"Pure Python loop : {t1 - t0:.4f} s")
print(f"NumPy vectorized : {t3 - t2:.4f} s")
print(f"NumPy is ~{(t1 - t0) / (t3 - t2):.0f}x faster here")
""")

nb.md(r"""
**Takeaway:** *"Vectorized"* means you express the operation on the **whole array
at once** and let NumPy's C code do the loop. Rule of thumb: if you're writing a
`for` loop over a NumPy array, pause — there's usually a vectorized way.
""")

nb.md(r"""
## 0.2 Creating arrays and understanding `dtype` & `shape`

Two attributes you will check constantly:
- `.shape` — the size along each dimension, e.g. `(rows, cols)`.
- `.dtype` — the type of every element (`int64`, `float64`, `bool`, ...).
""")

nb.code(r"""
a = np.array([1, 2, 3])                       # 1-D, inferred int
b = np.array([[1.0, 2.0], [3.0, 4.0]])        # 2-D float
zeros = np.zeros((2, 3))                       # filled with 0.0
ones = np.ones(4)
rng = np.random.default_rng(0)                 # modern random generator
randoms = rng.normal(0, 1, size=(2, 2))        # standard-normal samples

for name, x in [("a", a), ("b", b), ("zeros", zeros), ("randoms", randoms)]:
    print(f"{name:8s} shape={x.shape}  dtype={x.dtype}")
""")

nb.md(r"""
**Gotcha:** integer arrays can't silently hold `NaN`. `NaN` is a *float*. So the
moment you introduce missing values, your column becomes `float64`. This explains
why pandas often shows your integer column as `float` after cleaning.
""")

nb.code(r"""
ints = np.array([1, 2, 3])
print("int array dtype:", ints.dtype)
floaty = ints.astype(float)
floaty[0] = np.nan          # only possible because it's float now
print("with NaN:", floaty, "dtype:", floaty.dtype)
""")

nb.md(r"""
## 0.3 Indexing & slicing — and the view-vs-copy trap

Slicing a NumPy array gives a **view**: a window into the *same memory*. Editing
the view edits the original. This is fast (no copying) but bites beginners.
""")

nb.code(r"""
base = np.arange(10)
window = base[2:5]      # a VIEW, not a copy
window[:] = 99          # write into the view...
print("base after editing the view:", base)   # ...base changed too!

safe = base[2:5].copy() # force an independent copy
safe[:] = -1
print("base after editing the copy:", base)    # base unchanged
""")

nb.md(r"""
**Takeaway:** if you need to modify a slice *without* touching the source, call
`.copy()`. In pandas the analog is the `SettingWithCopyWarning` — same root cause.
""")

nb.md(r"""
## 0.4 Boolean masking — the workhorse of data filtering

You will use this constantly for cleaning and EDA. A comparison on an array gives
a boolean array; indexing with it keeps only the `True` rows.
""")

nb.code(r"""
data = np.array([12, -3, 7, 0, -8, 21, 5])
mask = data > 0
print("mask :", mask)
print("positives      :", data[mask])
print("count positive :", mask.sum())          # True==1, so sum = count
print("any negative?  :", (data < 0).any())
print("all positive?  :", (data > 0).all())

# Combine conditions with & (and), | (or), ~ (not). Parentheses are REQUIRED.
print("between 0 and 10:", data[(data > 0) & (data < 10)])
""")

nb.md(r"""
**Gotcha:** use `&`, `|`, `~` (bitwise) on arrays — **not** Python's `and`, `or`,
`not`. And always wrap each condition in parentheses because `&` binds tighter
than `<`.
""")

nb.md(r"""
## 0.5 Broadcasting — how arrays of different shapes combine

Broadcasting lets NumPy stretch a smaller array across a larger one *without
copying data*, as long as their shapes are compatible (dimensions equal, or one
of them is 1). This is how you subtract a mean from every row, or scale columns.
""")

nb.code(r"""
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])

col_means = X.mean(axis=0)          # mean of each COLUMN -> shape (3,)
print("column means:", col_means)

centered = X - col_means            # (3,3) - (3,) broadcasts across rows
print("centered:\n", centered)
print("new column means (~0):", centered.mean(axis=0).round(10))
""")

nb.md(r"""
This exact pattern — subtract the mean, divide by the std — **is** what
`StandardScaler` does internally. You now understand the machinery under a tool
you already use.
""")

nb.code(r"""
col_std = X.std(axis=0)
standardized = (X - col_means) / col_std
print("standardized:\n", standardized.round(3))
print("means ~0:", standardized.mean(axis=0).round(10))
print("stds  ~1:", standardized.std(axis=0).round(10))
""")

nb.md(r"""
## 0.6 The `axis` argument — the #1 confusion, solved

`axis=0` means **"collapse the rows"** → you get one value **per column**.
`axis=1` means **"collapse the columns"** → you get one value **per row**.

Mnemonic: `axis` is the dimension that **disappears**.
""")

nb.code(r"""
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print("M shape:", M.shape)                 # (2, 3)
print("sum axis=0 (per column):", M.sum(axis=0))   # -> shape (3,)
print("sum axis=1 (per row)   :", M.sum(axis=1))   # -> shape (2,)
print("sum no axis (grand)    :", M.sum())
""")

nb.md(r"""
## 0.7 Reshaping & the meaning of `-1`

`reshape` reinterprets the same data with new dimensions (total size must match).
`-1` says "figure this dimension out for me".
""")

nb.code(r"""
v = np.arange(12)
print("original:", v.shape)
print(v.reshape(3, 4))
print("as column vector:\n", v.reshape(-1, 1)[:4], "...")   # -1 => 12 here
""")

nb.md(r"""
`reshape(-1, 1)` (turn a 1-D array into a single column) is something scikit-learn
constantly demands, because it expects a 2-D feature matrix `X` of shape
`(n_samples, n_features)`. Now you know *why* that error appears.
""")

nb.md(r"""
## 0.8 A few Python idioms every data scientist must own

These show up in real code and interviews.
""")

nb.code(r"""
# List comprehension (build a list declaratively)
squares = [x**2 for x in range(6)]
print("squares:", squares)

# Dict comprehension (great for mapping/renaming)
mapping = {c: i for i, c in enumerate(["low", "mid", "high"])}
print("ordinal map:", mapping)

# enumerate + zip (loop with index / loop over pairs)
names = ["Ann", "Ben", "Cid"]
scores = [90, 75, 88]
for i, (name, score) in enumerate(zip(names, scores)):
    print(f"{i}: {name} -> {score}")

# f-strings with formatting (clean reporting)
acc = 0.8072
print(f"accuracy = {acc:.2%}")   # -> 80.72%
""")

nb.md(r"""
## 0.9 Mini-exercises (do these before moving on)

1. Create a `5x5` array of random integers 0–9 with a fixed seed. Print its
   column means and row maxima.
2. From `np.arange(20)`, extract all values divisible by 3 **and** greater than 5
   using a single boolean mask.
3. Given `A` shape `(4, 3)`, subtract the *row* means from each row using
   broadcasting. (Hint: you'll need `reshape(-1, 1)`.)
4. Explain in one sentence why editing a slice can change the original array.
""")

nb.code(r"""
# Scratch space for the exercises — try before peeking at any solution.
rng = np.random.default_rng(7)
grid = rng.integers(0, 10, size=(5, 5))
print(grid)
print("col means:", grid.mean(axis=0))
print("row maxima:", grid.max(axis=1))
""")

nb.md(r"""
## Summary — what you can now claim to *know*

- NumPy is fast because of **contiguous typed memory + compiled vectorized loops**.
- **Broadcasting** aligns shapes so you operate on whole arrays without copying.
- Slices are **views**; use `.copy()` to detach.
- **Boolean masks** filter data; combine with `&`, `|`, `~` and parentheses.
- `axis` = the dimension that collapses (`0` = down columns, `1` = across rows).
- scikit-learn wants `X` as 2-D `(n_samples, n_features)` — hence `reshape(-1, 1)`.

Next: **Module 01 — Pandas**, where these ideas power real tabular data work.
""")

out = nb.save("notebooks/00_python_numpy_foundations.ipynb")
print("saved", out)
