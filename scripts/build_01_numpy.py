"""Builder for Module 1: NumPy — fast maths on lots of numbers (4-layer rewrite)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 1 — NumPy: Fast Maths on Lots of Numbers

In **Module 0** you learned plain Python: variables, lists, loops, functions. That's
perfect for a *few* values. But data science means *millions* of numbers, and plain
Python lists are too slow and clumsy for that. Enter **NumPy** — the library that
makes number-crunching fast and easy. Almost every other tool (pandas, scikit-learn,
matplotlib) is built on top of it, so this is bedrock.

By the end you'll understand, and be able to explain simply:
- what a NumPy **array** is and why it beats a plain list for numbers
- **vectorization** — doing maths on a whole column at once (and why it's fast)
- **boolean masking** — filtering data with a yes/no test
- **broadcasting** — combining differently-shaped arrays
- the **axis** argument (the classic beginner confusion — we'll make it obvious)

> Same teaching rhythm as Module 0: 🌱 plain English → 🔤 code line-by-line →
> 🎓 go deeper → ✅ takeaway.
""")

# ---------------------------------------------------------------------------
nb.md("## 1.1 What is an array, and why not just a list?")

nb.plain(r"""
A **NumPy array** is like a Python **list**, but with two superpowers:
1. Every item is the **same type** (all numbers), packed tightly in memory.
2. You can do maths on the **whole thing at once** — `prices * 1.16` adds 16% VAT to
   *every* price in one line, no loop needed.

Analogy: a Python list is a **mixed drawer** (socks, keys, a phone) — flexible but
slow to search. A NumPy array is a **loaded ammo clip** — all identical rounds,
lined up, ready to fire together. That uniformity is what makes it fast.

We import NumPy with its universal nickname `np` (everyone writes `np`).
""")

nb.code(r"""
import numpy as np                       # load NumPy, nickname it 'np'

prices_list = [100, 200, 300]            # a plain Python list
prices = np.array([100, 200, 300])       # the same numbers as a NumPy array

print("list :", prices_list)
print("array:", prices)

# The magic: maths on the WHOLE array at once (no loop!)
print("with 16% VAT:", prices * 1.16)
print("all doubled  :", prices * 2)
""")

nb.readcode(r"""
- `import numpy as np` → load the toolbox, call it `np`.
- `np.array([100, 200, 300])` → build an array from a list of numbers.
- `prices * 1.16` → multiply **every** element by 1.16 in one go → the VAT-inclusive
  prices. With a plain list, `[100,200,300] * 1.16` would **error** — lists don't do
  maths like that. This "do it to everything at once" is the whole point of NumPy.
""")

nb.jargon("array", "NumPy's list-like container where every item is the same type, allowing whole-array maths")
nb.jargon("np", "the standard nickname for NumPy (from `import numpy as np`)")
nb.takeaway("A NumPy array holds same-typed numbers and lets you do maths on all of them at once.")
nb.try_this("Make an array `[10, 20, 30]` and print it with 10% added to each value.")

# ---------------------------------------------------------------------------
nb.md("## 1.2 Why arrays are FAST — 'vectorization'")

nb.plain(r"""
"**Vectorization**" is a fancy word for a simple idea: **do the operation on the
whole array at once**, instead of looping item by item. NumPy then runs that loop
in fast, pre-compiled code under the hood (written in the C language) rather than in
slower Python.

Analogy: paying 1,000 people one-by-one at a counter (a Python loop) vs sending one
bank instruction that pays all 1,000 at once (vectorized). Same result — wildly
different speed. Let's *measure* it.
""")

nb.code(r"""
import time
n = 1_000_000                                  # one million numbers

py_list = list(range(n))
t0 = time.perf_counter()
squared_py = [x * x for x in py_list]          # plain Python: loop, square each
t1 = time.perf_counter()

arr = np.arange(n)                             # np.arange = array version of range
t2 = time.perf_counter()
squared_np = arr ** 2                          # vectorized: square the WHOLE array
t3 = time.perf_counter()

print(f"Python loop     : {t1 - t0:.4f} seconds")
print(f"NumPy vectorized: {t3 - t2:.4f} seconds")
print(f"NumPy was about {(t1 - t0) / (t3 - t2):.0f}x faster")
""")

nb.readcode(r"""
- `np.arange(n)` → like `range(n)` from Module 0, but produces an **array** of
  `0..n-1` instead of a plain range.
- `[x * x for x in py_list]` → the Python way: a loop that squares each number
  (this is a *list comprehension* — a compact loop).
- `arr ** 2` → the NumPy way: `**` means "to the power of", so this squares **every**
  element at once. `** 2` = squared.
- `time.perf_counter()` → a stopwatch; we subtract start from end to time each way.
""")

nb.deeper(r"""
Why is plain Python slow here? In a Python list, each number is a full Python
*object* carrying extra baggage (its type, a reference count, etc.), and every `*`
goes through the interpreter one value at a time. NumPy instead stores the numbers
as a single tight block of one type (its **dtype**) and runs the multiply loop in
compiled C. Rule of thumb you'll live by: **if you're writing a `for` loop over a
NumPy array or a pandas column, pause — there's almost always a faster vectorized
way.**
""")

nb.jargon("vectorization", "doing an operation on a whole array at once instead of looping element by element")
nb.jargon("dtype", "the single data type of every element in an array (e.g. int64, float64)")
nb.takeaway("Vectorized whole-array maths is both shorter to write AND far faster than a Python loop.")

# ---------------------------------------------------------------------------
nb.md("## 1.3 Making arrays; checking `shape` and `dtype`")

nb.plain(r"""
Arrays can be 1-D (a single row of numbers), 2-D (a grid — rows and columns, like a
spreadsheet), or more. Two things you'll check *constantly*:
- **`.shape`** — how big it is along each dimension, e.g. `(3, 2)` = 3 rows, 2 columns.
- **`.dtype`** — the type of every element (like `int64` or `float64`).

Knowing an array's shape is like knowing a room's dimensions before buying a carpet
— it prevents almost every "sizes don't match" error later.
""")

nb.code(r"""
a = np.array([1, 2, 3])                    # 1-D: just a row of 3 numbers
b = np.array([[1.0, 2.0],                  # 2-D: 2 rows, 2 columns (a grid)
              [3.0, 4.0]])
zeros = np.zeros((2, 3))                   # a 2x3 grid filled with 0.0
ones  = np.ones(4)                         # four 1.0s

rng = np.random.default_rng(0)             # a random-number generator (seed 0 = repeatable)
randoms = rng.normal(0, 1, size=(2, 2))    # 2x2 of random 'normal' numbers

for name, x in [("a", a), ("b", b), ("zeros", zeros), ("randoms", randoms)]:
    print(f"{name:8s} shape={x.shape}  dtype={x.dtype}")
""")

nb.readcode(r"""
- `np.array([[...],[...]])` → nested lists make a 2-D array (a grid).
- `np.zeros((2, 3))` → a grid of zeros; the `(2, 3)` is the shape you want.
- `np.random.default_rng(0)` → make a random generator. The `0` is a **seed**: it
  makes the "random" numbers the *same every run*, so your work is reproducible.
- `rng.normal(0, 1, size=(2,2))` → 4 random numbers from a bell curve centred at 0.
- `x.shape` / `x.dtype` → report the size and element type of each array.
""")

nb.warn("Integer arrays can't hold `NaN` (the 'missing value' marker), because `NaN` "
        "is technically a *float*. So the moment a column gets a missing value it "
        "quietly turns into `float`. That's why a whole-number column often shows as "
        "`27.0` after cleaning — not a bug.")
nb.jargon("shape", "the size of an array along each dimension, e.g. (rows, columns)")
nb.jargon("seed", "a fixed starting number that makes 'random' results repeatable")
nb.takeaway("Always know an array's `.shape` and `.dtype` — they prevent most size/type bugs.")

# ---------------------------------------------------------------------------
nb.md("## 1.4 Indexing & slicing — and the 'view vs copy' trap")

nb.plain(r"""
Getting items out of an array works like lists from Module 0: positions start at
**0**, and `start:stop` grabs a slice (stop is *not* included).

But here's a surprise beginners get bitten by: slicing a NumPy array gives you a
**view** — a *window* onto the *same* memory, not a fresh copy. So if you change the
window, you change the original too! To get an independent copy, call `.copy()`.

Analogy: a view is a **live security-camera feed** of a room — draw on the feed and
you're really drawing in the room. `.copy()` is a **photograph** — scribble on it and
the room is untouched.
""")

nb.code(r"""
base = np.arange(10)              # array 0,1,2,...,9
print("base       :", base)
print("base[0]    :", base[0])    # first item (index 0)
print("base[2:5]  :", base[2:5])  # positions 2,3,4  (5 is NOT included)

window = base[2:5]                # this is a VIEW into base
window[:] = 99                    # overwrite the window...
print("base after editing view:", base)   # ...and base changed too!

safe = base[2:5].copy()           # a real, independent COPY
safe[:] = -1
print("base after editing copy:", base)    # base is untouched this time
""")

nb.readcode(r"""
- `base[0]` → first element. `base[2:5]` → elements at 2, 3, 4 (the `stop` index 5
  is excluded — a Python rule worth memorising).
- `window = base[2:5]` → `window` is a **view**, sharing memory with `base`.
- `window[:] = 99` → `[:]` means "all elements of window"; setting it to 99 writes
  *through* the view into `base`, so `base` changes too.
- `.copy()` → makes a detached copy; editing `safe` leaves `base` alone.
""")

nb.warn("If you slice an array (or a pandas column later) and then modify it, and the "
        "original changes unexpectedly — this view/copy behaviour is why. Add "
        "`.copy()` when you want to be safe. In pandas the same issue shows up as the "
        "famous `SettingWithCopyWarning`.")
nb.jargon("view", "a slice that shares memory with the original array (editing it edits the original)")
nb.jargon("copy", "an independent duplicate; editing it does NOT affect the original")
nb.takeaway("Slices are live views into the same memory; use `.copy()` when you need an independent one.")

# ---------------------------------------------------------------------------
nb.md("## 1.5 Boolean masking — filtering with a yes/no test")

nb.plain(r"""
This is the tool you'll use *most* for cleaning and exploring data. The idea:
1. Write a **condition** on the array (like `data > 0`). NumPy checks it for every
   element and returns an array of `True`/`False` — a **mask**.
2. Put that mask in the square brackets to **keep only the `True` items**.

Analogy: shine a filter over your data — everything that passes the test stays,
everything else is hidden. "Give me only the positive numbers" becomes one line.
""")

nb.code(r"""
data = np.array([12, -3, 7, 0, -8, 21, 5])

mask = data > 0                       # a True/False for each element
print("the mask      :", mask)
print("positives only:", data[mask])  # keep only where mask is True
print("count positive:", mask.sum())  # True counts as 1, so sum = how many True
print("any negative? :", (data < 0).any())   # is at least one negative?
print("all positive? :", (data > 0).all())   # are they ALL positive?

# Combine conditions with & (and), | (or), ~ (not) — wrap each in ( ):
print("between 0 and 10:", data[(data > 0) & (data < 10)])
""")

nb.readcode(r"""
- `data > 0` → compare every element to 0 → an array like `[True, False, True, ...]`.
- `data[mask]` → indexing with the mask keeps only the `True` positions.
- `mask.sum()` → since `True==1` and `False==0`, summing a mask **counts** the Trues.
- `.any()` / `.all()` → "is any True?" / "are all True?".
- `(data > 0) & (data < 10)` → two tests joined with `&` ("and"); keeps numbers that
  are *both* positive *and* under 10.
""")

nb.warn("On arrays use `&`, `|`, `~` — NOT the words `and`, `or`, `not` (those are "
        "for single True/False values, and error on arrays). Also wrap each condition "
        "in parentheses: `(a) & (b)`, because `&` otherwise binds too tightly.")
nb.jargon("boolean mask", "an array of True/False used to filter another array")
nb.takeaway("Make a True/False mask with a condition, then `array[mask]` keeps the matching items.")
nb.try_this("From `np.arange(20)`, keep only values that are even (hint: `x % 2 == 0`).")

# ---------------------------------------------------------------------------
nb.md("## 1.6 Broadcasting — combining different-shaped arrays")

nb.plain(r"""
**Broadcasting** is NumPy automatically "stretching" a smaller array so it lines up
with a bigger one — so you can, say, subtract one number from every element, or
subtract a row of column-averages from every row of a grid, without writing loops.

Analogy: a single **rubber stamp** (the small array) pressed across a whole sheet of
paper (the big array). NumPy repeats the small one to fit, *conceptually*, without
actually copying the data (so it stays fast).
""")

nb.code(r"""
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])

col_means = X.mean(axis=0)          # average of each column -> [4, 5, 6]
print("column means:", col_means)

centered = X - col_means            # subtract that row of means from EVERY row
print("centered:\n", centered)
print("new column means (~0):", centered.mean(axis=0).round(10))
""")

nb.readcode(r"""
- `X.mean(axis=0)` → the mean **down each column** (more on `axis` in 1.7) → shape
  `(3,)`, one number per column.
- `X - col_means` → `X` is 3x3, `col_means` is length 3. Broadcasting lines the
  length-3 row up against **each** of the 3 rows and subtracts — "centre" every
  column on 0.
- The new column means are essentially 0, confirming each column was recentred.
""")

nb.deeper(r"""
This exact pattern — subtract the mean, then divide by the standard deviation — **is
what `StandardScaler` does** in scikit-learn (you'll meet it in the scaling module).
So you're not learning trivia; you're seeing the machinery inside a tool you'll use
for real models.

Broadcasting rule (for later): two dimensions are compatible when they're **equal**,
or **one of them is 1**. NumPy stretches the size-1 dimension to match.
""")

nb.code(r"""
col_std = X.std(axis=0)                     # spread of each column
standardized = (X - col_means) / col_std     # centre, then scale — 'standardizing'
print("standardized:\n", standardized.round(3))
print("means ~0:", standardized.mean(axis=0).round(10))
print("stds  ~1:", standardized.std(axis=0).round(10))
""")

nb.jargon("broadcasting", "NumPy auto-stretching a smaller array to combine with a larger one")
nb.takeaway("Broadcasting lets you combine different-shaped arrays (e.g. subtract column means from every row) with no loops.")

# ---------------------------------------------------------------------------
nb.md("## 1.7 The `axis` argument — the #1 confusion, made simple")

nb.plain(r"""
When you sum or average a 2-D array (a grid), you must say **which direction** to
squash. That direction is the **`axis`**:
- `axis=0` → squash **down the rows** → you get **one value per column**.
- `axis=1` → squash **across the columns** → you get **one value per row**.

The simplest memory trick: **`axis` is the dimension that disappears.** If your grid
is `(2 rows, 3 cols)` and you sum with `axis=0`, the 2 rows collapse away and you're
left with 3 numbers (one per column).
""")

nb.code(r"""
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print("M shape:", M.shape)                       # (2, 3)  -> 2 rows, 3 cols
print("sum axis=0 (down rows -> per column):", M.sum(axis=0))  # [5, 7, 9]
print("sum axis=1 (across cols -> per row) :", M.sum(axis=1))  # [6, 15]
print("sum with no axis (everything)       :", M.sum())         # 21
""")

nb.readcode(r"""
- `M.sum(axis=0)` → collapse the 2 rows → 3 column-totals `[1+4, 2+5, 3+6] = [5,7,9]`.
- `M.sum(axis=1)` → collapse the 3 columns → 2 row-totals `[1+2+3, 4+5+6] = [6,15]`.
- `M.sum()` → no axis given → add up everything → `21`.
Say it out loud: "axis=0 goes down the columns, axis=1 goes along the rows."
""")

nb.jargon("axis", "the direction to collapse in a 2-D array: 0 = down rows (per column), 1 = across columns (per row)")
nb.takeaway("`axis=0` = per-column (rows collapse); `axis=1` = per-row (columns collapse). Axis = the dimension that vanishes.")

# ---------------------------------------------------------------------------
nb.md("## 1.8 Reshaping, and what `-1` means")

nb.plain(r"""
**Reshaping** re-arranges the same numbers into a new grid shape (the total count
must stay the same — you can't fit 12 numbers into a 3x5 grid). You'll do this often
because scikit-learn wants your features as a 2-D grid.

The special value **`-1`** means *"you (NumPy) work out this dimension for me"* — handy
when you know you want, say, one column, but don't want to count the rows by hand.
""")

nb.code(r"""
v = np.arange(12)                    # 0..11  (12 numbers)
print("original shape:", v.shape)    # (12,)

print("as 3x4:\n", v.reshape(3, 4))  # same 12 numbers, arranged 3 rows x 4 cols

col = v.reshape(-1, 1)               # -1 => "figure out the rows"; 1 => one column
print("as a single column, first 4 rows:\n", col[:4], "...")
""")

nb.readcode(r"""
- `v.reshape(3, 4)` → lay the 12 numbers out as 3 rows of 4 (3x4 = 12 ✓).
- `v.reshape(-1, 1)` → "one column, and NumPy figures out it needs 12 rows". Result
  shape `(12, 1)` — a tall single column.
""")

nb.deeper(r"""
`reshape(-1, 1)` shows up all the time with scikit-learn, because models expect the
feature matrix `X` to be **2-D**: shape `(n_samples, n_features)` = (rows of data,
columns of features). If you pass a 1-D array you'll get an error like *"Expected 2D
array, got 1D array instead"* — the fix is exactly `.reshape(-1, 1)`. Now that error
will never confuse you.
""")

nb.jargon("reshape", "rearrange the same values into a new shape (total count must match)")
nb.takeaway("`reshape` re-lays-out the same numbers; `-1` tells NumPy to compute that dimension; `reshape(-1,1)` makes a single column scikit-learn likes.")

# ---------------------------------------------------------------------------
nb.md("## 1.9 A few Python-with-NumPy idioms worth knowing")

nb.plain(r"""
These compact patterns appear in real code and interviews. They're just tidy ways to
build lists, label things, and format numbers nicely. Skim, then reuse.
""")

nb.code(r"""
# List comprehension — build a list in one readable line (a compact for-loop):
squares = [x**2 for x in range(6)]
print("squares:", squares)

# enumerate — loop with the position number handy:
names = ["Ann", "Ben", "Cid"]
for i, name in enumerate(names):
    print(i, name)

# zip — loop over two lists side by side (pairs):
scores = [90, 75, 88]
for name, score in zip(names, scores):
    print(name, "scored", score)

# f-string formatting — clean reports (:.2% shows a percentage with 2 decimals):
acc = 0.8072
print(f"accuracy = {acc:.2%}")     # -> accuracy = 80.72%
""")

nb.readcode(r"""
- `[x**2 for x in range(6)]` → a **list comprehension**: build `[0,1,4,9,16,25]` in
  one line instead of a multi-line loop.
- `enumerate(names)` → gives `(0,'Ann'), (1,'Ben'), ...` so you get the index too.
- `zip(names, scores)` → pairs them up: `('Ann',90), ('Ben',75), ...`.
- `f"...{acc:.2%}"` → an **f-string**: drop a variable straight into text; `:.2%`
  formats it as a percentage with 2 decimals.
""")

nb.takeaway("List comprehensions, `enumerate`, `zip`, and f-strings make everyday data code short and readable.")

# ---------------------------------------------------------------------------
nb.md(r"""
## 1.10 Practice (do these before Module 2)

1. Make a `5x5` array of random integers 0–9 with seed 7. Print its **column
   means** and its **row maxima**.
2. From `np.arange(20)`, keep values that are divisible by 3 **and** greater than 5,
   using one boolean mask.
3. Given a `(4, 3)` array, subtract each **row's** mean from that row (hint: you'll
   need `reshape(-1, 1)` so the row-means broadcast correctly).
4. In one sentence, explain to a friend why editing a slice can change the original.
""")

nb.code(r"""
# Scratch space — try the exercises here before peeking anywhere.
rng = np.random.default_rng(7)
grid = rng.integers(0, 10, size=(5, 5))
print(grid)
print("col means :", grid.mean(axis=0).round(2))
print("row maxima:", grid.max(axis=1))
""")

nb.interview(r"""
"NumPy is fast because it stores one type in tight memory and runs vectorized C
loops, so I avoid Python loops over data. I'm fluent with boolean masking for
filtering, broadcasting for shape-aligned maths, and I keep `axis` straight —
axis 0 collapses rows to give per-column results."
""")

nb.md(r"""
## Summary — what you can now do

- Build **arrays** and check their **`shape`** and **`dtype`**.
- Use **vectorization** (whole-array maths) instead of slow loops.
- **Slice** arrays, and remember slices are **views** (use `.copy()` to detach).
- **Filter** with boolean **masks** (`&`, `|`, `~`, with parentheses).
- **Broadcast** to combine different shapes (the guts of standardizing data).
- Keep **`axis`** straight (0 = per column, 1 = per row).
- **Reshape** data, including `reshape(-1, 1)` for scikit-learn.

Next: **Module 2 — Pandas**, where these ideas power real spreadsheet-style data.
""")

out = nb.save("notebooks/01_numpy.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
