"""Builder for Module 16: Probability, Combinatorics, Bayes & Naive Bayes (4-layer)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 16 — Probability, Combinatorics & Bayes

Probability is the **language of uncertainty** — the foundation under every
statistical test and every ML model. As a Math grad this is home turf; here we make
it **computational and intuitive**, then build up to **Bayes' theorem** and the
**Naive Bayes classifier** that powers spam filters and sentiment apps.

**What you'll be able to do by the end:**
- Count arrangements the right way (permutations vs combinations).
- Apply the probability rules without mixing up *exclusive* and *independent*.
- Update a belief with evidence using **Bayes' theorem** (and explain the famous
  medical-test paradox out loud).
- Build a **Naive Bayes** spam classifier — first by hand, then with scikit-learn.
""")

nb.plain(r"""
Probability is just a number between 0 and 1 that says *how sure* we are something
happens. 0 = "no chance", 1 = "certain", 0.5 = "coin flip".

Everything else in this module is bookkeeping: **counting** how many ways things can
happen, and **updating** our number when new information arrives. That's it. If you
can keep a tally and do a little arithmetic, you can do probability.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import factorial, comb, perm
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
print("Tools ready.")
""")

# ---------------------------------------------------------------------------
# 16.1 Counting
# ---------------------------------------------------------------------------
nb.md(r"""
## 16.1 Counting — permutations vs combinations

Before we can compute a probability, we often need to count *how many outcomes are
possible*. Two words trip everyone up:
""")

nb.analogy(r"""
Think of a race with 8 runners.
- **Permutation** = handing out **gold, silver, bronze**. Order matters — 1st place
  is different from 3rd place.
- **Combination** = picking **3 runners for a random drug test**. Order doesn't
  matter — the same 3 people is the same group no matter who you name first.
""")

nb.jargon("Permutation", "an arrangement where ORDER matters (gold/silver/bronze)")
nb.jargon("Combination", "a selection where order does NOT matter (a committee of 3)")

nb.md(r"""
The formulas:

$$ \text{Permutations: } P(n,r) = \frac{n!}{(n-r)!} \qquad
   \text{Combinations: } C(n,r) = \binom{n}{r} = \frac{n!}{r!\,(n-r)!} $$

The only difference is the extra `r!` on the bottom for combinations — it *divides
out* the orderings we don't want to count.
""")

nb.code(r"""
print("Ways to rank top-3 of 8 runners (order matters):", perm(8, 3))
print("Ways to pick a 3-person committee from 8 (order doesn't):", comb(8, 3))
print("Distinct 5-card poker hands from 52:", comb(52, 5))

# Verify by brute force on a small case (proof, not faith)
import itertools
items = ["A", "B", "C", "D"]
print("\nperm(4,2) =", perm(4,2), "==", len(list(itertools.permutations(items, 2))))
print("comb(4,2) =", comb(4,2), "==", len(list(itertools.combinations(items, 2))))
""")

nb.readcode(r"""
- `perm(8, 3)` = 336: gold/silver/bronze from 8 runners.
- `comb(8, 3)` = 56: a 3-person group from 8. Fewer, because we stopped
  double-counting the same group in different orders.
- `comb(52, 5)` = 2,598,960: every possible poker hand.
- The `itertools` lines *literally list out* every arrangement/selection for 4 items
  and count them — the counts match the formulas exactly. That's our proof.
""")

# WORKED EXAMPLE — by hand
nb.md(r"""
### ✏️ Worked example (by hand): a 3-person committee from 8

Let's grind `C(8,3)` on paper so the formula stops being magic.

$$ C(8,3) = \frac{8!}{3!\,(8-3)!} = \frac{8!}{3!\,5!} $$

Expand only what we need (cancel the `5!`):

$$ = \frac{8 \times 7 \times 6 \times \cancel{5!}}{(3 \times 2 \times 1)\times \cancel{5!}}
   = \frac{8 \times 7 \times 6}{6} = \frac{336}{6} = 56 $$

So there are **56** possible committees. Notice the top (`8×7×6 = 336`) is exactly the
*permutation* count — and dividing by `3! = 6` removes the orderings. That's the whole
idea in one line.
""")

nb.code(r"""
# Confirm the by-hand arithmetic step by step
top = 8 * 7 * 6            # this is perm(8,3)
bottom = 3 * 2 * 1         # this is 3!
print("top (8x7x6) =", top, " == perm(8,3)?", top == perm(8, 3))
print("bottom (3!) =", bottom)
print("C(8,3) = top / bottom =", top // bottom, " == comb(8,3)?", top // bottom == comb(8, 3))
""")

nb.takeaway("Order matters -> permutation; order doesn't -> combination (divide by r! to kill the orderings).")
nb.try_this("Compute C(52,5) by hand as (52x51x50x49x48)/5! and check it equals 2,598,960.")

# ---------------------------------------------------------------------------
# 16.2 Probability rules
# ---------------------------------------------------------------------------
nb.md(r"""
## 16.2 The probability rules

Four rules cover almost everything. Don't memorize symbols — learn the *stories*.
""")

nb.plain(r"""
- **Complement**: the chance something *doesn't* happen = 1 minus the chance it does.
  ("If there's a 30% chance of rain, there's a 70% chance of no rain.")
- **Addition** (the word *OR*): chance that A **or** B happens. Add them — but if they
  can overlap, *subtract the overlap once* so you don't count it twice.
- **Multiplication** (the word *AND*): chance that A **and** B both happen. Multiply.
- **Conditional** (the word *GIVEN*): chance of A *given* B already happened. It
  narrows the world down to only the cases where B is true.
""")

nb.md(r"""
The formal versions:

- Complement: $P(\text{not }A) = 1 - P(A)$
- Addition: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- Multiplication: $P(A \cap B) = P(A)\,P(B\mid A)$; if **independent**, $= P(A)\,P(B)$
- Conditional: $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$
""")

nb.warn(r"""
**Mutually exclusive is NOT the same as independent** — the #1 interview trap here.
- *Mutually exclusive* = they **can't both happen** (rolling a 2 and a 5 on one die).
  So P(both) = 0.
- *Independent* = one **doesn't affect** the other (two separate coin flips).
If two events are mutually exclusive and both possible, they are actually
*dependent* — knowing one happened tells you the other definitely didn't.
""")

nb.code(r"""
# Simulate a fair die to confirm P(even OR >4) via the addition rule
rolls = rng.integers(1, 7, size=200_000)
p_even = np.mean(rolls % 2 == 0)
p_gt4  = np.mean(rolls > 4)
p_both = np.mean((rolls % 2 == 0) & (rolls > 4))     # only {6}
p_union_sim = np.mean((rolls % 2 == 0) | (rolls > 4))
p_union_formula = p_even + p_gt4 - p_both
print(f"P(even)={p_even:.3f}  P(>4)={p_gt4:.3f}  P(both=6)={p_both:.3f}")
print(f"P(even OR >4): simulated={p_union_sim:.3f}, formula={p_union_formula:.3f}")
""")

nb.readcode(r"""
- We roll a die 200,000 times and just count.
- `p_even` ~ 0.5 (three of six faces), `p_gt4` ~ 0.333 (faces 5,6).
- `p_both` ~ 0.167 = only the **6** is both even AND >4.
- The addition rule says P(even OR >4) = 0.5 + 0.333 - 0.167 = 0.667. The simulation
  agrees. We subtracted the 6 once because adding the two groups counted it twice.
""")

# WORKED EXAMPLE — complement rule by hand
nb.md(r"""
### ✏️ Worked example (by hand): P(at least one 6 in 4 rolls)

"At least one" is the classic flag to **use the complement** — it's far easier to
count the *one* way it fails (no sixes at all) than the many ways it succeeds.

Chance a single roll is **not** a six: $\frac{5}{6}$.
Four independent rolls all miss (multiplication rule for independent events):

$$ P(\text{no six in 4}) = \left(\frac{5}{6}\right)^4 = \frac{625}{1296} \approx 0.482 $$

Therefore:

$$ P(\text{at least one six}) = 1 - 0.482 = 0.518 \approx 51.8\% $$

Slightly better than a coin flip — a famous gambling result (the *Chevalier de Méré*
problem that helped launch probability theory).
""")

nb.code(r"""
p_no_six_one = 5/6
p_no_six_four = p_no_six_one ** 4
p_at_least_one = 1 - p_no_six_four
print(f"P(no six in 4) = (5/6)^4 = {p_no_six_four:.4f}")
print(f"P(at least one six) = {p_at_least_one:.4f}")

# Verify by simulation
sims = rng.integers(1, 7, size=(1_000_000, 4))
got_a_six = (sims == 6).any(axis=1).mean()
print(f"simulated P(at least one six) = {got_a_six:.4f}")
""")

nb.takeaway("For 'at least one', flip to the complement: 1 - P(none). Multiply independent misses.")

# ---------------------------------------------------------------------------
# 16.3 Bayes
# ---------------------------------------------------------------------------
nb.md(r"""
## 16.3 Conditional probability & Bayes' theorem

**Conditional probability** narrows the world: $P(A\mid B)$ is "among only the cases
where B is true, how often is A also true?"

**Bayes' theorem** lets us *flip* a conditional — go from $P(E\mid H)$ (which we can
measure) to $P(H\mid E)$ (which we actually want):

$$ P(H \mid E) = \frac{P(E \mid H)\, P(H)}{P(E)} $$
""")

nb.plain(r"""
Bayes is just **updating a hunch with evidence**, the way a good detective works.
- **Prior** `P(H)` = how much you believed the theory *before* seeing the clue.
- **Likelihood** `P(E|H)` = how well the theory *explains* the clue.
- **Posterior** `P(H|E)` = your updated belief *after* the clue.
You start with a guess, see data, and revise. Do it again with the next clue. That's
literally how learning works.
""")

nb.jargon("Prior", "your belief BEFORE seeing the evidence")
nb.jargon("Likelihood", "how well a hypothesis explains the observed evidence")
nb.jargon("Posterior", "your UPDATED belief after combining prior and evidence")

nb.md(r"""
### The medical-test paradox (why base rates matter)

A disease affects **1%** of people. A test is **99% sensitive** ($P(+\mid\text{sick})=0.99$)
and **95% specific** ($P(-\mid\text{healthy})=0.95$, so a 5% false-positive rate). You
test **positive**. What's the chance you actually have the disease? Most people
blurt "99%". The real answer is shockingly lower — because the disease is *rare*.
""")

nb.md(r"""
### ✏️ Worked example (by hand): the positive test

Imagine **10,000 people** (round numbers make the paradox obvious):

| Group | Count | Test positive | Positives |
|---|---|---|---|
| **Sick** (1%) | 100 | 99% of them | ~**99** |
| **Healthy** (99%) | 9,900 | 5% of them | ~**495** |
| **Total positives** | | | **594** |

Now: of the 594 people who test positive, only **99** are truly sick.

$$ P(\text{sick} \mid +) = \frac{99}{594} \approx 0.167 = 16.7\% $$

Even after a positive result from a "99% accurate" test, there's only about a **1-in-6**
chance you're actually sick — because the 495 *false positives* from the huge healthy
group swamp the 99 true positives. **Base rates dominate.**
""")

nb.code(r"""
p_disease = 0.01
p_pos_given_disease = 0.99        # sensitivity
p_pos_given_healthy = 0.05        # 1 - specificity

# Bayes, matching the table above
num = p_pos_given_disease * p_disease
den = num + p_pos_given_healthy * (1 - p_disease)
posterior = num / den
print(f"P(disease | positive) = {posterior:.3f}  (~{posterior*100:.0f}%)")

# The table version, spelled out on 10,000 people
sick_pos    = 10000 * p_disease * p_pos_given_disease
healthy_pos = 10000 * (1 - p_disease) * p_pos_given_healthy
print(f"sick & positive   ~ {sick_pos:.0f}")
print(f"healthy & positive ~ {healthy_pos:.0f}")
print(f"P(sick | +) = {sick_pos:.0f}/{sick_pos+healthy_pos:.0f} = {sick_pos/(sick_pos+healthy_pos):.3f}")

# Confirm by simulating 1,000,000 people
N = 1_000_000
has = rng.random(N) < p_disease
tests_pos = np.where(has,
                     rng.random(N) < p_pos_given_disease,
                     rng.random(N) < p_pos_given_healthy)
sim = has[tests_pos].mean()
print(f"\nsimulated P(disease | positive) = {sim:.3f}")
""")

nb.interview(r"""
"A 99%-accurate test on a 1%-prevalence disease is still wrong most of the time it
fires, because false positives from the large healthy majority outnumber true
positives. That's base-rate neglect — and it's exactly why we track precision and
recall instead of raw accuracy on imbalanced problems."
""")

nb.takeaway("Bayes turns a prior into a posterior using the likelihood; with rare events, base rates rule.")

# ---------------------------------------------------------------------------
# 16.4 Naive Bayes
# ---------------------------------------------------------------------------
nb.md(r"""
## 16.4 Naive Bayes — Bayes turned into a classifier

To classify, we want $P(\text{class}\mid \text{features})$. Bayes gives:

$$ P(y \mid x_1,\dots,x_n) \;\propto\; P(y)\prod_i P(x_i \mid y) $$
""")

nb.plain(r"""
"Naive Bayes" sounds intimidating but it's just Bayes with one shortcut: pretend the
features don't interact. For spam detection, that means we treat each word as its own
independent little clue, multiply all the clues together, and see whether "spam" or
"ham" wins. The assumption is technically wrong (words *do* relate), yet it works
astonishingly well for text — and it's fast and needs very little data.
""")

nb.jargon("Naive Bayes", "a classifier that multiplies per-feature probabilities, pretending features are independent given the class")
nb.jargon("Laplace smoothing", "adding 1 to every count so an unseen word can't zero out the whole prediction")

nb.code(r"""
# Naive Bayes FROM SCRATCH on a tiny spam problem (word-presence features)
docs = [
    ("win money now", 1), ("cheap money offer", 1), ("win free prize", 1),
    ("meeting at noon", 0), ("project deadline today", 0), ("lunch meeting notes", 0),
]  # 1 = spam, 0 = ham
vocab = sorted(set(w for text, _ in docs for w in text.split()))

def word_vector(text):
    words = set(text.split())
    return np.array([1 if w in words else 0 for w in vocab])

X = np.array([word_vector(t) for t, _ in docs])
y = np.array([lab for _, lab in docs])

p_spam = y.mean(); p_ham = 1 - p_spam
def likelihoods(cls):
    subset = X[y == cls]
    return (subset.sum(0) + 1) / (len(subset) + 2)   # +1/+2 = Laplace smoothing

lik_spam, lik_ham = likelihoods(1), likelihoods(0)

def predict(text):
    v = word_vector(text)
    log_spam = np.log(p_spam) + np.sum(np.log(np.where(v==1, lik_spam, 1-lik_spam)))
    log_ham  = np.log(p_ham)  + np.sum(np.log(np.where(v==1, lik_ham,  1-lik_ham)))
    return "SPAM" if log_spam > log_ham else "HAM"

for msg in ["win free money", "project meeting today", "cheap prize now"]:
    print(f"'{msg}'  -> {predict(msg)}")
""")

nb.readcode(r"""
- We build a vocabulary of every word seen, then turn each message into a 0/1 vector
  ("does this word appear?").
- `p_spam` / `p_ham` are the **priors** — half the training set is spam.
- `likelihoods()` counts how often each word shows up in each class, with `+1`
  smoothing so a word never gets probability 0.
- `predict()` adds up **log**-probabilities (logs turn fragile multiplication into
  stable addition) and picks whichever class scores higher.
- Result: spammy words tip messages to SPAM, work words tip them to HAM.
""")

nb.warn(r"""
Without **Laplace smoothing** (the `+1`), any word never seen in a class gets
probability 0 — and a single 0 multiplied into the product zeroes *everything*,
letting one unseen word veto the whole prediction. Interviewers love to probe this.
""")

nb.code(r"""
# The professional version: same idea, scaled up with scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts  = [t for t, _ in docs]
labels = [l for _, l in docs]
vec = CountVectorizer()
Xc = vec.fit_transform(texts)
model = MultinomialNB().fit(Xc, labels)

tests = ["win money prize", "deadline meeting notes"]
for t, p in zip(tests, model.predict(vec.transform(tests))):
    print(f"'{t}' -> {'SPAM' if p==1 else 'HAM'}")
""")

nb.takeaway("Naive Bayes multiplies per-word probabilities with a prior; add Laplace smoothing so unseen words can't zero the product.")

# ---------------------------------------------------------------------------
# 16.5 Distributions
# ---------------------------------------------------------------------------
nb.md(r"""
## 16.5 Distributions you should name on sight

A **distribution** is the shape describing how likely each value is. Recognizing the
right one tells you which model and test to reach for.
""")

nb.plain(r"""
- **Bernoulli / Binomial** — yes/no trials. "Out of 10 emails, how many get clicked?"
- **Poisson** — counts of rare events in a window. "How many support calls per hour?"
- **Normal (Gaussian)** — the bell curve. Averages of lots of things drift toward it.
- **Uniform** — every outcome equally likely (a fair die).
- **Exponential** — waiting time until the next event. Right-skewed (long tail).
""")

nb.code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
sns.histplot(rng.binomial(10, 0.3, 5000), ax=ax[0], discrete=True); ax[0].set_title("Binomial(10, 0.3)")
sns.histplot(rng.poisson(2.0, 5000), ax=ax[1], discrete=True); ax[1].set_title("Poisson(2)")
sns.histplot(rng.exponential(2.0, 5000), ax=ax[2]); ax[2].set_title("Exponential(2)")
plt.tight_layout(); plt.show()
""")

nb.takeaway("Match the variable to its distribution (yes/no -> Binomial, rare counts -> Poisson, waiting time -> Exponential) to pick the right test/model.")

# ---------------------------------------------------------------------------
# Practice + summary
# ---------------------------------------------------------------------------
nb.md(r"""
## 16.6 Practice
""")

nb.try_this(r"""
1. Rework the medical test with **10%** prevalence. Does the posterior go up or down?
   Explain via the table (more truly-sick people, so fewer false positives dominate).
2. Add the word "free" to a ham message in the scratch Naive Bayes — does the
   prediction flip? Check the `lik_spam` vs `lik_ham` for "free".
3. In one sentence, explain why "naive" independence is usually wrong but still
   useful.
""")

nb.md(r"""
## Summary

- **Permutations** (order matters) vs **combinations** (order doesn't) — divide by
  `r!` to remove orderings. Use `math.perm` / `math.comb`.
- Probability rules: **complement** (1 - P), **addition** (subtract the overlap),
  **multiplication** (conditional in general). **Mutually exclusive ≠ independent.**
- For **"at least one"**, use the complement.
- **Bayes** turns a **prior** into a **posterior** via the **likelihood**; with rare
  events, **base rates dominate** (medical-test paradox = same logic as imbalanced ML).
- **Naive Bayes**: $P(y)\prod P(x_i\mid y)$, conditional independence, **Laplace
  smoothing**; fast, strong text baseline.
- Recognize Bernoulli/Binomial, Poisson, Normal, Uniform, Exponential on sight.

Next: **Module 17 — A/B Testing, ANOVA & Statistical Power** (proving what *causes* what).
""")

out = nb.save("notebooks/16_probability_combinatorics_bayes.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
