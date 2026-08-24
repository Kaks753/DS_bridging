"""Builder for Module 16: A/B Testing, ANOVA & Statistical Power."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 16 — A/B Testing, ANOVA & Statistical Power (Phase 2: T15–16)

This is how data scientists prove **causation**, not just correlation — the work
that drives real product decisions ("did the new button increase signups?"). It's
a favorite interview topic at product companies.

Goals:
- Why **randomized experiments** license causal claims.
- Design an **A/B test**: hypotheses, metric, and a **power analysis** for sample
  size *before* running.
- Analyze results: **z-test for proportions**, **t-test for means**.
- **ANOVA** for comparing **3+** groups, and why you can't just run many t-tests.
- **Type I/II errors, power**, and the peeking/multiple-comparisons traps.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)
""")

nb.md(r"""
## 16.1 Why A/B testing gives causation

In Module 04 we said correlation ≠ causation because of confounders. **Random
assignment** breaks confounding: if we randomly split users into control (A) and
treatment (B), the groups are equivalent *on average* in every respect except the
change we made. So a difference in outcome can be **attributed to the change** —
that's a causal claim.

Vocabulary: **control** (current version) vs **treatment/variant** (new version);
the metric we compare is the **conversion rate** or a mean.
""")

nb.md(r"""
## 16.2 Design FIRST: hypotheses, metric, and power analysis

**Before collecting a single data point**, decide:
1. **Hypotheses**: H0 = "no difference"; H1 = "B differs from A".
2. **Primary metric** (e.g. signup conversion rate). One metric — avoid fishing.
3. **α** (false-positive rate, usually 0.05) and desired **power** (usually 0.80).
4. **Minimum Detectable Effect (MDE)** — the smallest lift worth caring about.
5. **Required sample size** via a **power analysis** — so you're neither
   underpowered (can't detect real effects) nor wastefully over-sized.
""")

nb.code(r"""
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

baseline = 0.10           # current conversion 10%
target   = 0.12           # we care about detecting a lift to 12% (MDE = 2 points)
effect = proportion_effectsize(target, baseline)   # standardized effect size (h)

analysis = NormalIndPower()
n_per_group = analysis.solve_power(effect_size=effect, alpha=0.05, power=0.80,
                                   alternative="two-sided")
print(f"standardized effect size (Cohen's h) = {effect:.3f}")
print(f"required sample size PER GROUP ≈ {int(np.ceil(n_per_group)):,}")
print("Run the test until each arm reaches this size — decided IN ADVANCE.")
""")

nb.md(r"""
**Why this matters:** stopping early when you "see significance" (**peeking**)
massively inflates false positives. Committing to a sample size up front is the
discipline that keeps A/B tests honest.
""")

nb.md(r"""
## 16.3 Analyze a proportions A/B test (z-test)

We simulate a test where B truly converts slightly better, then analyze it the way
you would real logs.
""")

nb.code(r"""
n = 3000
conv_A = rng.random(n) < 0.10        # control: true 10%
conv_B = rng.random(n) < 0.12        # treatment: true 12%

from statsmodels.stats.proportion import proportions_ztest
successes = [conv_B.sum(), conv_A.sum()]
nobs = [n, n]
z, p = proportions_ztest(successes, nobs, alternative="larger")   # is B > A?

rate_A, rate_B = conv_A.mean(), conv_B.mean()
print(f"A (control)   conversion: {rate_A:.3%}")
print(f"B (treatment) conversion: {rate_B:.3%}")
print(f"absolute lift: {rate_B - rate_A:+.3%} | relative: {(rate_B/rate_A - 1):+.1%}")
print(f"z = {z:.3f}, p = {p:.4f}")
print("Decision:", "ship B — significant improvement" if p < 0.05
      else "not significant — keep A / gather more data")
""")

nb.md(r"""
Always report the **effect size** (the lift), not just significance. A tiny,
statistically-significant lift may not be worth the engineering cost — that's a
business judgment your analysis should enable.
""")

nb.md(r"""
## 16.4 Comparing means (t-test) — e.g. time-on-page

If the metric is a **mean** (revenue per user, time on page), use a **t-test**.
""")

nb.code(r"""
time_A = rng.normal(50, 15, 500)     # seconds on page, control
time_B = rng.normal(54, 15, 500)     # treatment slightly higher
t, p = stats.ttest_ind(time_B, time_A, equal_var=False)  # Welch's t-test
print(f"mean time A={time_A.mean():.1f}s, B={time_B.mean():.1f}s")
print(f"t = {t:.3f}, p = {p:.4f} ->",
      "significant" if p < 0.05 else "not significant")
""")

nb.md(r"""
## 16.5 ANOVA — comparing 3+ groups the right way

Suppose you test **three** landing pages (A/B/C). Why not run three t-tests
(A-B, A-C, B-C)? Because each test has a 5% false-positive chance, and running
several **inflates** the overall Type I error (the **multiple-comparisons
problem**). **ANOVA** (Analysis of Variance) tests "**are all group means equal?**"
in a *single* F-test, controlling that error.
""")

nb.code(r"""
page_A = rng.normal(50, 12, 300)
page_B = rng.normal(52, 12, 300)
page_C = rng.normal(57, 12, 300)     # C is genuinely better

f_stat, p_anova = stats.f_oneway(page_A, page_B, page_C)
print(f"ANOVA F = {f_stat:.3f}, p = {p_anova:.4g}")
print("Interpretation:", "at least one page differs" if p_anova < 0.05
      else "no evidence of any difference")
""")

nb.md(r"""
**ANOVA logic (intuition):** the F-statistic is the ratio of **variance *between*
groups** to **variance *within* groups**. If between-group spread is large relative
to within-group noise, the means likely truly differ (big F, small p).

ANOVA tells you *that* a difference exists, not *which* pair differs — for that run
a **post-hoc** test (e.g. Tukey's HSD) that corrects for multiple comparisons.
""")

nb.code(r"""
from statsmodels.stats.multicomp import pairwise_tukeyhsd
vals = np.concatenate([page_A, page_B, page_C])
grp = (["A"]*300) + (["B"]*300) + (["C"]*300)
tukey = pairwise_tukeyhsd(vals, grp, alpha=0.05)
print(tukey)
print("\n'reject=True' rows are the pairs that significantly differ.")
""")

nb.md(r"""
## 16.6 Power, errors, and honesty

- **Type I error (α)**: false positive — declare an effect that isn't real.
- **Type II error (β)**: false negative — miss a real effect.
- **Power (1−β)**: probability of detecting a true effect. Increases with **larger
  sample size**, **larger true effect**, and **higher α**.

Let's *see* how power grows with sample size for our 10%→12% test.
""")

nb.code(r"""
sizes = np.arange(500, 12001, 500)
powers = [NormalIndPower().solve_power(effect_size=effect, nobs1=nn,
                                       alpha=0.05, alternative="two-sided")
          for nn in sizes]
plt.figure(figsize=(7,4))
plt.plot(sizes, powers, "o-")
plt.axhline(0.80, color="red", ls="--", label="target power 0.80")
plt.xlabel("sample size per group"); plt.ylabel("statistical power")
plt.title("Power rises with sample size (for a fixed effect)"); plt.legend(); plt.show()
""")

nb.md(r"""
**Traps to name in interviews:**
- **Peeking / early stopping** — checking significance repeatedly inflates false
  positives. Fix: decide n up front, or use sequential-testing corrections.
- **Multiple comparisons** — many metrics/variants → some "significant" by chance.
  Fix: Bonferroni / FDR correction, or one pre-registered primary metric.
- **Novelty & seasonality effects** — run long enough to be representative.
- **Sample ratio mismatch** — if A/B split isn't ~50/50, the randomization/logging
  is broken; investigate before trusting results.
""")

nb.md(r"""
## 16.7 Mini-exercises

1. Recompute the required sample size for detecting a 10%→11% lift. Why is it so
   much larger?
2. Simulate an A/B test where A and B are *identical*; run it 1000× and confirm
   ~5% of runs produce p<0.05 (that's α in action).
3. Run ANOVA where all three groups are identical — what p-value do you expect?
4. Explain to a PM why you shouldn't stop an A/B test the moment it hits p<0.05.
""")

nb.md(r"""
## Summary

- **Randomization** → causal claims (A/B testing's whole point).
- **Design first**: hypotheses, one metric, α, power, MDE → **power analysis** for
  sample size.
- Analyze with **z-test** (proportions) or **t-test** (means); always report the
  **effect size**.
- **ANOVA** compares 3+ groups in one F-test (avoids multiple-comparison inflation);
  follow with **Tukey** post-hoc.
- Mind **power**, **Type I/II errors**, and the **peeking / multiple-comparisons**
  traps.

Next: **Module 17 — OOP + Linear Algebra & Calculus for ML**.
""")

out = nb.save("notebooks/16_ab_testing_anova_power.ipynb")
print("saved", out)
