"""Builder for Module 04: Statistics & Probability foundations."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 04 — Statistics & Probability for Data Science

You have a Mathematics degree — this module *connects that theory to practice* so
you can defend every number you report. Statistics is the grammar of data science:
it tells you when a pattern is **real** versus **noise**, and how **uncertain**
your estimates are.

We cover, with runnable intuition:
- Descriptive stats: mean/median/mode, variance/std, skew, quantiles.
- Distributions: normal, and why it matters.
- **Law of Large Numbers** & the **Central Limit Theorem** (simulated!).
- **Sampling distributions**, standard error, **confidence intervals**.
- **Hypothesis testing**: null/alternative, test statistic, **p-value**, errors.
- t-test, chi-square — the two you'll actually run.
- Correlation vs causation (again, because it matters).
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
df = pd.read_csv("../data/customers_clean.csv")
""")

nb.md(r"""
## 4.1 Center and spread — and why the *median* survives outliers

- **Mean** = balance point; sensitive to extremes.
- **Median** = 50th percentile; robust.
- **Mode** = most frequent (useful for categoricals).
- **Variance** = average squared deviation from the mean; **std** = its square
  root (same units as the data). Std answers "how spread out, typically?"
""")

nb.code(r"""
x = df["income"]
print(f"mean   : {x.mean():,.0f}")
print(f"median : {x.median():,.0f}   <- lower than mean => right skew")
print(f"std    : {x.std():,.0f}")
print(f"skew   : {x.skew():.2f}      (>0 right-skewed)")
print(f"IQR    : {x.quantile(.75) - x.quantile(.25):,.0f}")
""")

nb.md(r"""
**Takeaway:** when mean > median, the distribution leans right (a few big values).
Report the **median** for skewed money/'time' data; the mean can mislead.
""")

nb.md(r"""
## 4.2 The Normal distribution & the 68–95–99.7 rule

Many methods assume approximate normality. For a normal distribution, ~68% of
values fall within 1 std of the mean, ~95% within 2, ~99.7% within 3. This is the
basis of z-scores and many "is this unusual?" judgments.
""")

nb.code(r"""
samples = rng.normal(loc=100, scale=15, size=100_000)   # e.g. IQ-like
for k in (1, 2, 3):
    within = np.mean(np.abs(samples - 100) < k * 15) * 100
    print(f"within {k} std: {within:.1f}%")

plt.figure(figsize=(7,4))
sns.histplot(samples, bins=60, stat="density", color="steelblue")
xs = np.linspace(40, 160, 300)
plt.plot(xs, stats.norm.pdf(xs, 100, 15), "r", lw=2, label="theoretical pdf")
plt.title("Normal(100, 15): histogram vs true density"); plt.legend(); plt.show()
""")

nb.md(r"""
## 4.3 Law of Large Numbers — averages settle down

As sample size grows, the sample mean converges to the true mean. This is *why*
more data gives more reliable estimates. Watch it happen:
""")

nb.code(r"""
true_p = 0.3                      # true probability of "heads"
flips = rng.binomial(1, true_p, size=5000)
running_mean = np.cumsum(flips) / np.arange(1, 5001)

plt.figure(figsize=(8,4))
plt.plot(running_mean, color="darkgreen")
plt.axhline(true_p, color="red", ls="--", label=f"true p = {true_p}")
plt.xscale("log"); plt.xlabel("number of samples (log)")
plt.ylabel("running proportion of heads")
plt.title("Law of Large Numbers: estimate converges to truth")
plt.legend(); plt.show()
""")

nb.md(r"""
## 4.4 Central Limit Theorem — the crown jewel

**CLT:** the distribution of the *sample mean* becomes approximately **normal** as
sample size grows, **regardless of the population's shape**. This is what lets us
build confidence intervals and run t-tests on non-normal data.

We prove it by sampling from a *very* non-normal (exponential) population.
""")

nb.code(r"""
population = rng.exponential(scale=2.0, size=200_000)   # heavily right-skewed

def sample_means(n, reps=2000):
    return np.array([rng.choice(population, n).mean() for _ in range(reps)])

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
sns.histplot(population, bins=60, ax=axes[0], color="gray")
axes[0].set_title("Population (exponential, skewed)")
for ax, n in zip(axes[1:], (5, 50)):
    sns.histplot(sample_means(n), bins=40, ax=ax, color="steelblue", kde=True)
    ax.set_title(f"Means of samples (n={n})")
plt.tight_layout(); plt.show()
print("Even from a skewed population, the sample means look bell-shaped — and "
      "tighter for larger n.")
""")

nb.md(r"""
**Takeaway:** the sample mean has its own distribution (the *sampling
distribution*). Its spread is the **standard error** `SE = std / sqrt(n)` —
bigger `n` → smaller SE → more precise estimate. This single formula underlies
confidence intervals and p-values.
""")

nb.md(r"""
## 4.5 Confidence intervals — quantify your uncertainty

A 95% CI for the mean is a range that, under repeated sampling, would contain the
true mean ~95% of the time. Correct phrasing matters in interviews: it's about the
*procedure's* long-run coverage, **not** "95% probability the truth is in this
one interval".
""")

nb.code(r"""
sample = df["monthly_spend"]
n = len(sample)
mean = sample.mean()
se = sample.std(ddof=1) / np.sqrt(n)
# t-based CI (correct when population std unknown)
tcrit = stats.t.ppf(0.975, df=n-1)
lo, hi = mean - tcrit * se, mean + tcrit * se
print(f"n={n}, mean spend = {mean:.2f}")
print(f"standard error   = {se:.2f}")
print(f"95% CI for mean  = [{lo:.2f}, {hi:.2f}]")
""")

nb.md(r"""
## 4.6 Hypothesis testing — the logic

The machinery:
1. **H0 (null):** the boring default (e.g. "no difference").
2. **H1 (alternative):** what you suspect (e.g. "there IS a difference").
3. Compute a **test statistic** and its **p-value** = probability of data this
   extreme *if H0 were true*.
4. Compare p to a threshold **α** (usually 0.05). `p < α` → reject H0.

**Two errors:** Type I (false positive: reject a true H0; rate = α) and
Type II (false negative: fail to reject a false H0; rate = β). **Power** = 1 − β.

Crucial nuance: p-value is **not** the probability H0 is true, and "not
significant" ≠ "no effect".
""")

nb.md(r"""
### 4.6a Two-sample t-test — do churners and stayers differ in support calls?

Compares the **means** of two groups.
""")

nb.code(r"""
churn_calls = df.loc[df["churn"] == 1, "support_calls"]
stay_calls  = df.loc[df["churn"] == 0, "support_calls"]

t_stat, p_val = stats.ttest_ind(churn_calls, stay_calls, equal_var=False)  # Welch
print(f"mean calls — churned: {churn_calls.mean():.2f} | stayed: {stay_calls.mean():.2f}")
print(f"t = {t_stat:.3f}, p = {p_val:.4g}")
print("Decision:", "reject H0 (means differ)" if p_val < 0.05
      else "fail to reject H0")
""")

nb.md(r"""
### 4.6b Chi-square test — is churn associated with plan? (category vs category)

Compares observed vs expected counts in a contingency table.
""")

nb.code(r"""
table = pd.crosstab(df["plan"], df["churn"])
print("contingency table:\n", table)
chi2, p, dof, expected = stats.chi2_contingency(table)
print(f"\nchi2 = {chi2:.3f}, dof = {dof}, p = {p:.4g}")
print("Decision:", "association exists" if p < 0.05 else "no evidence of association")
""")

nb.md(r"""
### 4.6c Effect size — significance is not importance

With enough data, tiny, useless differences become "significant". Always report an
**effect size** (e.g. Cohen's d) alongside the p-value.
""")

nb.code(r"""
def cohens_d(a, b):
    na, nb_ = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb_-1)*b.var(ddof=1)) / (na+nb_-2))
    return (a.mean() - b.mean()) / pooled

d = cohens_d(churn_calls, stay_calls)
print(f"Cohen's d = {d:.2f}  (0.2 small, 0.5 medium, 0.8 large)")
""")

nb.md(r"""
## 4.7 Correlation ≠ causation (the sentence that saves careers)

Two variables can correlate because A→B, B→A, a hidden **confounder** C drives
both, or pure chance. To claim causation you need a randomized experiment (A/B
test) or careful causal inference — never a correlation alone.
""")

nb.md(r"""
## 4.8 Mini-exercises

1. Build a 95% CI for mean `income` using the cleaned data. Interpret it in one
   correct sentence.
2. Run a t-test: do Premium vs Basic customers differ in `monthly_spend`?
3. Simulate CLT from a **uniform** population; does the sample-mean still go
   normal?
4. Explain to a non-technical manager what a p-value of 0.03 means — in one line,
   without lying.
""")

nb.md(r"""
## Summary — statistics you can defend

- Report **median + IQR** for skewed data; mean + std for symmetric.
- **CLT** makes sample means normal → enables CIs and t-tests even for skewed data.
- **SE = std/√n**: more data → tighter estimates.
- A **CI** describes the procedure's long-run coverage, not one interval's probability.
- Hypothesis test = H0/H1 → statistic → **p-value** vs α; watch Type I/II errors.
- Use **t-test** (means), **chi-square** (categorical association); always add an
  **effect size**.
- **Correlation ≠ causation.**

Next: **Module 05 — Feature Engineering & Scaling**, turning raw columns into signal.
""")

out = nb.save("notebooks/04_statistics_probability.ipynb")
print("saved", out)
