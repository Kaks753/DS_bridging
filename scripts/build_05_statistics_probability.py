"""Builder for Module 5: Statistics & Probability (4-layer rewrite of old M04).
Adds beginner-friendly by-hand worked examples (SE, CI, p-value from scratch)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 5 — Statistics & Probability for Data Science

Statistics is the part that scares people, so let's demystify it up front:
**statistics is just the science of telling a real pattern apart from random luck,
and saying how sure you are.** That's the whole game.

Every time you claim "customers who call support churn more," someone can ask: *are
you sure, or did you get lucky with this sample?* Statistics is how you answer that
without bluffing — and it's the #1 thing that makes your conclusions defensible in
an interview or a meeting.

We'll build intuition by *simulating* everything (watching the math happen), and
we'll compute a couple of things **by hand** first so the formulas aren't magic.
""")

nb.analogy("Think of a food critic tasting one spoonful of soup to judge the whole pot. "
           "Statistics is the set of rules for how much you can trust that one spoonful "
           "(your sample) to tell you about the entire pot (the population).")

nb.md("## 5.1 Setup")

nb.plain("""
We load our cleaned data and `scipy.stats` — the standard toolbox of statistical
functions. `rng` is a random-number generator with a fixed **seed** so every run
gives the same 'random' numbers (reproducibility).
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
print("loaded:", df.shape)
""")

nb.jargon("population", "the entire group you care about (all customers ever)")
nb.jargon("sample", "the subset you actually measured (the customers in your data)")
nb.jargon("scipy.stats", "Python's library of statistical distributions and tests")

nb.md("## 5.2 Center and spread — and why the median survives outliers")

nb.plain("""
Two questions you ask of any number column: "what's a typical value?" (center) and
"how spread out are they?" (spread).
- **Mean** = the average; it's the balance point but gets yanked by extremes.
- **Median** = the middle value; it shrugs off extremes.
- **Standard deviation (std)** = the typical distance of a value from the mean.
""")

nb.code(r"""
x = df["income"]
print(f"mean   : {x.mean():,.0f}")
print(f"median : {x.median():,.0f}   <- lower than mean => right skew")
print(f"std    : {x.std():,.0f}")
print(f"skew   : {x.skew():.2f}      (>0 right-skewed)")
print(f"IQR    : {x.quantile(.75) - x.quantile(.25):,.0f}")
""")

nb.readcode("""
- `x.mean()` / `x.median()` → the average and the middle value.
- `x.std()` → standard deviation (typical spread around the mean).
- `x.skew()` → lopsidedness; > 0 means a long right tail.
- `x.quantile(.75) - x.quantile(.25)` → the IQR (middle-50% spread).
""")

nb.deeper("""
When the mean is BIGGER than the median, a few large values are pulling the average
up — the distribution 'leans right'. That's why for money and time data (almost
always right-skewed) you report the MEDIAN: it describes the typical person, not the
handful of whales. Saying this in an interview signals you understand your data.
""")

nb.takeaway("Mean > median ⇒ right-skewed. Report median + IQR for skewed data; "
            "mean + std for roughly symmetric data.")

nb.md("## 5.3 The Normal distribution & the 68–95–99.7 rule")

nb.plain("""
The **normal distribution** is the classic bell curve. It shows up everywhere, and
it has a handy rule of thumb: about 68% of values sit within 1 std of the mean, 95%
within 2 stds, and 99.7% within 3. That's how you judge whether a value is 'unusual'.
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

nb.readcode("""
- `rng.normal(loc=100, scale=15, size=...)` → draw random numbers from a normal
  with mean 100 and std 15.
- `np.abs(samples - 100) < k*15` → True where a value is within k stds of the mean;
  `.mean()*100` turns that into a percentage.
- `stats.norm.pdf(xs, 100, 15)` → the exact bell-curve height at each x (the red line).
""")

nb.jargon("normal distribution", "the symmetric bell curve, defined by its mean and std")
nb.jargon("pdf", "Probability Density Function — the curve giving a distribution's shape")

nb.md("## 5.4 Law of Large Numbers — averages settle down")

nb.plain("""
The Law of Large Numbers is common sense made formal: **the more data you collect,
the closer your sample average gets to the true value.** Flip a fair-ish coin a few
times and the proportion of heads bounces around; flip it thousands of times and it
settles near the truth. This is *why* more data = more reliable estimates.
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

nb.readcode("""
- `rng.binomial(1, true_p, size=5000)` → 5000 coin flips (1=heads) with P(heads)=0.3.
- `np.cumsum(flips) / np.arange(1, 5001)` → the running proportion of heads after
  each flip (cumulative heads ÷ number of flips so far).
- The green line wiggles a lot early, then hugs the red 'truth' line as n grows.
""")

nb.jargon("Law of Large Numbers", "as sample size grows, the sample average converges to the true average")

nb.md("## 5.5 Central Limit Theorem — the crown jewel")

nb.plain("""
Here's the most useful magic trick in statistics. Take ANY population — even a wildly
lopsided one — grab a sample, compute its average, and repeat many times. Those
**averages** form a nice bell curve, *even though the original data wasn't bell-shaped
at all.* That's the Central Limit Theorem (CLT). It's why we can use normal-based
tools (confidence intervals, t-tests) even on skewed data like income.
""")

nb.analogy("Individual people's incomes are wildly uneven. But the AVERAGE income of "
           "random groups of 50 people is remarkably predictable and bell-shaped. The CLT "
           "says averaging 'tames' almost any messiness.")

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

nb.readcode("""
- `rng.exponential(...)` → a deliberately skewed population (long right tail).
- `sample_means(n)` → take 2000 samples of size n, return each sample's mean.
- The 3 panels: the skewed population, then the means for n=5 and n=50 — watch them
  become bell-shaped AND narrower as n grows.
""")

nb.deeper("""
The set of all possible sample-means is called the **sampling distribution**. Its
spread has a name and a formula: the **standard error**, SE = std / √n. Bigger n →
smaller SE → a more precise estimate. This one formula is the engine behind
confidence intervals and p-values — everything below rests on it.
""")

nb.jargon("Central Limit Theorem", "sample means become normally distributed as sample size grows, whatever the population shape")
nb.jargon("sampling distribution", "the distribution of a statistic (like the mean) across many samples")
nb.jargon("standard error (SE)", "the std of a sample statistic; for the mean, SE = std/√n")

nb.md("## 5.6 Worked example BY HAND — standard error & a confidence interval")

nb.plain("""
Before we let a library do it, let's compute a confidence interval step by step with
plain arithmetic, so the formula stops being a black box. A **95% confidence
interval** is a range around our estimate that captures the true mean 95% of the
time (over many repeats). Recipe: estimate ± (critical value) × (standard error).
""")

nb.code(r"""
sample = df["monthly_spend"]

# --- step by step, like on paper ---
n       = len(sample)
mean    = sample.mean()
std     = sample.std(ddof=1)          # sample std (ddof=1 = divide by n-1)
se      = std / np.sqrt(n)            # standard error of the mean
tcrit   = stats.t.ppf(0.975, df=n-1)  # t critical value for 95% (two-sided)
margin  = tcrit * se                  # the +/- part
lo, hi  = mean - margin, mean + margin

print(f"step 1  n            = {n}")
print(f"step 2  mean         = {mean:.2f}")
print(f"step 3  std (n-1)     = {std:.2f}")
print(f"step 4  SE = std/sqrt(n) = {std:.2f}/sqrt({n}) = {se:.3f}")
print(f"step 5  t* (95%)     = {tcrit:.3f}")
print(f"step 6  margin = t*·SE = {tcrit:.3f} x {se:.3f} = {margin:.3f}")
print(f"RESULT  95% CI       = [{lo:.2f}, {hi:.2f}]")
""")

nb.readcode("""
- `std(ddof=1)` → sample standard deviation (divide by n-1, the honest version).
- `se = std/sqrt(n)` → standard error, straight from the CLT.
- `stats.t.ppf(0.975, df=n-1)` → the t-distribution's 97.5% point; for large n it's
  ~1.96. (Two-sided 95% leaves 2.5% in each tail → the 0.975 quantile.)
- `margin = t* × SE`, then CI = mean ± margin.
""")

nb.warn("Interpret a CI carefully: it means 'this PROCEDURE captures the true mean 95% "
        "of the time over many samples' — NOT '95% chance the truth is in THIS exact "
        "interval'. Interviewers love catching people on this.")

nb.interview("\"A 95% CI reflects the long-run coverage of the method, not the probability "
             "the parameter lies in one particular interval. Its width is driven by SE = std/√n, "
             "so more data narrows it.\"")

nb.md("## 5.7 Hypothesis testing — the logic (then a p-value by hand)")

nb.plain("""
A hypothesis test is a courtroom for data. You assume 'nothing interesting is going
on' (the **null hypothesis, H0**) and see whether your data is too surprising for
that story to hold. The measure of surprise is the **p-value**: the probability of
seeing data at least this extreme *if H0 were true*. Small p (usually < 0.05) →
'this would be too much of a coincidence' → reject H0.
""")

nb.md(r"""
The steps every test follows:
1. **H0 (null):** the boring default ("no difference").
2. **H1 (alternative):** what you suspect ("there IS a difference").
3. Compute a **test statistic** and its **p-value**.
4. Compare p to a threshold **α** (usually 0.05). If `p < α`, reject H0.
""")

nb.analogy("Court analogy: H0 = 'innocent' (assumed by default). The evidence (your data) "
           "has to be surprising enough under innocence to convict. The p-value is 'how "
           "likely is evidence this strong if the person were actually innocent?'")

nb.plain("""
Let's demystify a p-value by computing one from scratch. We ask: is a coin fair? We
flip it 100 times and get 63 heads. If it were truly fair (H0: p=0.5), how weird is
63+ heads? We'll simulate 'fair-coin' worlds and just COUNT how often we get a result
as extreme as ours — that count IS the p-value.
""")

nb.code(r"""
observed_heads = 63
n_flips = 100

# Simulate 100,000 fair-coin experiments and see how extreme 63 is.
sims = rng.binomial(n_flips, 0.5, size=100_000)
# two-sided: as far from 50 as 63 is (i.e. >=63 or <=37)
as_extreme = np.mean((sims >= 63) | (sims <= 37))
print(f"Observed: {observed_heads}/100 heads")
print(f"Simulated p-value (two-sided): {as_extreme:.4f}")

# Compare with the exact math (scipy binomial test):
exact = stats.binomtest(observed_heads, n_flips, 0.5).pvalue
print(f"Exact p-value (scipy):        {exact:.4f}")
print("Decision at α=0.05:", "reject H0 (coin looks biased)"
      if exact < 0.05 else "fail to reject H0 (could be fair)")
""")

nb.readcode("""
- `rng.binomial(100, 0.5, size=100_000)` → simulate 100,000 sets of 100 fair flips.
- `(sims >= 63) | (sims <= 37)` → count worlds at least as far from 50 as we saw
  (63 is 13 away from 50, so 37 is the mirror). `.mean()` = fraction = the p-value.
- `stats.binomtest(...)` → the exact formula; it matches our simulation closely.
""")

nb.deeper("""
Notice what the p-value is and isn't. It IS: 'if H0 were true, how often would I see
something this extreme?' It is NOT: 'the probability H0 is true.' A p of 0.03 means a
3% chance of data this extreme under the null — not a 3% chance the null is correct.
Also, 'not significant' (p ≥ 0.05) means 'not enough evidence', NOT 'proven no effect'.
""")

nb.jargon("null hypothesis (H0)", "the default 'nothing interesting here' assumption a test tries to disprove")
nb.jargon("p-value", "probability of data at least this extreme IF the null hypothesis were true")
nb.jargon("alpha (α)", "the significance threshold, usually 0.05; reject H0 when p < α")
nb.jargon("Type I error", "false positive: rejecting a true null (rate = α)")
nb.jargon("Type II error", "false negative: failing to reject a false null (rate = β); power = 1−β")

nb.md("## 5.8 The two tests you'll actually run")

nb.plain("""
Now the real, library versions on our data. Two workhorses:
- **t-test** → compares the MEANS of two groups (numeric outcome).
- **chi-square test** → checks whether two CATEGORIES are associated.
""")

nb.md("### 5.8a t-test — do churners and stayers differ in support calls?")

nb.code(r"""
churn_calls = df.loc[df["churn"] == 1, "support_calls"]
stay_calls  = df.loc[df["churn"] == 0, "support_calls"]

t_stat, p_val = stats.ttest_ind(churn_calls, stay_calls, equal_var=False)  # Welch
print(f"mean calls — churned: {churn_calls.mean():.2f} | stayed: {stay_calls.mean():.2f}")
print(f"t = {t_stat:.3f}, p = {p_val:.4g}")
print("Decision:", "reject H0 (means differ)" if p_val < 0.05
      else "fail to reject H0")
""")

nb.readcode("""
- `df.loc[df["churn"]==1, "support_calls"]` → support-call counts for churned customers.
- `stats.ttest_ind(a, b, equal_var=False)` → Welch's two-sample t-test (doesn't assume
  equal variances — the safer default). Returns the t statistic and p-value.
""")

nb.md("### 5.8b chi-square — is churn associated with plan? (category vs category)")

nb.code(r"""
table = pd.crosstab(df["plan"], df["churn"])
print("contingency table:\n", table)
chi2, p, dof, expected = stats.chi2_contingency(table)
print(f"\nchi2 = {chi2:.3f}, dof = {dof}, p = {p:.4g}")
print("Decision:", "association exists" if p < 0.05 else "no evidence of association")
""")

nb.readcode("""
- `pd.crosstab(plan, churn)` → a table counting customers in each (plan, churn) combo.
- `stats.chi2_contingency(table)` → tests whether the two categories are independent;
  compares observed counts to what you'd expect if there were NO association.
""")

nb.jargon("t-test", "test comparing the means of two groups")
nb.jargon("chi-square test", "test for association between two categorical variables")

nb.md("### 5.8c Effect size — significant ≠ important")

nb.plain("""
A trap: with enough data, even a microscopic, useless difference becomes
'statistically significant'. So always report an **effect size** — how BIG the
difference is — next to the p-value. Cohen's d expresses the gap in std units.
""")

nb.code(r"""
def cohens_d(a, b):
    na, nb_ = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb_-1)*b.var(ddof=1)) / (na+nb_-2))
    return (a.mean() - b.mean()) / pooled

d = cohens_d(churn_calls, stay_calls)
print(f"Cohen's d = {d:.2f}  (0.2 small, 0.5 medium, 0.8 large)")
""")

nb.interview("\"I always pair a p-value with an effect size. Significance says an effect is "
             "probably real; effect size says whether it's big enough to care about.\"")

nb.md("## 5.9 Correlation ≠ causation (the sentence that saves careers)")

nb.plain("""
Two things moving together does NOT mean one causes the other. Ice-cream sales and
drownings both rise in summer — ice cream doesn't cause drowning; hot weather (a
hidden **confounder**) drives both. To claim causation you need a randomized
experiment (an A/B test) or careful causal inference — never a correlation alone.
""")

nb.jargon("confounder", "a hidden variable that influences both things you see correlated, faking a link")

nb.warn("Never write 'X causes Y' from correlation alone. Say 'X is associated with Y' "
        "and note possible confounders. This one habit prevents embarrassing wrong claims.")

nb.md("## 5.10 Try it yourself")

nb.try_this("""
1. Build a 95% CI for mean `income` (reuse the by-hand recipe). Interpret it in one
   correct sentence.
2. Run a t-test: do Premium vs Basic customers differ in `monthly_spend`?
3. Redo the p-value simulation for 55/100 heads. Is it still 'significant'? Why not?
4. Explain to a non-technical manager what a p-value of 0.03 means — in one honest line.
""")

nb.md(r"""
## Summary — statistics you can defend

- Report **median + IQR** for skewed data; mean + std for symmetric.
- **CLT** makes sample means normal → enables CIs and t-tests even for skewed data.
- **SE = std/√n**: more data → tighter estimates.
- A **CI** describes the procedure's long-run coverage, not one interval's probability.
- A **p-value** = P(data this extreme | H0 true) — NOT P(H0 true).
- Use a **t-test** (means) and **chi-square** (categorical association); always add an
  **effect size**.
- **Correlation ≠ causation.**

Next: **Feature Engineering & Scaling**, turning raw columns into signal.
""")

out = nb.save("notebooks/05_statistics_probability.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
