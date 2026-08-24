# Exercises

Every notebook (`notebooks/NN_*.ipynb`) ends with a **Mini-exercises** section and a
**scratch cell**. That's where you practice — do them *before* looking anything up.

## How to practice (the method that actually builds mastery)

1. **Blank-page recall**: after finishing a module, close it and write — from memory —
   the 5-bullet summary. Compare to `notes/CHEATSHEET.md`. The gaps are your study list.
2. **Re-type, don't copy**: retype each code cell yourself. Muscle memory matters in
   live interviews.
3. **Break it on purpose**: change a parameter, remove a step, feed bad data. Predict
   what happens, then run it. Being *unsurprised* is understanding.
4. **Explain out loud**: record a 60-second voice note explaining the concept as if to
   a manager. Listen back. This is the single fastest confidence builder.
5. **Rebuild from scratch**: once per phase, open a blank notebook and reproduce a full
   workflow (load → clean → EDA → model → evaluate) on a new dataset (Kaggle/DrivenData).

## Suggested external practice sets

- **SQL**: LeetCode (Database, Easy→Medium), StrataScratch, DataLemur.
- **Python/pandas**: StrataScratch, Kaggle "Pandas" micro-course problems.
- **ML**: Kaggle Titanic → House Prices → your own DrivenData re-entry.
- **Stats/A-B**: explain a p-value, design an A/B test for a feature you use daily.
- **Bash/Git (M20)**: on a real CSV, get a value-count with only `cut|sort|uniq -c`;
  make a repo, branch, commit 3 times, merge; write a `.gitignore` from memory.
- **Spark (M21)**: rewrite one pandas `groupby` from an earlier module in PySpark and
  as Spark SQL; explain which lines are transformations vs actions.
- **Deployment/MLOps (M22)**: `joblib`-save a model you trained in M06–M09, write its
  `/predict` function + `Dockerfile`, and describe one data-drift and one concept-drift
  scenario for it.

There is no separate solutions file on purpose: the *worked examples inside each
notebook are the solutions template*. If you get stuck, re-read the section above the
exercise — the answer pattern is always there.
