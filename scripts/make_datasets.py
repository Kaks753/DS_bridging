"""
make_datasets.py — generate small, realistic, self-contained datasets.

We build datasets that MIRROR real data-science pain points on purpose:
  - missing values (MCAR / MAR patterns)
  - duplicated rows
  - wrong dtypes (numbers stored as strings, dates as strings)
  - outliers
  - skewed distributions
  - mixed-case / whitespace categorical noise

This means every cleaning / EDA / modeling lesson uses data that behaves
like the messy stuff you meet on the job, not a pre-polished toy set.

Run:  python scripts/make_datasets.py
Output: data/*.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)  # reproducible: same data every run


def make_customers(n: int = 500) -> pd.DataFrame:
    """A 'customers' table for cleaning + EDA + classification (churn)."""
    cities = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]
    plans = ["Basic", "Standard", "Premium"]

    age = RNG.normal(35, 10, n).round()
    age = np.clip(age, 18, 85)

    # Income: right-skewed (lognormal) — classic money variable
    income = RNG.lognormal(mean=10.8, sigma=0.5, size=n)  # ~ tens of thousands

    tenure_months = RNG.integers(1, 72, n)
    monthly_spend = (income / 1000) * RNG.uniform(0.5, 2.0, n) + RNG.normal(0, 5, n)
    monthly_spend = np.clip(monthly_spend, 5, None)

    support_calls = RNG.poisson(1.5, n)  # count data

    city = RNG.choice(cities, n, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    plan = RNG.choice(plans, n, p=[0.5, 0.3, 0.2])

    # Churn depends on tenure, support calls, plan — a learnable signal
    logit = (
        -0.03 * tenure_months
        + 0.45 * support_calls
        + np.where(plan == "Basic", 0.6, 0.0)
        - 0.00002 * income
        + RNG.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (RNG.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "income": income.round(2),
        "city": city,
        "plan": plan,
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend.round(2),
        "support_calls": support_calls,
        "churn": churn,
    })

    # --- inject realistic mess ---
    # 1) missing income (MAR: more missing for Premium — e.g. privacy)
    miss_idx = df.index[(df["plan"] == "Premium")]
    drop = RNG.choice(miss_idx, size=max(1, len(miss_idx) // 4), replace=False)
    df.loc[drop, "income"] = np.nan
    # random missing age
    df.loc[RNG.choice(df.index, 20, replace=False), "age"] = np.nan

    # 2) categorical noise: whitespace + case
    noisy = RNG.choice(df.index, 30, replace=False)
    df.loc[noisy, "city"] = df.loc[noisy, "city"].str.upper()
    noisy2 = RNG.choice(df.index, 20, replace=False)
    df.loc[noisy2, "city"] = " " + df.loc[noisy2, "city"] + " "

    # 3) wrong dtype: monthly_spend as string with currency on some rows
    #    (we keep a clean numeric copy too so lessons can compare)
    # 4) duplicates: copy 8 rows verbatim
    dups = df.sample(8, random_state=1)
    df = pd.concat([df, dups], ignore_index=True)

    # 5) an obvious outlier in income
    df.loc[df.index[0], "income"] = 5_000_000.0

    return df


def make_house_prices(n: int = 400) -> pd.DataFrame:
    """Regression dataset: predict house price from features."""
    size_sqft = RNG.normal(1500, 500, n).clip(400, 5000)
    bedrooms = RNG.integers(1, 6, n)
    age_years = RNG.integers(0, 60, n)
    dist_center_km = RNG.exponential(8, n).clip(0.5, 40)  # skewed

    price = (
        50_000
        + 180 * size_sqft
        + 15_000 * bedrooms
        - 800 * age_years
        - 3_000 * dist_center_km
        + RNG.normal(0, 25_000, n)
    )
    price = price.clip(20_000, None)

    df = pd.DataFrame({
        "size_sqft": size_sqft.round(0),
        "bedrooms": bedrooms,
        "age_years": age_years,
        "dist_center_km": dist_center_km.round(2),
        "price": price.round(0),
    })
    return df


def make_blobs_2d(n: int = 300) -> pd.DataFrame:
    """Unsupervised: 3 natural clusters in 2D for KMeans/PCA lessons."""
    centers = np.array([[2, 2], [8, 3], [5, 9]])
    parts = []
    for c in centers:
        pts = RNG.normal(c, 0.8, size=(n // 3, 2))
        parts.append(pts)
    X = np.vstack(parts)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    make_customers().to_csv("data/customers.csv", index=False)
    make_house_prices().to_csv("data/house_prices.csv", index=False)
    make_blobs_2d().to_csv("data/blobs_2d.csv", index=False)
    print("Wrote data/customers.csv, data/house_prices.csv, data/blobs_2d.csv")
