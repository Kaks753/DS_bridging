"""Builder for Module 22: AWS Deployment & MLOps."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 22 — Deployment & MLOps (getting the model OUT of the notebook)

A model that lives only in a notebook creates **zero** business value. This module
closes the last mile: **serialize → serve → containerize → deploy → monitor**. We
build a *real, runnable* local prediction service and explain exactly how each piece
maps onto AWS — the cloud you'll most likely meet in interviews.

Goals:
- **Persist** a trained model with `joblib` (and why *not* pickle raw).
- Wrap it in a **prediction API** (Flask/FastAPI shape) and *call it locally*.
- **Docker** in one page: why containers, and a real `Dockerfile`.
- The **AWS map**: S3, EC2, Lambda, ECR, SageMaker, API Gateway — what each does.
- **MLOps**: reproducibility, CI/CD, monitoring, and **data/concept drift**.

> No AWS account or network is needed: we run the serving logic **locally** so every
> cell executes, then annotate precisely what changes in the cloud.
""")

nb.code(r"""
import os, json, tempfile, numpy as np, pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ARTIFACTS = tempfile.mkdtemp(prefix="m22_")
print("artifact dir:", ARTIFACTS)
""")

nb.md(r"""
## 22.1 Train something worth deploying (a full Pipeline)

Deploy the **whole Pipeline** (preprocessing + model), never just the estimator. If
you deploy only the model, you must re-implement scaling/encoding at serving time —
a top source of **training/serving skew** bugs. The Pipeline guarantees the exact
same transforms run in production as in training.
""")

nb.code(r"""
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                           random_state=42)
feat_names = [f"f{i}" for i in range(5)]
X = pd.DataFrame(X, columns=feat_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("scaler", StandardScaler()),           # preprocessing travels WITH the model
    ("clf", LogisticRegression(max_iter=1000)),
]).fit(X_tr, y_tr)

print("test accuracy:", round(model.score(X_te, y_te), 3))
""")

nb.md(r"""
## 22.2 Serialize the model — `joblib` (the sklearn standard)

Serialization saves the fitted object to disk so a *different* process (the API
server) can load it without retraining. `joblib` is preferred over raw `pickle` for
scikit-learn because it stores large NumPy arrays far more efficiently.

**Production hygiene to state in interviews:**
- Version the artifact (`model_v3.joblib`) and log the training data + library
  versions (a model is only reproducible with matching `scikit-learn`).
- Security: only ever `joblib.load` files you **trust** (deserialization can execute
  code). Never load an artifact from an untrusted source.
""")

nb.code(r"""
model_path = os.path.join(ARTIFACTS, "churn_model_v1.joblib")
joblib.dump(model, model_path)
print("saved:", os.path.basename(model_path),
      f"({os.path.getsize(model_path)} bytes)")

# A fresh process would load it exactly like this:
loaded = joblib.load(model_path)
print("reloaded model predicts:", loaded.predict(X_te.iloc[:5]).tolist())

# Save metadata alongside the model (crucial for reproducibility & drift checks):
import sklearn
metadata = {
    "model_version": "v1",
    "features": feat_names,
    "sklearn_version": sklearn.__version__,
    "train_rows": int(len(X_tr)),
    "test_accuracy": round(float(model.score(X_te, y_te)), 4),
    "train_feature_means": X_tr.mean().round(4).to_dict(),  # baseline for drift
}
json.dump(metadata, open(os.path.join(ARTIFACTS, "metadata.json"), "w"), indent=2)
print("\nmetadata.json:"); print(json.dumps(metadata, indent=2)[:400], "...")
""")

nb.md(r"""
## 22.3 Wrap it in a prediction API (Flask / FastAPI shape)

A model server exposes an HTTP **endpoint** (e.g. `POST /predict`) that accepts JSON
features and returns a JSON prediction. Here's the real Flask app you'd deploy —
we write it to a file so you can read it, then we **run its core logic inline** so
this notebook executes without a running server.
""")

nb.code(r"""
app_code = '''\
# app.py  -- a minimal model-serving API (Flask)
import joblib, pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL = joblib.load("churn_model_v1.joblib")   # load ONCE at startup, not per request
FEATURES = ["f0", "f1", "f2", "f3", "f4"]

@app.route("/health")                          # liveness probe for load balancers
def health():
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()               # {"f0":..., ..., "f4":...}
    X = pd.DataFrame([payload])[FEATURES]       # order features consistently
    proba = float(MODEL.predict_proba(X)[0, 1])
    return jsonify(prediction=int(proba >= 0.5), probability=round(proba, 4))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)         # 0.0.0.0 so it's reachable in a container
'''
open(os.path.join(ARTIFACTS, "app.py"), "w").write(app_code)
print(app_code)
""")

nb.code(r"""
# Run the SAME logic inline (no server needed) to prove the request/response cycle:
def predict_endpoint(payload: dict, model, features):
    "Exactly what the Flask /predict route does, callable in-process."
    X = pd.DataFrame([payload])[features]
    proba = float(model.predict_proba(X)[0, 1])
    return {"prediction": int(proba >= 0.5), "probability": round(proba, 4)}

incoming_json = {f: float(v) for f, v in zip(feat_names, X_te.iloc[0])}
print("REQUEST  JSON:", json.dumps(incoming_json))
print("RESPONSE JSON:", json.dumps(predict_endpoint(incoming_json, loaded, feat_names)))
""")

nb.md(r"""
**FastAPI vs Flask:** FastAPI adds automatic request **validation** (via Pydantic),
async, and auto-generated docs — increasingly the default for ML APIs. Same shape:
load model once at startup, define `/predict`, return JSON. For heavy traffic you'd
run it behind **gunicorn/uvicorn** with multiple workers.
""")

nb.md(r"""
## 22.4 Docker — "it works on my machine" for everyone

A **container** packages your code + libraries + runtime into one portable image, so
it runs identically on your laptop, a teammate's, and AWS. This solves dependency
hell (the #1 deployment headache). The real `Dockerfile` for our API:
""")

nb.code(r"""
dockerfile = '''\
# Dockerfile
FROM python:3.11-slim                 # small, official base image
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # install deps
COPY app.py churn_model_v1.joblib .   # code + the model artifact
EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "app:app"]  # production server
'''
open(os.path.join(ARTIFACTS, "Dockerfile"), "w").write(dockerfile)
print(dockerfile)
print("Build & run locally:")
print("  docker build -t churn-api .")
print("  docker run -p 8080:8080 churn-api")
print("  curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \\\\")
print("       -d '{\"f0\":0.1,\"f1\":-0.3,\"f2\":1.2,\"f3\":0.0,\"f4\":-1.1}'")
""")

nb.md(r"""
## 22.5 The AWS map — what each service is *for*

You don't need to memorize AWS, but you must speak the vocabulary. The essentials
for a data scientist:

| Service | One-line purpose | DS use |
|---|---|---|
| **S3** | object storage (buckets) | store datasets, model artifacts, logs |
| **EC2** | virtual servers | run training / a persistent API host |
| **ECR** | Docker image registry | push your container image before deploy |
| **Lambda** | run code with **no server** (event-driven) | cheap, bursty predictions |
| **API Gateway** | managed HTTP front door | expose Lambda/containers as an API |
| **SageMaker** | end-to-end managed ML | train, host **endpoints**, monitor, autoscale |
| **ECS / Fargate** | run containers (serverless) | host the Docker API without managing EC2 |
| **IAM** | permissions & roles | who/what can access which resource (security) |
| **CloudWatch** | logs & metrics | monitor latency, errors, custom drift metrics |

**Two common deployment paths (name both):**
1. **Container path:** Docker image → push to **ECR** → run on **ECS/Fargate** (or
   **EKS**) → front with **API Gateway/ALB**. Maximum control.
2. **Managed path:** **SageMaker** — `model.deploy()` gives you a scalable HTTPS
   **endpoint** with monitoring baked in. Less ops work.
3. **Serverless path:** small model → **Lambda** + **API Gateway**. Pay per request,
   scales to zero. Great for spiky, low-volume workloads.
""")

nb.code(r"""
# Illustrate the S3 artifact convention (paths only — no real upload/network).
bucket = "s3://my-ml-artifacts"
print("Typical S3 layout for model artifacts:")
for p in ["models/churn/v1/churn_model_v1.joblib",
          "models/churn/v1/metadata.json",
          "data/raw/customers_2024-08.parquet",
          "logs/predictions/2024-08-24/*.jsonl"]:
    print(" ", f"{bucket}/{p}")
print("\n# boto3 (the AWS SDK) upload — pattern only:")
print("import boto3; boto3.client('s3').upload_file(")
print("    'churn_model_v1.joblib', 'my-ml-artifacts',")
print("    'models/churn/v1/churn_model_v1.joblib')")
""")

nb.md(r"""
## 22.6 MLOps — keeping a deployed model *healthy*

Deployment isn't the finish line; models **decay** as the world changes. MLOps is the
discipline of running ML reliably.

**Pillars (mention these):**
- **Reproducibility:** pin library versions, seed randomness, version data + models
  (tools: DVC, MLflow, Git). Anyone can rebuild the exact model.
- **CI/CD:** automated tests + retraining + deploy on every change (GitHub Actions).
- **Monitoring:** track latency, error rate, and **prediction quality** once true
  labels arrive.
- **Drift detection:** the big one below.

### Data drift vs concept drift
- **Data (covariate) drift:** the *input distribution* shifts (e.g. a new customer
  demographic). The model sees inputs unlike its training data.
- **Concept drift:** the *relationship* between inputs and target changes (e.g. a
  recession flips what predicts churn). Even the same inputs now mean something else.

You detect data drift by comparing **live feature stats to the training baseline**
(which is why we saved `train_feature_means` in the metadata). A large shift → alert
→ investigate → maybe retrain.
""")

nb.code(r"""
# Simple data-drift check: compare live feature means to the training baseline.
baseline = pd.Series(metadata["train_feature_means"])

# Simulate a month of live traffic that has DRIFTED on f0 and f2:
rng = np.random.default_rng(0)
live = X_te.copy()
live["f0"] = live["f0"] + 2.0     # a real shift
live["f2"] = live["f2"] * 1.8     # scale change

live_means = live.mean()
drift = pd.DataFrame({
    "train_mean": baseline,
    "live_mean": live_means.round(4),
    "abs_shift": (live_means - baseline).abs().round(4),
})
drift["ALERT"] = drift["abs_shift"] > 0.5     # simple threshold rule
print(drift)
print("\nFeatures flagged for drift:",
      drift.index[drift["ALERT"]].tolist(),
      "-> investigate / consider retraining.")
""")

nb.md(r"""
**Production-grade tools:** `evidently`, `whylogs`, or SageMaker **Model Monitor**
do this with statistical tests (PSI, KS-test) and dashboards — but the *idea* is
exactly the comparison above. Showing you understand **why** monitoring exists beats
naming any tool.

**Interview soundbite:**
> "Shipping the model is the start, not the end. I serialize the full pipeline,
> serve it behind a versioned API in a container, and monitor input drift against a
> saved training baseline plus live accuracy once labels land — retraining when
> drift or performance degradation crosses a threshold."
""")

nb.code(r"""
import shutil
shutil.rmtree(ARTIFACTS, ignore_errors=True)
print("cleaned up temporary artifacts.")
""")

nb.md(r"""
## 22.7 Mini-exercises

1. Re-serialize the model as `v2` after retraining on more data; how would you roll
   back to `v1` in production if `v2` underperforms?
2. Convert the Flask app to **FastAPI** with a Pydantic request model that validates
   the 5 features are floats.
3. Write the `docker build` / `docker run` commands and explain what `-p 8080:8080`
   does.
4. Describe, for the churn model, one realistic **data drift** and one **concept
   drift** scenario, and how you'd detect each.
5. Which AWS path (container / SageMaker / Lambda) fits: (a) 5 requests/day,
   (b) steady 1000 req/s, (c) a team already on SageMaker? Justify each.

## Summary

- Deploy the **whole Pipeline**; serialize with **joblib** + save **metadata**
  (versions, feature baseline) for reproducibility and drift checks.
- Serve via a **`/predict` API** (Flask/FastAPI): load model once, return JSON.
- **Docker** makes it portable; the same image runs on your laptop and AWS.
- **AWS map:** S3 (storage), EC2/ECS/Fargate (compute), ECR (images), Lambda+API
  Gateway (serverless), **SageMaker** (managed endpoints), IAM (access), CloudWatch
  (monitoring).
- **MLOps** = reproducibility + CI/CD + monitoring + **drift detection** (data vs
  concept). A model is a *living* system, not a one-off artifact.

**This completes the bootcamp's engineering track.** You can now take an idea from
messy CSV → cleaned data → model → *deployed, monitored service* — and explain every
step. That end-to-end story is exactly what turns interviews into offers.
""")

out = nb.save("notebooks/22_aws_deployment_mlops.ipynb")
print("saved", out)
