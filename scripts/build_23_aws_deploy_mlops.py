"""Builder for Module 23: AWS Deployment & MLOps (4-layer)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 23 — Deployment & MLOps (getting the model OUT of the notebook)

A model that lives only in a notebook creates **zero** business value. This final
module closes the last mile: **serialize -> serve -> containerize -> deploy ->
monitor**. We build a *real, runnable* local prediction service and explain exactly how
each piece maps onto AWS — the cloud you'll most likely meet in interviews.

**What you'll be able to do by the end:**
- **Persist** a trained model with `joblib` (and why to deploy the whole Pipeline).
- Wrap it in a **prediction API** (Flask/FastAPI shape) and call it locally.
- Read a real **Dockerfile** and know why containers matter.
- Speak the **AWS map**: S3, EC2, Lambda, ECR, SageMaker, API Gateway.
- Explain **MLOps**: reproducibility, CI/CD, monitoring, and **data/concept drift**.
""")

nb.plain(r"""
So far you've built models. This module is about *shipping* them — turning a trained
model into a live service other apps can call, running 24/7 in the cloud. The journey
has four plain steps: (1) **save** the model to a file, (2) put it behind a little web
**API** so anything can ask it for predictions, (3) pack it in a **container** so it
runs the same everywhere, and (4) **watch** it in production because models quietly go
stale as the world changes. That last-mile skill is what separates "I can train a model"
from "I can run one in production".
""")

nb.md(r"""
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

ARTIFACTS = tempfile.mkdtemp(prefix="m23_")
print("artifact dir:", ARTIFACTS)
""")

# ---------------------------------------------------------------------------
# 23.1 Train a pipeline
# ---------------------------------------------------------------------------
nb.md(r"""
## 23.1 Train something worth deploying (a full Pipeline)
""")

nb.warn(r"""
Deploy the **whole Pipeline** (preprocessing + model), never just the estimator. If you
ship only the model, you must re-implement scaling/encoding at serving time — and any
tiny mismatch causes **training/serving skew**, a top source of silent production bugs.
The Pipeline guarantees the exact same transforms run in production as in training.
""")

nb.jargon("Training/serving skew", "when preprocessing in production differs from training, quietly corrupting predictions")

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

nb.takeaway("Deploy the whole Pipeline (preprocessing + model) so production applies the exact same transforms as training.")

# ---------------------------------------------------------------------------
# 23.2 Serialize
# ---------------------------------------------------------------------------
nb.md(r"""
## 23.2 Serialize the model — `joblib` (the sklearn standard)

Serialization saves the fitted object to disk so a *different* process (the API server)
can load it without retraining. `joblib` beats raw `pickle` for scikit-learn because it
stores large NumPy arrays far more efficiently.
""")

nb.analogy(r"""
Serializing is like **freezing a cooked meal**. You did the hard work (training) once;
freezing (`joblib.dump`) lets you reheat it later (`joblib.load`) in a totally different
kitchen (the server) without cooking from scratch. Save the *recipe card* next to it
(metadata: versions, feature list) so anyone can reheat it correctly.
""")

nb.jargon("Serialization", "saving a trained object to a file so another process can reload it without retraining")

nb.code(r"""
model_path = os.path.join(ARTIFACTS, "churn_model_v1.joblib")
joblib.dump(model, model_path)
print("saved:", os.path.basename(model_path), f"({os.path.getsize(model_path)} bytes)")

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

nb.readcode(r"""
- `joblib.dump(model, path)` freezes the fitted Pipeline to a file.
- `joblib.load(path)` reloads it in a fresh process and it predicts immediately — no
  retraining.
- We also save `metadata.json`: model version, feature list, library version, and the
  **training feature means** — the baseline we'll later compare against to detect drift.
""")

nb.warn(r"""
Only ever `joblib.load` files you **trust** — deserialization can execute arbitrary
code, so a malicious artifact is a security hole. And a model is only reproducible with
a **matching scikit-learn version**, which is why we log it in the metadata.
""")

nb.takeaway("Save the fitted Pipeline with joblib + a metadata file (versions, features, baseline stats) for reproducibility and drift checks.")

# ---------------------------------------------------------------------------
# 23.3 API
# ---------------------------------------------------------------------------
nb.md(r"""
## 23.3 Wrap it in a prediction API (Flask / FastAPI shape)

A model server exposes an HTTP **endpoint** (e.g. `POST /predict`) that accepts JSON
features and returns a JSON prediction. Here's the real Flask app you'd deploy — we
write it to a file so you can read it, then **run its core logic inline** so this
notebook executes without a running server.
""")

nb.analogy(r"""
An API is a **restaurant order window**. The kitchen (your model) stays hidden; the
window (`/predict`) takes a written order (JSON features) and hands back a dish (the
prediction JSON). Callers never touch the kitchen — they just use the window. That clean
separation is why any app, in any language, can use your model over HTTP.
""")

nb.jargon("API endpoint", "a URL (like POST /predict) that accepts a request and returns a response")

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

nb.readcode(r"""
- `MODEL = joblib.load(...)` runs **once at startup**, not per request — loading on
  every call would be painfully slow.
- `/health` is a tiny endpoint load balancers ping to check the server is alive.
- `/predict` reads JSON features, arranges them in the right column order, gets a
  probability, and returns a clean JSON verdict. That's the whole serving contract.
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

nb.deeper(r"""
**FastAPI vs Flask:** FastAPI adds automatic request **validation** (via Pydantic),
async support, and auto-generated interactive docs — increasingly the default for ML
APIs. Same shape: load the model once at startup, define `/predict`, return JSON. For
heavy traffic you run it behind **gunicorn/uvicorn** with multiple worker processes so
requests are handled in parallel.
""")

nb.takeaway("A prediction API loads the model once, exposes POST /predict (+ /health), and returns JSON -- any app can then call your model over HTTP.")

# ---------------------------------------------------------------------------
# 23.4 Docker
# ---------------------------------------------------------------------------
nb.md(r"""
## 23.4 Docker — "it works on my machine" for everyone

A **container** packages your code + libraries + runtime into one portable image, so it
runs identically on your laptop, a teammate's, and AWS. This kills dependency hell — the
#1 deployment headache.
""")

nb.analogy(r"""
A container is a **shipping container** for software. Before standardized containers,
loading a ship was chaos — every cargo was a different shape. Standardize the box and
any crane, truck, or ship handles it identically. Docker does that for code: whatever's
inside, the "box" runs the same on every machine.
""")

nb.jargon("Container", "a portable box bundling your code + libraries + runtime so it runs identically everywhere")
nb.jargon("Dockerfile", "a recipe listing the steps to build a container image")

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

nb.readcode(r"""
- `FROM python:3.11-slim` starts from a lean official Python image.
- `COPY requirements.txt` + `RUN pip install` bakes in the exact libraries.
- `COPY app.py churn_model_v1.joblib` bundles the code *and* the frozen model.
- `EXPOSE 8080` + `CMD gunicorn ...` launch a production server when the container runs.
- `docker run -p 8080:8080` maps the container's port 8080 to your machine's 8080 so you
  can `curl` it. That image now runs identically on AWS.
""")

nb.takeaway("Docker packages code + libs + model into one image that runs identically on your laptop and in the cloud -- no more dependency hell.")

# ---------------------------------------------------------------------------
# 23.5 AWS map
# ---------------------------------------------------------------------------
nb.md(r"""
## 23.5 The AWS map — what each service is *for*

You don't need to memorize AWS, but you must speak the vocabulary. The essentials for a
data scientist:

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
""")

nb.plain(r"""
Three common ways to deploy — pick by traffic and how much ops work you want:
1. **Container path**: Docker image -> push to **ECR** -> run on **ECS/Fargate** ->
   front with **API Gateway**. Maximum control.
2. **Managed path**: **SageMaker** — `model.deploy()` gives a scalable HTTPS endpoint
   with monitoring built in. Least ops work.
3. **Serverless path**: small model -> **Lambda** + **API Gateway**. Pay per request,
   scales to zero. Great for spiky, low-volume workloads.
""")

nb.code(r"""
# Illustrate the S3 artifact convention (paths only -- no real upload/network).
bucket = "s3://my-ml-artifacts"
print("Typical S3 layout for model artifacts:")
for p in ["models/churn/v1/churn_model_v1.joblib",
          "models/churn/v1/metadata.json",
          "data/raw/customers_2024-08.parquet",
          "logs/predictions/2024-08-24/*.jsonl"]:
    print(" ", f"{bucket}/{p}")
print("\n# boto3 (the AWS SDK) upload -- pattern only:")
print("import boto3; boto3.client('s3').upload_file(")
print("    'churn_model_v1.joblib', 'my-ml-artifacts',")
print("    'models/churn/v1/churn_model_v1.joblib')")
""")

nb.takeaway("Know the AWS map: S3 (storage), EC2/ECS/Fargate (compute), ECR (images), Lambda+API Gateway (serverless), SageMaker (managed endpoints).")

# ---------------------------------------------------------------------------
# 23.6 MLOps + drift
# ---------------------------------------------------------------------------
nb.md(r"""
## 23.6 MLOps — keeping a deployed model *healthy*

Deployment isn't the finish line; models **decay** as the world changes. MLOps is the
discipline of running ML reliably.
""")

nb.plain(r"""
Think of a deployed model like a car: shipping it is just driving off the lot. It needs
ongoing care — MLOps is that maintenance. Four pillars:
- **Reproducibility**: pin versions, seed randomness, version data + models so anyone
  can rebuild the exact model.
- **CI/CD**: automated tests + retraining + deploy on every change.
- **Monitoring**: watch latency, errors, and prediction quality once true labels arrive.
- **Drift detection**: catch the world changing under your model (the big one).
""")

nb.jargon("MLOps", "the practice of deploying, monitoring and maintaining ML models reliably in production")
nb.jargon("Data (covariate) drift", "the input distribution shifts -- the model sees inputs unlike its training data")
nb.jargon("Concept drift", "the input-to-target relationship changes -- the same inputs now mean something different")

nb.analogy(r"""
**Data drift** vs **concept drift**, plainly: imagine predicting ice-cream sales from
temperature. *Data drift* = your city suddenly gets way hotter summers (the *inputs*
shifted). *Concept drift* = a new health craze means people now buy less ice cream even
on hot days (the *rule linking* temperature to sales changed). Both hurt the model, but
you fix them differently — so you name them separately.
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

nb.readcode(r"""
- We compare each feature's **live** mean to the **training baseline** we saved in the
  metadata.
- `abs_shift` is how far each feature has moved; a simple threshold flags big movers.
- Here `f0` and `f2` (which we deliberately shifted) get flagged — the signal to
  investigate and possibly retrain. Real tools use statistical tests, but this *is* the
  idea.
""")

nb.deeper(r"""
Production-grade tools — `evidently`, `whylogs`, or SageMaker **Model Monitor** — do
this with proper statistical tests (PSI, KS-test) and dashboards. But the *concept* is
exactly the comparison above: live stats vs a saved baseline. Interviewers care far more
that you understand **why** monitoring exists than that you can name a specific library.
""")

nb.interview(r"""
"Shipping the model is the start, not the end. I serialize the full pipeline, serve it
behind a versioned API in a container, and monitor input drift against a saved training
baseline plus live accuracy once labels land — retraining when drift or performance
degradation crosses a threshold."
""")

nb.takeaway("MLOps = reproducibility + CI/CD + monitoring + drift detection; a deployed model is a living system that decays and must be watched.")

nb.code(r"""
import shutil
shutil.rmtree(ARTIFACTS, ignore_errors=True)
print("cleaned up temporary artifacts.")
""")

# ---------------------------------------------------------------------------
# Practice + capstone summary
# ---------------------------------------------------------------------------
nb.md(r"""
## 23.7 Practice
""")

nb.try_this(r"""
1. Re-serialize the model as `v2` after retraining; how would you roll back to `v1` in
   production if `v2` underperforms?
2. Convert the Flask app to **FastAPI** with a Pydantic model validating the 5 features
   are floats.
3. Write the `docker build` / `docker run` commands and explain what `-p 8080:8080` does.
4. Describe one realistic **data drift** and one **concept drift** scenario for the churn
   model, and how you'd detect each.
5. Which AWS path (container / SageMaker / Lambda) fits: (a) 5 requests/day, (b) steady
   1000 req/s, (c) a team already on SageMaker? Justify each.
""")

nb.md(r"""
## Summary

- Deploy the **whole Pipeline**; serialize with **joblib** + save **metadata**
  (versions, feature baseline) for reproducibility and drift checks.
- Serve via a **`/predict` API** (Flask/FastAPI): load model once, return JSON.
- **Docker** makes it portable; the same image runs on your laptop and AWS.
- **AWS map:** S3 (storage), EC2/ECS/Fargate (compute), ECR (images), Lambda+API Gateway
  (serverless), **SageMaker** (managed endpoints), IAM (access), CloudWatch (monitoring).
- **MLOps** = reproducibility + CI/CD + monitoring + **drift detection** (data vs
  concept). A model is a *living* system, not a one-off artifact.

**This completes the bootcamp.** You can now take an idea from messy CSV -> cleaned data
-> model -> *deployed, monitored service* — and explain every step. That end-to-end
story is exactly what turns interviews into offers. Congratulations, Stephen — go build!
""")

out = nb.save("notebooks/23_aws_deployment_mlops.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
