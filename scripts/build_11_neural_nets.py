"""Builder for Module 11: Neural Networks intro."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 11 — Neural Networks (from a neuron to a trained net)

Neural nets sound intimidating; they're not. A neural net is just **logistic
regression stacked and repeated**, with a clever way to train it (**backprop**).
Because your foundation is now solid, this will feel like a natural next step
rather than magic. We'll build a tiny net **from scratch in NumPy** so you *see*
forward + backward passes, then use scikit-learn's `MLPClassifier`.

Goals:
- The artificial **neuron**: weighted sum → **activation**.
- Why we need **non-linear** activations (ReLU, sigmoid, tanh).
- **Forward pass**, **loss**, **backpropagation**, **gradient descent** — the
  training loop.
- Train a real (small) net; understand epochs, learning rate, overfitting.
- Where deep learning fits vs classical ML (be honest in interviews).
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
""")

nb.md(r"""
## 11.1 A single neuron

A neuron computes `z = w·x + b` (a weighted sum plus bias — identical to linear
regression), then applies an **activation** `a = f(z)`. With a sigmoid activation,
a single neuron **is** logistic regression. Stacking many neurons in layers is what
gives networks their power.
""")

nb.code(r"""
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z):    return np.maximum(0, z)
def tanh(z):    return np.tanh(z)

z = np.linspace(-6, 6, 200)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for a, f, name in zip(ax, [sigmoid, tanh, relu], ["sigmoid","tanh","relu"]):
    a.plot(z, f(z)); a.set_title(name); a.axhline(0, color="gray", lw=0.5)
plt.suptitle("Activation functions add the non-linearity", y=1.03)
plt.tight_layout(); plt.show()
""")

nb.md(r"""
**Why non-linear activations are essential:** if every layer were linear, stacking
them would collapse into a *single* linear map — no matter how deep, you'd only
ever fit a line/plane. Non-linear activations let the network bend decision
boundaries and approximate very complex functions (the *universal approximation*
idea).
""")

nb.md(r"""
## 11.2 A network from scratch — learn XOR (the classic non-linear problem)

XOR can't be solved by a single linear model (it's not linearly separable). A tiny
2-layer net can. We implement forward pass, binary cross-entropy loss, and
**backpropagation** (the chain rule computing gradients layer by layer), then train
with gradient descent.
""")

nb.code(r"""
# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)      # XOR truth table

# Network: 2 inputs -> 4 hidden (tanh) -> 1 output (sigmoid)
def init(n_in, n_hidden, n_out, seed=1):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, 1, (n_in, n_hidden)),
        "b1": np.zeros((1, n_hidden)),
        "W2": r.normal(0, 1, (n_hidden, n_out)),
        "b2": np.zeros((1, n_out)),
    }

def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.tanh(z1)                 # hidden activation
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = sigmoid(z2)                 # output probability
    cache = (X, z1, a1, z2, a2)
    return a2, cache

def bce(y, yhat, eps=1e-9):
    return -np.mean(y*np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))

def backward(p, cache, y):
    X, z1, a1, z2, a2 = cache
    m = X.shape[0]
    dz2 = (a2 - y) / m                       # d loss / d z2 (sigmoid+BCE simplifies)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(0, keepdims=True)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (1 - np.tanh(z1)**2)         # tanh derivative
    dW1 = X.T @ dz1
    db1 = dz1.sum(0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

params = init(2, 4, 1)
lr = 0.5
losses = []
for epoch in range(5000):
    yhat, cache = forward(params, X)
    losses.append(bce(y, yhat))
    grads = backward(params, cache, y)
    for k in params:                          # gradient descent step
        params[k] -= lr * grads[k]

final, _ = forward(params, X)
print("Learned XOR predictions (rounded):", final.round(2).ravel())
print("Targets                          :", y.ravel())
""")

nb.code(r"""
plt.figure(figsize=(7,4))
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("loss (binary cross-entropy)")
plt.title("Training loss falls as the net learns XOR"); plt.show()
print(f"final loss: {losses[-1]:.4f}  (started at {losses[0]:.4f})")
""")

nb.md(r"""
**What just happened (the training loop, in words):**
1. **Forward pass** — push inputs through layers to get a prediction.
2. **Loss** — measure how wrong it is (cross-entropy).
3. **Backward pass (backprop)** — use the **chain rule** to compute how each weight
   contributed to the error (the gradient).
4. **Update** — nudge every weight *against* its gradient (gradient descent),
   scaled by the **learning rate**.
5. Repeat for many **epochs** (passes over the data) until loss stops falling.

That five-step loop is *all* of deep learning, from this toy net to GPT.
""")

nb.md(r"""
## 11.3 The same, professionally — scikit-learn's MLPClassifier

You won't hand-code backprop at work; you'll use a library. Here's a real net on
our churn data. The concepts (hidden layers, activation, learning rate, epochs)
map one-to-one to what we just built.
""")

nb.code(r"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("../data/customers_clean.csv")
feats = ["age","income","tenure_months","monthly_spend","support_calls"]
X = df[feats]; y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)

# Scaling is ESSENTIAL for neural nets (gradient descent behaves badly otherwise)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                    alpha=1e-3, max_iter=800, random_state=42)
mlp.fit(X_tr_s, y_tr)
print("test AUC:", round(roc_auc_score(y_te, mlp.predict_proba(X_te_s)[:,1]), 3))
print(classification_report(y_te, mlp.predict(X_te_s), target_names=["stay","churn"]))
""")

nb.code(r"""
plt.figure(figsize=(7,4))
plt.plot(mlp.loss_curve_)
plt.xlabel("iteration"); plt.ylabel("training loss")
plt.title("MLP training loss curve"); plt.show()
""")

nb.md(r"""
## 11.4 Key hyperparameters & concepts (interview-ready)

- **Architecture**: `hidden_layer_sizes` — width & depth = model capacity.
- **Activation**: **ReLU** is the modern default (fast, avoids vanishing
  gradients); sigmoid/tanh for special cases; **softmax** on the output for
  multi-class.
- **Learning rate**: too high → diverges/oscillates; too low → crawls. The single
  most important knob.
- **Epochs / early stopping**: train until validation loss stops improving; stop
  to avoid overfitting.
- **Regularization**: `alpha` (L2), **dropout** (in deep-learning libraries),
  more data.
- **Optimizer**: Adam (adaptive) is the common default over plain SGD.
""")

nb.md(r"""
## 11.5 When to use deep learning (be honest — interviewers respect this)

- **Use classical ML (trees/boosting)** for most **tabular** business data — it's
  usually as good or better, faster, and more interpretable. (Your water-wells and
  churn problems live here.)
- **Use deep learning** for **unstructured** data at scale: images (CNNs), text /
  sequences (Transformers), audio, and where you have lots of data + compute.

Saying "I'd start with XGBoost on tabular data and only reach for a neural net when
the data or scale demands it" signals maturity, not ignorance.
""")

nb.md(r"""
## 11.6 Mini-exercises

1. In the from-scratch net, drop hidden units to 1. Can it still learn XOR? Why not?
2. Change the learning rate to 5.0 and to 0.01 — describe the loss curve in each.
3. On the churn MLP, try `hidden_layer_sizes=(64,64,64)`. Does test AUC improve or
   does it overfit?
4. Explain backpropagation to an interviewer in three sentences.
""")

nb.md(r"""
## Summary

- A neuron = weighted sum + **non-linear activation**; a sigmoid neuron *is*
  logistic regression. Non-linearity is what makes depth meaningful.
- Training = **forward → loss → backprop (chain rule) → gradient-descent update**,
  repeated over epochs. That loop scales from XOR to LLMs.
- **Always scale** inputs; tune **learning rate**, architecture, regularization;
  use **early stopping**.
- **Classical ML usually wins on tabular data**; deep learning shines on images,
  text, and large unstructured datasets.

Next: **Module 12 — Job Readiness & Interview Prep** — turning knowledge into
offers.
""")

out = nb.save("notebooks/11_neural_networks_intro.ipynb")
print("saved", out)
