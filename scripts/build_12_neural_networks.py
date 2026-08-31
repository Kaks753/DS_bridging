"""Builder for Module 12: Neural Networks (4-layer rewrite of old M11)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 12 — Neural Networks: from a single neuron to a trained net

'Neural network' sounds like a force field. It isn't. A neural net is just
**logistic regression, stacked and repeated**, with one clever trick for training it
(**backpropagation**). Everything you learned about weighted sums, sigmoids, and
gradient descent carries straight over.

To make it concrete rather than mystical, we'll build a tiny network **from scratch
in NumPy** — so you literally watch the forward and backward passes happen — then
switch to scikit-learn's `MLPClassifier` for the professional version.
""")

nb.analogy("A neuron is a tiny committee member: it listens to several inputs, weighs each by "
           "how much it trusts them, adds its own bias, then gives a yes/no-ish opinion. A "
           "neural network is committees feeding into committees — each layer forms opinions "
           "that the next layer reasons about, until the final vote comes out.")

nb.md("## 12.1 Setup")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)
print("ready")
""")

nb.jargon("neuron", "a unit that computes a weighted sum of inputs + bias, then an activation")
nb.jargon("activation function", "the non-linear squashing applied to a neuron's weighted sum")
nb.jargon("layer", "a group of neurons that all read the same inputs and feed the next group")

nb.md("## 12.2 A single neuron")

nb.plain("""
A neuron does two steps. First: z = w·x + b — a weighted sum of the inputs plus a
bias. That's identical to linear regression. Second: it passes z through an
**activation function** that bends the output. If that activation is the sigmoid,
a single neuron IS logistic regression — you've already built one. Stacking many of
them in layers is where the extra power comes from.
""")

nb.code(r"""
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z):    return np.maximum(0, z)
def tanh(z):    return np.tanh(z)

z = np.linspace(-6, 6, 200)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for a, f, name in zip(ax, [sigmoid, tanh, relu], ["sigmoid", "tanh", "relu"]):
    a.plot(z, f(z)); a.set_title(name); a.axhline(0, color="gray", lw=0.5)
plt.suptitle("Activation functions: the non-linearity that makes depth worthwhile", y=1.03)
plt.tight_layout(); plt.show()
""")

nb.readcode("""
- sigmoid squashes to (0,1) — good for probabilities; tanh to (-1,1) — zero-centred.
- ReLU just zeroes out negatives and passes positives through — cheap and the modern
  default for hidden layers.
""")

nb.deeper("""
Why are non-linear activations ESSENTIAL, not optional? Because a stack of purely
linear layers collapses into a single linear layer — a line composed with a line is
still a line. No matter how deep, without non-linearity you could only ever fit a
flat plane. The activation is what lets the network BEND its decision boundaries and,
in principle, approximate almost any function (the 'universal approximation' idea).
Depth without non-linearity is theatre.
""")

nb.jargon("ReLU", "activation that outputs max(0, z); the standard choice for hidden layers")
nb.jargon("bias", "a per-neuron constant added to the weighted sum, shifting the activation")

nb.md("## 12.3 A network from scratch — learning XOR")

nb.plain("""
XOR ('exactly one input is 1') is the classic problem a single linear model CANNOT
solve — you can't separate its classes with one straight line. A tiny two-layer
network can. We'll code the forward pass, the loss, and backpropagation by hand, then
train it and watch it crack XOR. Seeing this once demystifies all of deep learning.
""")

nb.code(r"""
# XOR truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Network shape: 2 inputs -> 4 hidden (tanh) -> 1 output (sigmoid)
def init(n_in, n_hidden, n_out, seed=1):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 1, (n_in, n_hidden)), "b1": np.zeros((1, n_hidden)),
            "W2": r.normal(0, 1, (n_hidden, n_out)), "b2": np.zeros((1, n_out))}

def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.tanh(z1)      # hidden layer
    z2 = a1 @ p["W2"] + p["b2"]; a2 = sigmoid(z2)     # output probability
    return a2, (X, z1, a1, z2, a2)

def bce(y, yhat, eps=1e-9):                            # binary cross-entropy loss
    return -np.mean(y*np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))

def backward(p, cache, y):                            # backprop = chain rule, layer by layer
    X, z1, a1, z2, a2 = cache; m = X.shape[0]
    dz2 = (a2 - y) / m
    dW2 = a1.T @ dz2; db2 = dz2.sum(0, keepdims=True)
    dz1 = (dz2 @ p["W2"].T) * (1 - np.tanh(z1)**2)     # tanh derivative
    dW1 = X.T @ dz1; db1 = dz1.sum(0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

params, lr, losses = init(2, 4, 1), 0.5, []
for epoch in range(5000):
    yhat, cache = forward(params, X)
    losses.append(bce(y, yhat))
    grads = backward(params, cache, y)
    for k in params:                                  # gradient-descent step
        params[k] -= lr * grads[k]

final, _ = forward(params, X)
print("Learned XOR predictions (rounded):", final.round(2).ravel())
print("Targets                          :", y.ravel())
""")

nb.readcode("""
- `forward` pushes inputs through hidden (tanh) then output (sigmoid) to a prediction.
- `backward` applies the chain rule to find how each weight contributed to the error
  (its gradient); `dz2 = a2 - y` is the tidy sigmoid+BCE shortcut.
- The loop repeats forward → loss → backward → nudge-weights 5000 times. By the end
  the net outputs ~0,1,1,0 — it solved the 'impossible for one line' problem.
""")

nb.code(r"""
plt.figure(figsize=(7, 4))
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("loss (binary cross-entropy)")
plt.title("Training loss falls as the net learns XOR"); plt.show()
print(f"final loss: {losses[-1]:.4f}  (started at {losses[0]:.4f})")
""")

nb.takeaway("""
The whole of deep learning — from this toy net to GPT — is FIVE steps on repeat:
(1) forward pass to a prediction, (2) measure the loss, (3) backprop the error via
the chain rule to get gradients, (4) nudge every weight against its gradient scaled
by the learning rate, (5) repeat for many epochs. That's it. Everything else is scale.
""")

nb.jargon("forward pass", "pushing inputs through the layers to produce a prediction")
nb.jargon("backpropagation", "using the chain rule to compute each weight's contribution to the error")
nb.jargon("epoch", "one full pass over the training data")
nb.jargon("learning rate", "how big a step gradient descent takes each update; the key training knob")

nb.md("## 12.4 The same thing, professionally — MLPClassifier")

nb.plain("""
You won't hand-code backprop at work — a library does it. Here's a real network on
the churn data using scikit-learn's MLPClassifier. Every concept maps one-to-one to
the toy net: hidden layers, activation, learning rate, epochs.
""")

nb.code(r"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("../data/customers_clean.csv")
feats = ["age", "income", "tenure_months", "monthly_spend", "support_calls"]
X = df[feats]; y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scaling is ESSENTIAL for neural nets — gradient descent misbehaves on unscaled inputs
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                    alpha=1e-3, max_iter=800, random_state=42)
mlp.fit(X_tr_s, y_tr)
print("test AUC:", round(roc_auc_score(y_te, mlp.predict_proba(X_te_s)[:, 1]), 3))
print(classification_report(y_te, mlp.predict(X_te_s), target_names=["stay", "churn"]))
""")

nb.code(r"""
plt.figure(figsize=(7, 4))
plt.plot(mlp.loss_curve_)
plt.xlabel("iteration"); plt.ylabel("training loss")
plt.title("MLP training loss curve (the library ran our five-step loop for us)"); plt.show()
""")

nb.warn("Always SCALE the inputs to a neural net. Gradient descent takes steps proportional "
        "to feature magnitude, so an unscaled big-range feature causes wild, unstable steps "
        "and slow or failed training. This is the same scaling lesson from earlier modules — "
        "it just matters even more here.")

nb.md("## 12.5 Key hyperparameters (interview-ready)")

nb.interview("""
The knobs an interviewer expects you to name:
- Architecture (hidden_layer_sizes): width & depth = model capacity.
- Activation: ReLU is the modern default (fast, avoids vanishing gradients);
  softmax on the OUTPUT for multi-class.
- Learning rate: the single most important knob — too high diverges/oscillates, too
  low crawls.
- Epochs + early stopping: train until validation loss stops improving, then stop to
  avoid overfitting.
- Regularization: alpha (L2 penalty), dropout (in deep-learning libraries), more data.
- Optimizer: Adam (adaptive) is the usual default over plain SGD.
""")

nb.md("## 12.6 When to actually use deep learning")

nb.deeper("""
The honest, senior take — and interviewers respect honesty here:
- For most TABULAR business data, classical ML (Random Forest / XGBoost) is usually
  as good or BETTER than a neural net, while being faster and more interpretable.
  Your water-wells and churn problems live squarely here.
- Deep learning earns its keep on UNSTRUCTURED data at scale: images (CNNs),
  text/sequences (Transformers), audio — and when you have lots of data and compute.

Saying 'I'd start with XGBoost on tabular data and only reach for a neural net when
the data type or scale demands it' signals maturity, not a gap in knowledge.
""")

nb.takeaway("A neural net is not automatically 'the advanced/better' choice. On a spreadsheet "
            "of customers, a gradient-boosted tree will usually beat it. Match the tool to the "
            "data: nets for images/text/audio, trees for tables.")

nb.md("## 12.7 Try it yourself")

nb.try_this("""
1. In the from-scratch net, drop the hidden layer to 1 unit. Can it still learn XOR?
   Why not? (Hint: how many lines does it take to carve XOR?)
2. Set the learning rate to 5.0, then to 0.01 — describe the loss curve in each case.
3. On the churn MLP, try hidden_layer_sizes=(64,64,64). Does test AUC improve, or does
   it start to overfit?
4. Explain backpropagation to an interviewer in exactly three sentences.
""")

nb.md("## Summary")

nb.takeaway("""
- A neuron = weighted sum + **non-linear activation**; a sigmoid neuron *is* logistic regression. Non-linearity is what makes depth meaningful.
- Training = **forward → loss → backprop (chain rule) → gradient-descent update**, repeated over epochs — the same loop from XOR to LLMs.
- **Always scale** inputs; tune **learning rate**, architecture, and regularization; use **early stopping**.
- **Classical ML usually wins on tabular data**; deep learning shines on images, text, and large unstructured datasets.
""")

nb.md(r"""
Next: **Module 13 — Job Readiness & Interview Prep** — turning all this knowledge
into offers.
""")

out = nb.save("notebooks/12_neural_networks_intro.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
