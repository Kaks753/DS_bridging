"""Builder for Module 11: Clustering & PCA (4-layer rewrite of old M10)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 11 — Unsupervised Learning: Clustering & PCA

Everything so far had an answer key (labels). **Unsupervised learning** has none —
the algorithm hunts for structure on its own. This powers customer segmentation,
anomaly detection, and — your NSE project — **risk clustering of stocks**.

We'll master **K-Means** (with honest ways to pick the number of clusters), meet
three other clustering algorithms for when K-Means fails, and then **PCA** for
squeezing many features into a few — with the linear-algebra intuition your degree
will enjoy.
""")

nb.analogy("Imagine tipping a box of mixed Lego onto the floor with no instructions. "
           "Clustering is grouping the pieces into natural piles (by colour, by size). "
           "PCA is realising that although each brick has many attributes, two axes — "
           "say 'bigness' and 'redness' — capture most of what makes them differ.")

nb.md("## 11.1 Setup")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

sns.set_theme(style="whitegrid")
blobs = pd.read_csv("../data/blobs_2d.csv")   # we secretly KNOW there are 3 groups
plt.figure(figsize=(6, 5))
plt.scatter(blobs["x1"], blobs["x2"], alpha=0.6)
plt.title("Raw 2-D data — how many natural groups do YOU see?")
plt.show()
""")

nb.jargon("unsupervised learning", "finding structure in data that has no labels/answer key")
nb.jargon("clustering", "grouping points so members of a group are more alike than across groups")
nb.jargon("centroid", "the mean point of a cluster — its centre of gravity")

nb.md("## 11.2 K-Means — the algorithm in plain words")

nb.plain("""
K-Means splits points into **k** groups, each represented by its **centroid** (the
average of its members). The loop is beautifully simple: guess k centre points,
assign every dot to its nearest centre, move each centre to the average of the dots
it caught, and repeat until nothing moves. It's chasing one goal: make every dot as
close as possible to its own centre.
""")

nb.md(r"""
Formally it minimises the **within-cluster sum of squares** (inertia):

$$ \text{minimize} \sum_{j=1}^{k} \sum_{x \in C_j} \| x - \mu_j \|^2 $$
""")

nb.code(r"""
X = blobs[["x1", "x2"]].values
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
labels = km.labels_

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c="red", s=200, marker="X", label="centroids")
plt.title("K-Means (k=3): points coloured by cluster"); plt.legend(); plt.show()
print("inertia (within-cluster sum of squares):", round(km.inertia_, 1))
""")

nb.readcode("""
- `n_init=10` restarts the algorithm 10 times from different seeds and keeps the best
  — K-Means can land in a bad local optimum from an unlucky start.
- Red X's are the final centroids; each colour is a cluster.
- Lower inertia = tighter clusters (but it always drops as k rises — see below).
""")

nb.interview("""
Name K-Means' assumptions and failure modes and you sound senior: it assumes clusters
are roughly SPHERICAL and SIMILAR in size (it uses straight-line distance to a mean),
so it struggles with elongated/odd shapes or very different densities — reach for
DBSCAN or a Gaussian Mixture there. It's sensitive to initialisation (use n_init > 1
or k-means++), it needs YOU to choose k, and being distance-based it demands scaling.
""")

nb.jargon("K-Means", "clustering into k groups by repeatedly assigning to nearest centroid then re-centring")
nb.jargon("inertia", "total squared distance of points to their cluster centroid; K-Means minimises it")

nb.md("## 11.3 Choosing k — the elbow method")

nb.plain("""
K-Means won't tell you how many clusters exist — you decide. The elbow method plots
inertia against k. Inertia always falls as k grows (more centres = tighter fit), but
there's usually a k where the improvement suddenly flattens — the 'elbow'. That bend
is your best guess at the natural number of groups.
""")

nb.code(r"""
ks = range(1, 10)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
plt.figure(figsize=(7, 4))
plt.plot(list(ks), inertias, "o-")
plt.xlabel("k"); plt.ylabel("inertia"); plt.title("Elbow method (the bend is near k=3)")
plt.show()
""")

nb.warn("The elbow is a GUIDE, not gospel — real data often has a soft, ambiguous bend. "
        "Treat it as one piece of evidence and confirm with the silhouette score below.")

nb.md("## 11.4 Choosing k — the silhouette score (more rigorous)")

nb.plain("""
The silhouette score asks, for each point: 'am I closer to my own cluster-mates than
to the nearest OTHER cluster?' It ranges from +1 (snugly inside my cluster) through 0
(on the border) to negative (probably in the wrong cluster). Average it over all
points and you get a single, honest score for the whole clustering — higher is better.
""")

nb.code(r"""
for k in range(2, 7):
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"k={k}: silhouette = {silhouette_score(X, km_k.labels_):.3f}")
print("\nHighest silhouette => best-separated clustering (should favour k=3).")
""")

nb.deeper("""
Stephen — this is the exact metric you drove from 0.32 to 0.717 on the NSE project.
In silhouette terms that's a leap from 'weak, overlapping clusters that barely hang
together' (0.32) to 'clearly separated, genuinely meaningful groups' (0.717). Being
able to say 'I improved silhouette from 0.32 to 0.717 by scaling and re-choosing k'
is a portfolio sentence that lands — it shows you measured cluster QUALITY, not just
ran the algorithm.
""")

nb.code(r"""
# Silhouette PLOT: diagnose each cluster's quality visually
k = 3
km3 = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
sv = silhouette_samples(X, km3.labels_)

plt.figure(figsize=(7, 4.5))
y_lower = 0
for i in range(k):
    vals = np.sort(sv[km3.labels_ == i])
    y_upper = y_lower + len(vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
    plt.text(-0.05, (y_lower + y_upper) / 2, str(i))
    y_lower = y_upper
plt.axvline(silhouette_score(X, km3.labels_), color="red", ls="--", label="mean silhouette")
plt.xlabel("silhouette value"); plt.ylabel("samples (grouped by cluster)")
plt.title("Silhouette plot (wide, positive 'knives' = good clusters)")
plt.legend(); plt.show()
""")

nb.jargon("silhouette score", "how well each point fits its cluster vs the nearest other, from -1 to +1")

nb.md("## 11.5 The scaling lesson for clustering (real customer data)")

nb.warn("""
Cluster customers by income (tens of thousands) and support_calls (0–6) WITHOUT
scaling, and K-Means effectively sees only income — the calls are numerically tiny
and get ignored. Because clustering is distance-based, you must scale first, or one
big-range feature quietly runs the whole show.
""")

nb.code(r"""
cust = pd.read_csv("../data/customers_clean.csv")
feat = cust[["income", "support_calls"]].dropna().values

lab_raw    = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(feat)
feat_s     = StandardScaler().fit_transform(feat)
lab_scaled = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(feat_s)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(feat[:, 0], feat[:, 1], c=lab_raw, cmap="viridis", alpha=0.6)
ax[0].set_title("UNSCALED: clusters split on income only")
ax[0].set_xlabel("income"); ax[0].set_ylabel("support_calls")
ax[1].scatter(feat[:, 0], feat[:, 1], c=lab_scaled, cmap="viridis", alpha=0.6)
ax[1].set_title("SCALED: both features actually matter"); ax[1].set_xlabel("income")
plt.tight_layout(); plt.show()
print("Unscaled silhouette:", round(silhouette_score(feat,   lab_raw), 3),
      "| Scaled silhouette:", round(silhouette_score(feat_s, lab_scaled), 3))
""")

nb.takeaway("Left panel slices purely by income (vertical bands); right panel, after scaling, "
            "forms real 2-D groups. Same data, same algorithm — scaling was the whole difference.")

nb.md("## 11.6 Three more clustering algorithms — matching the tool to the shape")

nb.plain("""
K-Means is not the only game in town, and it's the wrong tool for some data shapes.
Three alternatives worth knowing:
- **Agglomerative (hierarchical)** — build a merge-tree; no need to pre-pick k.
- **DBSCAN** — density-based; finds arbitrary shapes and flags noise; no k needed.
- **Gaussian Mixture (GMM)** — soft, elliptical clusters with membership probabilities.
""")

nb.code(r"""
# Agglomerative: repeatedly merge the two closest clusters, forming a dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, method="ward")
plt.figure(figsize=(9, 4))
dendrogram(Z, truncate_mode="level", p=5, color_threshold=0.7*max(Z[:, 2]))
plt.title("Dendrogram (Ward) — the tallest vertical gap suggests 3 clusters")
plt.xlabel("samples (merged bottom-up)"); plt.ylabel("merge distance")
plt.show()
""")

nb.readcode("""
- The y-axis is the distance at which clusters merge. Find the TALLEST vertical gap
  that no horizontal line crosses and 'cut' there — that gives the natural k.
- `method='ward'` merges the pair that increases within-cluster variance least; it's
  the go-to linkage and gives compact, K-Means-like clusters.
""")

nb.code(r"""
# DBSCAN vs K-Means on two interleaving moons — a shape K-Means CANNOT do
from sklearn.datasets import make_moons
Xm, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
km_m = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Xm)
db_m = DBSCAN(eps=0.20, min_samples=5).fit_predict(Xm)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(Xm[:, 0], Xm[:, 1], c=km_m, cmap="coolwarm", alpha=0.7)
ax[0].set_title("K-Means: splits the moons WRONG (it needs spheres)")
ax[1].scatter(Xm[:, 0], Xm[:, 1], c=db_m, cmap="coolwarm", alpha=0.7)
ax[1].set_title(f"DBSCAN: correct shapes; noise points = {(db_m==-1).sum()}")
plt.tight_layout(); plt.show()
""")

nb.deeper("""
DBSCAN has two knobs: `eps` (how close counts as 'near') and `min_samples` (how many
near-neighbours make a point a dense 'core'). Points are core, border, or noise (−1).
That noise label is a gift — automatic outlier detection. Its weakness: it's very
sensitive to eps and struggles when clusters have very different densities.

A **GMM** goes soft: instead of 'you're in cluster 2', it says 'you're 80% cluster 2,
20% cluster 1', and each cluster can be a stretched/rotated ellipse. It's fit by
Expectation-Maximization and you pick the number of components by BIC (lower is
better). In fact K-Means is just a GMM with spherical, equal covariances and hard
assignments — so GMM is the more general tool when clusters overlap or aren't round.
""")

nb.code(r"""
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(X)
proba = gmm.predict_proba(X)     # soft memberships, each row sums to 1
print("example soft memberships (first 3 points):")
print(np.round(proba[:3], 3))
bics = [GaussianMixture(n, covariance_type="full", random_state=42).fit(X).bic(X) for n in range(1, 7)]
print("\nBIC by n_components:", [round(b) for b in bics], "-> best =", int(np.argmin(bics)) + 1)
""")

nb.md(r"""
### Clustering cheat-sheet — match algorithm to cluster shape
| Algorithm | Needs k? | Cluster shape | Probabilities? | Detects noise? |
|---|---|---|---|---|
| **K-Means** | yes | spherical, equal size | no (hard) | no |
| **Agglomerative** | cut the tree | flexible (linkage) | no | no |
| **DBSCAN** | no | **arbitrary** | no | **yes (−1)** |
| **GMM** | yes | **elliptical** | **yes (soft)** | no |
""")

nb.jargon("DBSCAN", "density-based clustering: finds arbitrary shapes and labels sparse points as noise")
nb.jargon("Gaussian Mixture (GMM)", "models data as overlapping Gaussian bells; gives soft, probabilistic memberships")
nb.jargon("dendrogram", "the merge-tree from hierarchical clustering; cut it at a height to choose k")

nb.md("## 11.7 PCA — Principal Component Analysis")

nb.plain("""
When data has many feature columns, PCA finds a smaller set of new axes that still
capture most of the variation. The first axis (PC1) points along the direction the
data spreads out most; PC2 is the next-biggest spread at right angles to PC1; and so
on. Keep the first few and you've compressed many columns into a handful with little
information lost.
""")

nb.md(r"""
For your linear-algebra background: the principal components are the **eigenvectors
of the covariance matrix**, and each component's captured variance is its
**eigenvalue**. PCA is literally an eigendecomposition wearing a data-science hat.
""")

nb.warn("Always SCALE before PCA. It's variance-based, so an unscaled large-range feature "
        "will look like it has the most 'spread' purely because of its units, and hijack PC1.")

nb.code(r"""
from sklearn.datasets import load_iris
iris = load_iris()
Xi = StandardScaler().fit_transform(iris.data)     # 4 features, scaled

pca = PCA().fit(Xi)
ev = pca.explained_variance_ratio_
print("explained variance ratio per PC:", ev.round(3))
print("cumulative:", ev.cumsum().round(3))

plt.figure(figsize=(7, 4))
plt.bar(range(1, len(ev)+1), ev, alpha=0.6, label="individual")
plt.plot(range(1, len(ev)+1), ev.cumsum(), "ro-", label="cumulative")
plt.axhline(0.95, color="gray", ls="--", label="95% threshold")
plt.xlabel("principal component"); plt.ylabel("explained variance ratio")
plt.title("Scree plot: how many PCs to keep?"); plt.legend(); plt.show()
""")

nb.readcode("""
- `explained_variance_ratio_` is the slice of total variance each PC captures.
- The cumulative line crossing ~95% early means the first 2 PCs already hold almost
  everything: we can turn 4 features into 2 with negligible loss.
""")

nb.code(r"""
proj = PCA(n_components=2).fit_transform(Xi)
plt.figure(figsize=(7, 5))
for cls, name, c in zip(range(3), iris.target_names, ["navy", "darkorange", "green"]):
    m = iris.target == cls
    plt.scatter(proj[m, 0], proj[m, 1], label=name, alpha=0.7, color=c)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Iris squeezed from 4 features to 2 PCs — classes still separate cleanly")
plt.legend(); plt.show()
""")

nb.deeper("""
The honest caveat: each PC is a LINEAR COMBINATION of the originals
('PC1 = 0.5·petal_length + 0.4·petal_width + …'), so PCs are harder to explain than
raw features. Reach for PCA when compression, denoising, or 2-D visualisation matters
more than per-feature interpretability. And a supervised cousin, **LDA (Linear
Discriminant Analysis)**, finds axes that maximise CLASS SEPARATION rather than total
variance — often better when the downstream job is classification, and it doubles as
a fast linear classifier.
""")

nb.jargon("PCA", "re-express data on new orthogonal axes ordered by how much variance each captures")
nb.jargon("explained variance ratio", "the fraction of total variance a principal component captures")
nb.jargon("scree plot", "bar/line chart of explained variance per PC, used to decide how many to keep")

nb.md("## 11.8 Try it yourself")

nb.try_this("""
1. Cluster customers on 4 SCALED features; pick k by silhouette; then profile each
   cluster (mean of each feature) and give it a business name ('young high-spenders').
2. Run PCA on those 4 features — how many PCs reach 90% variance?
3. Reduce to 2 PCs, then K-Means on that projection — does silhouette improve?
4. Explain why K-Means fails on two concentric rings, and what you'd use instead.
""")

nb.md("## Summary")

nb.takeaway("""
- **K-Means** minimises within-cluster variance; assumes spherical, similar-size clusters; needs **scaling** and a chosen **k**.
- Pick k with the **elbow** (inertia bend) and, more rigorously, the **silhouette** score — the metric behind the NSE 0.32→0.717 win.
- **Agglomerative** builds a dendrogram (cut to pick k); **DBSCAN** finds arbitrary shapes + noise without k; **GMM** gives soft, elliptical clusters. Match the algorithm to the cluster shape.
- **PCA** re-expresses data on max-variance orthogonal axes (eigenvectors of covariance); keep the top PCs for compression/denoising/plots. **Scale first.**
""")

nb.md(r"""
Next: **Module 12 — Neural Networks**, where we stack simple units into models that
learn their own features.
""")

out = nb.save("notebooks/11_clustering_pca.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
