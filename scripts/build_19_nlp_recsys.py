"""Builder for Module 19: NLP + Recommendation Systems."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 19 — NLP & Recommendation Systems (Phase 4: T32, T37)

Two high-impact applied areas that power two of your own projects: the Twitter
sentiment analyzer (NLP) and LexiGenius document search (TF-IDF + clustering). We
build both from the ground up so you can explain the machinery, not just call a
library.

Goals:
- The NLP preprocessing pipeline (tokenize, stopwords, stem/lemmatize).
- **Bag-of-Words** and **TF-IDF** vectorization — turning text into numbers.
- **Cosine similarity** for semantic/document search.
- A working **sentiment classifier**.
- **Recommendation systems**: content-based vs collaborative filtering.
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
""")

nb.md(r"""
## 19.1 Why text needs special handling

Models need **numbers**, but text is unstructured. The NLP pipeline converts raw
strings into numeric vectors while preserving meaning:

1. **Lowercase & clean** — strip URLs, mentions, punctuation.
2. **Tokenize** — split into words (tokens).
3. **Remove stopwords** — drop low-information words ("the", "is", "and").
4. **Stem / lemmatize** — reduce words to a root ("running"→"run") so variants
   collapse.
5. **Vectorize** — Bag-of-Words or TF-IDF.
""")

nb.code(r"""
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove @mentions
    text = re.sub(r"[^a-z\s]", "", text)          # keep letters/spaces only
    return text.strip()

raw = "Check out https://x.com @user This MOVIE was absolutely AMAZING!!! 10/10"
print("raw   :", raw)
print("clean :", clean_text(raw))

# simple tokenization + stopword removal (no external downloads needed)
STOPWORDS = {"the","is","a","an","and","or","this","was","were","of","to","it",
             "in","on","for","with","out","check"}
tokens = [w for w in clean_text(raw).split() if w not in STOPWORDS]
print("tokens:", tokens)
""")

nb.md(r"""
## 19.2 Bag-of-Words — the simplest vectorization

Represent each document by **word counts**, ignoring order. The vocabulary becomes
the feature columns; each document is a row of counts.
""")

nb.code(r"""
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "the movie was great and the acting was great",
    "terrible movie boring acting",
    "great acting wonderful film",
    "boring and terrible film",
]
bow = CountVectorizer(stop_words="english")
X_bow = bow.fit_transform(corpus)
print("vocabulary:", bow.get_feature_names_out().tolist())
print(pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names_out()))
""")

nb.md(r"""
**Limitation of raw counts:** common words dominate even when uninformative. A word
appearing in *every* document tells us nothing about which document is which.
**TF-IDF** fixes this.
""")

nb.md(r"""
## 19.3 TF-IDF — weight words by informativeness

**TF-IDF** = Term Frequency × Inverse Document Frequency:
- **TF**: how often a word appears in a document (local importance).
- **IDF**: `log(N / documents-containing-the-word)` — down-weights words common
  across *all* documents, up-weights rare, distinctive ones.

So a word that's frequent in *this* document but rare overall gets a high score —
exactly the words that characterize the document. This is the core of LexiGenius.
""")

nb.code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(X_tfidf.toarray().round(2),
                        columns=tfidf.get_feature_names_out())
print("TF-IDF matrix (higher = more distinctive to that document):")
print(df_tfidf)
""")

nb.md(r"""
## 19.3b N-grams — putting *word order* back in (unigrams, bigrams, trigrams)

Bag-of-Words and plain TF-IDF throw away order, so **"not good"** and **"good"**
look almost identical — a disaster for sentiment. **N-grams** fix this by treating
sequences of *n* adjacent words as features:
- **unigram** (n=1): `good`, `not`, `movie`
- **bigram** (n=2): `not good`, `good movie`
- **trigram** (n=3): `was not good`

Set `ngram_range=(1, 2)` to keep unigrams **and** bigrams. Now `not good` is its own
feature and the model can learn it's negative.
""")

nb.code(r"""
neg = "the movie was not good not funny"
uni = CountVectorizer(ngram_range=(1,1)).fit([neg])
bi  = CountVectorizer(ngram_range=(2,2)).fit([neg])
tri = CountVectorizer(ngram_range=(3,3)).fit([neg])
print("unigrams:", uni.get_feature_names_out().tolist())
print("bigrams :", bi.get_feature_names_out().tolist())
print("trigrams:", tri.get_feature_names_out().tolist())
""")

nb.code(r"""
# Proof it helps: 'not good' vs 'good' become DIFFERENT features with bigrams.
from sklearn.metrics.pairwise import cosine_similarity
pair = ["this is good", "this is not good"]
uni_v = TfidfVectorizer(ngram_range=(1,1)).fit_transform(pair)
bi_v  = TfidfVectorizer(ngram_range=(1,2)).fit_transform(pair)
print("cosine similarity of the two sentences:")
print(f"  unigrams only : {cosine_similarity(uni_v)[0,1]:.3f}  (look almost identical!)")
print(f"  with bigrams  : {cosine_similarity(bi_v)[0,1]:.3f}  (correctly more different)")
""")

nb.md(r"""
**Trade-off:** higher n captures more context but **explodes** the vocabulary
(sparsity, memory) and risks overfitting. Practical defaults: `ngram_range=(1,2)`
plus `min_df` (ignore ultra-rare n-grams) and `max_df` (ignore near-universal ones).
Character n-grams (`analyzer="char_wb"`) are great for typos and language ID.
""")

nb.md(r"""
## 19.4 Cosine similarity — semantic/document search

To find similar documents, measure the **angle** between their TF-IDF vectors.
**Cosine similarity** = 1 when identical direction, 0 when orthogonal (no shared
words). It ignores document length (unlike raw distance), which is why it's the
standard for text search and clustering.
""")

nb.code(r"""
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(X_tfidf)
print("document-document cosine similarity:")
print(pd.DataFrame(sim.round(2),
                   index=[f"doc{i}" for i in range(4)],
                   columns=[f"doc{i}" for i in range(4)]))

# 'Search': find the document most similar to a query
query = tfidf.transform(["wonderful great acting"])
scores = cosine_similarity(query, X_tfidf).ravel()
best = scores.argmax()
print(f"\nquery best-matches doc{best}: '{corpus[best]}' (score {scores[best]:.2f})")
""")

nb.md(r"""
## 19.4b Topic modeling with LDA (Latent Dirichlet Allocation)

*(Careful: this "LDA" is **not** Linear Discriminant Analysis from Module 10 — same
acronym, totally different method. This one is **unsupervised topic discovery**.)*

**Latent Dirichlet Allocation** finds hidden **topics** in a collection of
documents with **no labels**. Its model of the world:
- each **topic** is a distribution over words (e.g. a "sports" topic puts weight on
  *game, team, score*),
- each **document** is a mixture of topics (60% sports + 40% politics).

You give it the number of topics; it returns the word-mix per topic and the
topic-mix per document. Great for exploring a large unlabeled corpus (news, reviews,
support tickets). Feed it **counts** (CountVectorizer), not TF-IDF.
""")

nb.code(r"""
from sklearn.decomposition import LatentDirichletAllocation

docs = [
    "the team won the game with a great score",
    "players scored goals in the football match",
    "the election results show the new policy vote",
    "government policy and the election campaign debate",
    "the match ended with a winning goal by the team",
    "voters chose the party in the national election",
]
cv = CountVectorizer(stop_words="english")
dtm = cv.fit_transform(docs)

lda_topics = LatentDirichletAllocation(n_components=2, random_state=0).fit(dtm)

# Show the top words defining each discovered topic:
vocab = cv.get_feature_names_out()
for k, comp in enumerate(lda_topics.components_):
    top = [vocab[i] for i in comp.argsort()[-5:][::-1]]
    print(f"Topic {k}: {top}")

# Which topic dominates each document?
doc_topics = lda_topics.transform(dtm)
for i, dist in enumerate(doc_topics):
    print(f"doc{i} topic mix -> {dist.round(2)}  (dominant: Topic {dist.argmax()})")
""")

nb.md(r"""
**Reading it:** one discovered topic should cluster the *sports* words
(team/game/goal), the other the *politics* words (election/policy/vote) — with **no
labels given**. Each document then gets a topic-mixture. That's unsupervised
structure discovery over text, complementing the supervised sentiment model next.
""")

nb.md(r"""
## 19.5 A working sentiment classifier

Combine TF-IDF with a classifier (Logistic Regression or Naive Bayes from Module
15) inside a Pipeline — the exact architecture of your Twitter sentiment app.
""")

nb.code(r"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# tiny labeled set: 1 = positive, 0 = negative
train_texts = [
    "great movie loved it", "amazing film wonderful acting", "best i have seen",
    "fantastic and brilliant", "really enjoyed this",
    "terrible boring waste", "awful acting hated it", "worst film ever",
    "so bad and disappointing", "do not watch this",
]
train_labels = [1,1,1,1,1, 0,0,0,0,0]

sentiment = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression()),
]).fit(train_texts, train_labels)

tests = ["wonderful and amazing", "boring waste of time", "i loved the acting"]
for t in tests:
    p = sentiment.predict([t])[0]
    prob = sentiment.predict_proba([t])[0][1]
    print(f"'{t}' -> {'POSITIVE' if p==1 else 'NEGATIVE'} (P(pos)={prob:.2f})")
""")

nb.md(r"""
**Note:** with tiny data this is illustrative; real sentiment needs thousands of
examples, careful preprocessing, and often modern embeddings/transformers. But the
*pipeline shape* — vectorize → classify — is production-real.
""")

nb.md(r"""
## 19.6 Recommendation systems — two core paradigms

**Content-based filtering:** recommend items *similar to what a user liked*, using
item **features**. "You liked this action movie → here's another action movie."
Uses the same cosine-similarity idea on item feature vectors.

**Collaborative filtering:** recommend based on *what similar users liked*, using
the **user–item interaction matrix** — no item features needed. "Users like you
also bought X." Two flavors: user-based and item-based; matrix factorization (SVD)
learns latent factors.
""")

nb.code(r"""
# CONTENT-BASED: recommend movies by genre-feature similarity
movies = pd.DataFrame({
    "title": ["Action A","Action B","Romance C","Romance D","ComedyE"],
    "action":[1,1,0,0,0], "romance":[0,0,1,1,0], "comedy":[0,0,0,0,1],
}).set_index("title")

from sklearn.metrics.pairwise import cosine_similarity
item_sim = cosine_similarity(movies.values)
item_sim = pd.DataFrame(item_sim, index=movies.index, columns=movies.index)

liked = "Action A"
recs = item_sim[liked].drop(liked).sort_values(ascending=False)
print(f"Because you liked '{liked}', we recommend:")
print(recs.round(2))
""")

nb.code(r"""
# COLLABORATIVE: user-item ratings; find users similar to a target and recommend
ratings = pd.DataFrame({
    "Action A":[5, 4, 1, np.nan, 5],
    "Action B":[4, 5, np.nan, 1, 4],
    "Romance C":[1, np.nan, 5, 4, 1],
    "Romance D":[np.nan, 1, 4, 5, np.nan],
}, index=["Ann","Ben","Cara","Dan","Eve"])
print("user-item rating matrix:")
print(ratings)

# similarity between users (fill NaN with 0 for the demo)
user_sim = cosine_similarity(ratings.fillna(0))
user_sim = pd.DataFrame(user_sim, index=ratings.index, columns=ratings.index)
target = "Eve"
closest = user_sim[target].drop(target).idxmax()
print(f"\n'{target}' is most similar to '{closest}'.")
# recommend an item the neighbor rated highly that the target hasn't seen
unseen = ratings.loc[target].isna()
recommend = ratings.loc[closest][unseen].idxmax()
print(f"Recommend '{recommend}' to {target} (highly rated by {closest}, unseen by {target}).")
""")

nb.md(r"""
**The cold-start problem (name it):** collaborative filtering fails for brand-new
users or items with no interaction history. Fixes: fall back to content-based, use
popularity defaults, or ask a few onboarding questions. Real systems are
**hybrids** of both paradigms.
""")

nb.md(r"""
## 19.7 Mini-exercises

1. Add bigrams to the TF-IDF (`ngram_range=(1,2)`). Does "not good" now separate
   from "good"? Why does that matter for sentiment?
2. Build a mini document-search over 5 sentences of your own; return the top-2
   matches for a query.
3. In the collaborative example, recommend for "Cara" and explain the logic.
4. Explain content-based vs collaborative filtering to a non-technical manager, and
   when each fails.
""")

nb.md(r"""
## Summary

- NLP pipeline: clean → tokenize → stopwords → stem/lemmatize → **vectorize**.
- **Bag-of-Words** counts words; **TF-IDF** weights them by informativeness
  (TF × IDF) — the basis of search and text classification.
- **N-grams** (`ngram_range=(1,2)`) restore word order so **"not good" ≠ "good"** —
  essential for sentiment.
- **LDA topic modeling** discovers hidden topics in unlabeled text (distinct from
  Linear Discriminant Analysis — same acronym, different method).
- **Cosine similarity** compares text vectors by angle → semantic search &
  clustering (your LexiGenius).
- Sentiment = **TF-IDF → classifier** Pipeline (your Twitter app).
- Recommenders: **content-based** (item features + similarity) vs **collaborative**
  (user–item matrix); watch the **cold-start** problem; real systems are hybrids.

**This completes the bootcamp's applied track.** Return to Module 12 to rehearse how
you'll *talk* about all of it — that's what converts this knowledge into offers.
""")

out = nb.save("notebooks/19_nlp_recommendation_systems.ipynb")
print("saved", out)
