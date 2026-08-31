"""Builder for Module 20: NLP + Recommendation Systems (4-layer)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 20 — NLP & Recommendation Systems

Two high-impact applied areas that power real products: a Twitter sentiment analyzer
(NLP) and a document-search engine (TF-IDF + clustering). We build both from the ground
up so you can explain the machinery, not just call a library.

**What you'll be able to do by the end:**
- Run the NLP preprocessing pipeline (clean, tokenize, stopwords, stem/lemmatize).
- Turn text into numbers with **Bag-of-Words** and **TF-IDF**.
- Use **N-grams** so "not good" ≠ "good", and **cosine similarity** for search.
- Build a working **sentiment classifier**.
- Explain **content-based** vs **collaborative filtering** recommenders.
""")

nb.plain(r"""
Computers only do math on numbers, but language is messy words. NLP (Natural Language
Processing) is the art of turning text into numbers *without throwing away the meaning*,
then doing normal machine learning on those numbers. Recommendation systems are a
close cousin: they turn "who liked what" into numbers and use similarity to suggest the
next thing. Both boil down to: **represent as vectors, then compare with similarity.**
""")

nb.code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
print("Tools ready.")
""")

# ---------------------------------------------------------------------------
# 20.1 Preprocessing
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.1 Why text needs special handling

The NLP pipeline turns raw strings into clean tokens before we vectorize:
1. **Lowercase & clean** — strip URLs, @mentions, punctuation.
2. **Tokenize** — split into words (tokens).
3. **Remove stopwords** — drop low-information words ("the", "is", "and").
4. **Stem / lemmatize** — collapse variants ("running" -> "run").
5. **Vectorize** — Bag-of-Words or TF-IDF.
""")

nb.analogy(r"""
Think of prepping vegetables before cooking: you wash off dirt (clean), chop into
pieces (tokenize), throw out the stems nobody eats (stopwords), and cut everything to a
uniform size (stemming) so it all cooks evenly. Only *then* do you start the actual
recipe (the model).
""")

nb.jargon("Token", "one unit of text, usually a single word, after splitting")
nb.jargon("Stopword", "a super-common, low-information word like 'the' or 'is' that we usually drop")
nb.jargon("Stemming / Lemmatization", "cutting words down to a root so 'run', 'running', 'ran' count as the same thing")

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

STOPWORDS = {"the","is","a","an","and","or","this","was","were","of","to","it",
             "in","on","for","with","out","check"}
tokens = [w for w in clean_text(raw).split() if w not in STOPWORDS]
print("tokens:", tokens)
""")

nb.readcode(r"""
- `clean_text` lowercases everything, then uses regular expressions to delete URLs,
  @mentions, and anything that isn't a letter or space.
- After cleaning, we split on spaces and drop stopwords, leaving only the words that
  carry meaning: `movie`, `absolutely`, `amazing`.
- That short list of tokens is what we'll turn into numbers next.
""")

nb.takeaway("Clean -> tokenize -> drop stopwords -> stem -> vectorize: get from messy strings to meaningful tokens first.")

# ---------------------------------------------------------------------------
# 20.2 BoW
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.2 Bag-of-Words — the simplest vectorization

Represent each document by **word counts**, ignoring order. The vocabulary becomes the
feature columns; each document is a row of counts.
""")

nb.analogy(r"""
Imagine dumping each sentence into a bag and shaking it — you keep *which* words are
present and *how many*, but lose the order. "Dog bites man" and "man bites dog" look
identical in the bag. Crude, but surprisingly useful as a starting point.
""")

nb.jargon("Bag-of-Words", "representing text as word counts, ignoring word order")

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

nb.warn(r"""
Raw counts let **common words dominate** even when they're uninformative. A word that
appears in *every* document tells you nothing about which document is which. TF-IDF
fixes exactly this.
""")

# ---------------------------------------------------------------------------
# 20.3 TF-IDF
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.3 TF-IDF — weight words by informativeness

**TF-IDF** = Term Frequency x Inverse Document Frequency:
- **TF**: how often a word appears in *this* document (local importance).
- **IDF**: $\log(N / \text{docs containing the word})$ — down-weights words common
  across *all* documents, up-weights rare, distinctive ones.

A word that's frequent *here* but rare *overall* scores high — exactly the words that
characterize the document.
""")

nb.analogy(r"""
At a data-science meetup, everyone says "model" — so hearing "model" tells you nothing
about a person. But if someone keeps saying "astrophysics", that's *distinctive* and
tells you a lot about them. TF-IDF gives high scores to the "astrophysics" words and low
scores to the "model" words.
""")

nb.jargon("TF-IDF", "term frequency times inverse document frequency; scores words high when frequent here but rare overall")

nb.code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(X_tfidf.toarray().round(2),
                        columns=tfidf.get_feature_names_out())
print("TF-IDF matrix (higher = more distinctive to that document):")
print(df_tfidf)
""")

nb.takeaway("Bag-of-Words counts words; TF-IDF weights them by how distinctive they are -- the basis of text search & classification.")

# ---------------------------------------------------------------------------
# 20.4 N-grams
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.4 N-grams — putting *word order* back in

Bag-of-Words and plain TF-IDF throw away order, so **"not good"** and **"good"** look
almost identical — a disaster for sentiment. **N-grams** treat sequences of *n*
adjacent words as features:
- **unigram** (n=1): `good`, `not`, `movie`
- **bigram** (n=2): `not good`, `good movie`
- **trigram** (n=3): `was not good`

Set `ngram_range=(1, 2)` to keep unigrams **and** bigrams, so `not good` becomes its own
feature the model can learn is negative.
""")

nb.jargon("N-gram", "a sequence of n adjacent words treated as one feature (bigram = 2 words)")

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

nb.readcode(r"""
- "this is good" and "this is not good" share almost all their single words, so with
  **unigrams only** the similarity is high — the model can't tell praise from criticism.
- Adding **bigrams** creates the feature "not good", which only the second sentence has,
  so the similarity drops — the model can now tell them apart. That's why n-grams matter
  for sentiment.
""")

nb.deeper(r"""
Higher n captures more context but **explodes** the vocabulary (more sparsity, more
memory) and risks overfitting. Practical defaults: `ngram_range=(1,2)` plus `min_df`
(ignore ultra-rare n-grams) and `max_df` (ignore near-universal ones). Character n-grams
(`analyzer="char_wb"`) shine for typos and language identification.
""")

nb.takeaway("N-grams (ngram_range=(1,2)) restore local word order so 'not good' != 'good' -- essential for sentiment.")

# ---------------------------------------------------------------------------
# 20.5 Cosine similarity & search
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.5 Cosine similarity — semantic/document search

To find similar documents, measure the **angle** between their TF-IDF vectors.
**Cosine similarity** = 1 when they point the same way, 0 when they share no words. It
ignores document length, which is why it's the standard for text search and clustering.
""")

nb.analogy(r"""
Imagine each document as an arrow in space, one axis per word. Two documents about the
same thing point in nearly the same direction (small angle -> cosine near 1); unrelated
ones point at right angles (cosine near 0). We compare *direction*, not *length*, so a
short tweet and a long article about the same topic still match.
""")

nb.jargon("Cosine similarity", "how aligned two vectors are by the angle between them: 1 = same direction, 0 = unrelated")

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

nb.readcode(r"""
- The square matrix compares every document to every other; the diagonal is 1.0 (each
  doc is identical to itself).
- For "search", we turn a query into the same TF-IDF space, compute its cosine to every
  document, and pick the highest score with `argmax`. That's a tiny search engine.
""")

nb.takeaway("Cosine similarity compares text vectors by angle (ignores length) -> the core of document search and clustering.")

# ---------------------------------------------------------------------------
# 20.6 LDA topic modeling
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.6 Topic modelling with LDA (Latent Dirichlet Allocation)
""")

nb.warn(r"""
This "LDA" is **not** the Linear Discriminant Analysis from the clustering/PCA module —
same acronym, totally different method. This one is **unsupervised topic discovery**.
""")

nb.plain(r"""
LDA reads a pile of documents with *no labels* and figures out the hidden themes. Its
worldview: every topic is a favourite set of words (a "sports" topic loves
*game/team/score*), and every document is a blend of topics (60% sports + 40%
politics). You tell it how many topics to look for; it hands back the word-mix per topic
and the topic-mix per document. Great for exploring big unlabelled text.
""")

nb.jargon("LDA (topic modelling)", "unsupervised method that discovers hidden topics as word-distributions across documents")

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

vocab = cv.get_feature_names_out()
for k, comp in enumerate(lda_topics.components_):
    top = [vocab[i] for i in comp.argsort()[-5:][::-1]]
    print(f"Topic {k}: {top}")

doc_topics = lda_topics.transform(dtm)
for i, dist in enumerate(doc_topics):
    print(f"doc{i} topic mix -> {dist.round(2)}  (dominant: Topic {dist.argmax()})")
""")

nb.readcode(r"""
- We feed LDA **counts** (from CountVectorizer, not TF-IDF) and ask for 2 topics.
- `components_` holds each topic's word weights; we print the top-5 words per topic —
  one clusters the *sports* words, the other the *politics* words, with no labels given.
- `transform` then reports each document's topic mixture; `argmax` names its dominant
  topic. That's unsupervised structure discovery over text.
""")

nb.takeaway("LDA discovers hidden topics in unlabelled text (feed it counts, not TF-IDF); distinct from Linear Discriminant Analysis.")

# ---------------------------------------------------------------------------
# 20.7 Sentiment classifier
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.7 A working sentiment classifier

Combine TF-IDF with a classifier (Logistic Regression) inside a Pipeline — the exact
architecture of a Twitter sentiment app: **vectorize -> classify**.
""")

nb.code(r"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

train_texts = [
    "great movie loved it", "amazing film wonderful acting", "best i have seen",
    "fantastic and brilliant", "really enjoyed this",
    "terrible boring waste", "awful acting hated it", "worst film ever",
    "so bad and disappointing", "do not watch this",
]
train_labels = [1,1,1,1,1, 0,0,0,0,0]   # 1 = positive, 0 = negative

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

nb.readcode(r"""
- The `Pipeline` chains TF-IDF (text -> numbers) directly into Logistic Regression
  (numbers -> label), so calling `.fit`/`.predict` on raw text just works.
- `predict_proba(...)[0][1]` gives the model's confidence that the text is positive.
- Positive-leaning phrases score high P(pos); negative ones score low. Same shape as a
  production sentiment service.
""")

nb.deeper(r"""
With only ten examples this is illustrative, not production-grade. Real sentiment needs
thousands of labelled examples, careful preprocessing, and often modern **embeddings /
transformers** (BERT and friends) that understand context far better than TF-IDF. But
the pipeline *shape* — vectorize then classify — is exactly what ships in production;
you'd just swap the vectorizer/model.
""")

nb.takeaway("Sentiment = a TF-IDF -> classifier Pipeline; upgrade the vectorizer to embeddings/transformers for real accuracy.")

# ---------------------------------------------------------------------------
# 20.8 Recommenders
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.8 Recommendation systems — two core paradigms
""")

nb.plain(r"""
- **Content-based**: "You liked this action movie, here's another action movie." It
  compares *item features* — same cosine-similarity trick on genre vectors.
- **Collaborative filtering**: "People who are similar to you also liked X." It uses
  only the *who-rated-what* table, no item features needed.
Both are just similarity again — one on items, one on users.
""")

nb.jargon("Content-based filtering", "recommend items similar to what you liked, using item features")
nb.jargon("Collaborative filtering", "recommend based on what similar users liked, using the user-item rating matrix")

nb.code(r"""
# CONTENT-BASED: recommend movies by genre-feature similarity
movies = pd.DataFrame({
    "title": ["Action A","Action B","Romance C","Romance D","ComedyE"],
    "action":[1,1,0,0,0], "romance":[0,0,1,1,0], "comedy":[0,0,0,0,1],
}).set_index("title")

item_sim = cosine_similarity(movies.values)
item_sim = pd.DataFrame(item_sim, index=movies.index, columns=movies.index)

liked = "Action A"
recs = item_sim[liked].drop(liked).sort_values(ascending=False)
print(f"Because you liked '{liked}', we recommend:")
print(recs.round(2))
""")

nb.code(r"""
# COLLABORATIVE: user-item ratings; find similar users and recommend
ratings = pd.DataFrame({
    "Action A":[5, 4, 1, np.nan, 5],
    "Action B":[4, 5, np.nan, 1, 4],
    "Romance C":[1, np.nan, 5, 4, 1],
    "Romance D":[np.nan, 1, 4, 5, np.nan],
}, index=["Ann","Ben","Cara","Dan","Eve"])
print("user-item rating matrix:")
print(ratings)

user_sim = cosine_similarity(ratings.fillna(0))
user_sim = pd.DataFrame(user_sim, index=ratings.index, columns=ratings.index)
target = "Eve"
closest = user_sim[target].drop(target).idxmax()
print(f"\n'{target}' is most similar to '{closest}'.")
unseen = ratings.loc[target].isna()
recommend = ratings.loc[closest][unseen].idxmax()
print(f"Recommend '{recommend}' to {target} (highly rated by {closest}, unseen by {target}).")
""")

nb.readcode(r"""
- **Content-based**: each movie is a genre vector; cosine similarity ranks other movies
  by how genre-alike they are to the one you liked (Action A -> Action B).
- **Collaborative**: we find the user most similar to Eve by comparing their rating
  rows, then recommend an item that neighbour rated highly which Eve hasn't seen yet.
""")

nb.warn(r"""
**Cold-start problem**: collaborative filtering fails for brand-new users or items with
no ratings yet — there's nothing to compare. Fixes: fall back to content-based, use
popularity defaults, or ask a few onboarding questions. Real systems are **hybrids** of
both paradigms.
""")

nb.takeaway("Content-based compares item features; collaborative compares user behaviour; combine them and handle cold-start.")

# ---------------------------------------------------------------------------
# Practice + summary
# ---------------------------------------------------------------------------
nb.md(r"""
## 20.9 Practice
""")

nb.try_this(r"""
1. Add bigrams to the sentiment TF-IDF (`ngram_range=(1,2)`). Does "not good" now behave
   differently from "good"?
2. Build a mini document-search over 5 sentences of your own; return the top-2 matches
   for a query.
3. In the collaborative example, recommend for "Cara" and explain the logic.
4. Explain content-based vs collaborative filtering to a non-technical manager, and when
   each fails.
""")

nb.md(r"""
## Summary

- NLP pipeline: clean -> tokenize -> stopwords -> stem/lemmatize -> **vectorize**.
- **Bag-of-Words** counts words; **TF-IDF** weights them by informativeness -- the basis
  of search and text classification.
- **N-grams** (`ngram_range=(1,2)`) restore word order so **"not good" != "good"**.
- **LDA topic modelling** discovers hidden topics in unlabelled text (distinct from
  Linear Discriminant Analysis).
- **Cosine similarity** compares text vectors by angle -> semantic search & clustering.
- Sentiment = **TF-IDF -> classifier** Pipeline.
- Recommenders: **content-based** (item features) vs **collaborative** (user-item
  matrix); watch **cold-start**; real systems are hybrids.

Next: **Module 21 — Bash & Git for Data Scientists**.
""")

out = nb.save("notebooks/20_nlp_recommendation_systems.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
