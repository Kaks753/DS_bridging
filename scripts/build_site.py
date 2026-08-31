#!/usr/bin/env python3
"""
build_site.py — Static-site generator for the DS Bridging Bootcamp.

Reads notebooks/*.ipynb, understands the 4-layer tagged-markdown convention
(> 🌱 In plain English / > 🔤 Reading the code / > 🎓 Go deeper / ✅ Takeaway /
🗣️ interview / 🧠 analogy / ⚠️ warn / ✍️ try_this / 📖 jargon), and renders each
lesson into a styled HTML page inside site/lessons/.

It also builds:
  - site/index.html         (landing page + module grid)
  - site/lessons/<slug>.html (one per notebook)
  - a shared shell with sidebar nav, prev/next, progress, and client-side search
  - site/assets/search-index.json

No external Python deps beyond the stdlib + `markdown` (fallback to a tiny
converter if markdown isn't installed).
"""
import json, os, re, html, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
# Output dir is overridable so we can build both the local-preview `site/`
# and the Vercel-served `public/` from one generator:
#   SITE_OUT=public python3 scripts/build_site.py
SITE = ROOT / os.environ.get("SITE_OUT", "site")
LESSONS = SITE / "lessons"
ASSETS = SITE / "assets"

# --- GitHub / Colab config (public repo so Colab & Binder can fetch) --------
GH_OWNER = "Kaks753"
GH_REPO = "DS_bridging"
GH_BRANCH = "main"

# --- brand logo (inline SVG "bridge" mark) ----------------------------------
# A suspension bridge from a low bank up to a high bank = "bridging" into DS.
def logo_svg(size=34):
    return (
        f'<svg class="logo" width="{size}" height="{size}" viewBox="0 0 48 48" '
        'fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<defs><linearGradient id="lg" x1="0" y1="0" x2="48" y2="48" '
        'gradientUnits="userSpaceOnUse">'
        '<stop stop-color="#7c5cff"/><stop offset="1" stop-color="#4dd0e1"/>'
        '</linearGradient></defs>'
        # two towers
        '<rect x="9" y="12" width="3.2" height="26" rx="1.4" fill="url(#lg)"/>'
        '<rect x="35.8" y="7" width="3.2" height="31" rx="1.4" fill="url(#lg)"/>'
        # deck (rising from left bank to right bank)
        '<path d="M4 34 L44 27" stroke="url(#lg)" stroke-width="3.2" '
        'stroke-linecap="round"/>'
        # main suspension cable (curved)
        '<path d="M10.6 13 C18 26, 30 22, 37.4 8.5" stroke="url(#lg)" '
        'stroke-width="2.2" fill="none" stroke-linecap="round"/>'
        # hanger cables
        '<path d="M15 20 L15 33 M20 22 L20 32 M25 22 L25 31 M30 20 L30 30" '
        'stroke="url(#lg)" stroke-width="1.2" opacity=".7"/>'
        '</svg>'
    )

# Data-URI favicon using the same bridge mark.
FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'%3E"
    "%3Crect x='9' y='12' width='3.2' height='26' rx='1.4' fill='%237c5cff'/%3E"
    "%3Crect x='35.8' y='7' width='3.2' height='31' rx='1.4' fill='%234dd0e1'/%3E"
    "%3Cpath d='M4 34 L44 27' stroke='%237c5cff' stroke-width='3.2' stroke-linecap='round'/%3E"
    "%3Cpath d='M10.6 13 C18 26, 30 22, 37.4 8.5' stroke='%234dd0e1' stroke-width='2.2' fill='none' stroke-linecap='round'/%3E"
    "%3C/svg%3E"
)

def colab_url(stem):
    return (f"https://colab.research.google.com/github/{GH_OWNER}/{GH_REPO}"
            f"/blob/{GH_BRANCH}/notebooks/{stem}.ipynb")

def github_nb_url(stem):
    return (f"https://github.com/{GH_OWNER}/{GH_REPO}"
            f"/blob/{GH_BRANCH}/notebooks/{stem}.ipynb")

# --- markdown backend -------------------------------------------------------
try:
    import markdown as _md
    def md_to_html(text):
        return _md.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])
except Exception:
    def md_to_html(text):
        # extremely small fallback: paragraphs + inline code + bold/italic
        out = []
        for para in text.split("\n\n"):
            p = html.escape(para)
            p = re.sub(r"`([^`]+)`", r"<code>\1</code>", p)
            p = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", p)
            p = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", p)
            out.append("<p>" + p.replace("\n", "<br>") + "</p>")
        return "\n".join(out)

# --- layer tag registry -----------------------------------------------------
# Each blockquote whose first line matches a header becomes a styled callout.
LAYERS = [
    ("🌱", "plain",     "In plain English"),
    ("🔤", "readcode",  "Reading the code"),
    ("🎓", "deeper",    "Go deeper"),
    ("✅", "takeaway",  "Takeaway"),
    ("🗣️", "interview", "Say this in an interview"),
    ("🗣", "interview", "Say this in an interview"),
    ("🧠", "analogy",   "Analogy"),
    ("⚠️", "warn",      "Watch out"),
    ("⚠", "warn",       "Watch out"),
    ("✍️", "try",       "Try this yourself"),
    ("✍", "try",        "Try this yourself"),
    ("📖", "jargon",    "Jargon"),
]

def detect_layer(first_line):
    for emoji, cls, label in LAYERS:
        if emoji in first_line:
            return cls, label
    return None, None

# --- notebook parsing -------------------------------------------------------
def strip_blockquote(text):
    """Remove leading '> ' from each line of a blockquote."""
    lines = text.split("\n")
    out = []
    for ln in lines:
        if ln.startswith("> "):
            out.append(ln[2:])
        elif ln.strip() == ">":
            out.append("")
        else:
            out.append(ln)
    return "\n".join(out)

def render_markdown_cell(src):
    """Turn a markdown cell into one or more HTML blocks.
    Blockquotes that start with a known layer emoji become styled callouts."""
    blocks = []
    # Split the cell into blockquote runs vs normal markdown.
    lines = src.split("\n")
    buf, mode = [], "normal"  # mode: normal | quote

    def flush(buf, mode):
        if not buf:
            return
        chunk = "\n".join(buf)
        if mode == "quote":
            inner = strip_blockquote(chunk).strip("\n")
            first = inner.strip().split("\n", 1)[0]
            cls, label = detect_layer(first)
            jm = re.match(r"\s*📖\s*\*\*([^*]+)\*\*\s*=\s*(.*)$", first.strip()) if cls == "jargon" else None
            if jm:
                # Format: > 📖 **term** = definition  → keep BOTH term and def.
                term, definition = jm.group(1).strip(), jm.group(2).strip()
                blocks.append(
                    f'<div class="callout jargon"><div class="callout-h">Jargon</div>'
                    f'<p><span class="jargon-term">{html.escape(term)}</span> = '
                    f'{html.escape(definition)}</p></div>'
                )
            elif cls:
                rest = inner.split("\n", 1)[1] if "\n" in inner else ""
                # Single-line callouts (Takeaway/Interview/Jargon) carry their
                # text ON the header line after a — or =. Multi-line callouts
                # (plain/readcode/deeper) put the body on the following lines.
                inline = ""
                # strip the leading emoji + **Bold label** from the first line,
                # keeping whatever real text follows it.
                m = re.match(r"\s*[^\w`*]*\*\*[^*]+\*\*\s*(.*)$", first)
                if m:
                    inline = m.group(1).strip()
                    # drop a leading separator dash/equals if present
                    inline = re.sub(r"^[—\-=:]\s*", "", inline)
                body_src = (inline + ("\n\n" if inline and rest.strip() else "")
                            + rest.strip()).strip()
                body_html = md_to_html(body_src)
                blocks.append(
                    f'<div class="callout {cls}"><div class="callout-h">'
                    f'{html.escape(label)}</div>{body_html}</div>'
                )
            else:
                blocks.append('<blockquote>' + md_to_html(inner) + '</blockquote>')
        else:
            h = md_to_html(chunk)
            if h.strip():
                blocks.append(h)

    for ln in lines:
        is_q = ln.startswith(">")
        if is_q and mode != "quote":
            flush(buf, mode); buf, mode = [ln], "quote"
        elif not is_q and mode == "quote":
            # blank line inside a quote group is tolerated
            if ln.strip() == "":
                buf.append(ln)
            else:
                flush(buf, mode); buf, mode = [ln], "normal"
        else:
            buf.append(ln)
    flush(buf, mode)
    return "\n".join(blocks)

def render_output(o):
    t = o.get("output_type")
    if t == "stream":
        txt = "".join(o.get("text", []))
        return f'<pre class="out stream">{html.escape(txt)}</pre>'
    if t in ("execute_result", "display_data"):
        data = o.get("data", {})
        if "image/png" in data:
            b64 = data["image/png"]
            if isinstance(b64, list):
                b64 = "".join(b64)
            return f'<img class="out img" src="data:image/png;base64,{b64.strip()}" alt="chart">'
        if "text/html" in data:
            h = data["text/html"]
            return f'<div class="out html">{"".join(h) if isinstance(h, list) else h}</div>'
        if "text/plain" in data:
            txt = "".join(data["text/plain"]) if isinstance(data["text/plain"], list) else data["text/plain"]
            return f'<pre class="out result">{html.escape(txt)}</pre>'
    if t == "error":
        tb = "\n".join(o.get("traceback", []))
        tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)  # strip ANSI colours
        return f'<pre class="out error">{html.escape(tb)}</pre>'
    return ""

def render_code_cell(cell):
    src = "".join(cell.get("source", []))
    code_html = html.escape(src)
    outs = "".join(render_output(o) for o in cell.get("outputs", []))
    out_block = f'<div class="outputs">{outs}</div>' if outs.strip() else ""
    return (
        '<div class="codecell">'
        f'<pre class="code"><code class="language-python">{code_html}</code></pre>'
        f'{out_block}</div>'
    )

def parse_notebook(path):
    nb = json.loads(Path(path).read_text())
    title, subtitle = None, None
    blocks = []
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            if title is None and src.lstrip().startswith("# "):
                # first H1 = page title; keep the rest as intro
                m = re.match(r"#\s+(.+)", src.lstrip())
                title = m.group(1).strip()
                rest = src.lstrip()[m.end():].strip()
                if rest:
                    blocks.append(f'<div class="intro">{md_to_html(rest)}</div>')
                continue
            blocks.append(render_markdown_cell(src))
        elif cell["cell_type"] == "code":
            if "".join(cell.get("source", [])).strip():
                blocks.append(render_code_cell(cell))
    # plain-text version for search
    text = ""
    for cell in nb.get("cells", []):
        text += "".join(cell.get("source", [])) + " "
    return title or Path(path).stem, "\n".join(blocks), text

# --- module registry --------------------------------------------------------
# Order matters — this drives the sidebar, prev/next, and the landing grid.
# (file stem, short label for the sidebar, emoji)
MODULES = [
    ("0_absolute_basics_python",           "Absolute Basics: Python from Zero", "🐣"),
    ("01_numpy",                            "NumPy: Fast Arrays",                "🔢"),
    ("02_pandas",                           "Pandas: Spreadsheets in Python",    "🐼"),
    ("03_data_cleaning",                    "Data Cleaning",                     "🧹"),
    ("04_eda_visualization",                "EDA & Visualization",               "📊"),
    ("05_statistics_probability",           "Statistics & Probability",          "📈"),
    ("06_feature_engineering_scaling",      "Feature Engineering & Scaling",     "⚙️"),
    ("07_regression",                       "Regression",                        "📉"),
    ("08_classification_knn",               "Classification & KNN",              "🎯"),
    ("09_evaluation_cv_gridsearch",         "Evaluation, CV & Grid Search",      "🔍"),
    ("10_trees_ensembles_boosting",         "Trees, Ensembles & Boosting",       "🌳"),
    ("11_clustering_pca",                   "Clustering & PCA",                  "🔮"),
    ("12_neural_networks_intro",            "Neural Networks Intro",             "🧠"),
    ("13_job_readiness_interview_prep",     "Job Readiness & Interviews",        "💼"),
    ("14_sql_for_data_science",             "SQL for Data Science",              "🗄️"),
    ("15_apis_web_scraping",                "APIs & Web Scraping",               "🕸️"),
    ("16_probability_combinatorics_bayes",  "Probability, Combinatorics, Bayes", "🎲"),
    ("17_ab_testing_anova_power",           "A/B Testing, ANOVA & Power",        "🧪"),
    ("18_oop_linear_algebra_calculus",      "OOP, Linear Algebra & Calculus",    "➗"),
    ("19_time_series",                      "Time Series",                       "⏳"),
    ("20_nlp_recommendation_systems",       "NLP & Recommender Systems",         "💬"),
    ("21_bash_git",                         "Bash & Git",                        "🐚"),
    ("22_bigdata_spark",                    "Big Data & Spark",                  "⚡"),
    ("23_aws_deployment_mlops",             "AWS Deployment & MLOps",            "☁️"),
]

def slug_for(stem):
    return stem

# Per-module metadata: one-line outcome, difficulty tag, est. minutes.
# Keyed by stem. Drives richer cards + lesson-header badges.
META = {
    "0_absolute_basics_python":          ("What code, variables, loops & functions actually are.", "Beginner", 60),
    "01_numpy":                          ("Vectorized thinking — why arrays beat Python loops.", "Beginner", 40),
    "02_pandas":                         ("Load, filter, group & join real tabular data.", "Beginner", 55),
    "03_data_cleaning":                  ("Handle missing values, dupes, dtypes & outliers.", "Beginner", 45),
    "04_eda_visualization":              "Ask questions with charts and read the story.", 
    "05_statistics_probability":         ("Distributions, CLT, p-values & confidence intervals.", "Intermediate", 55),
    "06_feature_engineering_scaling":    ("Encode, scale & transform features without leakage.", "Intermediate", 45),
    "07_regression":                     ("Predict numbers: linear regression from scratch + sklearn.", "Intermediate", 45),
    "08_classification_knn":             ("Predict classes: logistic regression & KNN.", "Intermediate", 45),
    "09_evaluation_cv_gridsearch":       ("Trust your model: CV, the right metric & tuning.", "Intermediate", 45),
    "10_trees_ensembles_boosting":       ("Decision trees, Random Forest & XGBoost.", "Intermediate", 50),
    "11_clustering_pca":                 ("Find structure without labels: KMeans & PCA.", "Intermediate", 45),
    "12_neural_networks_intro":          ("Neurons, activations & backprop intuition.", "Intermediate", 50),
    "13_job_readiness_interview_prep":   ("Tell your story & answer DS interview questions.", "Beginner", 40),
    "14_sql_for_data_science":           ("JOINs, CTEs & window functions for analysts.", "Intermediate", 50),
    "15_apis_web_scraping":              ("Pull data from APIs & scrape the web ethically.", "Intermediate", 45),
    "16_probability_combinatorics_bayes":("Counting, Bayes' theorem & Naive Bayes.", "Advanced", 55),
    "17_ab_testing_anova_power":         ("Design experiments: t-tests, ANOVA & power.", "Advanced", 55),
    "18_oop_linear_algebra_calculus":    ("Classes, eigenvectors & gradient descent by hand.", "Advanced", 60),
    "19_time_series":                    ("Decomposition, stationarity & ARIMA forecasting.", "Advanced", 50),
    "20_nlp_recommendation_systems":     ("TF-IDF, cosine similarity & recommenders.", "Advanced", 55),
    "21_bash_git":                       ("Shell wrangling & the git mental model.", "Beginner", 40),
    "22_bigdata_spark":                  ("Scale out with PySpark when data won't fit RAM.", "Advanced", 50),
    "23_aws_deployment_mlops":           ("Ship models: joblib, Docker, AWS & monitoring.", "Advanced", 55),
}
# normalize the one accidental short entry
META["04_eda_visualization"] = ("Ask questions with charts and read the story.", "Beginner", 45)

def meta_for(stem):
    m = META.get(stem)
    if not m:
        return ("", "", 0)
    return m

# Learning phases (tracks). Each: (title, one-line promise, list of module indices).
PHASES = [
    ("Phase 0 · Foundations", "Start from absolute zero — Python, then fast arrays & tables.", [0, 1, 2, 3]),
    ("Phase 1 · Stats & Modeling", "The math and the first models that turn data into predictions.", [4, 5, 6, 7, 8, 9]),
    ("Phase 2 · Machine Learning Core", "Ensembles, unsupervised learning and neural nets.", [10, 11, 12]),
    ("Phase 3 · Job-Ready Skills", "Interviews, SQL, APIs & the full analyst toolkit.", [13, 14, 15, 16, 17, 18, 19, 20]),
    ("Phase 4 · Production & Scale", "Engineering that gets your models into the real world.", [21, 22, 23]),
]

DIFF_CLASS = {"Beginner": "diff-b", "Intermediate": "diff-i", "Advanced": "diff-a"}

# --- HTML shell -------------------------------------------------------------
def _nav_item(i, active_slug):
    stem, label, emoji = MODULES[i]
    slug = slug_for(stem)
    cls = "active" if slug == active_slug else ""
    num = "0" if i == 0 else str(i)
    return (
        f'<a class="nav-item {cls}" href="/lessons/{slug}.html" data-slug="{slug}">'
        f'<span class="nav-emoji">{emoji}</span>'
        f'<span class="nav-num">M{num}</span>'
        f'<span class="nav-label">{html.escape(label)}</span></a>'
    )

def sidebar_html(active_slug):
    """Sidebar grouped into collapsible phases. The phase containing the active
    lesson stays open; others collapse to keep the nav short."""
    out = []
    active_idx = next((i for i, (s, _, _) in enumerate(MODULES) if s == active_slug), -1)
    for ptitle, _promise, idxs in PHASES:
        is_open = active_idx in idxs
        open_attr = " open" if is_open else ""
        rows = "\n".join(_nav_item(i, active_slug) for i in idxs)
        out.append(
            f'<details class="nav-phase"{open_attr}>'
            f'<summary class="nav-phase-h">{html.escape(ptitle)}</summary>'
            f'<div class="nav-phase-items">{rows}</div>'
            f'</details>'
        )
    return "\n".join(out)

def raw_nb_url(stem):
    return (f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}"
            f"/{GH_BRANCH}/notebooks/{stem}.ipynb")

def run_bar(stem):
    """The 'run this lesson for real' bar shown at the top of every lesson."""
    return (
        '<div class="runbar">'
        '<span class="runbar-label">▶ Run this lesson yourself:</span>'
        f'<a class="runbtn colab" href="{colab_url(stem)}" target="_blank" rel="noopener">'
        'Open in Colab</a>'
        f'<a class="runbtn gh" href="{github_nb_url(stem)}" target="_blank" rel="noopener">'
        'View on GitHub</a>'
        f'<a class="runbtn dl" href="{raw_nb_url(stem)}" download>Download .ipynb</a>'
        '<span class="runbar-hint">In Colab, press <kbd>Shift</kbd>+<kbd>Enter</kbd> to run each cell.</span>'
        '</div>'
    )

def lesson_header(module_num, stem, title):
    """Title + difficulty/time badges shown at the top of each lesson."""
    outcome, diff, mins = meta_for(stem)
    badges = []
    if diff:
        badges.append(f'<span class="badge {DIFF_CLASS.get(diff,"")}">{diff}</span>')
    if mins:
        badges.append(f'<span class="badge time">⏱ ~{mins} min</span>')
    badges_html = "".join(badges)
    outcome_html = f'<p class="lesson-goal">🎯 {html.escape(outcome)}</p>' if outcome else ""
    return (
        f'<div class="crumb">Module {module_num}</div>'
        f'<div class="lesson-badges">{badges_html}</div>'
        f'{outcome_html}'
    )

def page_shell(title, body, active_slug, prev_link, next_link, module_num, stem):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — DS Bridging Bootcamp</title>
<meta name="description" content="Data science from zero to job-ready — every idea in plain English, then the code, then the deep why.">
<link rel="icon" href="{FAVICON}">
<link rel="stylesheet" href="/assets/style.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
</head>
<body>
<button id="menu-toggle" aria-label="Toggle navigation menu" aria-expanded="false" aria-controls="sidebar">☰</button>
<div id="sidebar-overlay"></div>
<aside id="sidebar">
  <a class="brand" href="/index.html">
    {logo_svg(34)}
    <span class="brand-text">DS Bridging<br><small>Bootcamp</small></span>
  </a>
  <div class="search-box">
    <input id="search" type="search" placeholder="Search lessons…" autocomplete="off">
    <div id="search-results"></div>
  </div>
  <nav id="nav">
    {sidebar_html(active_slug)}
  </nav>
  <a class="nav-foot" href="https://github.com/{GH_OWNER}/{GH_REPO}/issues" target="_blank" rel="noopener">💬 Feedback / report an issue</a>
</aside>
<main id="content">
  <div class="lesson-wrap">
    {lesson_header(module_num, stem, title)}
    <h1 id="top">{html.escape(title)}</h1>
    {run_bar(stem)}
    <div id="lesson-toc" class="toc" aria-label="On this page"></div>
    {body}
    <div class="lesson-nav">
      {prev_link}
      {next_link}
    </div>
  </div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script src="/assets/app.js"></script>
</body>
</html>"""

# --- main build -------------------------------------------------------------
SRC_ASSETS = Path(__file__).resolve().parent / "site_assets"  # canonical css/js

def copy_static_assets():
    """Copy the hand-written CSS/JS into the output assets/ dir.
    Critical: without this a fresh build (e.g. Vercel) has no style.css/app.js,
    causing 404s and an unstyled page."""
    import shutil
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ("style.css", "app.js"):
        src = SRC_ASSETS / name
        if src.exists():
            shutil.copy2(src, ASSETS / name)
        else:
            print(f"WARNING: missing source asset {src}")

def build():
    import shutil
    # Start from a clean lessons/ dir so renamed/renumbered modules never leave
    # stale pages behind (which would otherwise ship dead links to Vercel).
    if LESSONS.exists():
        shutil.rmtree(LESSONS)
    LESSONS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    copy_static_assets()
    search_index = []
    built = []

    present = [(s, l, e) for (s, l, e) in MODULES if (NB_DIR / f"{s}.ipynb").exists()]
    print(f"Found {len(present)}/{len(MODULES)} notebooks present.")

    for idx, (stem, label, emoji) in enumerate(present):
        path = NB_DIR / f"{stem}.ipynb"
        title, body, text = parse_notebook(path)
        slug = slug_for(stem)
        module_num = "0" if MODULES.index((stem, label, emoji)) == 0 else str(MODULES.index((stem, label, emoji)))

        prev_link = next_link = ""
        if idx > 0:
            ps, pl, pe = present[idx - 1]
            prev_link = f'<a class="pn prev" href="/lessons/{slug_for(ps)}.html">← {html.escape(pl)}</a>'
        else:
            prev_link = '<a class="pn prev" href="/index.html">← Home</a>'
        if idx < len(present) - 1:
            ns, nl, ne = present[idx + 1]
            next_link = f'<a class="pn next" href="/lessons/{slug_for(ns)}.html">{html.escape(nl)} →</a>'

        page = page_shell(title, body, slug, prev_link, next_link, module_num, stem)
        (LESSONS / f"{slug}.html").write_text(page)
        built.append((slug, title, emoji, module_num, label))

        # search index: strip markdown/emoji, keep words
        clean = re.sub(r"\s+", " ", re.sub(r"[#>*`_\-]", " ", text)).strip()
        search_index.append({
            "slug": slug, "title": title, "label": label,
            "num": module_num, "emoji": emoji,
            "text": clean[:4000]
        })

    (ASSETS / "search-index.json").write_text(json.dumps(search_index, ensure_ascii=False))
    build_index(built)
    print(f"Built {len(built)} lesson pages + index.html")

def _card(built_by_idx, i):
    slug, title, emoji, num, label = built_by_idx[i]
    stem = MODULES[i][0]
    outcome, diff, mins = meta_for(stem)
    tags = []
    if diff:
        tags.append(f'<span class="badge {DIFF_CLASS.get(diff,"")}">{diff}</span>')
    if mins:
        tags.append(f'<span class="badge time">⏱ ~{mins}m</span>')
    tags_html = "".join(tags)
    return (
        f'<a class="card" href="/lessons/{slug}.html" data-slug="{slug}">'
        f'<div class="card-top"><span class="card-emoji">{emoji}</span>'
        f'<span class="card-num">Module {num}</span></div>'
        f'<div class="card-title">{html.escape(label)}</div>'
        f'<div class="card-outcome">{html.escape(outcome)}</div>'
        f'<div class="card-tags">{tags_html}</div>'
        f'<span class="card-go">Open lesson →</span></a>'
    )

def build_index(built):
    # index-aligned lookup (built preserves MODULES order)
    total = len(built)
    first_slug = built[0][0] if built else ""

    # phase-grouped roadmap
    phase_blocks = []
    for ptitle, promise, idxs in PHASES:
        cards = "\n".join(_card(built, i) for i in idxs if i < total)
        phase_blocks.append(
            f'<div class="phase">'
            f'<div class="phase-head"><h3>{html.escape(ptitle)}</h3>'
            f'<p class="phase-promise">{html.escape(promise)}</p></div>'
            f'<div class="grid">{cards}</div>'
            f'</div>'
        )
    roadmap_html = "\n".join(phase_blocks)

    index = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DS Bridging Bootcamp — Learn Data Science from Zero</title>
<meta name="description" content="A free {total}-module data science bootcamp: from absolute zero to job-ready. Every idea in plain English, then the code, then the deep why.">
<link rel="icon" href="{FAVICON}">
<link rel="stylesheet" href="/assets/style.css">
</head>
<body class="home">

<nav class="topnav" id="topnav">
  <a class="topnav-brand" href="#top">{logo_svg(28)}<span>DS Bridging</span></a>
  <div class="topnav-links">
    <a href="#how">How it works</a>
    <a href="#roadmap">Curriculum</a>
    <a href="#sample">Sample lesson</a>
    <a href="#about">About</a>
    <a class="topnav-cta" href="/lessons/{first_slug}.html">Start →</a>
  </div>
</nav>

<header class="hero" id="top">
  <div class="hero-inner">
    <div class="hero-logo">{logo_svg(72)}</div>
    <h1>Bridge into <span class="grad">Data Science</span></h1>
    <p class="tagline">A free, {total}-module path from <strong>absolute zero</strong> to
       <strong>job-ready</strong> — every idea in plain English first, then the code
       line-by-line, then the deep why.</p>
    <p class="cred">Built by a mathematics graduate for self-taught coders and career-switchers
       who have projects but shaky foundations.</p>
    <div class="hero-cta">
      <a class="cta" href="/lessons/{first_slug}.html">Start with Module 0 →</a>
      <a class="cta ghost" href="#sample">Preview a lesson</a>
    </div>
    <div class="hero-badges">
      <span>🎓 {total} modules</span>
      <span>🧩 4-layer method</span>
      <span>▶ Runnable notebooks</span>
      <span>💼 Interview-ready</span>
    </div>
  </div>
</header>

<section class="how" id="how">
  <h2>How every lesson works</h2>
  <p class="section-sub">The same four layers, every time — so you can stop at intuition, or go all the way to the math.</p>
  <div class="how-grid">
    <div class="how-box plain"><b>🌱 In plain English</b><p>The idea with zero jargon and an everyday analogy.</p></div>
    <div class="how-box readcode"><b>🔤 Reading the code</b><p>Every line explained, one at a time.</p></div>
    <div class="how-box deeper"><b>🎓 Go deeper</b><p>The rigorous why — for when you're ready.</p></div>
    <div class="how-box takeaway"><b>✅ Takeaway + 🗣️ interview</b><p>One line to remember, one line to say in an interview.</p></div>
  </div>
</section>

<section class="roadmap" id="roadmap">
  <h2>Your learning roadmap</h2>
  <p class="section-sub">Five phases, {total} modules — a guided journey, not a pile of tutorials.</p>
  {roadmap_html}
</section>

<section class="sample" id="sample">
  <h2>See the method in action</h2>
  <p class="section-sub">Example: <b>What is gradient descent?</b></p>
  <div class="sample-card">
    <div class="callout plain"><div class="callout-h">🌱 In plain English</div>
      <p>Like walking downhill in thick fog: you can't see the bottom, so you feel which way the ground slopes and take a small step that way — again and again — until it's flat.</p></div>
    <div class="callout readcode"><div class="callout-h">🔤 Reading the code</div>
      <pre class="code"><code class="language-python">w = w - lr * grad   # step against the slope; lr = how big a step</code></pre></div>
    <div class="callout deeper"><div class="callout-h">🎓 Go deeper</div>
      <p>Gradient descent minimises a loss function by repeatedly updating parameters in the direction opposite the gradient. The learning rate <code>lr</code> trades speed for stability.</p></div>
    <div class="callout takeaway"><div class="callout-h">✅ Takeaway + 🗣️ interview</div>
      <p><b>Remember:</b> follow the slope downhill in small steps. <b>Say it:</b> "Gradient descent iteratively moves parameters opposite the gradient of the loss to find a minimum."</p></div>
  </div>
  <div class="center"><a class="cta" href="/lessons/{first_slug}.html">Start learning →</a></div>
</section>

<section class="about" id="about">
  <h2>Who this is for</h2>
  <div class="about-grid">
    <div class="about-box"><h3>👤 Built for</h3><ul>
      <li>Absolute beginners starting from zero</li>
      <li>Self-taught coders with shaky foundations</li>
      <li>Students switching into data science</li>
      <li>Anyone prepping for DS interviews</li>
    </ul></div>
    <div class="about-box"><h3>🎁 You'll leave with</h3><ul>
      <li>Real Python & data intuition</li>
      <li>Mental models for core ML algorithms</li>
      <li>Interview-ready explanations you can defend</li>
      <li>Portfolio-ready, runnable notebooks</li>
    </ul></div>
    <div class="about-box"><h3>🧭 How to use it</h3><ul>
      <li>Work through the phases in order</li>
      <li>Run every notebook in Colab as you read</li>
      <li>Skim 🎓 boxes on the first pass; return for depth</li>
      <li>Explain each takeaway out loud</li>
    </ul></div>
  </div>
</section>

<footer class="foot" id="foot">
  <div class="center"><a class="cta" href="/lessons/{first_slug}.html">Begin at Module 0 →</a></div>
  <p>Built by <strong>Stephen Muema</strong> · Mathematics + Data Science ·
     <a href="https://github.com/{GH_OWNER}/{GH_REPO}">GitHub</a> ·
     <a href="https://github.com/{GH_OWNER}/{GH_REPO}/issues">Feedback</a></p>
</footer>
<script src="/assets/app.js"></script>
</body>
</html>"""
    (SITE / "index.html").write_text(index)

if __name__ == "__main__":
    build()
