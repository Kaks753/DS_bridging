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
SITE = ROOT / "site"
LESSONS = SITE / "lessons"
ASSETS = SITE / "assets"

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
    ("0_absolute_basics_python",      "Absolute Basics: Python from Zero", "🐣"),
    ("01_numpy",                       "NumPy: Fast Arrays",               "🔢"),
    ("02_pandas",                      "Pandas: Spreadsheets in Python",   "🐼"),
    ("02_data_cleaning",               "Data Cleaning",                    "🧹"),
    ("03_eda_visualization",           "EDA & Visualization",              "📊"),
    ("04_statistics_probability",      "Statistics & Probability",         "📈"),
    ("05_feature_engineering_scaling", "Feature Engineering & Scaling",    "⚙️"),
    ("06_regression",                  "Regression",                       "📉"),
    ("07_classification_knn",          "Classification & KNN",             "🎯"),
    ("08_evaluation_cv_gridsearch",    "Evaluation, CV & Grid Search",     "🔍"),
    ("09_trees_ensembles_boosting",    "Trees, Ensembles & Boosting",      "🌳"),
    ("10_clustering_pca",              "Clustering & PCA",                 "🔮"),
    ("11_neural_networks_intro",       "Neural Networks Intro",            "🧠"),
    ("12_job_readiness_interview_prep","Job Readiness & Interviews",       "💼"),
    ("13_sql_for_data_science",        "SQL for Data Science",             "🗄️"),
    ("14_apis_web_scraping",           "APIs & Web Scraping",              "🕸️"),
    ("15_probability_combinatorics_bayes","Probability, Combinatorics, Bayes","🎲"),
    ("16_ab_testing_anova_power",      "A/B Testing, ANOVA & Power",       "🧪"),
    ("17_oop_linear_algebra_calculus", "OOP, Linear Algebra & Calculus",   "➗"),
    ("18_time_series",                 "Time Series",                      "⏳"),
    ("19_nlp_recommendation_systems",  "NLP & Recommender Systems",        "💬"),
    ("20_bash_git",                    "Bash & Git",                       "🐚"),
    ("21_bigdata_spark",               "Big Data & Spark",                 "⚡"),
    ("22_aws_deployment_mlops",        "AWS Deployment & MLOps",           "☁️"),
]

def slug_for(stem):
    return stem

# --- HTML shell -------------------------------------------------------------
def sidebar_html(active_slug):
    items = []
    for i, (stem, label, emoji) in enumerate(MODULES):
        slug = slug_for(stem)
        cls = "active" if slug == active_slug else ""
        num = "0" if i == 0 else str(i)
        items.append(
            f'<a class="nav-item {cls}" href="{slug}.html" data-slug="{slug}">'
            f'<span class="nav-emoji">{emoji}</span>'
            f'<span class="nav-num">M{num}</span>'
            f'<span class="nav-label">{html.escape(label)}</span></a>'
        )
    return "\n".join(items)

def page_shell(title, body, active_slug, prev_link, next_link, module_num):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — DS Bridging Bootcamp</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%9A%80%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="../assets/style.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
</head>
<body>
<button id="menu-toggle" aria-label="Toggle menu">☰</button>
<aside id="sidebar">
  <a class="brand" href="../index.html">
    <span class="brand-logo">🚀</span>
    <span class="brand-text">DS Bridging<br><small>Bootcamp</small></span>
  </a>
  <div class="search-box">
    <input id="search" type="search" placeholder="Search lessons…" autocomplete="off">
    <div id="search-results"></div>
  </div>
  <nav id="nav">
    {sidebar_html(active_slug)}
  </nav>
</aside>
<main id="content">
  <div class="lesson-wrap">
    <div class="crumb">Module {module_num}</div>
    {body}
    <div class="lesson-nav">
      {prev_link}
      {next_link}
    </div>
  </div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script src="../assets/app.js"></script>
</body>
</html>"""

# --- main build -------------------------------------------------------------
def build():
    LESSONS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
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
            prev_link = f'<a class="pn prev" href="{slug_for(ps)}.html">← {html.escape(pl)}</a>'
        else:
            prev_link = '<a class="pn prev" href="../index.html">← Home</a>'
        if idx < len(present) - 1:
            ns, nl, ne = present[idx + 1]
            next_link = f'<a class="pn next" href="{slug_for(ns)}.html">{html.escape(nl)} →</a>'

        page = page_shell(title, body, slug, prev_link, next_link, module_num)
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

def build_index(built):
    cards = []
    for slug, title, emoji, num, label in built:
        cards.append(
            f'<a class="card" href="lessons/{slug}.html">'
            f'<div class="card-top"><span class="card-emoji">{emoji}</span>'
            f'<span class="card-num">Module {num}</span></div>'
            f'<div class="card-title">{html.escape(label)}</div></a>'
        )
    cards_html = "\n".join(cards)
    total = len(built)
    index = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DS Bridging Bootcamp — Learn Data Science from Zero</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%9A%80%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="assets/style.css">
</head>
<body class="home">
<header class="hero">
  <div class="hero-inner">
    <div class="hero-logo">🚀</div>
    <h1>DS Bridging Bootcamp</h1>
    <p class="tagline">Data science from <strong>absolute zero</strong> to job-ready —
       every idea in plain English first, then the code line-by-line, then the deep why.</p>
    <div class="hero-badges">
      <span>{total} modules</span>
      <span>4-layer teaching</span>
      <span>Runnable notebooks</span>
      <span>Interview-ready</span>
    </div>
    <a class="cta" href="lessons/{built[0][0] if built else ''}.html">Start with Module 0 →</a>
  </div>
</header>
<section class="how">
  <h2>How each lesson works</h2>
  <div class="how-grid">
    <div class="how-box plain"><b>🌱 In plain English</b><p>The idea with zero jargon and an everyday analogy.</p></div>
    <div class="how-box readcode"><b>🔤 Reading the code</b><p>Every line explained, one at a time.</p></div>
    <div class="how-box deeper"><b>🎓 Go deeper</b><p>The rigorous why — for when you're ready.</p></div>
    <div class="how-box takeaway"><b>✅ Takeaway + 🗣️ interview</b><p>One line to remember, one line to say in an interview.</p></div>
  </div>
</section>
<section class="grid-wrap">
  <h2>The curriculum</h2>
  <div class="grid">
    {cards_html}
  </div>
</section>
<footer class="foot">
  <p>Built by <strong>Stephen Muema</strong> · Mathematics + Data Science ·
     <a href="https://github.com/Kaks753/DS_bridging">GitHub</a></p>
</footer>
</body>
</html>"""
    (SITE / "index.html").write_text(index)

if __name__ == "__main__":
    build()
