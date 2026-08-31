# DS Bridging Bootcamp — Learning Website

A static site generated from the course notebooks. Every lesson is rendered with
the **4-layer teaching pattern** styled into colour-coded callouts:

- 🌱 **In plain English** (green) — the idea with zero jargon + an analogy
- 🔤 **Reading the code** (blue) — every line explained
- 🎓 **Go deeper** (purple) — the rigorous why
- ✅ **Takeaway** (amber) + 🗣️ **interview line** (pink)
- 🧠 analogy · ⚠️ watch out · ✍️ try this · 📖 jargon

## Rebuild the site

From the repo root:

```bash
pip install markdown
python3 scripts/build_site.py            # -> site/   (local preview)
SITE_OUT=public python3 scripts/build_site.py   # -> public/ (Vercel serves this)
```

This reads `notebooks/*.ipynb` and writes into the output dir:

- `index.html` — landing page + module grid
- `lessons/<module>.html` — one page per module
- `assets/style.css`, `assets/app.js` — copied from `scripts/site_assets/`
- `assets/search-index.json` — client-side search index

> The generator **copies** `style.css`/`app.js` from `scripts/site_assets/`
> into the output `assets/` on every build. (Editing those canonical files is
> how you change the theme.) This copy step is what makes a clean Vercel
> checkout render styled — without it the deploy 404s on the CSS.

## Preview locally

```bash
cd site && python3 -m http.server 8099
# open http://localhost:8099
```

## Deploy to Vercel

> ⚠️ Never paste your Vercel token into chat. Log in locally instead.

The **root** `vercel.json` tells Vercel this is a **pre-built static site**:
it skips install/build and serves the committed **`public/`** folder as
`outputDirectory`. A root `.vercelignore` hides the Python files so Vercel
does NOT mistake the repo for a Python app (that was the cause of the
"No python entrypoint found" error).

> ⚠️ If your Vercel **project dashboard** has an Output Directory or Root
> Directory set from a previous import, it OVERRIDES `vercel.json`. In the
> dashboard set: Framework = **Other**, Root Directory = **(empty)**,
> Output Directory = **public**, Build Command = **(empty/override off)**.

### Option 1 — GitHub auto-deploy (easiest)
1. Go to https://vercel.com/new and import `Kaks753/DS_bridging`.
2. Framework preset: **Other**. Leave build settings empty (root `vercel.json` handles it).
3. Deploy. Every push to `main` re-deploys automatically.

### Option 2 — CLI from your machine
```bash
npm i -g vercel        # once
vercel login           # opens browser, uses YOUR account (no token in chat)
vercel --prod          # run from the REPO ROOT (not site/)
```

Because the site is committed pre-built, no Python runs on Vercel at all.
