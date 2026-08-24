"""Builder for Module 14: APIs & Web Scraping."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 14 — APIs & Web Scraping (Phase 1: Topics 9–10)

Real data rarely arrives as a clean CSV. You fetch it from **APIs** (structured,
preferred) or **scrape** it from web pages (last resort). Your Kenya Economic Pulse
project pulled from the World Bank API — this module is the foundation under that.

Goals:
- What an API is; HTTP methods, status codes, JSON, params, headers, auth.
- Use `requests` correctly (with error handling & rate-limit awareness).
- Parse HTML with **BeautifulSoup** (fully reproducible on a local page here).
- Scrape **ethically & legally** (robots.txt, ToS, throttling).
- Turn messy responses into a tidy DataFrame.
""")

nb.md(r"""
## 14.1 What is an API?

An **API** (Application Programming Interface) lets your code request data from a
server. A **REST** API works over HTTP:

- **GET** — read data (most common for us). **POST** — send/create data.
- The server replies with a **status code** and (usually) a **JSON** body.
- **Status codes**: `200` OK · `201` created · `400` bad request · `401/403`
  auth problem · `404` not found · `429` too many requests (rate-limited) ·
  `500` server error.
- **Query params** refine the request (`?country=KE&year=2020`); **headers** carry
  metadata like an **API key** for auth.

Think of it as a restaurant: you (client) order from a menu (endpoints), the
kitchen (server) returns your dish (JSON).
""")

nb.code(r"""
import requests
import pandas as pd
import json

# We call a real, key-free public API (World Bank), but wrap it so the notebook
# STILL RUNS if the sandbox has no internet — falling back to a saved sample.
URL = "https://api.worldbank.org/v2/country/KE/indicator/NY.GDP.MKTP.CD"
params = {"format": "json", "date": "2015:2020", "per_page": 100}

def safe_get(url, params=None, timeout=8):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()          # raises on 4xx/5xx
        return r.json(), r.status_code, True
    except Exception as e:
        print(f"[offline fallback] network call failed ({type(e).__name__}); "
              f"using embedded sample data.")
        return None, None, False

payload, status, online = safe_get(URL, params)
print("online:", online, "| status:", status)
""")

nb.code(r"""
# World Bank returns [metadata, data]. Provide a small embedded sample as fallback.
sample = [
    {"date": "2020", "value": 100380000000},
    {"date": "2019", "value": 100550000000},
    {"date": "2018", "value": 92210000000},
    {"date": "2017", "value": 82040000000},
    {"date": "2016", "value": 74820000000},
    {"date": "2015", "value": 70120000000},
]

if online and isinstance(payload, list) and len(payload) == 2 and payload[1]:
    records = payload[1]
    rows = [{"year": int(d["date"]), "gdp_usd": d["value"]} for d in records
            if d.get("value") is not None]
else:
    rows = [{"year": int(d["date"]), "gdp_usd": d["value"]} for d in sample]

gdp = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
gdp["gdp_usd_bn"] = (gdp["gdp_usd"] / 1e9).round(1)
print("Kenya GDP (current US$) — tidy DataFrame from a JSON API:")
gdp
""")

nb.md(r"""
**What just happened:** we sent a GET with query params, checked the status, parsed
JSON, and flattened the nested structure into a tidy DataFrame — the exact loop
behind your Economic Pulse dashboard. Note the **error handling**: production code
never assumes the network works.
""")

nb.md(r"""
## 14.2 Reading API docs (the real skill)

You'll spend more time reading docs than coding. For any API, find:
1. **Base URL** and **endpoints** (what data is available).
2. **Auth**: none / API key in header / OAuth token.
3. **Parameters**: filtering, date ranges, fields.
4. **Pagination**: how to get page 2, 3, … (`per_page`, `offset`, `cursor`).
5. **Rate limits**: requests/min — respect them or get `429`.

Pattern for a key + pagination (pseudocode you can adapt):
```python
headers = {"Authorization": f"Bearer {API_KEY}"}
all_rows, page = [], 1
while True:
    r = requests.get(url, params={"page": page, "per_page": 100}, headers=headers)
    data = r.json()
    if not data: break
    all_rows.extend(data); page += 1
    time.sleep(0.5)            # be polite: throttle between requests
```
""")

nb.md(r"""
## 14.3 Web scraping with BeautifulSoup

When there's **no API**, you parse HTML. We use a local HTML string here so the
lesson is 100% reproducible, but the parsing code is identical for a real
`requests.get(url).text`.
""")

nb.code(r"""
from bs4 import BeautifulSoup

# Imagine this came from: html = requests.get(url).text
html = '''
<html><body>
  <h1>Top Stocks</h1>
  <table id="stocks">
    <tr><th>Ticker</th><th>Price</th><th>Sector</th></tr>
    <tr><td class="tk">SCOM</td><td class="pr">18.50</td><td>Telecom</td></tr>
    <tr><td class="tk">EQTY</td><td class="pr">45.20</td><td>Banking</td></tr>
    <tr><td class="tk">KCB</td><td class="pr">38.75</td><td>Banking</td></tr>
    <tr><td class="tk">EABL</td><td class="pr">155.00</td><td>Consumer</td></tr>
  </table>
  <a href="/page/2">Next</a>
</body></html>
'''

soup = BeautifulSoup(html, "html.parser")
print("page title (h1):", soup.find("h1").text)
print("the 'Next' link:", soup.find("a")["href"])
""")

nb.code(r"""
# Extract the table into a DataFrame — the core scraping task
rows = []
for tr in soup.select("table#stocks tr")[1:]:      # skip header row
    cells = tr.find_all("td")
    rows.append({
        "ticker": cells[0].text.strip(),
        "price":  float(cells[1].text.strip()),
        "sector": cells[2].text.strip(),
    })

stocks = pd.DataFrame(rows)
print(stocks)
print("\naverage price by sector:")
print(stocks.groupby("sector")["price"].mean().round(2))
""")

nb.md(r"""
**Selectors you'll use most:**
- `soup.find("tag")` — first match; `soup.find_all("tag")` — all matches.
- `soup.select("css selector")` — powerful CSS querying (`#id`, `.class`, `a > b`).
- `.text` / `.get_text()` — the visible text; `element["attr"]` — an attribute.

**Tip:** many tables can be grabbed in one line with `pd.read_html(html)` — try it
before hand-parsing.
""")

nb.code(r"""
# pandas can often read HTML tables directly:
tables = pd.read_html(html)          # returns a list of DataFrames
print("pd.read_html found", len(tables), "table(s):")
tables[0]
""")

nb.md(r"""
## 14.4 Scrape ethically and legally (say this in interviews)

- **Check `robots.txt`** (`site.com/robots.txt`) and the site's **Terms of
  Service** — some forbid scraping.
- **Throttle**: add `time.sleep()`; never hammer a server (that's abusive and can
  get you IP-banned).
- **Identify yourself** with a `User-Agent` header; consider caching responses.
- **Prefer an API** if one exists — it's more stable and permitted.
- Don't scrape **personal/copyrighted** data you have no right to use.

> "I always check robots.txt and ToS, prefer an official API, throttle my requests,
> and cache results so I never re-hit a server unnecessarily."
""")

nb.md(r"""
## 14.5 Mini-exercises

1. Modify `safe_get` to retry up to 3 times with a short `time.sleep` on failure.
2. From the stocks table, scrape only **Banking** rows into a DataFrame.
3. Given the `<a href="/page/2">`, write the loop logic to follow pagination
   (pseudocode is fine).
4. Explain the difference between using an API and scraping, and when you'd choose
   each.
""")

nb.md(r"""
## Summary

- APIs return structured **JSON**; use `requests.get(url, params, headers)`, check
  **status codes**, and **handle errors**. Mind **auth**, **pagination**,
  **rate limits**.
- **Scrape** only when there's no API; parse with **BeautifulSoup** (`find`,
  `select`) or `pd.read_html`; then tidy into a DataFrame.
- Always scrape **ethically/legally**: robots.txt, ToS, throttle, identify, cache.

Next: **Module 15 — Probability, Combinatorics & Bayes** (the math of uncertainty).
""")

out = nb.save("notebooks/14_apis_web_scraping.ipynb")
print("saved", out)
