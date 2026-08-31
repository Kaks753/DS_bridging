"""Builder for Module 15: APIs & Web Scraping (4-layer rewrite of old M14)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 15 — APIs & Web Scraping: getting data that isn't handed to you

Real data rarely arrives as a tidy CSV. You fetch it from **APIs** (structured, and
always the preferred route) or, as a last resort, you **scrape** it off web pages.
Your Kenya Economic Pulse project pulled live figures from the World Bank API — this
module is the foundation under exactly that kind of work.
""")

nb.analogy("An API is ordering from a restaurant menu: you ask for a specific dish (endpoint) "
           "with options (parameters) and the kitchen hands you a neat plated meal (JSON). "
           "Web scraping is being let into the kitchen to pick food off the counters yourself "
           "— messier, more fragile, and only done when there's no menu.")

nb.md("## 15.1 What is an API?")

nb.plain("""
An API (Application Programming Interface) lets your code ask a server for data. The
common web kind is a REST API over HTTP: you send a GET request to read data, the
server replies with a status code and usually a JSON body. Query parameters refine
what you ask for; headers carry metadata like an API key for authentication.
""")

nb.md(r"""
The status codes worth memorising:
- **200** OK · **201** created
- **400** bad request · **401 / 403** authentication problem · **404** not found
- **429** too many requests (you're being rate-limited) · **500** server error
""")

nb.code(r"""
import requests
import pandas as pd

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

nb.readcode("""
- `requests.get(url, params=..., timeout=...)` sends the GET; `params` becomes the
  ?key=value query string automatically.
- `r.raise_for_status()` turns any 4xx/5xx into an exception so we notice failures.
- The try/except wraps it so a dead network doesn't crash the lesson — production code
  NEVER assumes the network works.
""")

nb.code(r"""
# World Bank returns [metadata, data]. Provide a small embedded sample as fallback.
sample = [
    {"date": "2020", "value": 100380000000}, {"date": "2019", "value": 100550000000},
    {"date": "2018", "value": 92210000000},  {"date": "2017", "value": 82040000000},
    {"date": "2016", "value": 74820000000},  {"date": "2015", "value": 70120000000},
]

if online and isinstance(payload, list) and len(payload) == 2 and payload[1]:
    records = payload[1]
    rows = [{"year": int(d["date"]), "gdp_usd": d["value"]} for d in records
            if d.get("value") is not None]
else:
    rows = [{"year": int(d["date"]), "gdp_usd": d["value"]} for d in sample]

gdp = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
gdp["gdp_usd_bn"] = (gdp["gdp_usd"] / 1e9).round(1)
print("Kenya GDP (current US$) — tidy DataFrame built from a JSON API:")
gdp
""")

nb.takeaway("The whole API loop: send a GET with params → check the status → parse the JSON → "
            "flatten the nested structure into a tidy DataFrame. That's the exact pipeline "
            "behind your Economic Pulse dashboard, error-handling included.")

nb.jargon("API", "an interface that lets your code request data/actions from a server")
nb.jargon("REST API", "an API over HTTP using GET/POST etc. and usually returning JSON")
nb.jargon("JSON", "JavaScript Object Notation — the standard nested text format APIs return")
nb.jargon("status code", "the number an HTTP server returns saying how the request went (200, 404, 429...)")

nb.md("## 15.2 Reading API docs — the real skill")

nb.deeper("""
You'll spend more time reading API docs than writing request code. For any API, hunt
down five things: the base URL and endpoints (what's available); the auth method
(none / API key in a header / OAuth); the parameters (filters, date ranges, fields);
the pagination scheme (how to get page 2, 3, … via per_page / offset / cursor); and
the rate limits (requests per minute — exceed them and you get 429s). Master those
five and you can consume almost any API in the wild, not just the ones with tutorials.
""")

nb.md(r"""
The pattern for an API key + pagination (adapt this shape to any service):
```python
headers = {"Authorization": f"Bearer {API_KEY}"}
all_rows, page = [], 1
while True:
    r = requests.get(url, params={"page": page, "per_page": 100}, headers=headers)
    data = r.json()
    if not data:
        break
    all_rows.extend(data)
    page += 1
    time.sleep(0.5)            # be polite: throttle between requests
```
""")

nb.jargon("endpoint", "a specific URL path of an API that returns a particular resource")
nb.jargon("pagination", "splitting a large result across pages you fetch one at a time")
nb.jargon("rate limit", "a cap on how many requests you may send per time window")

nb.md("## 15.3 Web scraping with BeautifulSoup")

nb.plain("""
When a site has no API, you fall back to parsing its HTML. We use a local HTML string
here so the lesson runs identically every time, but the parsing code is exactly what
you'd run on `requests.get(url).text` from a live page.
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
print("the 'Next' link :", soup.find("a")["href"])
""")

nb.code(r"""
# Extract the table into a DataFrame — the core scraping task
rows = []
for tr in soup.select("table#stocks tr")[1:]:      # [1:] skips the header row
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

nb.readcode("""
- `soup.find('tag')` grabs the first match; `soup.select('css selector')` uses CSS
  querying (#id, .class, 'a > b') — often the cleanest way to target elements.
- `.text` gets an element's visible text; `element['href']` reads an attribute.
- We loop table rows (skipping the header), pull each <td>, and assemble a tidy DataFrame.
""")

nb.code(r"""
# Shortcut: pandas can often read HTML tables in ONE line — try this before hand-parsing
tables = pd.read_html(html)          # returns a list of DataFrames
print("pd.read_html found", len(tables), "table(s):")
tables[0]
""")

nb.deeper("""
Scraping is inherently fragile: it breaks the moment the site changes its HTML, so
treat it as a last resort and isolate the parsing so it's easy to fix. Reach for
`pd.read_html` first for straightforward tables; drop to BeautifulSoup when you need
specific elements, attributes, or messy non-table content. And whatever you extract,
the goal is always the same — a clean DataFrame you can model on.
""")

nb.jargon("web scraping", "extracting data by parsing a web page's HTML (when no API exists)")
nb.jargon("BeautifulSoup", "a Python library for navigating and searching HTML/XML")

nb.md("## 15.4 Scrape ethically and legally")

nb.warn("""
Scraping carelessly can get you IP-banned or into legal trouble. The rules: check the
site's robots.txt and Terms of Service (some forbid scraping outright); THROTTLE with
time.sleep() so you never hammer a server; identify yourself with a User-Agent header;
prefer an official API if one exists; and never scrape personal or copyrighted data
you have no right to use.
""")

nb.interview("""
Say this and you sound responsible, not reckless: 'I always check robots.txt and the
Terms of Service, prefer an official API when one exists, throttle my requests with a
delay, set a descriptive User-Agent, and cache results so I never re-hit a server
unnecessarily.' Ethics awareness is itself a hiring signal for data roles.
""")

nb.md("## 15.5 Try it yourself")

nb.try_this("""
1. Modify `safe_get` to retry up to 3 times with a short time.sleep on failure.
2. From the stocks table, scrape only the Banking rows into a DataFrame.
3. Given `<a href="/page/2">`, write the loop logic to follow pagination (pseudocode fine).
4. Explain in two sentences when you'd use an API vs scraping, and why the API wins.
""")

nb.md("## Summary")

nb.takeaway("""
- APIs return structured **JSON**; use `requests.get(url, params, headers)`, check **status codes**, and **handle errors**. Mind **auth**, **pagination**, and **rate limits**.
- **Scrape** only when there's no API; parse with **BeautifulSoup** (`find`, `select`) or `pd.read_html`, then tidy into a DataFrame.
- Always scrape **ethically and legally**: robots.txt, ToS, throttle, identify, cache.
- The end goal of either route is the same: a clean DataFrame you can analyse.
""")

nb.md(r"""
Next: **Module 16 — Probability, Combinatorics & Bayes** — the mathematics of
uncertainty that underpins every model you've built.
""")

out = nb.save("notebooks/15_apis_web_scraping.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
