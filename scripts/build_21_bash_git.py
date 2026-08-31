"""Builder for Module 21: Bash & Git for Data Science (4-layer)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 21 — Bash & Git for Data Scientists

These aren't "data science" per se, but you'll use them **every single day**, and they
show up in interviews and take-home reviews. This module is hands-on: every command
below actually runs inside a throwaway sandbox directory so you see real output —
nothing to memorize blind.

**What you'll be able to do by the end:**
- Navigate and *wrangle files from the command line* (often faster than opening Python
  to peek at a 5 GB CSV).
- Explain Git's mental model — working dir -> staging -> commit -> remote — fluently.
- Run the everyday Git loop: pull, branch, commit small, push, open a PR.
""")

nb.plain(r"""
Two tools, one plain description each:
- **Bash** is the text-based way to boss your computer around — type a short command,
  it does a job (list files, peek inside a huge file, filter rows) instantly.
- **Git** is a *save-game system for code*: it takes snapshots so you can experiment
  fearlessly, rewind mistakes, and work alongside teammates without overwriting each
  other.
You'll touch both constantly, so a little fluency here pays off forever.
""")

nb.code(r"""
import subprocess, os, tempfile, textwrap

# A tiny helper so every cell shows the command AND its output.
def sh(cmd, cwd=None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd,
                       capture_output=True, text=True, executable="/bin/bash")
    out = (r.stdout or "") + (r.stderr or "")
    print(out.rstrip() if out.strip() else "(no output)")
    return out

# Work in an isolated temp directory so we never touch the real repo.
SANDBOX = tempfile.mkdtemp(prefix="m21_")
print("sandbox:", SANDBOX)
""")

nb.readcode(r"""
- `sh(cmd)` runs a shell command from Python and prints both the command and its
  output, so the notebook is fully reproducible.
- `SANDBOX` is a throwaway folder — everything we do happens there, never in your real
  project. In real life you'd type these straight into a terminal.
""")

# ---------------------------------------------------------------------------
# 21.1 Navigation
# ---------------------------------------------------------------------------
nb.md(r"""
## 21.1 Bash navigation & inspection — the commands you'll actually type

A handful of commands cover 90% of daily use.
""")

nb.analogy(r"""
The terminal is like giving directions to an extremely fast, extremely literal
assistant. "Where am I?" (`pwd`), "show me what's here" (`ls`), "make a folder"
(`mkdir`), "go in there" (`cd`). It does exactly what you say, instantly — which is why
it's faster than clicking through folders once you know the words.
""")

nb.code(r"""
sh("pwd")                          # where am I?
sh("mkdir -p project/data project/src && ls -R project", cwd=SANDBOX)  # make dirs
sh("echo 'hello from bash' > project/notes.txt", cwd=SANDBOX)          # write file
sh("cat project/notes.txt", cwd=SANDBOX)                               # read file
""")

nb.md(r"""
**The navigation core:**
- `pwd` — print working directory. `cd` — change directory (`cd ..` = up one level).
- `ls -la` — list *all* files (incl. hidden) with details. `mkdir -p a/b/c` — make
  nested dirs. `rm -r dir` — remove recursively (**careful, no undo**).
- `cp src dst`, `mv src dst` — copy / move (move = rename).
- `cat` (whole file), `head -n 5` (first 5 lines), `tail -n 5` (last 5), `less`
  (scroll).
""")

nb.warn("`rm -r` deletes permanently -- there's no Recycle Bin. Double-check the path before you hit Enter, especially with wildcards like `rm -r *`.")

# ---------------------------------------------------------------------------
# 21.2 Inspecting data files
# ---------------------------------------------------------------------------
nb.md(r"""
## 21.2 Inspecting data files from the shell (a real DS superpower)

Before loading a giant CSV into pandas, *peek* at it from the shell — instant, no
memory cost.
""")

nb.plain(r"""
Opening a 5 GB CSV in pandas can freeze your laptop. But you often just want to know:
what do the columns look like, how many rows, what values are in one column? The shell
answers all three in under a second without loading the file into memory. That's a
genuine edge in interviews and on the job.
""")

nb.code(r"""
csv = textwrap.dedent('''\
    id,city,plan,income
    1,Nairobi,Premium,92000
    2,Nairobi,Basic,41000
    3,Mombasa,Standard,55000
    4,Mombasa,Premium,120000
    5,Kisumu,Basic,38000
    6,Nairobi,Standard,60000
''')
open(os.path.join(SANDBOX, "customers.csv"), "w").write(csv)

sh("head -n 3 customers.csv", cwd=SANDBOX)        # first rows (incl. header)
sh("wc -l customers.csv", cwd=SANDBOX)            # count lines (rows+header)
sh("cut -d, -f2 customers.csv | tail -n +2 | sort | uniq -c", cwd=SANDBOX)  # value counts of 'city'!
""")

nb.readcode(r"""
- `head`/`tail` glimpse the top/bottom without loading the whole file.
- `wc -l` counts rows in milliseconds, even for a 50 GB file.
- The last line is the clever one: `cut -d, -f2` pulls column 2 (comma-delimited),
  `tail -n +2` drops the header, `sort | uniq -c` counts each distinct value — a
  `value_counts()` straight from the shell. Chaining small tools with `|` (a "pipe") is
  the whole Unix philosophy.
""")

nb.jargon("Pipe (|)", "sends the output of one command straight into the next, letting you chain small tools")

nb.code(r"""
sh("grep Premium customers.csv", cwd=SANDBOX)                 # filter rows
sh("awk -F, 'NR>1 {sum+=$4} END {print \"total income:\", sum}' customers.csv",
   cwd=SANDBOX)                                               # sum a column with awk
""")

nb.deeper(r"""
`grep` = pattern search across lines; `awk` = a tiny column-processing language (`-F,`
sets the comma delimiter, `$4` is the 4th field, `NR>1` skips the header). You don't
need to master `awk`, but being able to filter and sum columns without opening Python is
a real productivity edge. Other must-knows: `chmod +x script.sh` (make executable),
`./script.sh` (run), `export VAR=value` (set an env var), and
`python script.py > log.txt 2>&1 &` (run in the background, capturing output).
""")

nb.takeaway("head/tail/wc/cut/grep/awk + pipes let you inspect and summarize huge files instantly, before Python ever loads them.")

# ---------------------------------------------------------------------------
# 21.3 Git mental model
# ---------------------------------------------------------------------------
nb.md(r"""
## 21.3 Git — the mental model first (this is what interviews test)

Git tracks **snapshots** of your project. A file lives in one of four "places":

```
 working directory  --git add-->  staging area  --git commit-->  local repo  --git push-->  remote (GitHub)
   (your edits)                    (marked to save)              (history)                  (shared)
```
""")

nb.analogy(r"""
Think of packing a shipping box:
- **Working directory** = your desk, covered in stuff you're editing.
- **Staging area** = the box you're carefully choosing what to put in (`git add`).
- **Local repo** = you seal and label the box, saving a snapshot (`git commit`).
- **Remote (GitHub)** = you ship the box to the shared warehouse (`git push`).
Staging is the key insight: it lets you pack *only some* changes into this commit and
leave the rest for later.
""")

nb.jargon("Staging area (index)", "the set of changes you've marked (git add) to include in the next commit")
nb.jargon("Commit", "a saved snapshot of your project at a point in time, with a message")
nb.jargon("Remote", "the shared copy of the repo on GitHub/GitLab that teammates pull from and push to")

nb.code(r"""
repo = os.path.join(SANDBOX, "demo_repo")
os.makedirs(repo, exist_ok=True)

sh("git init -q", cwd=repo)                                   # start a repo
sh("git config user.email 'you@example.com'", cwd=repo)       # identity (once)
sh("git config user.name 'DS Learner'", cwd=repo)
sh("git branch -m main", cwd=repo)                            # name default branch 'main'
sh("git status", cwd=repo)
""")

nb.code(r"""
# 1) Create a file (it's UNTRACKED in the working directory)
open(os.path.join(repo, "analysis.py"), "w").write("print('v1')\n")
sh("git status --short", cwd=repo)          # '??' = untracked

# 2) STAGE it
sh("git add analysis.py", cwd=repo)
sh("git status --short", cwd=repo)          # 'A' = staged/added

# 3) COMMIT it (snapshot saved to local history)
sh("git commit -q -m 'Add analysis script (v1)'", cwd=repo)
sh("git log --oneline", cwd=repo)
""")

nb.readcode(r"""
- Creating `analysis.py` leaves it **untracked** (`??`) — Git sees it but isn't
  watching it yet.
- `git add` moves it into staging (`A` = added).
- `git commit` seals the snapshot into history; `git log --oneline` shows the saved
  commit with its message. That three-step add -> commit is the heartbeat of Git.
""")

nb.deeper(r"""
**Commit-message hygiene** is a quiet signal of seniority: write imperative, specific
messages — `Add income outlier capping`, not `stuff`. Many teams follow **Conventional
Commits**: `feat:` (new feature), `fix:` (bug fix), `docs:`, `refactor:`, `test:`.
Good messages turn `git log` into a readable changelog.
""")

nb.code(r"""
# Change the file, then see WHAT changed with git diff
open(os.path.join(repo, "analysis.py"), "w").write("print('v2 - added feature')\n")
sh("git diff", cwd=repo)                     # unstaged changes vs last commit
sh("git add -A && git commit -q -m 'feat: update analysis to v2'", cwd=repo)
sh("git log --oneline", cwd=repo)
""")

nb.takeaway("Git flow: edit (working dir) -> git add (staging) -> git commit (local repo) -> git push (remote). Staging lets you pack exactly the right changes.")

# ---------------------------------------------------------------------------
# 21.4 Branching
# ---------------------------------------------------------------------------
nb.md(r"""
## 21.4 Branching & merging — work without breaking `main`

A **branch** is an independent line of work. You build a feature on its own branch, then
**merge** it into `main` when it's ready — the basis of **pull requests**.
""")

nb.analogy(r"""
A branch is like writing a chapter in a *copy* of the manuscript. The published book
(`main`) stays clean and readable while you experiment. When your chapter is polished,
you fold it back in (merge). If it flops, you just throw the copy away — the book was
never at risk.
""")

nb.jargon("Branch", "an independent line of work that keeps experiments off the stable main branch")
nb.jargon("Merge", "combining another branch's changes into your current branch")

nb.code(r"""
# Create and switch to a feature branch
sh("git checkout -q -b feature/add-readme", cwd=repo)   # -b = create + switch
sh("git branch", cwd=repo)                              # '*' marks current branch

open(os.path.join(repo, "README.md"), "w").write("# Demo Project\nAnalysis code.\n")
sh("git add README.md && git commit -q -m 'docs: add README'", cwd=repo)

# Switch back to main and MERGE the feature in
sh("git checkout -q main", cwd=repo)
sh("git merge -q feature/add-readme", cwd=repo)
sh("git log --oneline --graph --all", cwd=repo)
sh("ls", cwd=repo)                                      # README.md is now on main
""")

nb.warn(r"""
**Merge conflicts** happen when two branches change the *same lines*. Git marks them
with `<<<<<<<`, `=======`, `>>>>>>>`; you edit to keep the right version, then `git add`
+ `git commit`. Don't panic — conflicts are normal, local, and fixable.
""")

nb.takeaway("Branch to isolate work, commit on the branch, then merge into main; conflicts are just same-line clashes you resolve by hand.")

# ---------------------------------------------------------------------------
# 21.5 .gitignore
# ---------------------------------------------------------------------------
nb.md(r"""
## 21.5 `.gitignore` — never commit junk (or secrets!)

Some files should **never** be tracked: large data, notebook checkpoints, virtual envs,
and — critically — **credentials**. A `.gitignore` tells Git to skip them.
""")

nb.warn("Leaking an API key or password to a public repo is a classic, costly mistake -- bots scan GitHub for them within minutes. A `.gitignore` for secrets is non-negotiable.")

nb.code(r"""
gitignore = textwrap.dedent('''\
    # data & artifacts
    data/*.csv
    *.parquet
    # python
    __pycache__/
    *.pyc
    .venv/
    # notebooks
    .ipynb_checkpoints/
    # secrets -- NEVER commit these
    .env
    *.key
    credentials.json
''')
open(os.path.join(repo, ".gitignore"), "w").write(gitignore)

# Prove it works: create an ignored file and a tracked one
open(os.path.join(repo, ".env"), "w").write("API_KEY=super-secret\n")
open(os.path.join(repo, "keep.py"), "w").write("x = 1\n")

sh("git add -A && git status --short", cwd=repo)   # .env is absent -> correctly ignored!
""")

nb.readcode(r"""
- The `.gitignore` lists patterns Git should skip: bulky data, caches, virtual envs, and
  secret files like `.env`.
- After `git add -A`, notice `.env` does **not** appear in `git status` — Git is
  ignoring it, so your API key can't be committed by accident. `keep.py` shows up
  normally. This one habit prevents a whole category of security incidents.
""")

nb.takeaway("A .gitignore keeps data, caches, and especially secrets out of history -- set it up before your first commit.")

# ---------------------------------------------------------------------------
# 21.6 Remote workflow
# ---------------------------------------------------------------------------
nb.md(r"""
## 21.6 The remote workflow (GitHub) — the loop you'll run daily

No network here, but this is the exact cycle on a real project:

```bash
git clone https://github.com/you/project.git   # get the repo (once)
# ... edit files ...
git pull                       # get teammates' latest changes FIRST
git add -A                     # stage your changes
git commit -m "feat: add churn model"
git push                       # publish to GitHub
# then open a Pull Request on GitHub for review + merge to main
```
""")

nb.plain(r"""
The daily rhythm: **pull** first (grab everyone's latest so you don't clash), work on a
**branch**, **commit** in small logical chunks with clear messages, **push**, then open
a **Pull Request** so a teammate reviews before it merges into `main`. Never push
straight to `main` on a team, and never commit secrets or huge data.
""")

nb.interview(r"""
"My loop is pull, branch, commit in small logical chunks, push, open a PR. Git's three
areas — working directory, staging, and the repo — let me stage exactly what belongs in
each commit, and branches keep experiments off main until they're reviewed."
""")

nb.code(r"""
# Tidy up the sandbox
import shutil
shutil.rmtree(SANDBOX, ignore_errors=True)
print("cleaned up sandbox -- none of this touched your real repo.")
""")

# ---------------------------------------------------------------------------
# Practice + summary
# ---------------------------------------------------------------------------
nb.md(r"""
## 21.7 Practice
""")

nb.try_this(r"""
1. From the shell, count how many customers are on the `Premium` plan using only `grep`
   + `wc -l`.
2. Initialize a repo, make three commits, then show the history as a graph
   (`git log --oneline --graph`).
3. Create a branch, add a file, merge it into `main`, then delete the branch
   (`git branch -d name`).
4. Write a `.gitignore` for a Python DS project and explain each entry.
5. Explain the difference between `git fetch` and `git pull` in one sentence.
""")

nb.md(r"""
## Summary

- **Bash**: `pwd/cd/ls/mkdir/cp/mv/rm`, plus the data-peeking crew
  `head/tail/wc/cut/grep/awk` and pipes `|` — inspect huge files without Python.
- **Git model**: working dir -> `add` -> staging -> `commit` -> local repo -> `push` ->
  remote. Branches isolate work; merges/PRs combine it.
- Core loop: **pull -> branch -> commit small -> push -> PR**.
- **`.gitignore`** keeps junk and **secrets** out of history — a security must.

Next: **Module 22 — Big Data & Spark**.
""")

out = nb.save("notebooks/21_bash_git.ipynb", glossary_path="notes/glossary.json")
print("saved", out)
