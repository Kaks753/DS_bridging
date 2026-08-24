"""Builder for Module 20: Bash & Git for Data Science."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 20 — Bash & Git for Data Science (the daily-driver tools)

These aren't "data science" per se, but you'll use them **every single day** and
they show up in interviews and take-home reviews. This module is hands-on: every
command below actually runs inside a throwaway sandbox directory so you see real
output — nothing to memorize blind.

Goals:
- **Bash:** navigate, inspect, and *wrangle files from the command line* (often
  faster than opening Python for a quick peek at a 5 GB CSV).
- **Git:** the mental model (working dir → staging → commit → remote), the everyday
  commands, branching/merging, and how to talk about it fluently.

> We drive the shell from Python with `subprocess` so the notebook is fully
> reproducible. In real life you'd type these directly into a terminal.
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
SANDBOX = tempfile.mkdtemp(prefix="m20_")
print("sandbox:", SANDBOX)
""")

nb.md(r"""
## 20.1 Bash navigation & inspection — the commands you'll actually type

The terminal is where data engineering lives. A few commands cover 90% of daily use.
""")

nb.code(r"""
sh("pwd")                          # where am I?
sh("mkdir -p project/data project/src && ls -R project", cwd=SANDBOX)  # make dirs
sh("echo 'hello from bash' > project/notes.txt", cwd=SANDBOX)          # write file
sh("cat project/notes.txt", cwd=SANDBOX)                               # read file
""")

nb.md(r"""
**The navigation core:**
- `pwd` — print working directory. `cd` — change directory (`cd ..` up one level).
- `ls -la` — list *all* files (incl. hidden) with details. `mkdir -p a/b/c` — make
  nested dirs. `rm -r dir` — remove recursively (**careful**).
- `cp src dst`, `mv src dst` — copy / move (move = rename).
- `cat` (whole file), `head -n 5` (first 5 lines), `tail -n 5` (last 5),
  `less` (scroll).
""")

nb.md(r"""
## 20.2 Inspecting data files from the shell (a real DS superpower)

Before loading a giant CSV into pandas, *peek* at it from the shell — instant, no
memory cost. These four commands answer "what's in this file?" in seconds.
""")

nb.code(r"""
# Create a sample CSV to inspect
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

nb.md(r"""
**What just happened (this is the good stuff):**
- `head`/`tail` → glimpse the top/bottom without loading the whole file.
- `wc -l` → row count in milliseconds, even for a 50 GB file.
- `cut -d, -f2` → pull column 2 (delimiter `,`); `tail -n +2` drops the header;
  `sort | uniq -c` → **a `value_counts()` straight from the shell**. Piping small
  tools together (`|`) is the Unix philosophy.
- Also huge: `grep "Premium" customers.csv` filters rows matching a pattern.
""")

nb.code(r"""
sh("grep Premium customers.csv", cwd=SANDBOX)                 # filter rows
sh("awk -F, 'NR>1 {sum+=$4} END {print \"total income:\", sum}' customers.csv",
   cwd=SANDBOX)                                               # sum a column with awk
""")

nb.md(r"""
`grep` = pattern search; `awk` = mini column-processing language (`-F,` sets the
comma delimiter, `$4` is the 4th field). You don't need to master `awk`, but knowing
it can sum/filter columns without Python is a genuine edge.

**Other must-knows:** `chmod +x script.sh` (make executable), `./script.sh` (run),
`export VAR=value` (env var), `python script.py > log.txt 2>&1 &` (run in background,
redirect output), `history`, and piping into `| less` for long output.
""")

nb.md(r"""
## 20.3 Git — the mental model first (this is what interviews test)

Git tracks **snapshots** of your project so you can experiment fearlessly and
collaborate without overwriting each other. Four "places" a file can be:

```
 working directory  --git add-->  staging area  --git commit-->  local repo  --git push-->  remote (GitHub)
   (your edits)                    (marked to save)              (history)                  (shared)
```

- **Working directory:** your actual files as you edit them.
- **Staging area (index):** changes you've *marked* to include in the next commit
  (`git add`). Lets you commit *some* changes, not all.
- **Local repository:** committed history on your machine (`git commit`).
- **Remote:** the shared copy on GitHub/GitLab (`git push` / `git pull`).

Say this out loud until it's automatic — interviewers love "walk me through git".
""")

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

nb.md(r"""
**Commit message hygiene (a real signal of seniority):** write imperative,
specific messages — `Add income outlier capping` not `stuff`. Many teams use
**Conventional Commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
""")

nb.code(r"""
# Change the file, then see WHAT changed with git diff
open(os.path.join(repo, "analysis.py"), "w").write("print('v2 - added feature')\n")
sh("git diff", cwd=repo)                     # unstaged changes vs last commit
sh("git add -A && git commit -q -m 'feat: update analysis to v2'", cwd=repo)
sh("git log --oneline", cwd=repo)
""")

nb.md(r"""
## 20.4 Branching & merging — work without breaking `main`

A **branch** is an independent line of work. You develop a feature on its own
branch, then **merge** it into `main` when it's ready. This is how teams collaborate
without stepping on each other — and it's the basis of **pull requests**.
""")

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

nb.md(r"""
**Everyday branch commands:**
- `git branch` (list) · `git checkout -b name` / `git switch -c name` (create+switch)
- `git merge other` (bring `other` into current branch)
- `git checkout main -- file` (restore one file from another branch)

**Merge conflicts** happen when two branches change the *same lines*. Git marks them
with `<<<<<<<`, `=======`, `>>>>>>>`; you edit to keep the right version, then
`git add` + `git commit`. Don't panic — conflicts are normal and local.
""")

nb.md(r"""
## 20.5 `.gitignore` — never commit junk (or secrets!)

Some files should **never** be tracked: large data, notebook checkpoints, virtual
envs, and — critically — **credentials**. A `.gitignore` tells Git to skip them.
Leaking an API key to a public repo is a classic, costly mistake.
""")

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
    # secrets — NEVER commit these
    .env
    *.key
    credentials.json
''')
open(os.path.join(repo, ".gitignore"), "w").write(gitignore)

# Prove it works: create an ignored file and a tracked one
open(os.path.join(repo, "secret.env"), "w").write("API_KEY=super-secret\n")
os.rename(os.path.join(repo, "secret.env"), os.path.join(repo, ".env"))
open(os.path.join(repo, "keep.py"), "w").write("x = 1\n")

sh("git add -A && git status --short", cwd=repo)   # .env is absent -> correctly ignored!
""")

nb.md(r"""
Notice `.env` did **not** appear in `git status` — Git is ignoring it, so your
secret can't be committed by accident. This single habit prevents a whole category
of security incidents.
""")

nb.md(r"""
## 20.6 The remote workflow (GitHub) — the loop you'll run daily

You won't have network here, but this is the exact cycle on a real project:

```bash
git clone https://github.com/you/project.git   # get the repo (once)
# ... edit files ...
git pull                       # get teammates' latest changes FIRST
git add -A                     # stage your changes
git commit -m "feat: add churn model"
git push                       # publish to GitHub
# then open a Pull Request on GitHub for review + merge to main
```

**Golden rules:**
- `git pull` **before** you start and **before** you push (avoid conflicts).
- Commit small and often with clear messages.
- Never commit secrets or big data (that's what `.gitignore` + tools like DVC/Git
  LFS are for).
- Feature branch → Pull Request → review → merge. Never push straight to `main` on a
  team.

**Interview soundbite:**
> "My loop is pull, branch, commit in small logical chunks, push, open a PR. Git's
> three areas — working dir, staging, and the repo — let me stage exactly what
> belongs in each commit, and branches keep experiments off `main` until reviewed."
""")

nb.code(r"""
# Tidy up the sandbox
import shutil
shutil.rmtree(SANDBOX, ignore_errors=True)
print("cleaned up sandbox — none of this touched your real repo.")
""")

nb.md(r"""
## 20.7 Mini-exercises

1. From the shell, count how many customers are on the `Premium` plan in a CSV
   using only `grep` + `wc -l`.
2. Initialize a repo, make three commits, then show the history as a graph.
3. Create a branch, add a file, merge it back to `main`, and delete the branch
   (`git branch -d name`).
4. Write a `.gitignore` for a Python DS project and explain each entry.
5. Explain the difference between `git fetch` and `git pull`.

## Summary

- **Bash**: `pwd/cd/ls/mkdir/cp/mv/rm`, and the data-peeking crew
  `head/tail/wc/cut/grep/awk` + pipes `|` — inspect huge files without Python.
- **Git model**: working dir → `add` → staging → `commit` → local repo → `push` →
  remote. Branches isolate work; merges/PRs combine it.
- Core loop: **pull → branch → commit small → push → PR**.
- **`.gitignore`** keeps junk and **secrets** out of history — a security must.

Next: **Module 22 — AWS Deployment & MLOps**.
""")

out = nb.save("notebooks/20_bash_git.ipynb")
print("saved", out)
