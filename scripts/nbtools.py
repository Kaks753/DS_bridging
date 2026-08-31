"""
nbtools.py — tiny helper to build clean, consistent Jupyter notebooks in code.

Why this exists:
  Writing raw .ipynb JSON by hand is error-prone. This helper lets each module
  script declare cells in order (markdown or code), then we serialize to a valid
  notebook and (optionally) execute it so the committed .ipynb contains real,
  verified outputs — not just source that *might* run.

Usage (inside a module builder script):
    from nbtools import Notebook
    nb = Notebook()
    nb.md("# Title")
    nb.code("print('hello')")
    nb.save("notebooks/00_foo.ipynb")

------------------------------------------------------------------------------
THE 4-LAYER TEACHING PATTERN
------------------------------------------------------------------------------
Every concept in this bootcamp is taught in four ascending layers so an absolute
beginner is never lost, while a pro still gets the depth. Helper methods emit
consistently-tagged markdown so the website renderer can style each layer into a
coloured box.

  1. nb.plain(...)      🌱 "In plain English" — the idea with an everyday analogy,
                            NO jargon, BEFORE any code.
  2. nb.code(...)       the runnable code (annotate EVERY line with a `#` comment).
     nb.readcode(...)   🔤 "Reading the code" — a line-by-line plain translation.
  3. nb.deeper(...)     🎓 "Go deeper" — the rigorous why/how (the interview edge).
  4. nb.takeaway(...)   ✅ one-sentence anchor.
     nb.interview(...)  🗣️ a sentence you can say out loud in an interview.

Extra helpers:
  nb.analogy(...)   🧠 a standalone real-life analogy box.
  nb.warn(...)      ⚠️ a gotcha / common-mistake box.
  nb.try_this(...)  ✍️ a tiny inline practice prompt.
  nb.jargon(term, plain)  defines a scary word in one line (also collect for glossary).

These are THIN wrappers over md(): they just prepend a tagged heading line, so the
notebook stays 100% standard Jupyter and renders fine anywhere.
"""
from __future__ import annotations
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def _bq(text: str) -> str:
    """Render multi-line text as the body of a markdown blockquote (each line '> ')."""
    lines = text.strip("\n").split("\n")
    return "\n".join(("> " + ln) if ln.strip() else ">" for ln in lines)


class Notebook:
    def __init__(self):
        self.cells = []
        self.glossary = {}   # term -> plain definition (collected across the module)

    def md(self, text: str) -> "Notebook":
        """Add a markdown cell. Leading/trailing blank lines are trimmed."""
        self.cells.append(new_markdown_cell(text.strip("\n")))
        return self

    def code(self, src: str) -> "Notebook":
        """Add a code cell."""
        self.cells.append(new_code_cell(src.strip("\n")))
        return self

    # ---- 4-layer teaching helpers -------------------------------------------
    def plain(self, text: str) -> "Notebook":
        """🌱 Layer 1: plain-English intro with an analogy, no jargon."""
        return self.md("> 🌱 **In plain English**\n>\n" + _bq(text))

    def readcode(self, text: str) -> "Notebook":
        """🔤 Layer 2 companion: line-by-line plain translation of the code above."""
        return self.md("> 🔤 **Reading the code (line by line)**\n>\n" + _bq(text))

    def deeper(self, text: str) -> "Notebook":
        """🎓 Layer 3: the rigorous why/how (kept, for depth)."""
        return self.md("> 🎓 **Go deeper**\n>\n" + _bq(text))

    def takeaway(self, text: str) -> "Notebook":
        """✅ Layer 4: one-sentence anchor."""
        return self.md("> ✅ **Takeaway** — " + text.strip())

    def interview(self, text: str) -> "Notebook":
        """🗣️ Layer 4 companion: a line to say out loud in an interview."""
        return self.md("> 🗣️ **Say this in an interview** — " + text.strip())

    def analogy(self, text: str) -> "Notebook":
        """🧠 standalone analogy box."""
        return self.md("> 🧠 **Analogy** — " + text.strip())

    def warn(self, text: str) -> "Notebook":
        """⚠️ gotcha / common-mistake box."""
        return self.md("> ⚠️ **Watch out** — " + text.strip())

    def try_this(self, text: str) -> "Notebook":
        """✍️ tiny inline practice prompt."""
        return self.md("> ✍️ **Try this yourself** — " + text.strip())

    def jargon(self, term: str, plain: str) -> "Notebook":
        """Define a scary word in one line AND record it for the glossary."""
        self.glossary[term.strip()] = plain.strip()
        return self.md(f"> 📖 **{term.strip()}** = {plain.strip()}")

    def to_node(self):
        nb = new_notebook()
        nb.cells = self.cells
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        }
        return nb

    def save(self, path: str, glossary_path: str | None = None):
        nb = self.to_node()
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        if glossary_path and self.glossary:
            import json, os
            existing = {}
            if os.path.exists(glossary_path):
                try:
                    existing = json.load(open(glossary_path, encoding="utf-8"))
                except Exception:
                    existing = {}
            existing.update(self.glossary)
            with open(glossary_path, "w", encoding="utf-8") as f:
                json.dump(dict(sorted(existing.items())), f, indent=2, ensure_ascii=False)
        return path
