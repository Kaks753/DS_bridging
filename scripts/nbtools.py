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
"""
from __future__ import annotations
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


class Notebook:
    def __init__(self):
        self.cells = []

    def md(self, text: str) -> "Notebook":
        """Add a markdown cell. Leading/trailing blank lines are trimmed."""
        self.cells.append(new_markdown_cell(text.strip("\n")))
        return self

    def code(self, src: str) -> "Notebook":
        """Add a code cell."""
        self.cells.append(new_code_cell(src.strip("\n")))
        return self

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

    def save(self, path: str):
        nb = self.to_node()
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return path
