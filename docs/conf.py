"""Sphinx configuration for the *genriesz* project."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Allow importing from the local `src/` tree when building docs from a source
# checkout (e.g., `sphinx-build -b html docs docs/_build/html`).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "genriesz"
author = "Masahiro Kato"

version_text = (Path(SRC) / "genriesz" / "__init__.py").read_text(encoding="utf-8")
match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', version_text, re.MULTILINE)
if match is None:
    raise RuntimeError("src/genriesz/__init__.py does not define __version__.")
release = match.group(1)



# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    # Optional: notebooks / markdown
    "nbsphinx",
    "myst_parser",
]

autosummary_generate = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Do not execute notebooks during docs builds.
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "source"]


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
# No custom static assets yet; an empty list avoids a warning for the missing
# ``_static`` directory.
html_static_path = []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
nitpicky = False
