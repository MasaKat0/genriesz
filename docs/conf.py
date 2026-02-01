"""Sphinx configuration for the *genriesz* project.

This configuration is designed to build on Read the Docs and locally.
"""

from __future__ import annotations

import os
import sys


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

try:
    from importlib import metadata as importlib_metadata

    release = importlib_metadata.version("genriesz")
except Exception:  # pragma: no cover
    # Fallback for editable/non-installed builds.
    release = "0.0.0"


# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
# Keep strictness low by default; we can turn this on later if desired.
nitpicky = False
