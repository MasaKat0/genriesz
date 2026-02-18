from __future__ import annotations

project = "genriesz"
author = "Masahiro Kato"

extensions = [
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

# Do not execute notebooks on build by default.
nbsphinx_execute = "never"

# Allow both .rst and .md sources
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
