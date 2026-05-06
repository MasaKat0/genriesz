from __future__ import annotations

from pathlib import Path


def test_repo_notebooks_are_mirrored_in_docs() -> None:
    """Keep notebooks/ and docs/notebooks/ in sync.

    The documentation build uses ``docs/notebooks/*.ipynb`` (via nbsphinx),
    while users often look for runnable notebooks under ``notebooks/``.
    To avoid drift, we require the two directories to contain byte-identical
    copies of the same notebook files.
    """

    repo_root = Path(__file__).resolve().parents[1]
    nb_root = repo_root / "notebooks"
    nb_docs = repo_root / "docs" / "notebooks"

    assert nb_root.is_dir(), f"Missing directory: {nb_root}"
    assert nb_docs.is_dir(), f"Missing directory: {nb_docs}"

    for nb in sorted(nb_root.glob("*.ipynb")):
        mirror = nb_docs / nb.name
        assert mirror.is_file(), f"Missing mirrored notebook in docs: {mirror}"
        assert nb.read_bytes() == mirror.read_bytes(), (
            f"Notebook mismatch between {nb} and {mirror}. "
            "Please sync notebooks/ and docs/notebooks/."
        )


def test_notebook_markdown_math_delimiters() -> None:
    import json

    repo_root = Path(__file__).resolve().parents[1]
    for nb in sorted((repo_root / "notebooks").glob("*.ipynb")):
        data = json.loads(nb.read_text())
        for cell in data.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            text = "".join(cell.get("source", []))
            assert "\\\\[" not in text
            assert "\\\\]" not in text
            assert "\\\\(" not in text
            assert "\\\\)" not in text
