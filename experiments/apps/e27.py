"""E-27: Display saved results for unknown density-ratio shape."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, mo, os, pd


@app.cell
def _(mo, os):
    saved_directory = mo.ui.text(
        value=os.environ.get("GRREXP_E27_RESULTS", ""),
        label="Saved E-27 results directory",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("# E-27: Unknown density-ratio shape and fitted ridge regression"),
            saved_directory,
            mo.md(
                "This app displays saved numerical results. Execute the complete design with "
                "`uv run --locked python run_e27.py --n-jobs 2 --output NEW_DIRECTORY`."
            ),
        ]
    )
    return (saved_directory,)


@app.cell
def _(Path, mo, pd, saved_directory):
    mo.stop(not saved_directory.value, mo.md("Enter a saved results directory."))
    _directory = Path(saved_directory.value).expanduser().resolve()
    _summary = _directory / "summary.csv"
    mo.stop(not _summary.is_file(), mo.md(f"No saved summary in `{_directory}`."))
    _frame = pd.read_csv(_summary)
    mo.output.append(mo.ui.table(_frame, selection=None))
    _tables = {
        _path.stem: mo.md("```latex\n" + _path.read_text(encoding="utf-8") + "\n```")
        for _path in sorted(_directory.glob("tab_E27*.tex"))
    }
    if _tables:
        mo.output.append(mo.accordion(_tables))
    for _figure in sorted(_directory.glob("fig_E27*.pdf")):
        mo.output.append(mo.pdf(_figure))
    return


if __name__ == "__main__":
    app.run()
