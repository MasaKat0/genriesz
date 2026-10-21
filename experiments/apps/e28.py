import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, mo, os, pd


@app.cell
def _(Path, os, pd):
    result_directory = Path(os.environ["GRREXP_E28_RESULTS"])
    summary = pd.read_csv(result_directory / "summary.csv")
    return result_directory, summary


@app.cell
def _(mo, summary):
    mo.ui.table(summary, pagination=True)
    return


@app.cell
def _(mo, result_directory):
    mo.pdf(result_directory / "fig_E28.pdf")
    return


if __name__ == "__main__":
    app.run()
