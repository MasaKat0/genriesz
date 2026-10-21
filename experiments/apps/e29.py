"""E-29: Increasing density-ratio and regression classes."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    runner_script = Path(__file__).resolve().parents[1] / "run_e29.py"
    python_executable = sys.executable

    return Path, mo, os, pd, python_executable, runner_script, subprocess


@app.cell
def _(mo, os):
    saved_directory = mo.ui.text(
        value=os.environ.get("GRREXP_E29_RESULTS", ""),
        label="Saved results directory",
        full_width=True,
    )
    output_directory = mo.ui.text(label="New output directory", full_width=True)
    source_commit = mo.ui.text(label="Reviewed source commit", full_width=True)
    n_jobs = mo.ui.number(start=1, stop=32, value=4, step=1, label="Worker processes")
    run_button = mo.ui.run_button(label="Run the complete E-29 design")
    mo.vstack(
        [
            mo.md("# E-29: Increasing density-ratio and regression classes"),
            saved_directory,
            output_directory,
            source_commit,
            n_jobs,
            run_button,
            mo.md(
                "Opening this app displays saved results. The button executes all twelve "
                "conditions with 2000 replications each. SQ and UKL have the same fitted "
                "ratio on the count-smoothed cellwise class."
            ),
        ]
    )
    return n_jobs, output_directory, run_button, saved_directory, source_commit


@app.cell
def _(
    mo,
    n_jobs,
    output_directory,
    python_executable,
    run_button,
    runner_script,
    source_commit,
    subprocess,
):
    mo.stop(not run_button.value)
    _completed = subprocess.run(
        [
            python_executable,
            str(runner_script),
            "--output",
            output_directory.value,
            "--expected-source-commit",
            source_commit.value,
            "--n-jobs",
            str(int(n_jobs.value)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mo.md(f"Completed: `{_completed.stdout.strip()}`")
    return


@app.cell
def _(Path, mo, pd, saved_directory):
    _directory = Path(saved_directory.value).expanduser().resolve()
    _summary = _directory / "summary.csv"
    mo.stop(not _summary.is_file(), mo.md("Choose a directory containing a saved E-29 summary."))
    mo.output.append(mo.ui.table(pd.read_csv(_summary), selection=None))
    for _figure in sorted(_directory.glob("fig_E29*.pdf")):
        mo.output.append(mo.pdf(_figure))
    return


if __name__ == "__main__":
    app.run()
