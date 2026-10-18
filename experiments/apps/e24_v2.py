"""E-24: Controlled nonlinear benchmark."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from grrexp import outputs_v2 as outputs
    from grrexp import runner_v2 as runner

    return Path, mo, outputs, pd, runner


@app.cell
def _(outputs):
    headless_manifest = outputs.execute_app(24)
    return (headless_manifest,)


@app.cell
def _(headless_manifest, mo, outputs):
    mo.stop(headless_manifest is not None)
    _current = outputs.stage_dir(24)
    _saved = _current
    saved_directory = mo.ui.text(
        value=str(_saved), label="Saved results directory", full_width=True
    )
    output_directory = mo.ui.text(
        value=str(_current), label="New output directory", full_width=True
    )
    n_jobs = mo.ui.number(start=1, stop=4, value=2, step=1, label="Worker processes")
    run_button = mo.ui.run_button(label="Run the complete E-24 design")
    mo.vstack([
        mo.md("# E-24: Controlled nonlinear benchmark"),
        saved_directory,
        output_directory,
        n_jobs,
        run_button,
        mo.md("Opening this app displays saved results. The button executes every condition "
              "and replication in the paper design. Choose a new output directory for each run."),
    ])
    return n_jobs, output_directory, run_button, saved_directory


@app.cell
def _(mo, n_jobs, output_directory, run_button, runner):
    mo.stop(not run_button.value)
    completed_manifest = runner.run(
        24, directory=output_directory.value, n_jobs=int(n_jobs.value)
    )
    mo.md(f"Completed: `{completed_manifest}`")
    return


@app.cell
def _(Path, mo, pd, saved_directory):
    _directory = Path(saved_directory.value).expanduser().resolve()
    _failure = _directory / "FAILED.txt"
    if _failure.is_file():
        mo.output.append(mo.md("### Recorded failure\n```text\n" +
                               _failure.read_text(encoding="utf-8") + "\n```"))
    _summary = _directory / "summary.csv"
    mo.stop(not _summary.is_file(), mo.md(f"No saved summary in `{_directory}`."))
    _frame = pd.read_csv(_summary)
    mo.output.append(mo.ui.table(_frame, selection=None))
    _tables = {
        _path.stem: mo.md("```latex\n" + _path.read_text(encoding="utf-8") + "\n```")
        for _path in sorted(_directory.glob("tab_*.tex"))
    }
    if _tables:
        mo.output.append(mo.accordion(_tables))
    _figure = _directory / "fig_E24.pdf"
    if _figure.is_file():
        mo.output.append(mo.pdf(_figure))
    return


if __name__ == "__main__":
    app.run()
