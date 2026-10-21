"""Source identity and failure-atomic publication for the E-28 runner."""

import hashlib
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import run_e28 as runner


def git(root, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Source Test",
            "-c",
            "user.email=source-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def source(tmp_path):
    names = (
        "src/genriesz/__init__.py",
        "src/genriesz/density_ratio_pilot.py",
        "src/genriesz/shape_regression.py",
        "experiments/grrexp/__init__.py",
        "experiments/grrexp/seeds.py",
        "experiments/grrexp/env.py",
        "experiments/grrexp/parallel.py",
        "experiments/grrexp/study.py",
        "experiments/grrexp/e25.py",
        "experiments/grrexp/e27.py",
        "experiments/grrexp/e28.py",
        "experiments/run_e27.py",
        "experiments/run_e28.py",
        "experiments/pyproject.toml",
        "experiments/uv.lock",
        "experiments/.python-version",
        "pyproject.toml",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
    git(tmp_path, "init", "-q", "--object-format=sha1")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Pin scientific inputs")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD"), names


def test_source_verification_covers_packages_runners_and_environment(source):
    root, commit, names = source
    hashes = runner._verify_source(root, commit)
    assert set(hashes) == set(names)
    for name, value in hashes.items():
        assert value == hashlib.sha256((root / name).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "name",
    (
        "experiments/grrexp/seeds.py",
        "experiments/grrexp/env.py",
        "experiments/grrexp/parallel.py",
        "experiments/grrexp/study.py",
        "experiments/grrexp/e25.py",
        "experiments/grrexp/__init__.py",
        "src/genriesz/__init__.py",
        "src/genriesz/shape_regression.py",
        "experiments/run_e27.py",
        "experiments/uv.lock",
        "experiments/.python-version",
        "pyproject.toml",
    ),
)
def test_changed_dependency_is_rejected(source, name):
    root, commit, _ = source
    (root / name).write_text("different scientific input\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the expected reviewed commit"):
        runner._verify_source(root, commit)


@pytest.mark.parametrize("change", ("new", "deleted"))
def test_python_dependency_set_must_match_reviewed_commit(source, change):
    root, commit, _ = source
    if change == "new":
        (root / "src/genriesz/imported_later.py").write_text("value = 1\n", encoding="utf-8")
    else:
        (root / "experiments/grrexp/seeds.py").unlink()
    with pytest.raises(RuntimeError, match="Python source files differ"):
        runner._verify_source(root, commit)


def test_current_head_cannot_silently_replace_reviewed_commit(source):
    root, commit, _ = source
    (root / "unrelated.txt").write_text("another commit\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Add another file")
    with pytest.raises(RuntimeError, match="HEAD differs"):
        runner._verify_source(root, commit)


@pytest.mark.parametrize("commit", (None, "HEAD", "a" * 39, "g" * 40))
def test_expected_commit_is_required_before_output_creation(tmp_path, commit):
    destination = tmp_path / "results"
    with pytest.raises(ValueError, match="exact 40-digit Git commit"):
        runner.run(directory=destination, expected_source_commit=commit)
    assert not destination.exists()


def test_cli_requires_explicit_reviewed_commit(tmp_path):
    destination = tmp_path / "results"
    with pytest.raises(SystemExit):
        runner.main(["--output", str(destination)])
    assert not destination.exists()


@pytest.fixture
def outputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    shard = raw / "part-000001.parquet"
    frame = pd.DataFrame({"rep": [0, 1], "value": [0.25, 0.5]})
    frame.to_parquet(shard, index=False)
    summary = pd.DataFrame({"coverage": [0.95]})
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    yield tmp_path, shard, frame, summary, figure
    plt.close(figure)


def test_companions_are_closed_and_verified_before_summary_publication(outputs, monkeypatch):
    directory, shard, frame, summary, figure = outputs
    publications = []
    replace = runner.os.replace

    def verify():
        assert not (directory / "summary.csv").exists()
        assert not (directory / "table.tex").exists()
        assert not (directory / "figure.pdf").exists()
        assert (directory / ".table.tex").read_text() == "complete table\n"
        assert (directory / ".figure.pdf").read_bytes().startswith(b"%PDF")
        pd.testing.assert_frame_equal(pd.read_csv(directory / ".summary.csv"), summary)
        publications.append("verified")

    def publish(partial, target):
        if Path(target).parent == directory:
            if Path(target).name == "summary.csv":
                assert (directory / "table.tex").exists()
                assert (directory / "figure.pdf").exists()
            publications.append(Path(target).name)
        return replace(partial, target)

    monkeypatch.setattr(runner.os, "replace", publish)
    runner._publish_derived(
        directory, summary, {"table": "complete table\n"}, {"figure": figure}, verify
    )
    assert publications == ["verified", "table.tex", "figure.pdf", "summary.csv"]
    assert not any(p.name.startswith(".") for p in directory.iterdir())
    assert not plt.fignum_exists(figure.number)
    pd.testing.assert_frame_equal(pd.read_parquet(shard), frame)


@pytest.mark.parametrize("phase", ("table", "figure", "summary", "source", "publication"))
def test_output_failures_never_publish_summary_and_preserve_closed_raw(
    outputs, monkeypatch, phase
):
    directory, shard, frame, summary, figure = outputs
    message = "publication stopped"
    if phase == "table":
        write_text = Path.write_text

        def fail_table(path, value, **kwargs):
            result = write_text(path, value, **kwargs)
            if path.name == ".table.tex":
                raise OSError(message)
            return result

        monkeypatch.setattr(Path, "write_text", fail_table)
    elif phase == "figure":

        def fail_figure(path, **kwargs):
            Path(path).write_bytes(b"%PDF incomplete")
            raise OSError(message)

        monkeypatch.setattr(figure, "savefig", fail_figure)
    elif phase == "summary":

        def fail_summary(self, path, **kwargs):
            Path(path).write_text("coverage\n", encoding="utf-8")
            raise OSError(message)

        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_summary)
    elif phase == "publication":
        replace = runner.os.replace

        def fail_publication(partial, target):
            if Path(target).name == "figure.pdf":
                raise OSError(message)
            return replace(partial, target)

        monkeypatch.setattr(runner.os, "replace", fail_publication)

    def verify():
        if phase == "source":
            raise OSError(message)

    with pytest.raises(OSError, match=message):
        runner._publish_derived(
            directory, summary, {"table": "complete table\n"}, {"figure": figure}, verify
        )
    assert not (directory / "summary.csv").exists()
    assert not (directory / "figure.pdf").exists()
    if phase == "publication":
        assert (directory / "table.tex").read_text() == "complete table\n"
    else:
        assert not (directory / "table.tex").exists()
    assert not any(p.name.startswith(".") for p in directory.iterdir())
    assert not plt.fignum_exists(figure.number)
    pd.testing.assert_frame_equal(pd.read_parquet(shard), frame)
