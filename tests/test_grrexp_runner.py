"""Failures after a marimo subprocess must leave an unusable run record."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from grrexp import outputs, runner  # noqa: E402


def test_runner_records_manifest_validation_failure(tmp_path, monkeypatch):
    directory = tmp_path / "run"

    def process(command, **kwargs):
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"experiment": "E-19"}))
        return subprocess.CompletedProcess(command, 0, "", "")

    def reject(*args):
        raise ValueError("required provenance missing")

    monkeypatch.setattr(subprocess, "run", process)
    monkeypatch.setattr(outputs, "validate_manifest", reject)
    with pytest.raises(ValueError, match="required provenance missing"):
        runner.run(19, directory=directory)
    assert "ValueError: required provenance missing" in (directory / "FAILED.txt").read_text()


def test_runner_records_process_interruption(tmp_path, monkeypatch):
    directory = tmp_path / "run"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run(19, directory=directory)
    assert "KeyboardInterrupt" in (directory / "FAILED.txt").read_text()
