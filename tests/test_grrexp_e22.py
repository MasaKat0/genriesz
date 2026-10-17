"""LaLonde partitions and diagnostics on estimation weights."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e22
from grrexp.seeds import Seeds, fold_ids


def test_partitions_and_specifications():
    assert e22.R == 20 and e22.K == 5 and e22.PRIMARY_PARTITION == 0
    assert len(e22.ARM_LABELS) == 6 and e22.DICTIONARIES == ("D1", "D2")
    partitions = [fold_ids(614, 5, Seeds(22).folds(0, r)) for r in range(20)]
    assert len({p.tobytes() for p in partitions}) == 20
    np.testing.assert_array_equal(partitions[0], fold_ids(614, 5, Seeds(22).folds(0, 0)))


def test_primary_failure_is_never_replaced_and_no_aggregated_inference():
    raw = pd.DataFrame(
        [
            {
                "arm": "SQ-D1",
                "n": 614,
                "rep": r,
                "estimate": np.nan if r == 0 else float(r),
                "se": np.nan if r == 0 else 1.0,
                "status": "optimizer_failure" if r == 0 else "ok",
            }
            for r in range(20)
        ]
    )
    row = e22.summarise(raw).iloc[0]
    assert np.isnan(row.primary_estimate) and row.primary_status == "optimizer_failure"
    assert row.sensitivity_partitions == 19 and row.sensitivity_failures == 0
    assert "estimate" not in row and "coverage" not in row
    assert "optimizer\\_failure" in e22.tables(e22.summarise(raw))["tab_E22"]


def test_actual_sample_weights_and_common_outcome():
    sample = e22.data()
    result = e22.replicate((0, 0, 20261007))
    assert set(result["arms"]) == set(e22.ARM_LABELS)
    for arm, record in result["arms"].items():
        if record["status"] != "ok":
            continue
        a = np.asarray(record["alpha"])
        dictionary = arm.split("-")[-1]
        diagnostic = e22.weight_summaries(
            sample["D"], sample["Zs"], sample["X"], a, e22.ATTArmBasis(dictionary)
        )
        assert record["ess_c"] == diagnostic["ess_c"]
        assert record["smd_raw_max"] == diagnostic["smd_raw_max"]
        assert len(a) == 614 and record["se"] > 0


def test_score_concentration_is_preserved_in_primary_and_sensitivity_outputs():
    concentration = [0.9906, *np.linspace(0.01, 0.99, 19)]
    raw = pd.DataFrame({
        "arm": "UKL0-D2",
        "n": 614,
        "rep": np.arange(20),
        "estimate": -26993.0,
        "se": 28128.0,
        "status": "ok",
        "score_concentration": concentration,
    })
    summary = e22.summarise(raw)
    row = summary.iloc[0]
    assert row.primary_score_concentration == concentration[0]
    assert row.primary_estimate == -26993.0 and row.sensitivity_failures == 0
    for q in (5, 50, 95):
        assert row[f"sensitivity_score_concentration_q{q:02}"] == np.quantile(
            concentration[1:], q / 100
        )
    tables = e22.tables(summary)
    assert "primary\\_score\\_concentration" in tables["tab_E22"]
    assert "0.9906" in tables["tab_E22_score_concentration"]
