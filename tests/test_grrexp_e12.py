"""E-12 design, balance and shared-data cross-fitting."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e12, study


def test_design():
    assert e12.N_VALUES == (1000, 4000) and len(e12.CELLS) == 8
    assert (e12.R, e12.K) == (500, 5)
    assert e12.GRR_ARMS == ("SQ", "UKL1", "EB")
    assert len(e12.ARM_LABELS) == 7


def test_replication_has_exact_balance_and_common_folds():
    result = e12.replicate((0, 0, 20261007))
    assert set(result["arms"]) == set(e12.ARM_LABELS)
    assert np.bincount(result["folds"]).tolist() == [200] * 5
    for arm in e12.GRR_ARMS:
        row = result["arms"][f"{arm}|RW_full"]
        assert row["status"] == "ok"
        assert row["train_imbalance"] < 0.001
        assert abs(row["estimate"] - 1) < 0.2


def test_omission_changes_only_dictionary():
    Z = np.random.default_rng(0).normal(size=(10, 3))
    include, omit = e12.features(Z, "Include"), e12.features(Z, "Omit")
    np.testing.assert_array_equal(include[:, :4], omit)
    np.testing.assert_allclose(include[:, 4], Z[:, 1] ** 2 - 1)


def test_failure_is_not_discarded():
    raw = study.raw_frame(
        e12,
        [(0, 0, 20261007)],
        [{"folds": [0, 1], "arms": {"SQ|ARW_cf": study.failed("optimizer_failure")}}],
    )
    summary = e12.summarise(raw).iloc[0]
    assert summary.failure_rate == 1 and summary.coverage == 0
    assert np.isnan(summary.bias_conditional)
