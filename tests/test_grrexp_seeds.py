from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import seeds  # noqa: E402


def test_key_matches_registration_formula() -> None:
    expected = np.random.default_rng(
        np.random.SeedSequence(entropy=20261007, spawn_key=(12, 3, 7, 0))
    ).random(5)
    assert np.array_equal(seeds.rng(12, 3, 7, 0).random(5), expected)
    assert np.array_equal(seeds.Seeds(12).data(3, 7).random(5), expected)


def test_special_stream_keys() -> None:
    s = seeds.Seeds(24)

    def ref(key, entropy=20261007):
        return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=key)).random(3)

    assert np.array_equal(s.bootstrap(7).random(3), ref((24, 9999, 7, 4)))
    assert np.array_equal(s.pilot(2).random(3), ref((24, 2, 0, 5)))
    assert np.array_equal(s.evaluation(5).random(3), ref((24, 999, 5, 6)))
    assert np.array_equal(seeds.Seeds(14).evaluation(5, block=998).random(3), ref((14, 998, 5, 6)))
    p = seeds.Seeds(24, entropy=seeds.PILOT_ENTROPY)
    assert np.array_equal(p.data(1, 0).random(3), ref((24, 1, 0, 0), 20261008))


def test_streams_differ_and_inputs_validated() -> None:
    draws = {st: seeds.rng(13, 0, 0, st).random() for st in range(7)}
    assert len(set(draws.values())) == 7
    with pytest.raises(ValueError):
        seeds.rng(11, 0, 0, 0)
    with pytest.raises(ValueError):
        seeds.rng(12, 0, 0, 7)
    with pytest.raises(ValueError):
        seeds.Seeds(12).bootstrap(8)
    with pytest.raises(ValueError):
        seeds.Seeds(12).evaluation(0, block=998)


def test_fold_ids_balanced_partition() -> None:
    ids = seeds.fold_ids(503, 5, seeds.Seeds(12).folds(0, 0))
    counts = np.bincount(ids)
    assert counts.tolist() == [101, 101, 101, 100, 100]
    again = seeds.fold_ids(503, 5, seeds.Seeds(12).folds(0, 0))
    assert np.array_equal(ids, again)
