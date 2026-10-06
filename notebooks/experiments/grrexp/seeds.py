"""Random-number streams of registration §1.2.

Every generator is ``np.random.default_rng(SeedSequence(entropy, spawn_key=key))``
with ``key = (EXP, cell, rep, stream)``:

====== ===================================== ==========================
stream use                                   key
====== ===================================== ==========================
0      data                                  (EXP, cell, rep, 0)
1      cross-fitting folds (uniform, unstr.) (EXP, cell, rep, 1)
2      inner CV / selection splits           (EXP, cell, rep, 2)
3      baseline randomness (GBM, NN)         (EXP, cell, rep, 3)
4      bootstrap (family_id of §1.9)         (EXP, 9999, family_id, 4)
5      tuning pilot data                     (EXP, cell, 0, 5)
6      evaluation sample Z^eval              (EXP, 999, rep, 6)
====== ===================================== ==========================

Stage 0.5 (timing pilot) uses ``PILOT_ENTROPY`` so that it never overlaps the
confirmatory streams. E-14 re-evaluates flagged replications with the key
``(14, 998, rep, 6)``; pass ``block=998`` to :meth:`Seeds.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONFIRMATORY_ENTROPY = 20261007
PILOT_ENTROPY = 20261008

STREAMS = {
    "data": 0,
    "folds": 1,
    "inner_cv": 2,
    "baseline": 3,
    "bootstrap": 4,
    "pilot": 5,
    "evaluation": 6,
}

BOOTSTRAP_CELL = 9999
EVALUATION_CELL = 999
REGISTERED_EXPERIMENTS = range(12, 25)


def _check_key(exp: int, cell: int, rep: int, stream: int) -> None:
    for name, value in (("exp", exp), ("cell", cell), ("rep", rep), ("stream", stream)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if exp not in REGISTERED_EXPERIMENTS:
        raise ValueError(f"EXP must be in 12..24, got {exp}")
    if stream not in STREAMS.values():
        raise ValueError(f"stream must be one of {sorted(STREAMS.values())}, got {stream}")


def seed_sequence(
    exp: int, cell: int, rep: int, stream: int, *, entropy: int = CONFIRMATORY_ENTROPY
) -> np.random.SeedSequence:
    """``SeedSequence(entropy, spawn_key=(exp, cell, rep, stream))``."""
    _check_key(exp, cell, rep, stream)
    if entropy not in (CONFIRMATORY_ENTROPY, PILOT_ENTROPY):
        raise ValueError(f"entropy must be {CONFIRMATORY_ENTROPY} or {PILOT_ENTROPY}")
    return np.random.SeedSequence(
        entropy=entropy, spawn_key=(int(exp), int(cell), int(rep), int(stream))
    )


def rng(
    exp: int, cell: int, rep: int, stream: int, *, entropy: int = CONFIRMATORY_ENTROPY
) -> np.random.Generator:
    """A fresh generator for the key ``(exp, cell, rep, stream)``."""
    return np.random.default_rng(seed_sequence(exp, cell, rep, stream, entropy=entropy))


@dataclass(frozen=True)
class Seeds:
    """The registered streams of one experiment, bound to one entropy."""

    exp: int
    entropy: int = CONFIRMATORY_ENTROPY

    def data(self, cell: int, rep: int) -> np.random.Generator:
        return rng(self.exp, cell, rep, 0, entropy=self.entropy)

    def folds(self, cell: int, rep: int) -> np.random.Generator:
        return rng(self.exp, cell, rep, 1, entropy=self.entropy)

    def inner_cv(self, cell: int, rep: int) -> np.random.Generator:
        return rng(self.exp, cell, rep, 2, entropy=self.entropy)

    def baseline(self, cell: int, rep: int) -> np.random.Generator:
        return rng(self.exp, cell, rep, 3, entropy=self.entropy)

    def bootstrap(self, family_id: int) -> np.random.Generator:
        if family_id not in range(1, 8):
            raise ValueError(f"family_id must be in 1..7 (§1.9), got {family_id}")
        return rng(self.exp, BOOTSTRAP_CELL, family_id, 4, entropy=self.entropy)

    def pilot(self, cell: int) -> np.random.Generator:
        return rng(self.exp, cell, 0, 5, entropy=self.entropy)

    def evaluation(self, rep: int, block: int = EVALUATION_CELL) -> np.random.Generator:
        if block not in (EVALUATION_CELL, 998):
            raise ValueError("block must be 999 (Z^eval) or 998 (E-14 re-evaluation)")
        if block == 998 and self.exp != 14:
            raise ValueError("block 998 is registered only for E-14")
        return rng(self.exp, block, rep, 6, entropy=self.entropy)


def fold_ids(n: int, k: int, generator: np.random.Generator) -> np.ndarray:
    """Uniformly random, unstratified, data-independent partition into ``k`` folds.

    A random permutation assigns position ``j`` to fold ``j mod k``, so fold
    sizes differ by at most one. Pass the stream-1 generator.
    """
    if k < 2 or n < k:
        raise ValueError(f"need 2 <= k <= n, got n={n}, k={k}")
    ids = np.empty(n, dtype=np.int64)
    ids[generator.permutation(n)] = np.arange(n) % k
    return ids
