"""Bootstrap families of registration §1.9 (all on stream 4).

Each family has one generator, keyed ``(EXP, 9999, family_id, 4)``. Create one
:class:`FamilyBootstrap` per family in a notebook and call it in the registered
order of cells and comparisons: every call consumes the stream, so a second
instance of the same family would repeat the first instance's draws rather than
give independent ones.

``replicate`` resamples replication indices with replacement (``mode="indices"``,
exactly ``x[idx]``), or draws the equivalent multinomial counts
(``mode="counts"``) for the 200000-draw family 7, where materialising indices is
too slow. Draws are generated in row chunks; chunking does not change the
values (tested).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .seeds import Seeds

REGISTERED_B = {1: 2000, 2: 10000, 3: 10000, 4: 10000, 5: 10000, 6: 10000, 7: 200000}
DEFAULT_CHUNK = 2000


class FamilyBootstrap:
    def __init__(self, seeds: Seeds, family_id: int):
        if family_id not in REGISTERED_B:
            raise ValueError(f"family_id must be in 1..7, got {family_id}")
        self.family_id = family_id
        self.B = REGISTERED_B[family_id]
        self._rng = seeds.bootstrap(family_id)

    def replicate(
        self,
        statistic: Callable[[np.ndarray], np.ndarray],
        R: int,
        *,
        mode: str = "indices",
        chunk: int = DEFAULT_CHUNK,
    ) -> np.ndarray:
        """``B`` bootstrap values of ``statistic``.

        ``statistic`` maps a ``(rows, R)`` array (indices into the replications, or
        resample counts) to ``rows`` values.
        """
        if R < 2:
            raise ValueError(f"need R >= 2, got {R}")
        if mode not in ("indices", "counts"):
            raise ValueError("mode must be 'indices' or 'counts'")
        out = np.empty(self.B)
        done = 0
        p = np.full(R, 1.0 / R)
        while done < self.B:
            rows = min(chunk, self.B - done)
            if mode == "indices":
                draw = self._rng.integers(0, R, size=(rows, R))
            else:
                draw = self._rng.multinomial(R, p, size=rows)
            values = np.asarray(statistic(draw), dtype=float)
            if values.shape != (rows,):
                raise ValueError(f"statistic must return shape ({rows},), got {values.shape}")
            out[done : done + rows] = values
            done += rows
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite bootstrap statistic")
        return out


def percentile_interval(boot, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(boot, dtype=float), [a, 1.0 - a])
    return float(lo), float(hi)


def bootstrap_se(boot) -> float:
    return float(np.std(np.asarray(boot, dtype=float), ddof=1))


def centred_pvalue(boot, t_hat: float) -> float:
    """E-24: ``(1 + #{|T*_b - T_hat| >= |T_hat|}) / (B + 1)``."""
    boot = np.asarray(boot, dtype=float)
    return float((1 + np.sum(np.abs(boot - t_hat) >= abs(t_hat))) / (boot.size + 1))


def log_rmse_ratio_counts(err_a, err_b) -> Callable[[np.ndarray], np.ndarray]:
    """Statistic for family 7 on resample counts: ``log(RMSE_A / RMSE_B)``.

    ``err_a`` and ``err_b`` are the paired errors ``theta_hat - theta0`` on the
    replications where both methods succeeded.
    """
    sq = np.column_stack([np.asarray(err_a, float) ** 2, np.asarray(err_b, float) ** 2])

    def stat(counts: np.ndarray) -> np.ndarray:
        s = counts @ sq
        return 0.5 * (np.log(s[:, 0]) - np.log(s[:, 1]))

    return stat
