from __future__ import annotations

import numpy as np

from grr import GRR_ATE


def test_rkhs_grr_ate_smoke() -> None:
    """Smoke test: RKHS_GRR runs end-to-end on a tiny synthetic dataset."""

    rng = np.random.default_rng(0)

    n, d = 60, 3
    X = rng.normal(size=(n, d))

    beta = np.array([0.8, -0.5, 0.2])
    ps = 1.0 / (1.0 + np.exp(-(X @ beta)))
    T = rng.binomial(1, ps, size=n)

    tau = 1.0
    Y = tau * T + X @ np.array([1.0, -1.0, 0.5]) + rng.normal(scale=1.0, size=n)

    est = GRR_ATE()
    out = est.estimate(
        covariates=X,
        treatment=T,
        outcome=Y,
        method="RKHS_GRR",
        riesz_loss="SQ",
        riesz_with_D=True,
        riesz_link_name="Linear",
        folds=2,
        num_basis=20,
        sigma_list=np.array([1.0]),
        lda_list=np.array([0.1]),
        random_state=0,
    )

    assert len(out) == 9
    assert all(np.isfinite(v) for v in out)
