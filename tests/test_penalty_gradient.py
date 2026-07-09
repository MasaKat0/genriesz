from __future__ import annotations

import numpy as np

from genriesz.glm import _Penalty


def test_l1_penalty_value_and_gradient_are_consistent() -> None:
    penalty = _Penalty("l1", lam=0.7, p_norm=None)
    beta = np.array([-0.8, -0.2, 0.0, 0.3, 1.2], dtype=float)
    eps = 1e-6
    numeric = np.empty_like(beta)
    for j in range(beta.size):
        step = np.zeros_like(beta)
        step[j] = eps
        numeric[j] = (penalty.value(beta + step) - penalty.value(beta - step)) / (2.0 * eps)
    assert np.allclose(numeric, penalty.grad(beta), atol=2e-5, rtol=2e-5)
