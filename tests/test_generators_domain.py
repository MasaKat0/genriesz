"""Domain / clip diagnostics for the built-in Bregman generators.

Covers design-doc items E/P (clip visibility) and the conjugate identity
d g*(v)/dv = alpha(v), which guarantees that the GRR objective and gradient
are mutually consistent wherever the link is smooth.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import (
    BKLGenerator,
    BPGenerator,
    PUGenerator,
    SquaredGenerator,
    UKLGenerator,
)


def _branch(x: np.ndarray) -> int:
    return 1 if x[0] >= 0 else 0


def _fd_conjugate_gradient(gen, X, v, h=1e-6):
    gp, _ = gen.conjugate(X, v + h)
    gm, _ = gen.conjugate(X, v - h)
    return (gp - gm) / (2.0 * h)


@pytest.mark.parametrize(
    "gen, v_lo, v_hi",
    [
        (SquaredGenerator(C=0.0), -5.0, 5.0),
        (UKLGenerator(C=1.0, branch_fn=_branch), -3.0, 3.0),
        (BKLGenerator(C=1.0, branch_fn=_branch), -3.0, -0.1),
        (BPGenerator(C=1.0, omega=0.5, branch_fn=_branch), -2.0, 2.0),
        (PUGenerator(C=1.0, branch_fn=_branch), -3.0, 3.0),
    ],
)
def test_conjugate_gradient_identity_in_valid_region(gen, v_lo, v_hi):
    rng = np.random.default_rng(0)
    n = 9
    X = rng.normal(size=(n, 2))
    v = np.linspace(v_lo, v_hi, n)

    _, alpha = gen.conjugate(X, v)
    fd = _fd_conjugate_gradient(gen, X, v)
    rel = np.max(np.abs(fd - alpha) / np.maximum(1.0, np.abs(alpha)))
    assert rel < 1e-5


def test_bkl_domain_binding_flags_violations():
    gen = BKLGenerator(C=1.0, branch_fn=_branch)
    rng = np.random.default_rng(1)
    n = 6
    X = rng.normal(size=(n, 2))
    X[:, 0] = 1.0  # positive branch everywhere -> u = v

    ok = gen.domain_binding(X, np.full(n, -0.5))
    assert not np.any(ok)

    # u = s*v > 0 violates the BKL domain; the internal clip binds there.
    bad = gen.domain_binding(X, np.full(n, +0.5))
    assert np.all(bad)


def test_bp_domain_binding_flags_violations():
    gen = BPGenerator(C=1.0, omega=0.5, branch_fn=_branch)
    rng = np.random.default_rng(2)
    n = 5
    X = rng.normal(size=(n, 2))
    X[:, 0] = 1.0  # positive branch, k = 3 -> t = 1 + v/3

    assert not np.any(gen.domain_binding(X, np.full(n, 0.5)))
    assert np.all(gen.domain_binding(X, np.full(n, -10.0)))


def test_sq_reports_no_binding():
    gen = SquaredGenerator(C=0.0)
    X = np.zeros((4, 2))
    assert not np.any(gen.domain_binding(X, np.array([-1e6, -1.0, 1.0, 1e6])))


def test_bkl_and_pu_warn_without_branch_fn():
    with pytest.warns(UserWarning, match="branch_fn"):
        BKLGenerator(C=1.0)
    with pytest.warns(UserWarning, match="branch_fn"):
        PUGenerator(C=1.0)
