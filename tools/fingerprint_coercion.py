"""Fingerprint the numeric outputs of the basis and generator coercion paths.

Item X-2 changes error behaviour on purpose, but must leave the numbers alone
for every input that was already valid. Run this on two revisions and compare
the hashes:

    python3 tools/fingerprint_coercion.py

Covered: ``fit_density_ratio`` over the five generator names crossed with three
penalties, over five generator instances, and over a callable and a built-in
basis; ``grr_ate`` and ``grr_att`` over four generators. The estimates and the
standard errors go into the hash, rounded to 10 decimals; the routes go in as
strings.

One cell of the penalty grid does not converge (``pu`` under ``l1``), and an
unpenalised BKL or PU diverges by construction, so the grid stops at ``l2``,
``l1`` and ``lp``. A configuration that raises contributes its exception's class
name to the hash rather than being skipped: a change from one error type to
another must move the fingerprint too. That is why this file catches broadly --
it is recording failures, not handling them.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np

import genriesz
from genriesz.generators import (
    BKLGenerator,
    BoundedBKLGenerator,
    BPGenerator,
    PUGenerator,
    SquaredGenerator,
    UKLGenerator,
)

SEED = 20260710
DIGITS = 10


def _positive_branch(_x):
    return 1


def _treatment_branch(x):
    return int(x[0] == 1.0)


def main() -> str:
    warnings.filterwarnings("ignore")
    h = hashlib.sha256()

    def feed(tag: str, values) -> None:
        a = np.asarray(values, dtype=float).ravel()
        a = np.where(np.isfinite(a), a, np.nan)
        h.update(tag.encode())
        h.update(np.round(a, DIGITS).tobytes())

    rng = np.random.default_rng(SEED)
    Xn = rng.normal(size=(70, 2))
    Xd = rng.normal(loc=0.5, size=(90, 2))

    penalties = [("l2", {}), ("l1", {}), ("lp", {"p_norm": 1.5})]
    for name in ["sq", "ukl", "bkl", "bp", "pu"]:
        for penalty, extra in penalties:
            try:
                r = genriesz.fit_density_ratio(
                    Xn,
                    Xd,
                    generator=name,
                    penalty=penalty,
                    lam=0.1,
                    random_state=0,
                    max_iter=5000,
                    **extra,
                )
            except Exception as exc:  # noqa: BLE001 - the class name is the fingerprint
                h.update(f"dr:{name}:{penalty}:EXC={type(exc).__name__}".encode())
                continue
            feed(f"dr:{name}:{penalty}:beta", r.beta)
            feed(f"dr:{name}:{penalty}:ratio", r.predict_ratio(Xd[:20]))
            h.update(f"dr:{name}:{penalty}:route={r.route}".encode())

    instances = [
        ("sq", SquaredGenerator(C=0.0)),
        ("ukl", UKLGenerator(C=0.0, branch_fn=_positive_branch)),
        ("bkl", BKLGenerator(C=1.0, branch_fn=_positive_branch)),
        ("bp", BPGenerator(C=0.0, omega=0.5, branch_fn=_positive_branch)),
        ("pu", PUGenerator(C=1.0, branch_fn=_positive_branch)),
    ]
    for label, gen in instances:
        r = genriesz.fit_density_ratio(Xn, Xd, generator=gen, lam=0.1, random_state=0)
        feed(f"dri:{label}:beta", r.beta)
        feed(f"dri:{label}:ratio", r.predict_ratio(Xn[:15]))

    r = genriesz.fit_density_ratio(
        Xn,
        Xd,
        basis=lambda X: np.column_stack([np.ones(len(X)), X]),
        lam=0.1,
        random_state=0,
    )
    feed("dr:callable_basis:beta", r.beta)

    r = genriesz.fit_density_ratio(
        Xn, Xd, basis=genriesz.PolynomialBasis(degree=2), lam=0.1, random_state=0
    )
    feed("dr:polybasis:beta", r.beta)

    n = 400
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Z = rng.normal(size=(n, 2))
    X = np.column_stack([D, Z])
    Y = 2 * Z[:, 0] + 1.5 * D + rng.normal(scale=0.5, size=n)

    grr_generators = [
        ("sq", SquaredGenerator(C=0.0)),
        ("ukl", UKLGenerator(C=0.0, branch_fn=_treatment_branch)),
        ("bp", BPGenerator(C=0.0, omega=0.5, branch_fn=_treatment_branch)),
        ("bbkl", BoundedBKLGenerator(C=1e-2, alpha_max=20.0, branch_fn=_treatment_branch)),
    ]
    for label, gen in grr_generators:
        for fn_name in ["grr_ate", "grr_att"]:
            try:
                res = getattr(genriesz, fn_name)(
                    X=X, Y=Y, basis=genriesz.PolynomialBasis(degree=2), generator=gen
                )
            except Exception as exc:  # noqa: BLE001 - the class name is the fingerprint
                h.update(f"{fn_name}:{label}:EXC={type(exc).__name__}".encode())
                continue
            for key in sorted(res.estimates):
                e = res.estimates[key]
                feed(f"{fn_name}:{label}:{key}", [float(e.estimate), float(e.se)])

    return h.hexdigest()


if __name__ == "__main__":
    print(main())
