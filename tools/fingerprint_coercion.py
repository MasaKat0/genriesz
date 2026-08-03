"""Fingerprint successful basis and generator coercion paths.

Run this script on two revisions and compare the hashes. Every configuration
listed below is expected to fit; an error stops the script rather than being
encoded as a successful comparison.
"""

from __future__ import annotations

import hashlib

import numpy as np

import genriesz
from genriesz.generators import (
    BPGenerator,
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
    digest = hashlib.sha256()

    def feed(tag: str, values) -> None:
        array = np.asarray(values, dtype=float).ravel()
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{tag} contains a nonfinite value.")
        digest.update(tag.encode())
        digest.update(np.round(array, DIGITS).tobytes())

    rng = np.random.default_rng(SEED)
    X_num = rng.normal(size=(70, 2))
    X_den = rng.normal(loc=0.5, size=(90, 2))

    configurations = [
        ("sq_l2", "sq", "l2", {}),
        ("sq_l1", "sq", "l1", {}),
        ("sq_lp", "sq", "lp", {"p_norm": 1.5}),
        ("ukl_l2", "ukl", "l2", {}),
        ("ukl_lp", "ukl", "lp", {"p_norm": 1.5}),
        ("bkl_l2", "bkl", "l2", {}),
        ("bp_l2", "bp", "l2", {}),
        ("bp_lp", "bp", "lp", {"p_norm": 1.5}),
        ("pu_l2", "pu", "l2", {}),
        ("pu_lp", "pu", "lp", {"p_norm": 1.5}),
    ]
    for label, generator, penalty, extra in configurations:
        result = genriesz.fit_density_ratio(
            X_num,
            X_den,
            generator=generator,
            penalty=penalty,
            lam=0.1,
            random_state=0,
            max_iter=5000,
            **extra,
        )
        feed(f"dr:{label}:beta", result.beta)
        feed(f"dr:{label}:ratio", result.predict_ratio(X_den[:20]))
        digest.update(f"dr:{label}:route={result.route}".encode())

    instances = [
        ("sq", SquaredGenerator(C=0.0)),
        ("ukl", UKLGenerator(C=0.0, branch_fn=_positive_branch)),
        ("bp", BPGenerator(C=0.0, omega=0.5, branch_fn=_positive_branch)),
    ]
    for label, generator in instances:
        result = genriesz.fit_density_ratio(
            X_num, X_den, generator=generator, lam=0.1, random_state=0
        )
        feed(f"dri:{label}:beta", result.beta)
        feed(f"dri:{label}:ratio", result.predict_ratio(X_num[:15]))

    callable_result = genriesz.fit_density_ratio(
        X_num,
        X_den,
        basis=lambda X: np.column_stack([np.ones(len(X)), X]),
        lam=0.1,
        random_state=0,
    )
    feed("dr:callable_basis:beta", callable_result.beta)

    polynomial_result = genriesz.fit_density_ratio(
        X_num,
        X_den,
        basis=genriesz.PolynomialBasis(degree=2),
        lam=0.1,
        random_state=0,
    )
    feed("dr:polybasis:beta", polynomial_result.beta)

    n = 400
    treatment = (rng.uniform(size=n) < 0.5).astype(float)
    covariates = rng.normal(size=(n, 2))
    X = np.column_stack([treatment, covariates])
    Y = 2.0 * covariates[:, 0] + 1.5 * treatment + rng.normal(scale=0.5, size=n)

    grr_generators = [
        ("sq", SquaredGenerator(C=0.0)),
        ("ukl", UKLGenerator(C=0.0, branch_fn=_treatment_branch)),
        ("bp", BPGenerator(C=0.0, omega=0.5, branch_fn=_treatment_branch)),
    ]
    for label, generator in grr_generators:
        for function_name in ("grr_ate", "grr_att"):
            result = getattr(genriesz, function_name)(
                X=X,
                Y=Y,
                basis=genriesz.PolynomialBasis(degree=2),
                generator=generator,
            )
            for key in sorted(result.estimates):
                estimate = result.estimates[key]
                feed(
                    f"{function_name}:{label}:{key}",
                    [float(estimate.estimate), float(estimate.se)],
                )

    return digest.hexdigest()


if __name__ == "__main__":
    print(main())
