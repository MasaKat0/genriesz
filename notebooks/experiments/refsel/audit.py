"""Analytic audit of candidate bias and score variance.

Section 10 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``. Because the
simulation knows ``alpha_0``, ``gamma_0``, and the noise variance, the
conditional bias and the conditional score variance are evaluated in closed form
instead of by averaging noisy score contributions. That removes the outcome
noise from the Monte Carlo error, which matters because the earlier design's
audit error was the same order as the bias it was trying to measure.

For a candidate with fitted ``alpha_hat`` and the fold's ``gamma_hat``,

    B = E[(alpha_0 - alpha_hat) (gamma_hat - gamma_0)]
    V = Var[m_hat + alpha_hat (gamma_0 - gamma_hat)] + E[alpha_hat^2] sigma^2

where ``m_hat(X) = gamma_hat(1, Z) - gamma_hat(0, Z)``. The first identity is
equation ``candidate_bias`` of the manuscript; the second uses
``E[eps | X] = 0`` so the cross term vanishes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import Dictionary, ExperimentBasis, FoldLibrary, OutcomeFit
from .dgp import (
    NOISE_SD,
    Design,
    FloatArray,
    GeneratedData,
    generate_data,
    true_outcome,
    true_representer,
)

#: Rows processed at a time when evaluating candidates on the integration sample.
#:
#: This bounds the size of the temporaries created per candidate, not the cached
#: raw feature matrix, which is built once for the whole integration sample.
CHUNK_SIZE = 25_000

_SAMPLE_CACHE: dict[tuple, GeneratedData] = {}
_RAW_CACHE: dict[tuple, FloatArray] = {}
_RAW_CACHE_MAX = 3


def clear_caches() -> None:
    """Drop the cached integration sample and its raw feature blocks."""

    _SAMPLE_CACHE.clear()
    _RAW_CACHE.clear()


def scenario_key(
    *, design: Design, overlap_scale: float, hidden_scale: float, size: int, seed: int
) -> tuple:
    return (design, float(overlap_scale), float(hidden_scale), int(size), int(seed))


def integration_sample(
    *, design: Design, overlap_scale: float, hidden_scale: float, size: int, seed: int
) -> GeneratedData:
    """Return the integration sample for one scenario, generated once.

    The sample is shared by every replication of the scenario, so the audit
    comparison across candidates and across replications is paired.
    """

    key = scenario_key(
        design=design, overlap_scale=overlap_scale, hidden_scale=hidden_scale, size=size, seed=seed
    )
    cached = _SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    sample = generate_data(
        n=size,
        design=design,
        overlap_scale=overlap_scale,
        hidden_scale=hidden_scale,
        seed=seed,
        with_outcome=False,
    )
    _SAMPLE_CACHE.clear()
    _SAMPLE_CACHE[key] = sample
    return sample


def _raw_features(key: tuple, kind: Dictionary, X: FloatArray) -> FloatArray:
    cache_key = (key, kind)
    cached = _RAW_CACHE.get(cache_key)
    if cached is not None:
        return cached
    raw = ExperimentBasis(kind).raw_features(X)
    if len(_RAW_CACHE) >= _RAW_CACHE_MAX:
        _RAW_CACHE.pop(next(iter(_RAW_CACHE)))
    _RAW_CACHE[cache_key] = raw
    return raw


@dataclass(frozen=True)
class AuditResult:
    """Conditional bias, variance, and risk of every candidate."""

    bias: FloatArray
    variance: FloatArray
    risk: FloatArray

    def oracle_index(self) -> int:
        if not np.any(np.isfinite(self.risk)):
            raise RuntimeError("No candidate produced a finite audit risk.")
        return int(np.nanargmin(self.risk))


def audit_from_values(
    *,
    alpha_hat: FloatArray,
    alpha0: FloatArray,
    gamma_hat: FloatArray,
    gamma0: FloatArray,
    m_hat: FloatArray,
    n_evaluation: int,
    noise_sd: float = NOISE_SD,
) -> tuple[float, float, float]:
    """Audit one predictor given its values on the integration sample."""

    residual = gamma_hat - gamma0
    bias = float(np.mean((alpha0 - alpha_hat) * residual))
    centered = m_hat - alpha_hat * residual
    variance = float(np.var(centered, ddof=0) + np.mean(alpha_hat * alpha_hat) * noise_sd**2)
    return bias, variance, bias * bias + variance / n_evaluation


def audit_library(
    library: FoldLibrary,
    outcome: OutcomeFit,
    integration: GeneratedData,
    *,
    key: tuple,
    n_evaluation: int,
    noise_sd: float = NOISE_SD,
    chunk_size: int = CHUNK_SIZE,
) -> AuditResult:
    """Audit every candidate of a fold on the shared integration sample.

    The dictionaries are evaluated once per fold rather than once per candidate,
    and the thirty candidates that share a dictionary are reduced to a single
    matrix product.
    """

    X = integration.X
    n = X.shape[0]
    n_candidates = len(library)
    gamma_hat_all = outcome.predict(X)
    m_hat_all = outcome.contrast(X)
    residual_all = gamma_hat_all - integration.gamma0

    sum_w = np.zeros(n_candidates, dtype=float)
    sum_z = np.zeros(n_candidates, dtype=float)
    sum_zz = np.zeros(n_candidates, dtype=float)
    sum_aa = np.zeros(n_candidates, dtype=float)
    finite = np.array(library.success, dtype=bool)

    blocks = {
        kind: (columns, library.coefficient_matrix(kind)[1])
        for kind, columns in library.dictionary_columns.items()
    }

    with library.branch_caches():
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            X_chunk = X[start:stop]
            alpha0_chunk = integration.alpha0[start:stop]
            residual_chunk = residual_all[start:stop]
            m_chunk = m_hat_all[start:stop]
            for kind, (columns, beta) in blocks.items():
                Phi = library.bases[kind].standardize(_raw_features(key, kind, X)[start:stop])
                V = Phi @ beta
                for i, j in enumerate(columns):
                    if not finite[j]:
                        continue
                    alpha = library._safe_inv_grad(j, X_chunk, V[:, i])  # noqa: SLF001
                    if alpha is None:
                        finite[j] = False
                        continue
                    z = m_chunk - alpha * residual_chunk
                    sum_w[j] += float(np.sum((alpha0_chunk - alpha) * residual_chunk))
                    sum_z[j] += float(np.sum(z))
                    sum_zz[j] += float(np.sum(z * z))
                    sum_aa[j] += float(np.sum(alpha * alpha))

    bias = np.where(finite, sum_w / n, np.nan)
    mean_z = sum_z / n
    variance = np.where(
        finite,
        np.maximum(sum_zz / n - mean_z * mean_z, 0.0) + sum_aa / n * noise_sd**2,
        np.nan,
    )
    risk = bias * bias + variance / n_evaluation
    return AuditResult(bias=bias, variance=variance, risk=risk)


def audit_true_reference(
    integration: GeneratedData,
    *,
    design: Design,
    overlap_scale: float,
    hidden_scale: float,
    n_evaluation: int,
) -> tuple[float, float, float]:
    """Audit the infeasible truth reference.

    Both nuisances are the truth, so the bias is exactly zero and the variance
    reduces to the efficient score variance. The contrast must be the true one,
    ``tau(Z)``, to stay consistent with ``gamma_hat = gamma_0``.
    """

    alpha0 = true_representer(
        integration.X, design=design, overlap_scale=overlap_scale, hidden_scale=hidden_scale
    )
    gamma0 = true_outcome(integration.X, design=design, hidden_scale=hidden_scale)
    return audit_from_values(
        alpha_hat=alpha0,
        alpha0=integration.alpha0,
        gamma_hat=gamma0,
        gamma0=integration.gamma0,
        m_hat=integration.tau,
        n_evaluation=n_evaluation,
    )
