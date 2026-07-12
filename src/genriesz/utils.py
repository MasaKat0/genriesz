"""Small utility helpers used across *genriesz*.

The library tries to keep the core dependency footprint small. This module
therefore uses only NumPy and SciPy.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lapack
from scipy.stats import norm

# Gate for taking the Cholesky fast path in ``solve_stationarity``: the LAPACK
# reciprocal-condition estimate must clear both an absolute floor and a margin
# over the caller's numerical rank threshold. LAPACK's estimator is biased
# optimistic (it lower-bounds ||A^-1||), so the margin is an empirical cushion,
# not a proof -- the estimate would have to be off by more than 100x to matter.
# Realistic penalized Gram matrices sit far above the floor: a 400-center RKHS
# basis with lam=1e-3 estimates ~1e-6 to ~3e-5.
_FAST_PATH_RCOND = 1e-8
_FAST_PATH_SAFETY = 100.0

# The computed null basis spans a true invariant subspace of ``A`` only up to an
# angle, and that angle sets the finest null-space component ``solve_stationarity``
# can tell apart from decomposition noise (see ``_null_space_resolution``).
# _EIGVEC_NOISE is the safety factor on that bound; _MAX_NULL_TOL is where the
# bound gets so coarse that the range/null split stops being a decision and
# becomes a guess, and the rank is reported as ambiguous instead.
_EIGVEC_NOISE = 10.0
_MAX_NULL_TOL = 1e-6

# How far from symmetric ``solve_stationarity`` still calls a matrix symmetric,
# relative to its largest entry: the rounding that accumulating a Gram matrix in
# float64 leaves behind, and no more. It bounds how far the matrix actually solved,
# ``(A + A.T) / 2``, sits from the ``A`` handed in.
_SYM_ROUNDING = 10.0


def as_2d(x: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    """Convert an array-like object to a 2D float64 NumPy array."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n, d). Got shape {arr.shape}.")
    return arr


def as_1d(x: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert an array-like object to a 1D float64 NumPy array."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of shape (n,). Got shape {arr.shape}.")
    return arr


def as_1d_of_length(x: ArrayLike, *, n: int, name: str) -> NDArray[np.float64]:
    """Flatten ``x`` to 1D float64 and require it to have exactly ``n`` entries."""

    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}. Got shape {arr.shape}.")
    return arr


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic sigmoid, evaluated branch-wise on the sign of ``z``.

    Splitting on ``z >= 0`` keeps ``exp`` away from overflow on either tail:
    the positive branch exponentiates ``-z`` and the negative branch ``z``.
    """

    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out


def _safe_norm(x: NDArray[np.float64]) -> float:
    """Euclidean norm that neither underflows nor overflows on extreme inputs.

    ``np.linalg.norm`` squares before summing, so it returns 0.0 for a vector of
    ~1e-200 and inf for one of ~1e200.
    """

    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m == 0.0 or not np.isfinite(m):
        return m if np.isfinite(m) else float("inf")
    return m * float(np.linalg.norm(x / m))


def _rescale_or_raise(
    beta_unit: NDArray[np.float64], b_scale: float, a_max: float
) -> NDArray[np.float64]:
    """Undo the unit-scaling, refusing a solution float64 cannot represent.

    ``beta = (b_scale / a_max) * beta_unit`` can overflow (a vanishing ``A``
    against an ordinary ``b``) or underflow to exactly zero (a huge ``A`` against
    a subnormal ``b``). Either way the true solution is outside float64, and a
    finite-looking answer would be a wrong one: zeros do not solve the system any
    more than infinities do. The two associations are tried because only the
    *intermediate* may overflow.
    """

    if b_scale == 0.0:
        return np.zeros(beta_unit.shape[0], dtype=float)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        beta = (beta_unit * b_scale) / a_max
        if not np.isfinite(beta).all():
            beta = (beta_unit / a_max) * b_scale

    if not np.isfinite(beta).all():
        raise np.linalg.LinAlgError(
            "the solution overflows float64: A is vanishingly small relative to b, "
            "so no representable beta satisfies the stationarity condition. Rescale "
            "the features or add an l2 penalty."
        )
    if not np.any(beta):
        raise np.linalg.LinAlgError(
            "the solution underflows float64: A is enormous relative to b, so the "
            "only representable beta is zero, which does not solve the system. "
            "Rescale the features."
        )
    return np.asarray(beta, dtype=float)


def _null_space_resolution(
    A_unit: NDArray[np.float64],
    evals: NDArray[np.float64],
    evecs: NDArray[np.float64],
    keep: NDArray[np.bool_],
    lam_min_kept: float,
) -> float:
    """Smallest null-space component of ``b`` the decomposition can still resolve.

    The computed null basis ``V`` spans a true invariant subspace of ``A`` only up
    to an angle, and leakage across that angle mixes the (comparatively large)
    retained part of ``b`` into the measured null component. A component below the
    leak is therefore indistinguishable from decomposition noise -- a perturbation
    of ``b`` that small would make the system consistent -- so its absence of a
    solution cannot be certified.

    The sin-theta theorem bounds the angle by ``||R|| / gap``, where ``R = A V - V
    L`` is the residual of the computed null basis and ``gap`` its separation from
    the true retained spectrum. Both inputs have to be *bounds*, not readings:

    - ``R`` is evaluated in float64, and a residual below the rounding of its own
      evaluation says nothing. Every step of ``fl(A V - V L)`` -- the dot products,
      the scaling, the subtraction -- is bounded componentwise in the standard way
      (Higham, ASNA 2nd ed., §3.5: ``|fl(x'y) - x'y| <= gamma_n |x|'|y|`` with
      ``gamma_k = k u / (1 - k u)``), and that bound is added to what was measured.
      A backward stable eigensolver leaves ``||R|| ~ eps ||A||`` anyway, which is
      why this normally reproduces the familiar ``eps / gap``. What it does *not*
      do is trust an unmeasurably small residual: an ``A`` whose null column is
      exactly zero (a diagonal one, say) has ``|A| |V| = 0`` there, and only then
      -- when the rounding bound itself vanishes -- is the decomposition credited
      with resolving the direction exactly, rather than charged a worst case it did
      not pay and waving through a null-space component it resolved perfectly well.
    - the computed eigenvalues sit within ``~n eps ||A||`` of the true ones (Weyl),
      so the separation is that much smaller than the computed one.

    Rounding in forming the projections ``V' b`` puts a floor of ``~sqrt(n) eps``
    under all of this, no matter how clean the decomposition is.
    """

    n = A_unit.shape[0]
    eps = float(np.finfo(float).eps)  # 2u, so using it for u is already conservative
    floor = _EIGVEC_NOISE * eps * np.sqrt(n)
    if bool(np.all(keep)):
        return float(floor)  # no null space to leak into: A is (numerically) PD

    V = evecs[:, ~keep]
    lam_null = evals[~keep]
    measured = _safe_norm((A_unit @ V - V * lam_null).ravel())

    # gamma_{n+2}: n for the dot products in A @ V, one each for the scaling V L and
    # the subtraction. (n + 2) eps < 1 for any n float64 can index.
    gamma = (n + 2) * eps / (1.0 - (n + 2) * eps)
    rounding = gamma * (
        _safe_norm((np.abs(A_unit) @ np.abs(V)).ravel())
        + _safe_norm((np.abs(V) * np.abs(lam_null)).ravel())
    )

    # Null eigenvalues sitting below 0 (rounding on an exactly-singular direction)
    # only widen the separation; do not take credit for that. Weyl's bound on the
    # eigenvalue error eats into it from both ends.
    weyl = n * eps * max(float(evals[-1]), 0.0)  # ||A_unit||_2 = lambda_max
    gap = lam_min_kept - max(float(np.max(lam_null)), 0.0) - 2.0 * weyl
    if gap <= 0.0:
        return np.inf  # the two clusters are not even separated: nothing to certify
    return float(max(floor, _EIGVEC_NOISE * (measured + rounding) / gap))


def solve_stationarity(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    rcond: float = 1e-10,
    rtol: float = 1e-12,
) -> NDArray[np.float64]:
    """Solve the stationarity condition ``A beta = b`` of a convex quadratic.

    ``A`` is symmetric positive *semi*-definite here (a Gram matrix plus a
    possibly zero ridge), so it can be singular, and the two singular cases mean
    opposite things:

    - ``b`` lies in the range of ``A``: minimizers exist and form an affine set,
      and the minimum-norm one is returned (in the sense made precise below --
      *exactly* the minimum-norm minimizer is not something a float64
      decomposition can promise).
    - ``b`` has a component in the null space of ``A``: the quadratic has *no*
      stationary point and is unbounded below along that direction. A linear
      solver still hands back a finite vector, and passing that off as a solution
      would silently turn a divergent problem into a successful fit.
      :class:`numpy.linalg.LinAlgError` is raised instead, so the caller can fail
      honestly or drop the candidate.

    Everything below is stated for the symmetric matrix actually solved, which is
    ``(A + A.T) / 2``. That is not a hedge on the caller's behalf: an ``A`` more
    asymmetric than the rounding that forming a Gram matrix in float64 can leave
    (``_SYM_ROUNDING * n * eps`` relative to its largest entry) is a bug upstream
    and is rejected outright, precisely so that the difference between ``A`` and
    its symmetric part stays at the level of writing ``A`` down at all.

    Telling them apart cannot be done from the solver's residual alone: a
    near-singular ``A`` makes ``numpy.linalg.solve`` succeed *without* raising,
    returning a huge vector whose backward error is small but which solves a
    perturbed system, not this one. So a suspicious matrix is escalated to a
    symmetric eigendecomposition, which says where the range of ``A`` ends:
    ``rcond`` sets the numerical rank (eigenvalues at or below ``rcond *
    lambda_max`` count as null).

    What can be certified there is bounded by the accuracy of the
    eigendecomposition itself: when the computed null basis is off by an angle,
    leakage from the retained subspace can cancel a real null-space component of
    ``b``. The component is therefore tested against ``tol_b = max(rtol, leak)``,
    where ``leak`` bounds that angle on this particular matrix
    (:func:`_null_space_resolution`), and a component below it is accepted. What
    that buys is a backward guarantee on ``b`` alone:

        the system is reported solvable only if there is a ``b_tilde`` with
        ``||b_tilde - b||_2 <= 2 tol_b ||b||_2`` for which the system is consistent.
        The matrix is not perturbed to get there -- ``b`` alone absorbs it.

    ("The matrix" being ``(A + A.T) / 2``, per above. For an ``A`` that is
    symmetric as written, which is every ``A`` this library passes in, the two are
    the same matrix.)

    Two ``tol_b`` s of room, not one, because the component being compared is itself
    only known to within ``leak``. Both ends matter: ``rtol`` is what the caller is
    willing to discard, ``leak`` is what the decomposition cannot see, and the test
    -- so the guarantee -- runs on whichever is larger.

    What ``beta`` is *not* is the exact minimum-norm solution of that ``b_tilde``.
    It is the minimum-norm solution as the computed decomposition sees it: it is
    orthogonal to the *computed* null basis rather than the true one, so it keeps a
    component of order ``leak * ||beta||`` along the true null space, and it is
    reconstructed by dividing by the retained eigenvalues, so its residual carries
    the usual ``~eps lambda_max / lambda_min_kept`` of an ill-conditioned solve --
    which, for an eigenvalue barely above the rank threshold, is far coarser than
    ``rtol``. ``rtol`` bounds the component of ``b`` that is thrown away; it says
    nothing about the accuracy of what is handed back.

    If ``leak`` itself grows past ``_MAX_NULL_TOL`` the decomposition can certify
    nothing at all, and the rank is reported as ambiguous rather than guessed at.

    Note that a direction whose eigenvalue falls below the numerical rank
    threshold is treated as null even if ``b`` has a component there and an
    exact solution therefore exists on paper: recovering it would mean dividing
    by a numerically-zero eigenvalue, and the resulting ``beta`` is not a quantity
    an unpenalized fit should return. Add a ridge to get it.

    Both criteria are invariant to rescaling ``A`` or ``b``: the two sides are put
    on a unit sup-norm scale up front, which also keeps the raw norms of extreme
    inputs from underflowing to 0 or overflowing to inf and quietly vacating every
    test below.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if not np.isfinite(rcond) or not 0.0 <= float(rcond) < 1.0:
        raise ValueError(f"rcond must be finite and in [0, 1). Got {rcond!r}.")
    if not np.isfinite(rtol) or not 0.0 < float(rtol) < 1.0:
        # rtol >= 1 would accept a b that is purely null-space, and NaN compares
        # false against everything, so either would quietly disable the check
        # below. rtol = 0 is refused too: it would promise to detect *any*
        # null-space component, which no float64 decomposition can deliver.
        raise ValueError(f"rtol must be finite and in (0, 1). Got {rtol!r}.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square 2D array. Got shape {A.shape}.")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError(f"b must be 1D of length {n}. Got shape {b.shape}.")
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("A and b must be finite.")
    if n == 0:
        return np.zeros(0, dtype=float)

    a_max = float(np.max(np.abs(A)))
    b_scale = float(np.max(np.abs(b)))
    if a_max == 0.0:
        if b_scale == 0.0:
            return np.zeros(n, dtype=float)  # 0 @ beta = 0 = b
        raise np.linalg.LinAlgError(
            "A is zero and b is not: the objective has no stationary point and is "
            "unbounded below. Add or increase the l2 penalty."
        )

    # Normalize before symmetrizing: on the unit scale the entries are in [-1, 1]
    # and halving them cannot underflow, whereas 0.5 * A on a subnormal A would
    # collapse it to the zero matrix.
    A_unit = A / a_max
    asym = float(np.max(np.abs(A_unit - A_unit.T)))
    sym_tol = _SYM_ROUNDING * n * float(np.finfo(float).eps)
    if asym > sym_tol:
        # LAPACK reads one triangle of A and eigh the other, so a non-symmetric
        # argument would have them silently solve two different systems. Only the
        # rounding BLAS can leave on a Gram matrix is tolerated -- and then averaged
        # away, because tolerating it is not the same as making both routines see
        # the same matrix. The tolerance has to stay at that rounding scale: solving
        # the symmetrized system means perturbing A by asym/2, and every guarantee
        # made below is about the matrix actually solved, so a wide tolerance would
        # quietly widen the perturbation the caller is never told about.
        raise ValueError(
            f"A must be symmetric: |A - A.T| reaches {asym * a_max:.3g} "
            f"({asym:.3g} relative to its largest entry), past the {sym_tol:.3g} "
            "that rounding on a Gram matrix would explain."
        )
    A_unit = 0.5 * A_unit + 0.5 * A_unit.T

    b_unit = b / b_scale if b_scale > 0.0 else np.zeros(n, dtype=float)
    b_norm = _safe_norm(b_unit)  # 1 <= b_norm <= sqrt(n), or 0 when b is zero

    # Fast path for a comfortably positive-definite A -- the penalized case, and
    # the overwhelmingly common one. A numerically successful Cholesky gives pocon
    # the factor it needs to estimate the reciprocal 1-norm condition number in
    # O(p^2), and that is what lets us skip the ~6x more expensive
    # eigendecomposition below: for a symmetric matrix cond_2 <= cond_1, so
    # 1/cond_1 above the gate puts the smallest eigenvalue well above the
    # numerical rank threshold. (The Cholesky *pivots* do not bound the eigenvalue
    # ratio -- a matrix with pivot ratio 1 can still be singular to working
    # precision -- so they cannot be used for this.)
    #
    # The gate has to move with ``rcond``, or a caller who widens the numerical
    # null space would still get directions the exact path would have rejected.
    # pocon returns an *estimate*, biased optimistic, so the factor above ``rcond``
    # is an empirical margin rather than a proof.
    gate = max(_FAST_PATH_RCOND, _FAST_PATH_SAFETY * rcond)
    chol, info = lapack.dpotrf(A_unit, lower=0, clean=1)
    if info == 0:
        rcond_1, info_c = lapack.dpocon(chol, float(np.linalg.norm(A_unit, 1)))
        if info_c == 0 and float(rcond_1) > gate:
            beta_unit, info_s = lapack.dpotrs(chol, b_unit, lower=0)
            if info_s == 0:
                beta_unit = np.asarray(beta_unit, dtype=float).reshape(-1)
                return _rescale_or_raise(beta_unit, b_scale, a_max)

    evals, evecs = np.linalg.eigh(A_unit)
    tol = rcond * max(float(evals[-1]), 0.0)
    if float(evals[0]) < -tol:
        raise np.linalg.LinAlgError(
            f"A is not positive semi-definite (smallest eigenvalue {evals[0] * a_max:.3g}): "
            "the quadratic is unbounded below along that eigenvector."
        )
    if b_scale == 0.0:
        return np.zeros(n, dtype=float)  # beta = 0 solves it, and A is now known PSD

    keep = evals > tol
    lam_min_kept = float(np.min(evals[keep]))  # > 0: lambda_max > tol for rcond < 1

    # How small a null-space component this decomposition can still resolve.
    null_tol = max(rtol, _null_space_resolution(A_unit, evals, evecs, keep, lam_min_kept))
    if null_tol > _MAX_NULL_TOL:
        raise np.linalg.LinAlgError(
            f"the numerical rank of A is ambiguous: its smallest retained eigenvalue "
            f"({lam_min_kept * a_max:.3g}) sits so close to the rank threshold that "
            f"eigenvector error (~{null_tol:.1g}) would swamp any null-space component "
            "of b, so neither a solution nor its absence can be certified. Add or "
            "increase the l2 penalty."
        )

    coeffs = evecs.T @ b_unit
    null_norm = _safe_norm(coeffs[~keep])
    if null_norm > null_tol * b_norm:
        raise np.linalg.LinAlgError(
            "singular system with b outside the range of A: the objective has no "
            "stationary point and is unbounded below along the null space of A "
            f"(null-space component {null_norm:.3g} of ||b|| = {b_norm:.3g}, "
            "after rescaling b to unit sup-norm). Add or increase the l2 penalty."
        )
    beta_unit = evecs[:, keep] @ (coeffs[keep] / evals[keep])
    return _rescale_or_raise(np.asarray(beta_unit, dtype=float), b_scale, a_max)


def standardize_columns(
    X: NDArray[np.float64], *, enabled: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Center and scale the columns of ``X``; return ``(Xs, mean, std)``.

    Columns with zero (or non-positive) spread are left unscaled -- their ``std``
    entry is set to 1.0 -- so that a constant column maps to zeros rather than to
    NaN. When ``enabled`` is False the transform is the identity, but ``mean`` and
    ``std`` are still returned so callers can store them unconditionally.
    """

    d = X.shape[1]
    if enabled:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std = np.where(std > 0, std, 1.0)
    else:
        mean = np.zeros(d)
        std = np.ones(d)
    return (X - mean) / std, mean, std


def is_binary_y(y: NDArray[np.float64]) -> bool:
    """Return True iff ``y`` contains only values in {0, 1}."""

    if y.size == 0:
        return False
    uniq = np.unique(y)
    if uniq.size > 2:
        return False
    return bool(np.all(np.isin(uniq, [0.0, 1.0])))


@dataclass(frozen=True)
class Fold:
    """A single cross-fitting fold."""

    train: NDArray[np.int64]
    test: NDArray[np.int64]


def kfold_splits(
    n: int,
    *,
    folds: int,
    random_state: int | None = None,
    shuffle: bool = True,
) -> Iterator[Fold]:
    """Yield K-fold train/test splits.

    Parameters
    ----------
    n:
        Number of observations.
    folds:
        Number of folds (K).
    random_state:
        Random seed used when ``shuffle=True``.
    shuffle:
        Whether to shuffle indices before splitting.
    """

    n = int(n)
    folds = int(folds)
    if n < 2:
        raise ValueError("n must be >= 2 for cross-fitting")
    if folds <= 1:
        raise ValueError("folds must be >= 2 for cross-fitting")
    if folds > n:
        raise ValueError("folds must be <= n for cross-fitting")
    idx = np.arange(n, dtype=int)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    # Split into approximately equal folds
    parts = np.array_split(idx, folds)
    for k in range(folds):
        test = parts[k]
        train = np.concatenate([parts[j] for j in range(folds) if j != k])
        yield Fold(train=train.astype(int), test=test.astype(int))


def se_ci_pvalue(
    est: float,
    psi: NDArray[np.float64],
    *,
    alpha: float,
    null: float,
) -> tuple[float, float, float, float]:
    """Compute standard error, (1-alpha) CI and a two-sided p-value.

    We use the usual normal approximation and the empirical variance of the
    provided influence function (or score) values.
    """

    def _nan_result(reason: str) -> tuple[float, float, float, float]:
        warnings.warn(
            f"se_ci_pvalue returned NaN: {reason}. Downstream tables will show "
            "NaN standard errors and confidence intervals.",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("nan"), float("nan"), float("nan"), float("nan")

    n = int(len(psi))
    if n <= 1:
        return _nan_result("need at least two score values")

    var = float(np.var(psi, ddof=1))
    if not np.isfinite(var) or var < 0:
        return _nan_result("score variance is not finite")

    se = float(np.sqrt(var / n))
    if not np.isfinite(se) or se <= 0:
        return _nan_result("standard error is not finite or not positive")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci_low = float(est - z * se)
    ci_high = float(est + z * se)

    z_stat = float((est - null) / se)
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
    return se, ci_low, ci_high, p_value


def bias_proxy(held_out_imbalance: float, outcome_coef_norm: float) -> float:
    """Worst-case first-order bias proxy of a DML/ARW functional estimate.

    The augmented (ARW/one-step) estimator has a second-order remainder that is
    controlled by the product of the Riesz-representer imbalance on the working
    span and the size of the outcome regression on that span. Concretely, for a
    linear outcome model ``gamma(x) = phi(x)^T theta`` the remainder is bounded by::

        |E[(alpha_hat - alpha0)(gamma0 - gamma_hat)]|
            <= max_j |Delta_j| * ||theta||_1  (Hoelder),

    and a convenient scale-free surrogate uses the held-out working-span
    imbalance ``max_j |Delta_j|`` (see :func:`genriesz.estimation.grr_functional`)
    together with the outcome coefficient budget ``||theta||_2``.

    This is a *diagnostic* proxy, not an exact bias. It is used to flag cells
    where the reported interval may under-cover; it is never used to select
    hyper-parameters.

    Parameters
    ----------
    held_out_imbalance:
        The held-out (out-of-fold) working-span imbalance ``max_j |Delta_j|``.
    outcome_coef_norm:
        A norm of the outcome regression coefficients on the same span
        (the "coefficient budget").
    """

    b = float(np.abs(held_out_imbalance) * np.abs(outcome_coef_norm))
    return b if np.isfinite(b) else float("nan")


def coverage_decomposition(
    *,
    estimate: float,
    se: float,
    n: int,
    b_hat: float,
    truth: float | None = None,
) -> dict[str, float]:
    """Assemble a bias / variance / standardized-bias decomposition for a cell.

    Returns a dictionary with the pieces needed for the coverage-failure tables
    (revision plan section 5.2/5.3): the point estimate, its standard error, the
    implied score variance ``V_hat = n * se^2``, the bias proxy ``b_hat`` and the
    standardized bias ``sqrt(n) * b_hat / sqrt(V_hat) = b_hat / se``.

    When ``truth`` is provided (simulation only), the realized signed bias and a
    nominal coverage indicator for the Wald interval are added. These oracle
    quantities are for *evaluation only* and must not drive selection.
    """

    n = int(n)
    se = float(se)
    v_hat = float(n * se * se) if np.isfinite(se) else float("nan")
    std_bias = float(b_hat / se) if np.isfinite(se) and se > 0 else float("nan")

    out: dict[str, float] = {
        "estimate": float(estimate),
        "se": se,
        "v_hat": v_hat,
        "b_hat": float(b_hat),
        "std_bias": std_bias,
    }
    if truth is not None:
        from scipy.stats import norm as _norm

        z = float(_norm.ppf(0.975))
        bias = float(estimate) - float(truth)
        out["bias"] = bias
        out["covered"] = float(abs(bias) <= z * se) if np.isfinite(se) else float("nan")
    return out


def oracle_decomposition(
    *,
    y: ArrayLike,
    alpha_hat: ArrayLike,
    alpha0: ArrayLike,
    gamma_hat: ArrayLike,
    gamma0: ArrayLike,
    m_gamma_hat: ArrayLike,
    m_gamma0: ArrayLike,
) -> dict[str, float]:
    """Simulation-only decomposition of a coverage failure into its sources.

    Coverage collapse in the augmented (ARW/one-step) estimator is driven by the
    *product* of the Riesz error ``alpha_hat - alpha0`` and the outcome error
    ``gamma_hat - gamma0``. Given both true nuisances (available only in
    simulation), this returns the RMS nuisance errors, the empirical product
    drift ``E_n[(alpha0 - alpha_hat)(gamma_hat - gamma0)]`` (the leading bias
    term), and the one-step estimators obtained by substituting each true
    nuisance in turn -- which isolates whether the failure is on the Riesz side,
    the outcome side, or their interaction.

    These quantities are for **evaluation only** and must never be used to select
    hyper-parameters (that would be oracle selection).

    Parameters
    ----------
    y:
        Observed outcome.
    alpha_hat, alpha0:
        Estimated and true Riesz representer values at the sample points.
    gamma_hat, gamma0:
        Estimated and true outcome regression values at the sample points.
    m_gamma_hat, m_gamma0:
        The functional applied to the estimated and true outcome regressions,
        ``m(X_i, gamma_hat)`` and ``m(X_i, gamma0)`` (e.g. from
        ``functional.m_from_function``).
    """

    y = as_1d(np.asarray(y, dtype=float).reshape(-1), name="y")
    a_hat = np.asarray(alpha_hat, dtype=float).reshape(-1)
    a0 = np.asarray(alpha0, dtype=float).reshape(-1)
    g_hat = np.asarray(gamma_hat, dtype=float).reshape(-1)
    g0 = np.asarray(gamma0, dtype=float).reshape(-1)
    mg_hat = np.asarray(m_gamma_hat, dtype=float).reshape(-1)
    mg0 = np.asarray(m_gamma0, dtype=float).reshape(-1)

    n = y.shape[0]
    if not (a_hat.shape[0] == a0.shape[0] == g_hat.shape[0] == g0.shape[0] == n):
        raise ValueError("all nuisance arrays must have the same length as y")

    alpha_rmse = float(np.sqrt(np.mean((a_hat - a0) ** 2)))
    gamma_rmse = float(np.sqrt(np.mean((g_hat - g0) ** 2)))
    product_drift = float(np.mean((a0 - a_hat) * (g_hat - g0)))

    # One-step estimators substituting each true nuisance in turn.
    theta_true_alpha = float(np.mean(mg_hat + a0 * (y - g_hat)))
    theta_true_gamma = float(np.mean(mg0 + a_hat * (y - g0)))
    theta_true_both = float(np.mean(mg0 + a0 * (y - g0)))

    return {
        "alpha_rmse": alpha_rmse,
        "gamma_rmse": gamma_rmse,
        "product_drift": product_drift,
        "theta_true_alpha": theta_true_alpha,
        "theta_true_gamma": theta_true_gamma,
        "theta_true_both": theta_true_both,
    }
