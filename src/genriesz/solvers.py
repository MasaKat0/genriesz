r"""Strict solvers for the coefficient problem of generalized Riesz regression.

For a model that is linear in the dual coordinate,

.. math::

    u_\beta(x) = u_{\mathrm{ref}}(x) + \phi(x)^\top \beta ,

the empirical objective (``eq:beta_erm`` in the paper, up to the constant
``-mean m(W, u_ref)``) is

.. math::

    F(\beta) = \frac1n \sum_i g^*(u_\beta(X_i)) - b^\top\beta + \mathrm{pen}(\beta),
    \qquad b = \frac1n\sum_i m(W_i, \phi),

whose loss gradient is the empirical imbalance
:math:`\widehat\Delta_j = \frac1n\sum_i \alpha_i \phi_j(X_i) - b_j`. The
two-sample density-ratio objective has the same form with the sums taken over
the denominator sample and ``b`` the numerator mean of ``phi``. Coefficients are
always deviations from the offset; penalties and the ``l1`` ball act on them.

Feasibility. A coefficient vector is feasible when, at every fitting row *and*
at every additional evaluation point required by the problem (``check_*`` in
:class:`DualProblem`: the counterfactual rows at which the functional evaluates
the dual coordinate, or the numerator sample of a density ratio), the dual
coordinate lies in the open range of ``g'`` on the row's branch and the
representer value lies in the open domain of ``g`` on that branch (so an exp
underflow to ``|alpha| = C`` is infeasible) and is finite.

Two solvers are provided. Both keep every iterate feasible, never clip, and
return an explicit status instead of raising:

- :func:`newton_solve` -- damped Newton for no penalty, ridge, and ``l_q``
  penalties with ``q >= 2``. Backtracking halves the step until the trial point
  is feasible, finite, and satisfies the Armijo condition (coefficient
  ``1e-4``; at most ``max_halvings`` halvings, then ``"linesearch"``).
- :func:`fista_solve` -- projected proximal gradient with acceleration (FISTA)
  for ``h = lam ||.||_1 + I{||.||_1 <= R}`` (either part may be absent) with an
  optional ridge term in the smooth part, and for ``l_q`` with ``1 < q < 2``.
  The update is ``beta+ = prox_{h/L}(y - grad f(y)/L)``: soft-thresholding at
  ``max(lam/L, tau)``, with ``tau`` the level that puts the result on the ball.
  ``L`` doubles until ``beta+`` is feasible and the sufficient-decrease
  condition holds. As registered, the extrapolation is discarded (``y = beta_k``,
  ``t = 1``, and ``beta+`` is recomputed from ``beta_k``) when the extrapolated
  point is infeasible and at an adaptive restart. **Deviation from the
  registration (§1.3 A-3):** the restart test is the gradient-mapping form of
  O'Donoghue and Candes (2015), ``(y - beta+)'(beta+ - beta_k) > 0``, i.e.
  ``G(y)'(beta+ - beta_k) > 0`` with ``G(y) = L (y - beta+)``, instead of the
  registered ``grad f(y)'(beta+ - beta_k) > 0``. The two coincide when
  ``h = 0``; with an ``l1`` term, ``grad f`` does not vanish at the solution
  (it equals ``-lam * sign(beta)`` on the support), the registered test then
  fires at almost every step, and FISTA degrades to an unaccelerated proximal
  gradient (an RKHS ``l1`` example needed more than 100000 iterations instead
  of about 2300). The stopping quantity is the registered proximal
  residual ``rho = L ||beta - prox_{h/L}(beta - grad f(beta)/L)||_inf`` with the
  last ``L``; the subdifferential residual (with the normal cone of an active
  ball) is reported as well.

Statuses (:data:`STATUSES`):

``"ok"``
    The first-order criterion is met at a feasible point.
``"uncertified_numerical_boundary"``
    An iterate has a fitting row within ``boundary_tol`` of the boundary of the
    domain of ``g`` (see :meth:`BregmanGenerator.boundary_mask`; never for SQ).
    This is a numerical finding; it does not establish that no minimizer
    exists. (Exported as :data:`BOUNDARY`.)
``"maxit"``
    The iteration limit was reached.
``"linesearch"``
    Backtracking exhausted its budget without an acceptable point.
``"singular"``
    The Newton system is numerically singular (or, for the arbitrary-link
    objective, the final Hessian is not positive definite).
``"infeasible_start"``
    The starting point is infeasible (e.g. BKL with ``u_ref = 0``).
``"nonfinite"``
    A non-finite objective, gradient, curvature or representer value.
``"domain_prediction"``
    An additional evaluation point (a counterfactual row evaluated by the
    functional, or a numerator row of a density ratio) reached the boundary of
    the domain: the iterates are blocked by the requirement that these points
    stay in the domain.

A solver status is a statement about the optimization run on this sample. It is
not evidence that the population problem, or even this sample's problem, has no
solution; see :mod:`genriesz.certificates` for the sample-level weight program.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .generators import DomainError

OK = "ok"
BOUNDARY = "uncertified_numerical_boundary"
MAXIT = "maxit"
LINESEARCH = "linesearch"
SINGULAR = "singular"
INFEASIBLE_START = "infeasible_start"
NONFINITE = "nonfinite"
DEGENERATE_FUNCTIONAL = "degenerate_functional"
DOMAIN_PREDICTION = "domain_prediction"
CV_FAILED = "cv_failed"

#: Every status the strict solvers and the high-level estimators can report.
STATUSES = (
    OK,
    BOUNDARY,
    MAXIT,
    LINESEARCH,
    SINGULAR,
    INFEASIBLE_START,
    NONFINITE,
    DEGENERATE_FUNCTIONAL,
    DOMAIN_PREDICTION,
    CV_FAILED,
)

#: Statuses that count as a successful fit. ``"closed_form"`` and
#: ``"converged"`` are produced by the legacy ``solver="lbfgs"`` path only.
SUCCESS_STATUSES = frozenset({OK, "closed_form", "converged"})

_EPS = float(np.finfo(float).eps)


@dataclass
class _Point:
    beta: NDArray[np.float64]
    feasible: bool
    finite: bool = False
    loss: float = float("nan")
    grad: NDArray[np.float64] | None = None
    imbalance: NDArray[np.float64] | None = None
    v: NDArray[np.float64] | None = None
    alpha: NDArray[np.float64] | None = None
    dalpha: NDArray[np.float64] | None = None
    check_v: NDArray[np.float64] | None = None
    check_alpha: NDArray[np.float64] | None = None

    @property
    def ok(self) -> bool:
        return self.feasible and self.finite


def _alpha_ok(generator, X, alpha) -> tuple[bool, bool]:
    """``(finite, in_domain)`` of representer values on their rows."""

    fin = np.isfinite(alpha)
    if not np.all(fin):
        return False, True
    return True, bool(np.all(generator.alpha_domain(X, alpha)))


@dataclass
class DualProblem:
    """Smooth part ``mean g*(u_ref + Phi beta) - b' beta + (ridge/2)||beta||^2``.

    Parameters
    ----------
    generator:
        A :class:`~genriesz.generators.BregmanGenerator` (uses ``link_domain``,
        ``dual_eval``, ``alpha_domain`` and ``boundary_mask``).
    X:
        Fitting rows (used to select branches).
    Phi:
        Design matrix ``(n, p)``.
    offset:
        Offset ``u_ref(X_i)``, shape ``(n,)``.
    target:
        The vector ``b``, shape ``(p,)``.
    ridge:
        Ridge weight included in the smooth part (used by FISTA).
    check_X, check_Phi, check_offset:
        Additional evaluation points that must stay feasible (counterfactual
        rows evaluated by the functional, numerator rows of a density ratio).
    """

    generator: object
    X: NDArray[np.float64]
    Phi: NDArray[np.float64]
    offset: NDArray[np.float64]
    target: NDArray[np.float64]
    ridge: float = 0.0
    check_X: NDArray[np.float64] | None = None
    check_Phi: NDArray[np.float64] | None = None
    check_offset: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self.Phi = np.asarray(self.Phi, dtype=float)
        n, p = self.Phi.shape
        self.offset = np.asarray(self.offset, dtype=float).reshape(-1)
        self.target = np.asarray(self.target, dtype=float).reshape(-1)
        self.ridge = float(self.ridge)
        if self.offset.shape != (n,):
            raise ValueError(f"offset must have shape ({n},). Got {self.offset.shape}.")
        if self.target.shape != (p,):
            raise ValueError(f"target must have shape ({p},). Got {self.target.shape}.")
        if self.check_X is not None:
            self.check_Phi = np.asarray(self.check_Phi, dtype=float)
            self.check_offset = np.asarray(self.check_offset, dtype=float).reshape(-1)
            m = self.check_Phi.shape[0]
            if self.check_Phi.shape != (m, p) or self.check_offset.shape != (m,):
                raise ValueError("check_Phi/check_offset do not match check_X and p.")
            if len(self.check_X) != m:
                raise ValueError("check_X and check_Phi must have the same number of rows.")

    @property
    def n(self) -> int:
        return int(self.Phi.shape[0])

    @property
    def p(self) -> int:
        return int(self.Phi.shape[1])

    def restrict(self, S: NDArray[np.bool_], target: NDArray[np.float64]) -> DualProblem:
        """The same problem on the coordinates ``S`` with a new ``target``."""

        return DualProblem(
            generator=self.generator,
            X=self.X,
            Phi=self.Phi[:, S],
            offset=self.offset,
            target=target,
            ridge=self.ridge,
            check_X=self.check_X,
            check_Phi=None if self.check_Phi is None else self.check_Phi[:, S],
            check_offset=self.check_offset,
        )

    @property
    def has_checks(self) -> bool:
        return self.check_X is not None and len(self.check_X) > 0

    def check_points(self, beta: NDArray[np.float64]):
        """``(feasible, finite, v, alpha)`` of the additional evaluation points."""

        if not self.has_checks:
            return True, True, None, None
        assert self.check_Phi is not None and self.check_offset is not None
        v = self.check_offset + self.check_Phi @ beta
        if not np.all(self.generator.link_domain(self.check_X, v)):
            return False, True, v, None
        try:
            _, a, _ = self.generator.dual_eval(self.check_X, v)
        except DomainError:
            return False, True, v, None
        fin, dom = _alpha_ok(self.generator, self.check_X, a)
        return dom, fin, v, a

    def evaluate(self, beta: NDArray[np.float64]) -> _Point:
        beta = np.asarray(beta, dtype=float).reshape(-1)
        v = self.offset + self.Phi @ beta
        in_dom = np.asarray(self.generator.link_domain(self.X, v), dtype=bool)
        if not np.all(in_dom):
            return _Point(beta=beta, feasible=False, v=v)
        try:
            g_star, alpha, dalpha = self.generator.dual_eval(self.X, v)
        except DomainError:
            # A user-defined generator signals "outside my domain" by raising
            # DomainError from its link (the generic link_domain cannot know the
            # domain). That is a property of this trial point, handled exactly
            # like a link_domain violation: backtrack, or report
            # "infeasible_start". Any other exception propagates.
            return _Point(beta=beta, feasible=False, v=v)
        fin_a, dom_a = _alpha_ok(self.generator, self.X, alpha)
        if not dom_a:
            return _Point(beta=beta, feasible=False, v=v)
        dom_c, fin_c, cv, ca = self.check_points(beta)
        if not dom_c:
            return _Point(beta=beta, feasible=False, v=v)
        imbalance = self.Phi.T @ alpha / self.n - self.target
        loss = float(np.mean(g_star) - self.target @ beta) + 0.5 * self.ridge * float(beta @ beta)
        grad = imbalance + self.ridge * beta
        finite = bool(
            fin_a
            and fin_c
            and np.isfinite(loss)
            and np.all(np.isfinite(grad))
            and np.all(np.isfinite(dalpha))
        )
        return _Point(
            beta=beta,
            feasible=True,
            finite=finite,
            loss=loss,
            grad=grad,
            imbalance=imbalance,
            v=v,
            alpha=alpha,
            dalpha=dalpha,
            check_v=cv,
            check_alpha=ca,
        )

    def hessian(self, pt: _Point) -> NDArray[np.float64]:
        assert pt.dalpha is not None
        H = (self.Phi.T * pt.dalpha) @ self.Phi / self.n
        if self.ridge:
            H = H + self.ridge * np.eye(self.p)
        return H

    def at_boundary(self, pt: _Point, tol: float) -> NDArray[np.bool_]:
        assert pt.v is not None and pt.alpha is not None
        return np.asarray(self.generator.boundary_mask(self.X, pt.v, pt.alpha, tol=tol), bool)

    def check_at_boundary(self, pt: _Point, tol: float) -> int:
        """Number of additional evaluation points numerically at the domain boundary."""

        if not self.has_checks or pt.check_alpha is None:
            return 0
        return int(np.sum(self.generator.boundary_mask(
            self.check_X, pt.check_v, pt.check_alpha, tol=tol)))

    def _margins(self, X, v, alpha) -> NDArray[np.float64]:
        """Per-row margins (``alpha``-side and dual-side) stacked as columns."""

        cols = []
        fn = getattr(self.generator, "boundary_margin", None)
        if callable(fn):
            cols.append(np.asarray(fn(X, alpha), dtype=float))
        fn = getattr(self.generator, "dual_margin", None)
        if callable(fn):
            cols.append(np.asarray(fn(X, v), dtype=float))
        if not cols:
            return np.full((alpha.shape[0], 1), np.inf)
        return np.column_stack(cols)

    def margin_ok(self, ref: _Point, trial: _Point, shrink: float) -> bool:
        """Fraction-to-the-boundary rule over fitting rows and evaluation points.

        No row's distance to the domain boundary (in ``alpha`` and in the dual
        coordinate) may shrink below ``shrink`` times its current value.
        """

        if shrink <= 0.0:
            return True
        pairs = [(self.X, ref.v, ref.alpha, trial.v, trial.alpha)]
        if self.has_checks and ref.check_alpha is not None and trial.check_alpha is not None:
            pairs.append((self.check_X, ref.check_v, ref.check_alpha, trial.check_v,
                          trial.check_alpha))
        for X, v0, a0, v1, a1 in pairs:
            m0 = self._margins(X, v0, a0)
            fin = np.isfinite(m0)
            if not np.any(fin):
                continue
            m1 = self._margins(X, v1, a1)
            if not np.all(m1[fin] >= shrink * m0[fin]):
                return False
        return True


@dataclass
class SolverResult:
    """Outcome of a strict solve.

    Attributes
    ----------
    beta:
        Last iterate (the solution when ``status == "ok"``).
    status:
        One of :data:`STATUSES`.
    n_iter:
        Number of iterations performed.
    objective:
        Penalized objective at ``beta`` (exact ``l1``, no smoothing).
    gradient:
        Loss gradient at ``beta`` without penalty terms: the empirical
        imbalance ``Delta``.
    kkt_residual:
        Subdifferential residual of ``0 in grad L + lam subdiff||beta||_1 +
        N_B(beta)`` (with the smooth penalty in ``grad L``); for smooth
        penalties without a ball, ``max_j |dF/dbeta_j|``.
    prox_residual:
        FISTA only: ``L ||beta - prox_{h/L}(beta - grad f(beta)/L)||_inf`` with
        the last ``L`` (the registered stopping quantity).
    alpha, v:
        Representer values and dual coordinates on the fitting rows.
    n_boundary:
        Number of rows numerically at the domain boundary at ``beta``.
    l1_ball_active, ball_multiplier:
        Whether ``||beta||_1 = R1`` at the solution and the multiplier ``mu >= 0``
        of the ball constraint used in the residual.
    polished:
        Whether the FISTA solution was refined by Newton on its support.
    hessian_min_eig:
        Smallest eigenvalue of the (penalized) Hessian at ``beta`` when computed.
    """

    beta: NDArray[np.float64]
    status: str
    n_iter: int
    message: str = ""
    objective: float = float("nan")
    gradient: NDArray[np.float64] | None = None
    kkt_residual: float = float("nan")
    prox_residual: float = float("nan")
    alpha: NDArray[np.float64] | None = None
    v: NDArray[np.float64] | None = None
    n_boundary: int = 0
    l1_ball_active: bool = False
    ball_multiplier: float = 0.0
    polished: bool = False
    hessian_min_eig: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == OK


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------
def _smooth_penalty(
    lam: float, q: float | None
) -> tuple[
    Callable[[NDArray[np.float64]], float],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
]:
    """Value, gradient and Hessian diagonal of ``(lam/q) ||beta||_q^q`` (``q >= 2``)."""

    if q is None or lam == 0.0:
        return (
            lambda b: 0.0,
            lambda b: np.zeros_like(b),
            lambda b: np.zeros_like(b),
        )
    if q < 2.0:
        raise ValueError("newton_solve requires a twice-differentiable penalty (q >= 2).")
    return (
        lambda b: float((lam / q) * np.sum(np.abs(b) ** q)),
        lambda b: lam * np.sign(b) * np.abs(b) ** (q - 1.0),
        lambda b: lam * (q - 1.0) * np.abs(b) ** (q - 2.0),
    )


def soft_threshold(z: NDArray[np.float64], tau: float) -> NDArray[np.float64]:
    """Elementwise soft-thresholding ``sign(z) max(|z| - tau, 0)``."""

    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


def _l1_ball_level(z: NDArray[np.float64], radius: float) -> float:
    """Threshold ``tau >= 0`` with ``||soft_threshold(z, tau)||_1 = radius``.

    Requires ``||z||_1 > radius``. Standard sort-based projection onto the
    ``l1`` ball (Duchi et al., 2008).
    """

    u = np.sort(np.abs(z))[::-1]
    css = np.cumsum(u)
    j = np.arange(1, u.size + 1)
    cond = u - (css - radius) / j > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    return float((css[rho] - radius) / (rho + 1))


def prox_l1_ball(
    z: NDArray[np.float64], step: float, lam: float, radius: float | None
) -> NDArray[np.float64]:
    """Proximal map of ``step * (lam ||.||_1 + I{||.||_1 <= radius})`` at ``z``.

    This is soft-thresholding at ``max(step * lam, tau)``, where ``tau`` is the
    level at which the soft-thresholded vector has ``l1`` norm exactly
    ``radius`` (``tau = 0`` when ``||z||_1 <= radius``). The ``lam`` threshold is
    scaled by the step size ``step = 1/L``; the ball level is not, because the
    indicator is invariant to scaling.
    """

    w = soft_threshold(z, step * lam)
    if radius is None or float(np.sum(np.abs(w))) <= radius:
        return w
    tau = _l1_ball_level(z, radius)
    return soft_threshold(z, max(step * lam, tau))


def prox_lq(z: NDArray[np.float64], step: float, lam: float, q: float) -> NDArray[np.float64]:
    """Proximal map of ``step * (lam/q) ||.||_q^q`` for ``1 < q < 2`` (bisection).

    Coordinatewise ``x = sign(z) r`` with ``r + step*lam*r^(q-1) = |z|``.
    """

    c = step * lam
    a = np.abs(z)
    lo = np.zeros_like(a)
    hi = a.copy()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = mid + c * np.power(mid, q - 1.0) - a
        lo = np.where(f < 0.0, mid, lo)
        hi = np.where(f < 0.0, hi, mid)
    return np.sign(z) * 0.5 * (lo + hi)


def l1_kkt_residual(
    grad: NDArray[np.float64],
    beta: NDArray[np.float64],
    lam: float,
    radius: float | None = None,
    *,
    active_rtol: float = 1e-12,
) -> tuple[float, bool, float]:
    r"""Residual of ``0 in grad + lam * subdiff||beta||_1 + N_B(beta)``.

    For ``B = {||beta||_1 <= R1}`` and ``||beta||_1 = R1``, the normal cone is
    ``{mu s : mu >= 0, s in subdiff||beta||_1}``, so the condition is the ``l1``
    condition with ``lam`` replaced by some ``lam' = lam + mu >= lam``. The
    residual is

    .. math::

        \min_{\lambda'\ge\lambda}\max\Bigl(\max_{j\in S}|g_j+\lambda'\sigma_j|,\;
        \max_{j\notin S}(|g_j|-\lambda')_+\Bigr),

    minimized in closed form. Without an active ball, ``lam' = lam``.

    Returns
    -------
    (residual, ball_active, mu)
    """

    grad = np.asarray(grad, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if beta.size == 0:
        return 0.0, False, 0.0
    S = beta != 0.0
    sigma = np.sign(beta)

    def resid(lp: float) -> float:
        on = np.abs(grad[S] + lp * sigma[S]) if np.any(S) else np.zeros(0)
        off = np.maximum(np.abs(grad[~S]) - lp, 0.0) if np.any(~S) else np.zeros(0)
        return float(max(on.max(initial=0.0), off.max(initial=0.0)))

    active = radius is not None and float(np.sum(np.abs(beta))) >= radius * (1.0 - active_rtol)
    if not active:
        return resid(lam), False, 0.0
    if np.any(S):
        c = -sigma[S] * grad[S]
        hi = max(float(c.max()), float(np.abs(grad[~S]).max(initial=-np.inf)))
        lp = max(lam, 0.5 * (float(c.min()) + hi))
    else:  # pragma: no cover - beta = 0 cannot be on a sphere of positive radius
        lp = lam
    return resid(lp), True, lp - lam


def lq_residual(
    grad: NDArray[np.float64], beta: NDArray[np.float64], lam: float, q: float
) -> float:
    """Residual of ``grad + lam sign(beta)|beta|^(q-1) = 0`` (``q > 1``)."""

    if beta.size == 0:
        return 0.0
    return float(np.max(np.abs(grad + lam * np.sign(beta) * np.abs(beta) ** (q - 1.0))))



def _accept_newton(F: float, Ft: float, slope: float, step: float, armijo: float,
                   gnorm: float, gnorm_t: float) -> bool:
    """Armijo, or -- when the objective change is below rounding -- gradient decrease."""

    if Ft <= F - armijo * step * slope:
        return True
    roundoff = 8.0 * _EPS * max(1.0, abs(F))
    return abs(Ft - F) <= roundoff and gnorm_t < gnorm


def newton_solve(
    problem: DualProblem,
    *,
    beta0: NDArray[np.float64],
    lam: float = 0.0,
    q: float | None = None,
    max_iter: int = 500,
    max_halvings: int = 60,
    tol: float = 1e-10,
    tol_scale: float = 1.0,
    boundary_tol: float = 1e-8,
    armijo: float = 1e-4,
    singular_rtol: float = 1e-13,
    margin_shrink: float = 0.01,
) -> SolverResult:
    """Damped Newton for ``F = smooth part + (lam/q)||beta||_q^q`` with ``q >= 2``.

    Stops with ``"ok"`` when ``max_j |dF/dbeta_j| <= tol * tol_scale`` at a
    feasible point. ``tol_scale`` lets the caller express the registered
    unpenalized criterion ``max_j|Delta_j| <= tol * max(1, max_j|b_j|)``.

    A trial step is accepted when it is feasible (see :class:`DualProblem`),
    everything is finite, the Armijo condition holds (or, when the objective
    change is below rounding, the gradient norm decreases), and no fitting row's
    distance to the domain boundary shrinks below ``margin_shrink`` times its
    current value. The step is halved otherwise, at most ``max_halvings`` times.
    """

    pen_val, pen_grad, pen_hess = _smooth_penalty(float(lam), q)
    beta = np.asarray(beta0, dtype=float).reshape(-1).copy()
    pt = problem.evaluate(beta)
    if not pt.feasible:
        return SolverResult(
            beta=beta,
            status=INFEASIBLE_START,
            n_iter=0,
            message="the starting point is outside the domain of the generator at a "
            "fitting row or at an evaluation point required by the problem",
        )
    if not pt.finite:
        return SolverResult(beta=beta, status=NONFINITE, n_iter=0,
                            message="non-finite objective at the starting point")

    status = MAXIT
    message = ""
    hmin = float("nan")
    it = 0
    while True:
        F = pt.loss + pen_val(pt.beta)
        G = pt.grad + pen_grad(pt.beta)
        gnorm = float(np.max(np.abs(G))) if G.size else 0.0
        nb = int(np.sum(problem.at_boundary(pt, boundary_tol)))
        if nb > 0:
            status, message = BOUNDARY, f"{nb} row(s) at the domain boundary"
            break
        nc = problem.check_at_boundary(pt, boundary_tol)
        if nc > 0:
            status = DOMAIN_PREDICTION
            message = (f"{nc} evaluation point(s) (counterfactual or numerator rows) reached "
                       "the domain boundary")
            break
        H = problem.hessian(pt) + np.diag(pen_hess(pt.beta))
        if not np.all(np.isfinite(H)):
            status, message = NONFINITE, "non-finite Hessian"
            break
        evals, evecs = np.linalg.eigh(H) if H.size else (np.zeros(0), np.zeros((0, 0)))
        hmin = float(evals.min()) if evals.size else float("nan")
        if gnorm <= tol * tol_scale:
            status = OK
            break
        if it >= max_iter:
            status, message = MAXIT, f"reached max_iter={max_iter}"
            break
        emax = float(evals.max())
        if emax <= 0.0 or hmin <= singular_rtol * emax:
            status = SINGULAR
            message = f"Hessian numerically singular (min eig {hmin:.3e}, max eig {emax:.3e})"
            break
        d = evecs @ ((evecs.T @ G) / evals)
        slope = float(G @ d)
        step = 1.0
        accepted = None
        for _ in range(max_halvings + 1):
            trial = problem.evaluate(pt.beta - step * d)
            if trial.ok and problem.margin_ok(pt, trial, margin_shrink):
                Ft = trial.loss + pen_val(trial.beta)
                Gt = trial.grad + pen_grad(trial.beta)
                if _accept_newton(F, Ft, slope, step, armijo, gnorm,
                                  float(np.max(np.abs(Gt)))):
                    accepted = trial
                    break
            step *= 0.5
        it += 1
        if accepted is None:
            status = LINESEARCH
            message = f"no acceptable step after {max_halvings} halvings"
            break
        pt = accepted

    G = pt.grad + pen_grad(pt.beta)
    return SolverResult(
        beta=pt.beta,
        status=status,
        n_iter=it,
        message=message,
        objective=float(pt.loss + pen_val(pt.beta)),
        gradient=pt.imbalance,
        kkt_residual=float(np.max(np.abs(G))) if G.size else 0.0,
        alpha=pt.alpha,
        v=pt.v,
        n_boundary=int(np.sum(problem.at_boundary(pt, boundary_tol))),
        hessian_min_eig=hmin,
    )


def _polish_on_support(
    problem: DualProblem,
    beta: NDArray[np.float64],
    *,
    lam: float,
    radius: float | None,
    tol: float,
    boundary_tol: float,
    margin_shrink: float,
    max_iter: int = 50,
    max_halvings: int = 60,
) -> NDArray[np.float64] | None:
    """Newton refinement of a FISTA solution on its support with the signs fixed.

    With ``lam > 0`` or an active ball, the support ``S`` is the set of nonzero
    coordinates and ``loss(beta_S) + lam sigma' beta_S`` is minimized over it
    (with ``radius``, subject to ``sigma' beta_S = radius`` by
    equality-constrained Newton steps). With ``lam = 0`` and an inactive ball,
    every coordinate is refined. The ridge term of ``problem`` is included.
    Returns the refined full vector, or ``None`` when the refinement did not
    reach ``tol``; the caller re-validates it on the full problem.
    """

    if lam > 0.0 or radius is not None:
        S = beta != 0.0
    else:
        S = np.ones(beta.shape[0], dtype=bool)
    sigma = np.sign(beta[S])
    sub = problem.restrict(S, problem.target[S] - lam * sigma)
    if radius is None:
        sr = newton_solve(
            sub, beta0=beta[S], tol=tol, boundary_tol=boundary_tol, margin_shrink=margin_shrink
        )
        if sr.status != OK:
            return None
        out = np.zeros_like(beta)
        out[S] = sr.beta
        return out

    k = int(np.sum(S))
    pt = sub.evaluate(beta[S])
    if not pt.ok:
        return None
    for _ in range(max_iter):
        H = sub.hessian(pt)
        evals = np.linalg.eigvalsh(H)
        if not np.all(np.isfinite(evals)) or evals.min() <= 1e-13 * max(evals.max(), 0.0):
            return None
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = H
        K[:k, k] = sigma
        K[k, :k] = sigma
        rhs = np.concatenate([-pt.grad, [radius - float(sigma @ pt.beta)]])
        sol = np.linalg.solve(K, rhs)
        d, nu = sol[:k], float(sol[k])
        if float(np.max(np.abs(pt.grad + nu * sigma))) <= tol and abs(rhs[-1]) <= 1e-14 * radius:
            out = np.zeros_like(beta)
            out[S] = pt.beta
            return out
        step = 1.0
        accepted = None
        for _ in range(max_halvings + 1):
            trial = sub.evaluate(pt.beta + step * d)
            if trial.ok and sub.margin_ok(pt, trial, margin_shrink):
                if trial.loss <= pt.loss + 1e-4 * step * float(pt.grad @ d) + 8.0 * _EPS * max(
                    1.0, abs(pt.loss)
                ):
                    accepted = trial
                    break
            step *= 0.5
        if accepted is None:
            return None
        pt = accepted
    return None


# ---------------------------------------------------------------------------
# Projected proximal gradient with acceleration (FISTA)
# ---------------------------------------------------------------------------
def fista_solve(
    problem: DualProblem,
    *,
    beta0: NDArray[np.float64],
    lam: float,
    q: float = 1.0,
    radius: float | None = None,
    max_iter: int = 100_000,
    max_backtracks: int = 60,
    tol: float = 1e-8,
    boundary_tol: float = 1e-8,
    lipschitz_decrease: float = 0.9,
    polish: bool = True,
    polish_tol: float = 1e-12,
    margin_shrink: float = 0.01,
) -> SolverResult:
    """Projected FISTA for ``f + h``, ``f`` = smooth part of ``problem`` (incl. ridge).

    ``h = lam ||.||_1 + I{||.||_1 <= radius}`` for ``q = 1`` (``lam = 0``: the
    projection onto the ball; ``radius=None``: no ball), or
    ``(lam/q)||.||_q^q`` for ``1 < q < 2`` (no ball). Stops with ``"ok"`` when the
    proximal residual ``rho`` (see the module docstring) is ``<= tol`` at a
    feasible point. The registered tolerances are ``1e-6 * lam`` for ``l1`` and
    ``1e-8`` for ridge or ball only.

    With ``polish=True`` and ``q = 1``, an ``"ok"`` solution is refined by
    Newton on its support with the signs fixed (equality-constrained to the
    sphere ``||beta||_1 = R`` when the ball is active), toward ``polish_tol``.
    The refinement is kept only if it is feasible, stays in the ball, keeps the
    signs, and does not increase the proximal residual; ``polished`` records
    whether it was kept.
    """

    q = float(q)
    if q < 1.0:
        raise ValueError("q must be >= 1")
    if q != 1.0 and radius is not None:
        raise ValueError("an l1-ball restriction is supported only with the l1 penalty")
    lam = float(lam)

    def penalty(b: NDArray[np.float64]) -> float:
        if q == 1.0:
            return lam * float(np.sum(np.abs(b)))
        return (lam / q) * float(np.sum(np.abs(b) ** q))

    def prox(z: NDArray[np.float64], step: float) -> NDArray[np.float64]:
        if q == 1.0:
            return prox_l1_ball(z, step, lam, radius)
        return prox_lq(z, step, lam, q)

    def prox_residual(p: _Point, Lc: float) -> float:
        assert p.grad is not None
        if p.beta.size == 0:
            return 0.0
        return float(Lc * np.max(np.abs(p.beta - prox(p.beta - p.grad / Lc, 1.0 / Lc))))

    def subgrad_residual(p: _Point) -> tuple[float, bool, float]:
        assert p.grad is not None
        if q == 1.0:
            return l1_kkt_residual(p.grad, p.beta, lam, radius)
        return lq_residual(p.grad, p.beta, lam, q), False, 0.0

    beta = np.asarray(beta0, dtype=float).reshape(-1).copy()
    if radius is not None and float(np.sum(np.abs(beta))) > radius:
        return SolverResult(beta=beta, status=INFEASIBLE_START, n_iter=0,
                            message="beta0 lies outside the l1 ball")
    pt = problem.evaluate(beta)
    if not pt.feasible:
        return SolverResult(
            beta=beta,
            status=INFEASIBLE_START,
            n_iter=0,
            message="the starting point is outside the domain of the generator at a "
            "fitting row or at an evaluation point required by the problem",
        )
    if not pt.finite:
        return SolverResult(beta=beta, status=NONFINITE, n_iter=0,
                            message="non-finite objective at the starting point")

    # Initial Lipschitz estimate from the curvature at the start.
    H0 = problem.hessian(pt)
    L = float(np.linalg.eigvalsh(H0).max()) if H0.size else 1.0
    if not np.isfinite(L) or L <= 0.0:
        L = 1.0

    y_pt = pt
    t = 1.0
    status = MAXIT
    message = f"reached max_iter={max_iter}"
    it = 0
    n_restarts = 0
    n_extrapolation_resets = 0
    rho = prox_residual(pt, L)
    while it < max_iter:
        nb = int(np.sum(problem.at_boundary(pt, boundary_tol)))
        if nb > 0:
            status, message = BOUNDARY, f"{nb} row(s) at the domain boundary"
            break
        nc = problem.check_at_boundary(pt, boundary_tol)
        if nc > 0:
            status = DOMAIN_PREDICTION
            message = (f"{nc} evaluation point(s) (counterfactual or numerator rows) reached "
                       "the domain boundary")
            break
        if rho <= tol:
            status, message = OK, ""
            break
        L = L * lipschitz_decrease
        new = None
        for _ in range(max_backtracks + 1):
            b_new = prox(y_pt.beta - y_pt.grad / L, 1.0 / L)
            cand = problem.evaluate(b_new)
            if cand.ok and problem.margin_ok(y_pt, cand, margin_shrink):
                d = b_new - y_pt.beta
                bound = y_pt.loss + float(y_pt.grad @ d) + 0.5 * L * float(d @ d)
                if cand.loss <= bound + 8.0 * _EPS * max(1.0, abs(y_pt.loss)):
                    new = cand
                    break
            L *= 2.0
        it += 1
        if new is None:
            status = LINESEARCH
            message = f"no acceptable step after {max_backtracks} backtracking doublings"
            break
        if y_pt is not pt and float((y_pt.beta - new.beta) @ (new.beta - pt.beta)) > 0.0:
            # Adaptive restart (gradient-mapping test; see the module docstring
            # for the deviation from the registered test): y = beta_k, t = 1,
            # and the step is recomputed from beta_k.
            y_pt, t = pt, 1.0
            n_restarts += 1
            continue
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y_cand = problem.evaluate(new.beta + ((t - 1.0) / t_new) * (new.beta - pt.beta))
        if not y_cand.ok:
            # Registered: an infeasible extrapolated point is discarded.
            y_cand, t_new = new, 1.0
            n_extrapolation_resets += 1
        pt, y_pt, t = new, y_cand, t_new
        rho = prox_residual(pt, L)

    polished = False
    if status == OK and polish and q == 1.0 and np.any(pt.beta != 0.0):
        _, active0, _ = subgrad_residual(pt)
        b_pol = _polish_on_support(
            problem,
            pt.beta,
            lam=lam,
            radius=radius if active0 else None,
            tol=polish_tol,
            boundary_tol=boundary_tol,
            margin_shrink=margin_shrink,
        )
        if b_pol is not None:
            cand = problem.evaluate(b_pol)
            same_signs = bool(np.all(np.sign(b_pol)[pt.beta != 0.0] == np.sign(pt.beta)[
                pt.beta != 0.0]))
            in_ball = radius is None or float(np.sum(np.abs(b_pol))) <= radius * (1.0 + 1e-12)
            if cand.ok and same_signs and in_ball:
                rho_pol = prox_residual(cand, L)
                if (rho_pol <= rho and not np.any(problem.at_boundary(cand, boundary_tol))
                        and problem.check_at_boundary(cand, boundary_tol) == 0):
                    pt, rho = cand, rho_pol
                    polished = True

    res_sub, active, mu = subgrad_residual(pt)
    return SolverResult(
        beta=pt.beta,
        status=status,
        n_iter=it,
        message=message,
        objective=float(pt.loss + penalty(pt.beta)),
        gradient=pt.imbalance,
        kkt_residual=float(res_sub),
        prox_residual=float(rho),
        alpha=pt.alpha,
        v=pt.v,
        n_boundary=int(np.sum(problem.at_boundary(pt, boundary_tol))),
        l1_ball_active=bool(active),
        ball_multiplier=float(mu),
        polished=polished,
        extra={
            "n_restarts": n_restarts,
            "n_extrapolation_resets": n_extrapolation_resets,
            "lipschitz": L,
        },
    )
