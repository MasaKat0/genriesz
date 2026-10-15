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

The entry points ``newton_solve`` and ``fista_solve`` use
``scipy.optimize.minimize``. Smooth unconstrained objectives use its default
BFGS method. Exact l1 penalties use positive and negative coefficient parts,
with nonnegative bounds (default L-BFGS-B); an l1 ball adds a linear
constraint (default SLSQP). No solver is retried on failure. ``ok`` means the
library reported success at a finite, feasible solution. Library messages and
status codes are retained. Residuals and boundary counts are diagnostics.

A solver status is a statement about the optimization run on this sample. It is
not evidence that the population problem, or even this sample's problem, has no
solution; see :mod:`genriesz.certificates` for the sample-level weight program.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

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
OPTIMIZER_FAILURE = "optimizer_failure"

#: Every status the strict solvers and the high-level estimators can report.
STATUSES = (
    OPTIMIZER_FAILURE,
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
            # domain). This trial point has infinite objective value in the
            # optimizer. Any other exception propagates.
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
        Unavailable for the SciPy routines (NaN).
    alpha, v:
        Representer values and dual coordinates on the fitting rows.
    n_boundary:
        Number of rows numerically at the domain boundary at ``beta``.
    l1_ball_active, ball_multiplier:
        Whether ``||beta||_1 = R1`` at the solution and the multiplier ``mu >= 0``
        of the ball constraint used in the residual.
    polished:
        False for the SciPy routines.
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


def prox_lq(z: NDArray[np.float64], step: float, lam: float, q: float,
            *, options: dict | None = None) -> NDArray[np.float64]:
    """Proximal map of ``step * (lam/q) ||.||_q^q`` for ``1 < q < 2`` (SciPy Brent roots).

    Coordinatewise ``x = sign(z) r`` with ``r + step*lam*r^(q-1) = |z|``.
    """

    c = step * lam
    if c == 0.0:
        return z.copy()
    a = np.abs(z)
    roots = np.array([
        0.0 if v == 0.0 else optimize.brentq(lambda r, v=v: r + c * r ** (q - 1.0) - v,
                                              0.0, v, **({} if options is None else options))
        for v in a.flat
    ]).reshape(a.shape)
    return np.sign(z) * roots


def l1_kkt_residual(
    grad: NDArray[np.float64],
    beta: NDArray[np.float64],
    lam: float,
    radius: float | None = None,
    *,
    active_rtol: float | None = None,
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

    active = radius is not None and float(np.sum(np.abs(beta))) >= radius * (
        1.0 if active_rtol is None else 1.0 - active_rtol
    )
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



def minimize_coefficients(fun, jac, beta0, *, l1=0.0, radius=None,
                          max_iter=None, tol=None, options=None):
    """Minimize an objective with an exact (possibly weighted) l1 penalty.

    ``beta = positive - negative`` and ``positive, negative >= 0`` give an
    equivalent smooth constrained problem. With a ball, their sum is at most
    ``radius``. ``None`` leaves every SciPy solver setting at its default.
    """
    beta0 = np.asarray(beta0, dtype=float)
    opts = {} if options is None else dict(options)
    if max_iter is not None:
        opts["maxiter"] = int(max_iter)
    split = radius is not None or np.any(np.asarray(l1) != 0.0)
    if not split:
        return optimize.minimize(fun, beta0, jac=jac, tol=tol, options=opts)
    p = beta0.size
    weights = np.broadcast_to(l1, beta0.shape)
    z0 = np.concatenate((np.maximum(beta0, 0.0), np.maximum(-beta0, 0.0)))

    def value(z):
        return fun(z[:p] - z[p:]) + float(weights @ (z[:p] + z[p:]))

    def gradient(z):
        g = jac(z[:p] - z[p:])
        return np.concatenate((g + weights, -g + weights))

    constraints = () if radius is None else (
        optimize.LinearConstraint(np.ones((1, 2 * p)), -np.inf, radius),
    )
    res = optimize.minimize(value, z0, jac=gradient, bounds=[(0.0, None)] * (2 * p),
                            constraints=constraints, tol=tol, options=opts)
    res.split_x = res.x.copy()
    res.x = res.x[:p] - res.x[p:]
    return res


def scipy_status(res):
    """Translate SciPy termination without imposing a separate convergence test."""
    if res.success:
        return OK
    if ("iteration" in str(res.message).lower() and "limit" in str(res.message).lower()
            or "maximum number of iterations" in str(res.message).lower()):
        return MAXIT
    return OPTIMIZER_FAILURE


def _solve(problem, *, beta0, lam=0.0, q=None, radius=None,
           max_iter=None, tol=None, boundary_tol=None, options=None):
    q = 1.0 if q is None else float(q)
    if q < 1.0:
        raise ValueError("q must be >= 1")
    if radius is not None and q != 1.0:
        raise ValueError("an l1-ball restriction requires q=1")
    beta0 = np.asarray(beta0, dtype=float).reshape(-1)

    def value(b):
        pt = problem.evaluate(b)
        if not pt.ok:
            return np.inf
        penalty = 0.0 if q == 1.0 else (lam / q) * float(np.sum(np.abs(b) ** q))
        return pt.loss + penalty

    def gradient(b):
        pt = problem.evaluate(b)
        if not pt.ok:
            return np.full_like(b, np.nan)
        return pt.grad + (0.0 if q == 1.0 else lam * np.sign(b) * np.abs(b) ** (q - 1.0))

    res = minimize_coefficients(value, gradient, beta0, l1=lam if q == 1.0 else 0.0,
                                radius=radius, max_iter=max_iter, tol=tol, options=options)
    pt = problem.evaluate(res.x)
    status = scipy_status(res)
    in_ball = radius is None or float(np.sum(np.abs(res.x))) <= radius
    if not np.all(np.isfinite(res.x)):
        status = NONFINITE
    elif not pt.feasible or not in_ball:
        status = INFEASIBLE_START if not problem.evaluate(beta0).feasible else OPTIMIZER_FAILURE
    elif not pt.finite or not np.isfinite(res.fun) or not np.all(np.isfinite(res.jac)):
        status = NONFINITE
    residual, active, mu = float("nan"), False, 0.0
    if pt.ok:
        if q == 1.0:
            residual, active, mu = l1_kkt_residual(pt.grad, pt.beta, lam, radius)
        else:
            residual = lq_residual(pt.grad, pt.beta, lam, q)
    if pt.ok and hasattr(res, "split_x"):
        if radius is not None:
            mu = float(res.multipliers[0]) if hasattr(res, "multipliers") else float("nan")
            active = bool(mu > 0.0)
        split_grad = np.asarray(res.jac) + mu
        residual = float(np.max(np.abs(np.minimum(res.split_x, split_grad))))
    return SolverResult(
        beta=res.x, status=status, n_iter=int(getattr(res, "nit", 0)),
        message=str(res.message),
        objective=(float(pt.loss + (lam / q) * np.sum(np.abs(pt.beta) ** q))
                   if pt.ok else float(res.fun)), gradient=pt.imbalance,
        kkt_residual=residual, alpha=pt.alpha, v=pt.v,
        n_boundary=(int(np.sum(problem.at_boundary(pt, boundary_tol)))
                    if pt.ok and boundary_tol is not None else 0),
        l1_ball_active=active, ball_multiplier=mu,
        extra={"scipy_status": int(res.status), "scipy_success": bool(res.success)},
    )


def newton_solve(problem, *, beta0, lam=0.0, q=None, max_iter=None, tol=None,
                 boundary_tol=None, options=None):
    """Smooth coefficient objective using SciPy's default unconstrained method.

    ``options`` is passed to SciPy unchanged, except an explicit ``max_iter``.
    """
    if q is not None and q < 2.0:
        raise ValueError("newton_solve requires q >= 2")
    return _solve(problem, beta0=beta0, lam=0.0 if q is None else lam, q=q,
                  max_iter=max_iter, tol=tol,
                  boundary_tol=boundary_tol, options=options)


def fista_solve(problem, *, beta0, lam, q=1.0, radius=None, max_iter=None, tol=None,
                boundary_tol=None, options=None):
    """Exact l1 or differentiable lq objective using SciPy's default method.

    The l1 ball and penalty apply to deviations from the offset.
    """
    return _solve(problem, beta0=beta0, lam=lam, q=q, radius=radius,
                  max_iter=max_iter, tol=tol, boundary_tol=boundary_tol, options=options)
