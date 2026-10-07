"""E-14: the offset sparse-model bound and rate for the ATT (H14-Bnd, H14-Rate).

Design: doc/2026-10-07_experiment_registration.md (parent repository), E-14, with
§1.2 (streams), §1.3 A-1/A-3/B/C (offset, projected FISTA, statuses, tolerance),
§1.8 and §1.9. Since U28 (2026-10-08) the design document is not a binding
registration; where it does not fix a choice, the choice is made here and listed
in the parent repository's ``docs/spec/02`` §2.1 ("E-14 の" rows).

Planted design. ``Z ~ Unif(-sqrt3, sqrt3)^{p_c}``; the control dual coordinate is
``u0(z) = c + k L(z)`` with ``L = b'z_{1:4}``, ``b = (0.5, -0.4, 0.3, -0.2)``, which
fixes the control weight ``a0(z) = -(g')^{-1}(u0(z))`` on the negative branch. The
treated representer is the constant ``1/pi1`` and ``pi1`` solves
``E[a0 / (1 + pi1 a0)] = 1``; ``e(z) = pi1 a0 / (1 + pi1 a0)`` and
``D | Z ~ Bern(e(Z))``. No outcome is generated: ``m^ATT(W, f) = D (f(1, Z) - f(0, Z))
/ pi1`` with the true ``pi1``.

Dictionary ``phi(x) = (d, 1 - d, (1 - d) z_1, ..., (1 - d) z_{p_c})``
(``p = p_c + 2``, ``B_phi = sqrt3``), offset ``u_ref = g'(1/pi1)`` (treated) and
``g'(-1/(1 - pi1))`` (control), penalty on every coefficient (deviation from the
offset), l1 ball of radius ``R1``. The population minimizer is the planted
``beta*`` (treated 0; control ``(c - u_ref_c, k b, 0, ...)``, ``s = 5``).

Penalties: SN (self-normalised, main), TH (the theorem's), LTH (the two-step rule
of E-15). Every fit is the projected FISTA of genriesz with ``tol = 1e-6 lam``.
"""

from __future__ import annotations

import collections
import itertools
import json
import math

import numpy as np
from scipy import optimize, stats

from . import metrics
from .seeds import CONFIRMATORY_ENTROPY, Seeds

EXP = 14
SQRT3 = math.sqrt(3.0)
B_VEC = np.array([0.5, -0.4, 0.3, -0.2])
L_MAX = SQRT3 * float(np.abs(B_VEC).sum())  # 2.42487...
GENERATORS = ("SQ", "UKL", "BP", "BKL")
P_C_VALUES = (50, 200, 800)
N_VALUES = (4000, 8000, 16000, 32000, 64000)
CELLS = [list(c) for c in itertools.product(GENERATORS, P_C_VALUES, N_VALUES)]  # fixed order
R = 200
PENALTIES = ("SN", "TH", "LTH")
S_SUPPORT = 5
B_PHI = SQRT3
EPS_THEOREM = 0.05  # epsilon of the theorem's lambda (not fixed by the design document)
SN_FACTOR = 1.1
SN_ALPHA = 0.05
LTH_ALPHA = 0.05  # E-15: sqrt(2 log(2p / 0.05) / n)
FISTA_REL_TOL = 1e-6  # §1.3 C, l1: rho <= 1e-6 lam
N_EVAL = 10**6
N_REEVAL = 10**7
REEVAL_BLOCK = 998
N_CHECKS = len(PENALTIES) * len(GENERATORS) * len(P_C_VALUES) * len(N_VALUES) * R  # 36000
DELTA_EVAL = 0.001 / N_CHECKS
SLOPE_N = (16000, 32000, 64000)
SLOPE_BAND = (-0.65, -0.35)
RATIO_P = (800, 50)
RATIO_N = 64000
RATIO_LIMIT = 2.0
FAILURE_LIMIT = 0.02
HEALTH_LEVEL = 0.01
CERT_GRADIENT = 1e-12  # §1.9 (i)
CERT_DIFF = 1e-10  # §1.9 (i)
CERT_MARGIN = 1e-4  # §1.9 (iii)
QUAD_FROZEN = 40  # the design probe's 40^4 grid
QUAD_CERT = (48, 64)

#: Control dual coordinate u0 = c + k L (the planted design).
PLANT = {
    "SQ": {"c": -6.0, "k": -2.0},
    "UKL": {"c": -0.25, "k": -1.0},
    "BP": {"c": -1.8, "k": -1.5},
    "BKL": {"c": 1.2, "k": 0.35},
}
#: Generator shift C (SQ, UKL, BP: 0; BKL: 1.2) and the BP power.
SHIFT = {"SQ": 0.0, "UKL": 0.0, "BP": 0.0, "BKL": 1.2}
BP_OMEGA = 0.5
#: R_max of the bounded-domain generators (design document, 6 decimals).
R_MAX = {"BP": 2.675731, "BKL": 0.671260}

#: Frozen design values (design-document table). ``a_range`` and the constants are
#: rounded outward to 9 significant digits; the rest is displayed as computed.
FROZEN = {
    "SQ": {"pi1": 0.6522157218, "a_range": (0.575128869, 5.42487114), "E_alpha2": 4.5997051,
           "uref_c": -5.75069123, "uref_t": 3.06647009, "l1_beta": 3.049309, "R1": 3.8116},
    "UKL": {"pi1": 0.2966298312, "a_range": (0.113622795, 14.5104798), "E_alpha2": 5.6438489,
            "uref_c": -0.35187197, "uref_t": 1.21527028, "l1_beta": 1.501872, "R1": 1.8773},
    "BP": {"pi1": 0.5809783901, "a_range": (0.150206191, 7.90979381), "E_alpha2": 4.6387268,
           "uref_c": -1.63450168, "uref_t": 0.93587471, "l1_beta": 2.265498, "R1": 2.4706},
    "BKL": {"pi1": 0.5636416361, "a_range": (1.55514267, 6.90197787), "E_alpha2": 4.1498574,
            "uref_c": 1.16265616, "uref_t": -1.64478500, "l1_beta": 0.527344, "R1": 0.5993},
}  # fmt: skip
#: Displayed precision of each frozen value: a recomputed value must agree to half a
#: unit in the last displayed place (the design log keeps only these digits).
FROZEN_DIGITS = {"pi1": 10, "E_alpha2": 7, "uref_c": 8, "uref_t": 8, "l1_beta": 6}
#: Curvature constants, rounded outward (mu and lambda_min(Sigma) down; the rest up).
CONST = {
    "SQ": {"mu": 2.0, "C_g": 2.0, "kappa_g": 1.0, "A_alpha": 6.17628805,
           "B_m": 3.06647009, "lam_min_Sigma": 0.296299287},
    "UKL": {"mu": 0.0272295921, "C_g": 18.1688214, "kappa_g": 667.245446, "A_alpha": 36.7247514,
            "B_m": 6.74241021, "lam_min_Sigma": 0.296629831},
    "BP": {"mu": 0.504840500, "C_g": 12.6654617, "kappa_g": 25.0880460, "A_alpha": 8.82824022,
           "B_m": 3.44246884, "lam_min_Sigma": 0.318374976},
    "BKL": {"mu": 0.00648115843, "C_g": 3.14075251, "kappa_g": 484.597397, "A_alpha": 19.2806684,
            "B_m": 3.54835391, "lam_min_Sigma": 0.387205686},
}  # fmt: skip


def gr():
    import genriesz

    return genriesz


def sign(A):
    """Treated rows (``d = 1``) on the positive branch, control rows on the negative one."""
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


sign.vectorized = True


def generator(name):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL": lambda: g.UKLGenerator(C=0.0, branch_fn=sign),
        "BP": lambda: g.BPGenerator(C=0.0, omega=BP_OMEGA, branch_fn=sign),
        "BKL": lambda: g.BKLGenerator(C=SHIFT["BKL"], branch_fn=sign),
    }[name]()


def _rows(d, n):
    return np.full((n, 1), float(d))


def control_weight(name, u, gen=None):
    """``a = -(g')^{-1}(u)`` on the control (negative) branch."""
    gen = generator(name) if gen is None else gen
    u = np.asarray(u, dtype=float)
    return -np.asarray(gen.inv_grad(_rows(0, u.size), u))


def treated_alpha(name, u, gen=None):
    gen = generator(name) if gen is None else gen
    u = np.atleast_1d(np.asarray(u, dtype=float))
    return np.asarray(gen.inv_grad(_rows(1, u.size), u))


def dual(name, d, alpha, gen=None):
    gen = generator(name) if gen is None else gen
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    return np.asarray(gen.grad(_rows(d, alpha.size), alpha))


def planted_u(name, Z4):
    """Control dual coordinate ``c + k L(z)`` from the first four covariates."""
    p = PLANT[name]
    return p["c"] + p["k"] * (np.asarray(Z4)[:, :4] @ B_VEC)


def solve_pi1(a0, w):
    """Root of ``E[a0 / (1 + pi1 a0)] = 1`` (Brent, xtol 1e-15)."""
    return float(
        optimize.brentq(lambda q: float(np.sum(w * a0 / (1.0 + q * a0))) - 1.0, 1e-9, 1 - 1e-12,
                        xtol=1e-15)
    )  # fmt: skip


def p_total(p_c):
    return p_c + 2


# ---------------------------------------------------------------- Stage 0


def _grid4(points):
    x, w = np.polynomial.legendre.leggauss(points)
    z = SQRT3 * x
    wz = w / 2.0
    G = np.meshgrid(z, z, z, z, indexing="ij")
    W = np.prod(np.meshgrid(wz, wz, wz, wz, indexing="ij"), axis=0).ravel()
    Z4 = np.column_stack([g.ravel() for g in G])
    return Z4, W


def _half_unit(digits):
    return 0.5 * 10.0 ** (-digits)


def design_values(name, points=QUAD_FROZEN):
    """The design-document quantities recomputed on a ``points^4`` grid (unrounded)."""
    gen = generator(name)
    Z4, W = _grid4(points)
    u0 = planted_u(name, Z4)
    a0 = control_weight(name, u0, gen)
    pi1 = solve_pi1(a0, W)
    e = pi1 * a0 / (1.0 + pi1 * a0)
    E_alpha2 = 1.0 / pi1 + float(np.sum(W * (1.0 - e) * a0**2))
    k = PLANT[name]["k"]
    u_ext = PLANT[name]["c"] + np.array([-1.0, 1.0]) * abs(k) * L_MAX
    a_ext = control_weight(name, u_ext, gen)
    uref_c = float(dual(name, 0, -1.0 / (1.0 - pi1), gen)[0])
    uref_t = float(dual(name, 1, 1.0 / pi1, gen)[0])
    l1 = abs(PLANT[name]["c"] - uref_c) + abs(k) * float(np.abs(B_VEC).sum())
    if name in R_MAX:
        R1 = math.floor((l1 + 0.5 * (R_MAX[name] - l1)) * 1e4) / 1e4
    else:
        R1 = math.floor(1.25 * l1 * 1e4) / 1e4
    # induced range of the ball (control: u_ref_c +- B_phi R1; treated: u_ref_t +- R1)
    cu = np.array([uref_c - B_PHI * R1, uref_c + B_PHI * R1])
    tu = np.array([uref_t - R1, uref_t + R1])
    ca = control_weight(name, cu, gen)
    ta = np.abs(treated_alpha(name, tu, gen))
    vals = np.concatenate([ca, ta])
    if name == "SQ":
        mu = C_g = 2.0
    else:
        g2 = np.concatenate([
            np.asarray(gen.grad2(_rows(0, 2), -ca)), np.asarray(gen.grad2(_rows(1, 2), ta))
        ])  # fmt: skip
        mu, C_g = float(g2.min()), float(g2.max())
    F = np.column_stack([np.ones(len(Z4)), Z4])
    M5 = (F * (W * (1.0 - e))[:, None]).T @ F
    lam_min = min(pi1, float(np.linalg.eigvalsh(M5).min()), 1.0 - pi1)
    return {
        "pi1": pi1, "a_range": (float(a_ext.min()), float(a_ext.max())), "E_alpha2": E_alpha2,
        "uref_c": uref_c, "uref_t": uref_t, "l1_beta": l1, "R1": R1,
        "ball_u_control": cu.tolist(), "ball_u_treated": tu.tolist(),
        "mu": mu, "C_g": C_g, "kappa_g": C_g / mu, "A_alpha": float(vals.max()),
        "B_m": 2.0 / pi1, "lam_min_Sigma": lam_min, "M5": M5.tolist(),
    }  # fmt: skip


def _domain_margin(name, d, u):
    """Distance of dual coordinates ``u`` (branch ``d``) to the boundary of the range of g'."""
    u = np.asarray(u, dtype=float)
    if name in ("SQ", "UKL"):
        return np.full(u.shape, np.inf)
    if name == "BP":
        k = 1.0 + 1.0 / BP_OMEGA
        s = 1.0 if d == 1 else -1.0
        return k + s * u  # w = 1 + s u / k > 0  <=>  k + s u > 0
    s = 1.0 if d == 1 else -1.0  # BKL: s u < 0
    return -s * u


def frozen_checks(name, vals):
    """Recomputed values against the frozen table (displayed precision) and the outward
    constants (each must bound the recomputed value on the safe side)."""
    fz, cs = FROZEN[name], CONST[name]
    checks = {}
    for key, digits in FROZEN_DIGITS.items():
        checks[f"{key}_agrees"] = bool(abs(vals[key] - fz[key]) <= _half_unit(digits) * (1 + 1e-9))
    lo, hi = fz["a_range"]
    checks["a_range_outward"] = bool(lo <= vals["a_range"][0] and hi >= vals["a_range"][1])
    checks["a_range_tight"] = bool(
        abs(lo - vals["a_range"][0]) <= 1e-8 * max(1.0, abs(lo))
        and abs(hi - vals["a_range"][1]) <= 1e-8 * max(1.0, abs(hi))
    )
    checks["R1_rule"] = bool(vals["R1"] == fz["R1"])
    checks["mu_outward"] = bool(cs["mu"] <= vals["mu"])
    checks["C_g_outward"] = bool(cs["C_g"] >= vals["C_g"])
    checks["kappa_g_outward"] = bool(cs["kappa_g"] >= vals["kappa_g"])
    checks["A_alpha_outward"] = bool(cs["A_alpha"] >= vals["A_alpha"])
    checks["B_m_outward"] = bool(cs["B_m"] >= vals["B_m"])
    checks["lam_min_outward"] = bool(cs["lam_min_Sigma"] <= vals["lam_min_Sigma"])
    for key in ("mu", "C_g", "kappa_g", "A_alpha", "B_m", "lam_min_Sigma"):
        checks[f"{key}_tight"] = bool(abs(cs[key] - vals[key]) <= 1e-8 * max(1.0, abs(vals[key])))
    return checks


def certificate(name):
    """§1.9 (i)-(iv) for the planted ``beta*`` (an interior population minimizer)."""
    gen = generator(name)
    res = {}
    pis = {}
    pi_stored = design_values(name, points=QUAD_FROZEN)["pi1"]  # used by the replications
    for pts in QUAD_CERT:
        Z4, W = _grid4(pts)
        u0 = planted_u(name, Z4)
        a0 = control_weight(name, u0, gen)
        pi1 = solve_pi1(a0, W)
        pis[pts] = pi1
        e = pi1 * a0 / (1.0 + pi1 * a0)
        # gradient E[alpha phi_j - m(W, phi_j)] at beta*: treated column, control (1, z1..z4)
        g_t = float(np.sum(W * e) * (1.0 / pi1) - np.sum(W * e) / pi1)
        F = np.column_stack([np.ones(len(Z4)), Z4])
        g_c = (F * (W * ((1.0 - e) * (-a0) + e / pi1))[:, None]).sum(axis=0)
        res[f"gradient_max_{pts}"] = float(max(abs(g_t), np.abs(g_c).max()))
        res[f"pi1_consistency_{pts}"] = float(abs(np.sum(W * e) - pi1))
        # the stored (40-point) design on this grid: E[e] = pi1 is the only condition, since
        # (1 - e) a0 = e / pi1 makes every population imbalance vanish for any pi1
        e_s = pi_stored * a0 / (1.0 + pi_stored * a0)
        res[f"stored_design_consistency_{pts}"] = float(abs(np.sum(W * e_s) - pi_stored))
    pis[QUAD_FROZEN] = pi_stored
    res["pi1_diff"] = float(abs(pis[48] - pis[64]))
    res["pi1_diff_40"] = float(max(abs(pis[40] - pis[48]), abs(pis[40] - pis[64])))
    bstar = {}
    for pts, q in pis.items():
        uref_c = float(dual(name, 0, -1.0 / (1.0 - q), gen)[0])
        bstar[pts] = np.r_[0.0, PLANT[name]["c"] - uref_c, PLANT[name]["k"] * B_VEC]
    res["beta_star_diff"] = float(max(np.abs(bstar[a] - bstar[b]).max()
                                      for a, b in ((48, 64), (40, 48), (40, 64))))  # fmt: skip
    pi1 = pis[64]
    vals = design_values(name, points=64)
    c, k = PLANT[name]["c"], PLANT[name]["k"]
    u_ctrl = c + np.array([-1.0, 1.0]) * abs(k) * L_MAX
    res["margin_beta_star"] = float(
        min(_domain_margin(name, 0, u_ctrl).min(), _domain_margin(name, 1, [vals["uref_t"]]).min())
    )
    res["margin_ball"] = float(
        min(_domain_margin(name, 0, vals["ball_u_control"]).min(),
            _domain_margin(name, 1, vals["ball_u_treated"]).min())
    )  # fmt: skip
    for key in ("margin_beta_star", "margin_ball"):  # SQ and UKL: g' has range R (no edge)
        res[f"{key}_kind"] = "unbounded" if math.isinf(res[key]) else "finite"
    finite = {k: res[k] for k in ("margin_beta_star", "margin_ball")}
    for key in finite:
        if math.isinf(res[key]):
            res[key] = None
    res["beta_star_l1"] = vals["l1_beta"]
    res["hessian_lam_min_Sigma"] = vals["lam_min_Sigma"]
    checks = {
        "i_gradient": all(res[f"gradient_max_{p}"] <= CERT_GRADIENT for p in QUAD_CERT),
        "i_difference": res["beta_star_diff"] <= CERT_DIFF and res["pi1_diff_40"] <= CERT_DIFF,
        "i_stored_design": all(res[f"stored_design_consistency_{p}"] <= CERT_GRADIENT
                               for p in QUAD_CERT),
        "ii_hessian_pd": vals["lam_min_Sigma"] > 0.0 and vals["mu"] > 0.0,
        "iii_interior": math.isinf(finite["margin_beta_star"])
        or finite["margin_beta_star"] >= CERT_MARGIN,
        "iii_ball_in_domain": math.isinf(finite["margin_ball"]) or finite["margin_ball"] > 0.0,
        "iii_beta_in_ball": vals["l1_beta"] < FROZEN[name]["R1"],
        "iv_convex": True,  # compatible pair: the objective is convex in beta
    }
    res["pi1"] = pi1
    return res, checks


def bkl_zero_offset_infeasible():
    """§1.3 B / PR4a 2(c-1): with ``u_ref = 0`` the BKL start ``beta = 0`` is infeasible."""
    gen = generator("BKL")
    X = np.array([[1.0], [0.0]])
    return bool(not np.all(gen.link_domain(X, np.zeros(2))))


def stage0():
    """Stage 0 rows, one per generator. Every value the replications need is here."""
    rows = []
    for name in GENERATORS:
        vals = design_values(name, points=QUAD_FROZEN)
        checks = frozen_checks(name, vals)
        cert, cert_checks = certificate(name)
        cs = CONST[name]
        kappa = math.sqrt(cs["lam_min_Sigma"])  # kappa^2 = lambda_min(Sigma) (rounded down)
        max_abs_alpha0 = max(1.0 / FROZEN[name]["pi1"], FROZEN[name]["a_range"][1])
        rows.append({
            "generator": name, "design": vals, "frozen_checks": checks,
            "certificate": cert, "certificate_checks": cert_checks,
            "certified": bool(all(cert_checks.values())),
            "frozen_ok": bool(all(checks.values())),
            "pi1": vals["pi1"], "uref_c": vals["uref_c"], "uref_t": vals["uref_t"],
            "R1": FROZEN[name]["R1"], "kappa_g": cs["kappa_g"], "kappa": kappa,
            "A_alpha": cs["A_alpha"], "B_m": cs["B_m"], "B_eval": cs["A_alpha"] + max_abs_alpha0,
            "M5": vals["M5"],
        })  # fmt: skip
    return {"generators": rows, "bkl_zero_offset_infeasible": bkl_zero_offset_infeasible()}


# ---------------------------------------------------------------- one replication


class Design:
    """What a replication needs from Stage 0 for one generator (picklable)."""

    def __init__(self, row):
        self.name = row["generator"]
        self.pi1 = float(row["pi1"])
        self.uref_c = float(row["uref_c"])
        self.uref_t = float(row["uref_t"])
        self.R1 = float(row["R1"])
        self.kappa_g = float(row["kappa_g"])
        self.kappa = float(row["kappa"])
        self.A_alpha = float(row["A_alpha"])
        self.B_m = float(row["B_m"])
        self.B_eval = float(row["B_eval"])
        self.M5 = np.asarray(row["M5"], dtype=float)
        self.lam_min_Sigma = self.kappa**2

    def beta_star(self, p_c):
        c, k = PLANT[self.name]["c"], PLANT[self.name]["k"]
        b = np.zeros(p_total(p_c))
        b[1] = c - self.uref_c
        b[2:6] = k * B_VEC
        return b

    def sigma(self, p_c):
        """Population ``E[phi phi']``: ``diag(pi1, M5, (1 - pi1) I)``."""
        p = p_total(p_c)
        S = np.zeros((p, p))
        S[0, 0] = self.pi1
        S[1:6, 1:6] = self.M5
        idx = np.arange(6, p)
        S[idx, idx] = 1.0 - self.pi1
        return S


def designs_from_stage0(stage0_json):
    return {r["generator"]: Design(r) for r in stage0_json["generators"]}


def draw(rng, n, p_c, des):
    """``Z`` (``n x p_c``), then ``D`` (uniforms against ``e(Z)``), from stream 0."""
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, p_c))
    u = rng.uniform(size=n)
    a0 = control_weight(des.name, planted_u(des.name, Z))
    e = des.pi1 * a0 / (1.0 + des.pi1 * a0)
    D = (u < e).astype(float)
    return D, Z, a0


def design_matrix(D, Z):
    return np.column_stack([D, 1.0 - D, (1.0 - D)[:, None] * Z])


def target(D, Z, pi1):
    """``b_j = P_n m(W, phi_j)``: ``(D, -D, -D z_j) / pi1``."""
    n = len(D)
    return np.concatenate([[D.sum(), -D.sum()], -(D @ Z)]) / (pi1 * n)


def xi_matrix(alpha, Phi, D, Z, pi1):
    """``xi_ij = alpha_i phi_j(X_i) - m(W_i, phi_j)``."""
    M = np.column_stack([D, -D, -D[:, None] * Z]) / pi1
    return alpha[:, None] * Phi - M


def s_hat(xi):
    return float(np.sqrt(np.max(np.mean(xi**2, axis=0))))


def lam_sn(xi_star, n, p):
    q = stats.norm.ppf(1.0 - SN_ALPHA / (2.0 * p))
    return 2.0 * SN_FACTOR * q * s_hat(xi_star) / math.sqrt(n)


def lam_theorem(des, n, p):
    B_xi = B_PHI * (des.A_alpha + des.B_m)
    return 2.0 * B_xi * math.sqrt(2.0 * math.log(2.0 * p / EPS_THEOREM) / n)


def lth_factor(n, p):
    return 2.0 * math.sqrt(2.0 * math.log(2.0 * p / LTH_ALPHA) / n)


def bound(des, lam):
    """Theorem: ``6 kappa_g sqrt(s) lam / kappa`` (outward ``kappa_g``, rounded-down ``kappa``)."""
    return 6.0 * des.kappa_g * math.sqrt(S_SUPPORT) * lam / des.kappa


def fit(des, gen, D, Phi, b, lam, beta0=None):
    g = gr()
    offset = np.where(D == 1.0, des.uref_t, des.uref_c)
    prob = g.solvers.DualProblem(generator=gen, X=D[:, None], Phi=Phi, offset=offset, target=b)
    beta0 = np.zeros(Phi.shape[1]) if beta0 is None else beta0
    return g.solvers.fista_solve(prob, beta0=beta0, lam=lam, radius=des.R1,
                                 tol=FISTA_REL_TOL * lam)  # fmt: skip


def _eval_columns(entropy, rep, block, cols, N):
    """Column ``j`` of ``Z^eval`` from its own child stream of ``(14, block, rep, 6)``."""
    out = {}
    for j in cols:
        ss = np.random.SeedSequence(entropy=entropy, spawn_key=(EXP, block, rep, 6, int(j)))
        out[int(j)] = np.random.default_rng(ss).uniform(-SQRT3, SQRT3, size=N)
    return out


def evaluate(des, gen, beta, entropy, rep, *, block=999, N=N_EVAL):
    """``err^2 = E_Z[(1 - e)(a_hat - a0)^2 + e(alpha_t - 1/pi1)^2]`` on ``Z^eval``, with the
    one-sided empirical-Bernstein lower bound (Maurer and Pontil 2009, Thm 4)."""
    sup = [int(j) - 2 for j in np.flatnonzero(beta[2:] != 0.0) + 2]
    cols = sorted(set(range(4)) | set(sup))
    Zc = _eval_columns(entropy, rep, block, cols, N)
    Z4 = np.column_stack([Zc[j] for j in range(4)])
    a0 = control_weight(des.name, planted_u(des.name, Z4), gen)
    e = des.pi1 * a0 / (1.0 + des.pi1 * a0)
    u = np.full(N, des.uref_c + beta[1])
    for j in sup:
        u += beta[2 + j] * Zc[j]
    with np.errstate(all="ignore"):
        a_hat = control_weight(des.name, u, gen)
        at = float(treated_alpha(des.name, des.uref_t + beta[0], gen)[0])
        f = (1.0 - e) * (a_hat - a0) ** 2 + e * (at - 1.0 / des.pi1) ** 2
    finite = bool(np.all(np.isfinite(a_hat)) and math.isfinite(at) and np.all(np.isfinite(f)))
    if not finite:
        return {"eval_status": "nonfinite"}
    max_alpha = float(max(np.abs(a_hat).max(), abs(at)))
    if max_alpha > des.A_alpha or f.max() > des.B_eval**2:
        # beta_hat is in the l1 ball, whose induced range is certified at Stage 0: a value
        # outside it is an implementation error, not a statistical outcome (stop).
        raise RuntimeError(
            f"evaluation outside the certified range: max |alpha| = {max_alpha} "
            f"(A_alpha = {des.A_alpha}), max integrand = {float(f.max())} (B^2 = {des.B_eval**2})"
        )
    mean = float(f.mean())
    var = float(f.var(ddof=1))
    lg = math.log(2.0 / DELTA_EVAL)
    lower = mean - math.sqrt(2.0 * var * lg / N) - 7.0 * des.B_eval**2 * lg / (3.0 * (N - 1))
    if not (math.isfinite(mean) and math.isfinite(var) and math.isfinite(lower)):
        return {"eval_status": "nonfinite"}
    return {"eval_status": "ok", "err2": mean, "err2_var": var, "err2_lower": lower,
            "eval_in_range": True, "max_abs_alpha_eval": max_alpha}  # fmt: skip


def _fit_record(des, gen, res, lam, events, entropy, rep, xi_star_inf):
    rec = {
        "status": res.status, "lam": lam, "n_iter": int(res.n_iter),
        "support": int(np.sum(res.beta != 0.0)), "ball_active": bool(res.l1_ball_active),
        "prox_residual": float(res.prox_residual), "kkt_residual": float(res.kkt_residual),
        "E1": bool(lam >= 2.0 * xi_star_inf), **events,
        "bound": bound(des, lam),
    }  # fmt: skip
    rec["events"] = bool(rec["E1"] and events["E2"] and events["E3"])
    if res.status == "ok":
        nz = np.flatnonzero(res.beta)
        rec["beta_hat"] = json.dumps({int(j): float(res.beta[j]) for j in nz})
        ev = evaluate(des, gen, res.beta, entropy, rep)
        if ev.pop("eval_status") != "ok":
            rec["status"] = "nonfinite"  # §1.3 B: a non-finite prediction fails the fit
            return rec
        rec.update(ev)
        rec["l1_beta_hat"] = float(np.abs(res.beta).sum())
        rec["bnd_violation"] = bool(rec["events"] and rec["err2_lower"] > rec["bound"] ** 2)
    return rec


def replicate(task):
    """One replication of one cell: the three penalties (and, for BKL, the ``u_ref = 0``
    start). ``task = (cell, rep, entropy, design_row)``. Warnings are recorded (§1.1)."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _replicate(task)
    out["n_warnings"] = len(caught)
    out["warnings"] = json.dumps(sorted({f"{w.category.__name__}: {w.message}" for w in caught}))
    return out


def _replicate(task):
    cell_index, rep, entropy, row = task
    name, p_c, n = CELLS[cell_index]
    des = Design(row)
    gen = generator(name)
    p = p_total(p_c)
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, a0 = draw(seeds.data(cell_index, rep), n, p_c, des)
    Phi = design_matrix(D, Z)
    b = target(D, Z, des.pi1)
    alpha_star = np.where(D == 1.0, 1.0 / des.pi1, -a0)
    xi_star = xi_matrix(alpha_star, Phi, D, Z, des.pi1)
    grad_star = xi_star.mean(axis=0)
    xi_star_inf = float(np.abs(grad_star).max())
    if not (math.isfinite(xi_star_inf) and np.all(np.isfinite(Phi))):
        raise FloatingPointError("non-finite design or gradient at beta*")
    Sig_hat = Phi.T @ Phi / n
    ev_hat = float(np.linalg.eigvalsh(Sig_hat).min())
    ev_e3 = float(np.linalg.eigvalsh(Sig_hat - 0.5 * des.sigma(p_c)).min())
    if not (math.isfinite(ev_hat) and math.isfinite(ev_e3)):
        raise FloatingPointError("non-finite eigenvalue in the (E2)/(E3) checks")
    events = {"lam_min_Sigma_hat": ev_hat, "lam_min_E3": ev_e3,
              "E2": bool(ev_hat >= des.lam_min_Sigma / 2.0), "E3": bool(ev_e3 >= 0.0)}  # fmt: skip
    out = {"n_treated": int(D.sum()), "grad_star_inf": xi_star_inf}
    lam = lam_sn(xi_star, n, p)
    out["SN"] = _fit_record(des, gen, fit(des, gen, D, Phi, b, lam), lam, events, entropy, rep,
                            xi_star_inf)
    lam = lam_theorem(des, n, p)
    out["TH"] = _fit_record(des, gen, fit(des, gen, D, Phi, b, lam), lam, events, entropy, rep,
                            xi_star_inf)
    # LTH (E-15): lambda_1 from S_hat at the offset representer, then from the first fit.
    alpha_ref = np.where(D == 1.0, 1.0 / des.pi1, -1.0 / (1.0 - des.pi1))
    lam1 = s_hat(xi_matrix(alpha_ref, Phi, D, Z, des.pi1)) * lth_factor(n, p)
    first = fit(des, gen, D, Phi, b, lam1)
    if first.status == "ok":
        lam2 = s_hat(xi_matrix(first.alpha, Phi, D, Z, des.pi1)) * lth_factor(n, p)
        rec = _fit_record(des, gen, fit(des, gen, D, Phi, b, lam2, beta0=first.beta), lam2,
                          events, entropy, rep, xi_star_inf)
    else:
        # the failed first step is the LTH fit's status (a solver status, §1.3 B)
        rec = {"status": first.status, "lam": float("nan"), "E1": False, "events": False,
               "lth_first_step_failed": True, **events}  # fmt: skip
    rec["lam1"] = lam1
    out["LTH"] = rec
    if name == "BKL":
        g = gr()
        prob = g.solvers.DualProblem(generator=gen, X=D[:, None], Phi=Phi,
                                     offset=np.zeros(n), target=b)  # fmt: skip
        z = g.solvers.fista_solve(prob, beta0=np.zeros(p), lam=out["SN"]["lam"], radius=des.R1,
                                  tol=FISTA_REL_TOL * out["SN"]["lam"])  # fmt: skip
        out["BKL_zero_offset_status"] = z.status
    return out


def tasks_for(entropy, reps, stage0_json, cells=None):
    rows = {r["generator"]: r for r in stage0_json["generators"]}
    cells = range(len(CELLS)) if cells is None else cells
    return [(ci, r, entropy, rows[CELLS[ci][0]]) for ci in cells for r in range(reps)]


# ---------------------------------------------------------------- aggregation

FIT_NUMERIC = ("lam", "lam1", "n_iter", "support", "prox_residual", "kkt_residual", "bound",
               "err2", "err2_var", "err2_lower", "l1_beta_hat", "max_abs_alpha_eval",
               "lam_min_Sigma_hat", "lam_min_E3")  # fmt: skip
FIT_BOOL = ("E1", "E2", "E3", "events", "ball_active", "bnd_violation", "eval_in_range")


def raw_frame(tasks, results):
    """One row per replication x penalty (and the BKL zero-offset start); failures are rows."""
    import pandas as pd

    rows = []
    for (ci, rep, _, _), res in zip(tasks, results, strict=True):
        name, p_c, n = CELLS[ci]
        base = {"cell": ci, "generator": name, "p_c": p_c, "n": n, "rep": rep,
                "n_treated": res["n_treated"], "grad_star_inf": res["grad_star_inf"],
                "n_warnings": res["n_warnings"], "warnings": res["warnings"]}  # fmt: skip
        for pen in PENALTIES:
            r = res[pen]
            rows.append({
                **base, "penalty": pen, "status": r["status"],
                **{k: float(r.get(k, np.nan)) for k in FIT_NUMERIC},
                **{k: (bool(r[k]) if k in r else None) for k in FIT_BOOL},
                "beta_hat": r.get("beta_hat", ""),
            })  # fmt: skip
        if "BKL_zero_offset_status" in res:
            rows.append({**base, "penalty": "BKL_u0", "status": res["BKL_zero_offset_status"]})
    return pd.DataFrame(rows)


def _flag(series, default=False):
    """Boolean column with ``None`` (no fit) mapped to ``default``."""
    return series.map(lambda v: default if v is None or v != v else bool(v)).astype(bool)


def summarise(raw):
    """Per cell x penalty: failures, the events (E1)-(E3), err, and the H14-Bnd check."""
    import pandas as pd

    rows = []
    for ci, (name, p_c, n) in enumerate(CELLS):
        for pen in PENALTIES:
            g = raw[(raw["cell"] == ci) & (raw["penalty"] == pen)].sort_values("rep")
            if g.empty:
                continue
            ok = (g["status"] == "ok").to_numpy()
            err = np.sqrt(g.loc[ok, "err2"].to_numpy())
            ev = _flag(g["events"]).to_numpy()
            row = {
                "cell": ci, "generator": name, "p_c": p_c, "n": n, "penalty": pen,
                **metrics.failure_rate(ok),
                "status_counts": json.dumps(dict(sorted(collections.Counter(g["status"]).items()))),
                "E1_rate": float(_flag(g["E1"]).mean()),
                "E2_rate": float(_flag(g["E2"]).mean()),
                "E3_rate": float(_flag(g["E3"]).mean()),
                "events_rate": float(ev.mean()),
                "lam_median": float(g["lam"].median()),
                "bound_median": float(g["bound"].median()),
                "err_mean": float(err.mean()) if err.size else float("nan"),
                "err_median": float(np.median(err)) if err.size else float("nan"),
                "err_max": float(err.max()) if err.size else float("nan"),
                "support_median": (
                    float(g.loc[ok, "support"].median()) if ok.any() else float("nan")
                ),
                "ball_active_rate": float(_flag(g.loc[ok, "ball_active"]).mean())
                if ok.any() else float("nan"),
                "bnd_checked": int((ok & ev).sum()),
                "bnd_violations": int(_flag(g["bnd_violation"]).sum()),
                "max_err_over_bound": float(
                    (np.sqrt(np.maximum(g.loc[ok & ev, "err2_lower"], 0.0))
                     / g.loc[ok & ev, "bound"]).max()
                ) if (ok & ev).any() else float("nan"),
                "eval_out_of_range": int((ok & ~_flag(g["eval_in_range"], True).to_numpy()).sum()),
            }  # fmt: skip
            rows.append(row)
    return pd.DataFrame(rows)


def h14_bnd(summ):
    """Negative iff any replication with (E1)-(E3) has its err^2 lower bound above the bound."""
    v = summ[summ["bnd_violations"] > 0]
    return {
        "verdict": "negative" if len(v) else "no violation",
        "violations": int(summ["bnd_violations"].sum()),
        "checked": int(summ["bnd_checked"].sum()),
        "violating_cells": [
            f"{r.generator}|p_c={r.p_c}|n={r.n}|{r.penalty}" for r in v.itertuples()
        ],
    }  # fmt: skip


def rate_predictions(raw, seeds):
    """H14-Rate with family 4 (stream 4): per cell of the SN penalty at n in SLOPE_N (cells in
    the fixed order), one call that resamples the successful replications; the slopes of
    log mean err on log n per (generator, p_c) and the ratio p_c = 800 / 50 at n = 64000."""
    from .bootstrap import FamilyBootstrap, percentile_interval

    fb = FamilyBootstrap(seeds, 4)
    boot, point = {}, {}
    for ci, (name, p_c, n) in enumerate(CELLS):
        if n not in SLOPE_N:
            continue
        g = raw[(raw["cell"] == ci) & (raw["penalty"] == "SN") & (raw["status"] == "ok")]
        err = np.sqrt(g.sort_values("rep")["err2"].to_numpy())
        point[(name, p_c, n)] = float(err.mean()) if err.size else float("nan")
        if err.size >= 2:
            boot[(name, p_c, n)] = fb.replicate(lambda idx, x=err: x[idx].mean(axis=1), err.size)
    logn = np.log(np.array(SLOPE_N, dtype=float))
    xc = logn - logn.mean()

    def slope(ys):
        return (np.log(ys) - np.log(ys).mean(axis=0)).T @ xc / (xc @ xc)

    slopes, ratios = [], []
    for name in GENERATORS:
        for p_c in P_C_VALUES:
            keys = [(name, p_c, n) for n in SLOPE_N]
            if all(k in boot for k in keys):
                ys = np.vstack([boot[k] for k in keys])
                lo, hi = percentile_interval(slope(ys), 4)
                est = float(slope(np.array([[point[k]] for k in keys]))[0])
                ok = bool(hi >= SLOPE_BAND[0] and lo <= SLOPE_BAND[1])
                slopes.append({"generator": name, "p_c": p_c, "slope": est, "lo": lo, "hi": hi,
                               "intersects": ok})  # fmt: skip
            else:
                slopes.append({"generator": name, "p_c": p_c, "slope": None, "lo": None,
                               "hi": None, "intersects": None})  # fmt: skip
        ka, kb = (name, RATIO_P[0], RATIO_N), (name, RATIO_P[1], RATIO_N)
        if ka in boot and kb in boot:
            lo, hi = percentile_interval(boot[ka] / boot[kb], 4)
            ratios.append({"generator": name, "ratio": point[ka] / point[kb], "lo": lo, "hi": hi,
                           "below_limit": bool(lo < RATIO_LIMIT)})  # fmt: skip
        else:
            ratios.append({"generator": name, "ratio": None, "lo": None, "hi": None,
                           "below_limit": None})  # fmt: skip
    failed = [f"slope|{s['generator']}|p_c={s['p_c']}" for s in slopes if s["intersects"] is False]
    failed += [f"ratio|{r['generator']}" for r in ratios if r["below_limit"] is False]
    incomplete = [
        f"slope|{s['generator']}|p_c={s['p_c']}" for s in slopes if s["intersects"] is None
    ]
    incomplete += [f"ratio|{r['generator']}" for r in ratios if r["below_limit"] is None]
    verdict = "negative" if failed else ("incomplete" if incomplete else "no departure detected")
    return {"slopes": slopes, "ratios": ratios, "verdict": verdict, "failed": failed,
            "incomplete": incomplete}


def health(raw):
    """Failure rate <= 2% per generator x penalty (pooled over cells; one-sided exact binomial,
    Holm 0.01) and the BKL u_ref = 0 start (infeasible_start in every replication)."""
    from .inference import judge_family

    tests = {}
    counts = {}
    for name in GENERATORS:
        for pen in PENALTIES:
            g = raw[(raw["generator"] == name) & (raw["penalty"] == pen)]
            k = int((g["status"] != "ok").sum())
            tests[f"{name}|{pen}"] = float(stats.binom.sf(k - 1, len(g), FAILURE_LIMIT))
            counts[f"{name}|{pen}"] = [k, int(len(g))]
    verdict = judge_family("H14-health", tests, level=HEALTH_LEVEL)
    return {
        "failures": counts, "pvalues": tests, "rejected": list(verdict.rejected),
        "verdict": verdict.sentence(), "bkl_zero_offset": bkl_zero_offset(raw),
    }  # fmt: skip


def bkl_zero_offset(raw):
    """Deterministic (PR4a 2(c-1)): every BKL replication's ``u_ref = 0`` start is
    ``infeasible_start``. ``incomplete`` if a BKL replication has no such record."""
    bkl = raw[(raw["generator"] == "BKL") & raw["penalty"].isin(PENALTIES)]
    expected = set(zip(bkl["cell"], bkl["rep"], strict=True))
    z = raw[raw["penalty"] == "BKL_u0"]
    got = dict(zip(zip(z["cell"], z["rep"], strict=True), z["status"], strict=True))
    missing = sorted(k for k in expected if k not in got)
    mismatches = sorted((int(c), int(r), s) for (c, r), s in got.items() if s != "infeasible_start")
    verdict = "fail" if mismatches else ("incomplete" if missing or not expected else "pass")
    return {"verdict": verdict, "expected": len(expected), "replications": len(got),
            "infeasible_start": int(sum(s == "infeasible_start" for s in got.values())),
            "mismatches": mismatches[:50], "missing": [list(map(int, k)) for k in missing[:50]]}


# ---------------------------------------------------------------- reporting

LABEL = {"SN": "SN", "TH": "Theorem", "LTH": "L\\_th"}
END = " \\\\"


def _fmt(x, d=3):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def _sci(x):
    return "--" if x is None or x != x else f"{x:.1e}"


def tables(S, verdicts):
    """``tab_E14.tex`` (longtable body with repeated head): per generator x p_c x n x penalty."""
    heads = ["$g$", "$p_c$", "$n$", "Penalty", "Fail \\%", "CP up. \\%", "(E1)", "(E1)--(E3)",
             "$\\lambda$", "Bound", "Mean err", "Max $\\underline{\\mathrm{err}}$/bound",
             "Violations"]  # fmt: skip
    head = ["\\hline", " & ".join(heads) + END, "\\hline"]
    lines = ["\\begin{tabular}{lrrlrrrrrrrrr}", *head, "\\endfirsthead", *head, "\\endhead"]
    for r in S.itertuples():
        lines.append(" & ".join([
            r.generator, str(r.p_c), str(r.n), LABEL[r.penalty], f"{100 * r.failure_rate:.1f}",
            _fmt(100 * getattr(r, "failure_rate_cp_upper", float("nan")), 2),
            _fmt(r.E1_rate, 2), _fmt(r.events_rate, 2), _sci(r.lam_median), _sci(r.bound_median),
            _sci(r.err_mean), _fmt(r.max_err_over_bound, 3), str(int(r.bnd_violations)),
        ]) + END)  # fmt: skip
    lines += ["\\hline", "\\end{tabular}"]
    return {"tab_E14": "\n".join(lines) + "\n"}


def figure(S, verdicts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(GENERATORS), figsize=(12, 3.2), sharey=False)
    for ax, name in zip(axes, GENERATORS, strict=True):
        for p_c, mk in zip(P_C_VALUES, ("o", "s", "^"), strict=True):
            g = S[(S["generator"] == name) & (S["p_c"] == p_c) & (S["penalty"] == "SN")]
            ax.loglog(g["n"], g["err_mean"], marker=mk, label=f"$p_c={p_c}$")
        g = S[(S["generator"] == name) & (S["p_c"] == 800) & (S["penalty"] == "SN")]
        ref = g["err_mean"].iloc[0] * (g["n"] / g["n"].iloc[0]) ** -0.5
        ax.loglog(g["n"], ref, "k--", lw=0.8, label="slope $-1/2$")
        ax.set_title(name)
        ax.set_xlabel("$n$")
    axes[0].set_ylabel("mean $\\|\\widehat\\alpha-\\alpha_0\\|_{L_2}$ (SN)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def total_warnings(raw):
    return int(raw.drop_duplicates(["cell", "rep"])["n_warnings"].sum())


def reevaluate(stage0_json, raw, entropy=CONFIRMATORY_ENTROPY):
    """Re-evaluate every H14-Bnd violation on an independent ``N = 1e7`` sample, key
    ``(14, 998, rep, 6)`` (reported next to the verdict; it does not change it)."""
    designs = designs_from_stage0(stage0_json)
    out = []
    v = raw[_flag(raw["bnd_violation"])]
    for r in v.itertuples():
        des = designs[r.generator]
        beta = np.zeros(p_total(r.p_c))
        for j, x in json.loads(r.beta_hat).items():
            beta[int(j)] = x
        ev = evaluate(des, generator(r.generator), beta, entropy, int(r.rep), block=REEVAL_BLOCK,
                      N=N_REEVAL)  # fmt: skip
        if ev["eval_status"] != "ok":
            raise FloatingPointError("non-finite re-evaluation")
        out.append({"cell": int(r.cell), "rep": int(r.rep), "penalty": r.penalty,
                    "err2_lower_1e6": float(r.err2_lower), "bound": float(r.bound),
                    "err2_1e7": ev["err2"], "err2_lower_1e7": ev["err2_lower"]})  # fmt: skip
    return out
