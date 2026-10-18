"""E-24: nonlinear ATE benchmark with nested tuning and published RieszNet."""

from __future__ import annotations

import itertools

import numpy as np

from . import baselines, study
from .seeds import CONFIRMATORY_ENTROPY, Seeds, fold_ids

EXP, R, K, CV_FOLDS = 24, 500, 2, 3
THETA0 = 1.0
N_VALUES = (1000,)
CELLS = [[n] for n in N_VALUES]
PZ, SQRT3 = 6, np.sqrt(3.0)
GRR_ARMS = ("SQ", "UKL1", "UKL0", "BP_C0", "BKL1")
ARMS = GRR_ARMS + ("LogitAIPW", "DML-GBM", "RieszNet")
ARM_LABELS = ARMS
LAMBDA_GRID = (1e-3, 1e-2, 1e-1)
C_GRID = (0.1, 1.0, 10.0)


def gr():
    import genriesz

    return genriesz


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


def alpha_ref(X):
    return np.where(np.atleast_2d(X)[:, 0] == 1, 2.0, -2.0)


def eta(Z):
    Z = np.atleast_2d(Z)
    return (
        0.7 * Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * np.sin(np.pi * Z[:, 2] / SQRT3)
        + 0.4 * Z[:, 0] * Z[:, 3]
    )  # fmt: skip


def propensity(Z):
    return 1.0 / (1.0 + np.exp(-eta(Z)))


def tau(Z):
    Z = np.atleast_2d(Z)
    return 1.0 + 0.5 * np.tanh(Z[:, 0] + Z[:, 3])


def gamma0(D, Z):
    Z = np.atleast_2d(Z)
    return np.sin(Z[:, 0]) + 0.5 * Z[:, 1] ** 2 + 0.3 * Z[:, 2] * Z[:, 4] + D * tau(Z)


def draw(rng, n):
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, PZ))
    D = (rng.uniform(size=n) < propensity(Z)).astype(float)
    Y = gamma0(D, Z) + rng.standard_normal(n)
    return D, Z, Y


def _arms_rows(Z):
    n = len(Z)
    return np.column_stack([np.ones(n), Z]), np.column_stack([np.zeros(n), Z])


def poly2(Z):
    Z = np.atleast_2d(Z)
    cols = [Z[:, j] for j in range(PZ)]
    cols += [Z[:, i] * Z[:, j] for i, j in itertools.combinations_with_replacement(range(PZ), 2)]
    return np.column_stack(cols)


def cns_dictionary(X):
    X = np.atleast_2d(X)
    D, Z = X[:, 0], X[:, 1:]
    mono = poly2(Z)
    return np.column_stack([np.ones(len(X)), D, mono, D[:, None] * mono])


def interaction(base, X):
    X = np.atleast_2d(X)
    D = X[:, :1]
    B = base(X[:, 1:])
    return np.concatenate([D * B, (1.0 - D) * B], axis=1)


def generator(arm):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL0": lambda: g.UKLGenerator(C=0.0, branch_fn=sign),
        "UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
        "BKL1": lambda: g.BKLGenerator(C=1.0, branch_fn=sign),
        "BP_C0": lambda: g.BPGenerator(omega=0.5, C=0.0, branch_fn=sign),
    }[arm]()


def glm_model(arm, base, lam):
    g = gr()
    gen = generator(arm)
    basis = g.TreatmentInteractionBasis(base_basis=g.CallableBasis(base))
    return g.GRRGLM(
        basis=basis, generator=gen, functional=g.ATEFunctional(0),
        penalty=None if lam == 0.0 else "l2", lam=float(lam),
        offset=g.offset_from_alpha(gen, alpha_ref),
    )  # fmt: skip


def poly_base(Z):
    Z = np.atleast_2d(Z)
    return np.column_stack([np.ones(len(Z)), poly2(Z)])


class Rep:
    def __init__(self, status, fn=None, info=None):
        self.status, self.fn, self.info = status, fn, info or {}

    def at(self, X):
        X1, X0 = _arms_rows(X[:, 1:])
        return self.fn(X, X1, X0)


def fit_glm(arm, base, lam, X_tr):
    if not (np.any(X_tr[:, 0] == 1) and np.any(X_tr[:, 0] == 0)):
        return Rep("degenerate_functional")
    mdl = glm_model(arm, base, lam)
    fr = mdl.fit(X_tr)
    if fr.status != "ok":
        return Rep(str(fr.status), info={"message": str(fr.message), "status": str(fr.status)})

    def fn(X, X1, X0):
        a, outside, nonfinite = mdl.classify(np.vstack([X, X1, X0]))
        if np.any(nonfinite):
            return None, None, None, "nonfinite"
        if np.any(outside):
            return None, None, None, "domain_prediction"
        n = len(X)
        return a[:n], a[n : 2 * n], a[2 * n :], "ok"

    rep = Rep(
        "ok", fn, {"n_iter": int(fr.n_iter), "message": str(fr.message), "status": str(fr.status)}
    )

    def observed(X):
        a, outside, nonfinite = mdl.classify(X)
        status = (
            "nonfinite" if np.any(nonfinite) else ("domain_prediction" if np.any(outside) else "ok")
        )
        return a, status

    rep.observed = observed
    return rep


def _torch():
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def fit_riesznet(seed, X_tr, Y_tr):
    from . import riesznet

    torch = _torch()
    net, training = riesznet.fit(seed, X_tr, Y_tr)

    def fn(Xe, Xe1, Xe0):
        with torch.no_grad():
            a = [net(torch.as_tensor(r, dtype=torch.float64))[0].numpy() for r in (Xe, Xe1, Xe0)]
        if not all(np.all(np.isfinite(v)) for v in a):
            return None, None, None, "nonfinite"
        return a[0], a[1], a[2], "ok"

    rep = Rep("ok", fn, training)

    def outcome(X):
        with torch.no_grad():
            return net(torch.as_tensor(X, dtype=torch.float64))[1].numpy()

    rep.outcome = outcome
    return rep


class Ridge:
    def __init__(self, base, lam, X_tr, Y_tr):
        F = interaction(base, X_tr)
        n, q = F.shape
        pen = np.full(q, float(lam))
        half = q // 2
        pen[0] = pen[half] = 0.0  # D * 1 and (1 - D) * 1
        A = F.T @ F / n + np.diag(pen)
        self.base = base
        self.coef = np.linalg.solve(A, F.T @ Y_tr / n)

    def __call__(self, X):
        return interaction(self.base, X) @ self.coef


def riesz_criterion(a, a1, a0):
    return float(np.sum(a**2 - 2.0 * (a1 - a0)))


sign.vectorized = True


def fit_logit(C, X):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C)
    clf.fit(poly2(X[:, 1:]), X[:, 0])

    def predict(A, A1, A0):
        e = clf.predict_proba(poly2(A[:, 1:]))[:, 1]
        if not np.all((e > 0) & (e < 1)):
            return None, None, None, "propensity_range"
        return A[:, 0] / e - (1 - A[:, 0]) / (1 - e), 1 / e, -1 / (1 - e), "ok"

    rep = Rep("ok", predict)
    rep.propensity = lambda Z: clf.predict_proba(poly2(Z))[:, 1]
    return rep


def select(grid, evaluate):
    records = []
    for value in grid:
        result, warnings, converged = baselines.run_recording_warnings(
            lambda value=value: evaluate(value)
        )
        score, status = result[:2]
        if not converged:
            status = "convergence_warning"
        if status == "ok" and not np.isfinite(score):
            status = "nonfinite"
        records.append(
            {
                "value": value,
                "score": score,
                "status": status,
                "warnings": warnings,
                "fits": result[2] if len(result) > 2 else [],
            }
        )
    # A failed tuning calculation is retained as a failure of this procedure.
    if any(r["status"] != "ok" for r in records):
        return None, records
    return min(records, key=lambda r: (r["score"], -r["value"]))["value"], records


def tune_representer(arm, X, cv):
    from sklearn.metrics import log_loss

    def evaluate(value):
        total, fits = 0.0, []
        for k in range(CV_FOLDS):
            tr, va = cv != k, cv == k
            rep = (
                fit_logit(value, X[tr])
                if arm == "LogitAIPW"
                else fit_glm(arm, poly_base, value, X[tr])
            )
            fits.append({"fold": k, "status": rep.status, **rep.info})
            if rep.status != "ok":
                return None, rep.status, fits
            if arm == "LogitAIPW":
                total += len(X[va]) * log_loss(X[va, 0], rep.propensity(X[va, 1:]), labels=[0, 1])
            else:
                a, a1, a0, status = rep.at(X[va])
                if status != "ok":
                    return None, status, fits
                total += riesz_criterion(a, a1, a0)
        return total, "ok", fits

    choice, record = select(C_GRID if arm == "LogitAIPW" else LAMBDA_GRID, evaluate)
    if choice is None:
        return Rep("cv_failed", info={"cv": record})
    rep = fit_logit(choice, X) if arm == "LogitAIPW" else fit_glm(arm, poly_base, choice, X)
    rep.info.update(choice=choice, cv=record)
    return rep


def tune_outcome(X, Y, cv):
    def evaluate(lam):
        total = 0.0
        for k in range(CV_FOLDS):
            tr, va = cv != k, cv == k
            model = Ridge(poly_base, lam, X[tr], Y[tr])
            total += float(np.sum((Y[va] - model(X[va])) ** 2))
        return total, "ok"

    choice, record = select(LAMBDA_GRID, evaluate)
    return (None if choice is None else Ridge(poly_base, choice, X, Y)), {
        "choice": choice,
        "cv": record,
    }


def replicate(task):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cell, rep, entropy = task
    n = CELLS[cell][0]
    seeds = Seeds(EXP, entropy)
    D, Z, Y = draw(seeds.data(cell, rep), n)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell, rep))
    inner = seeds.inner_cv(cell, rep)
    learner_seeds = seeds.baseline(cell, rep).integers(0, 2**31 - 1, size=(K, len(ARMS)))
    psi = {a: np.empty(n) for a in ARMS}
    alpha = {a: np.empty(n) for a in ARMS}
    status = dict.fromkeys(ARMS, "ok")
    info = {a: [] for a in ARMS}
    inner_folds = []
    for k in range(K):
        tr, te = folds != k, folds == k
        Xt = X[te]
        X1, X0 = _arms_rows(Xt[:, 1:])
        cv = fold_ids(int(tr.sum()), CV_FOLDS, inner)
        inner_folds.append({"outer_fold": k, "folds": cv.tolist()})

        def fit_common_outcome(tr=tr, cv=cv):
            outcome, tuning = tune_outcome(X[tr], Y[tr], cv)
            return {
                "estimate": 0.0,
                "se": 0.0,
                "status": "ok" if outcome else "cv_failed",
                "model": outcome,
                "tuning": tuning,
            }

        common = study.recorded(fit_common_outcome)
        for arm in ARMS:
            if status[arm] != "ok":
                continue
            seed = int(learner_seeds[k, ARMS.index(arm)])

            def run(arm=arm, common=common, tr=tr, te=te, cv=cv, Xt=Xt, X1=X1, X0=X0, seed=seed):
                if arm in GRR_ARMS or arm == "LogitAIPW":
                    if common["status"] != "ok":
                        return study.failed(common["status"], outcome_tuning=common["tuning"])
                    representation = tune_representer(arm, X[tr], cv)
                    if representation.status != "ok":
                        return study.failed(representation.status, representer=representation.info)
                    if hasattr(representation, "observed"):
                        a, st = representation.observed(Xt)
                    else:
                        a, _, _, st = representation.at(Xt)
                    if st != "ok":
                        return study.failed(st, representer=representation.info)
                    outcome = common["model"]
                    details = {
                        "representer": representation.info,
                        "outcome_tuning": common["tuning"],
                    }
                elif arm == "DML-GBM":
                    clf = HistGradientBoostingClassifier(random_state=seed).fit(Z[tr], D[tr])
                    e = clf.predict_proba(Z[te])[:, 1]
                    if not np.all((e > 0) & (e < 1)):
                        return study.failed("propensity_range")
                    a = D[te] / e - (1 - D[te]) / (1 - e)
                    reg = HistGradientBoostingRegressor(random_state=seed).fit(X[tr], Y[tr])
                    outcome, details = reg.predict, {"seed": seed}
                else:
                    representation = fit_riesznet(seed, X[tr], Y[tr])
                    a, _, _, st = representation.at(Xt)
                    if st != "ok":
                        return study.failed(st, training=representation.info)
                    outcome, details = (
                        representation.outcome,
                        {"training": representation.info, "seed": seed},
                    )
                score = outcome(X1) - outcome(X0) + a * (Y[te] - outcome(Xt))
                if not np.isfinite(score).all():
                    return study.failed("nonfinite", **details)
                return {
                    "estimate": 0.0,
                    "se": 0.0,
                    "status": "ok",
                    "score": score,
                    "alpha": a,
                    **details,
                }

            record = study.recorded(run)
            status[arm] = record["status"]
            if status[arm] == "ok":
                psi[arm][te], alpha[arm][te] = record.pop("score"), record.pop("alpha")
            info[arm].append({"fold": k, **record})
    arms = {}
    for arm in ARMS:
        if status[arm] != "ok":
            arms[arm] = study.failed(status[arm], fits=info[arm])
        else:
            a = alpha[arm]
            arms[arm] = study.score_record(
                psi[arm],
                ddof=1 if arm == "RieszNet" else 0,
                fits=info[arm],
                max_abs_alpha=float(np.max(np.abs(a))),
                ess=float(np.sum(np.abs(a)) ** 2 / np.sum(a**2)),
            )
    return {"arms": arms, "folds": folds.tolist(), "inner_folds": inner_folds}


def tasks_for(entropy=CONFIRMATORY_ENTROPY):
    import sys

    return study.tasks_for(sys.modules[__name__], entropy)


def summarise(raw):
    return study.summarise(raw, THETA0)
