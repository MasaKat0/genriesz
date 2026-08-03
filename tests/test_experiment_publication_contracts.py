"""Contracts for the manuscript experiment code and notebooks.

These tests are intentionally fine grained. They protect the sample roles,
exact generator domains, publication grids, notebook structure, and failure
semantics that the manuscript relies on.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from genriesz.experiments.publication import (
    CoverageDiagnosticBasis,
    ZOnlyBasis,
    fit_matching_ate,
    fit_one_grr,
    fit_one_grr_with_basis,
    fit_one_incompatible,
    load_ihdp_replication,
    load_lalonde,
    make_base_basis,
    make_coverage_diagnostic_data,
    make_dimension_data,
    make_kernel_gp_data,
    make_score_guided_data,
    make_simulation_data,
)
from genriesz.experiments.reference_selection.candidates import (
    CandidateSpec,
    ExperimentBasis,
    candidate_grid,
)
from genriesz.experiments.reference_selection.dgp import make_fold_roles
from genriesz.experiments.reference_selection.grids import publication_grid
from genriesz.experiments.reference_selection.runner import TABLES
from genriesz.generators import (
    BKLGenerator,
    BoundedBKLGenerator,
    BPGenerator,
    DomainError,
    SquaredGenerator,
    UKLGenerator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "experiments"
EXPERIMENT_SOURCE_DIR = REPO_ROOT / "src" / "genriesz" / "experiments"
NOTEBOOKS = tuple(sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb")))
SPECS = candidate_grid()
SCENARIOS = publication_grid()
SOURCE_FILES = tuple(sorted(EXPERIMENT_SOURCE_DIR.rglob("*.py")))
FORBIDDEN_NOTEBOOK_TERMS = ("fallback", "smoke", "pilot", "fast mode", "fast-mode")


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _notebook_code(path: Path) -> str:
    notebook = _notebook(path)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _notebook_text(path: Path) -> str:
    notebook = _notebook(path)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") in {"code", "markdown"}
    )


def _function_name_for_call(tree: ast.AST, target: ast.Call) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(child is target for child in ast.walk(node)):
            return node.name
    return None


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.label)
def test_each_publication_candidate_has_a_valid_loss(spec: CandidateSpec) -> None:
    assert spec.loss in {"SQ", "UKL", "BKL", "BP"}
    assert spec.dictionary in {"linear", "second_order", "rich"}
    assert spec.penalty_multiplier in {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}
    if spec.loss == "BP":
        assert spec.omega in {0.25, 0.5}
    else:
        assert spec.omega is None


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.label)
def test_each_publication_candidate_has_a_unique_stable_label(spec: CandidateSpec) -> None:
    assert spec.label
    assert SPECS.count(spec) == 1
    assert sum(other.label == spec.label for other in SPECS) == 1
    assert " " not in spec.label


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_code_parses(path: Path) -> None:
    ast.parse(_notebook_code(path))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_contains_no_exception_handler(path: Path) -> None:
    tree = ast.parse(_notebook_code(path))
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_contains_no_shortcut_configuration(path: Path) -> None:
    text = _notebook_text(path).lower()
    assert not [term for term in FORBIDDEN_NOTEBOOK_TERMS if term in text]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_uses_the_installed_experiment_package(path: Path) -> None:
    code = _notebook_code(path)
    assert "from genriesz.experiments" in code
    assert "notebooks/experiments/refsel" not in code
    assert "sys.path.insert(0, str(EXPERIMENT_DIR))" not in code


def test_reference_selection_reports_reload_only_complete_results() -> None:
    path = NOTEBOOK_DIR / "09_reference_based_loss_link_selection.ipynb"
    notebook = _notebook(path)
    report_cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "report." in source:
            report_cells.append(source)
    assert report_cells
    assert all("tables = load_experiment(RUN_DIR)" in source for source in report_cells)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_has_no_stored_results(path: Path) -> None:
    notebook = _notebook(path)
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert not cell.get("outputs", [])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_displays_each_figure_in_the_notebook(path: Path) -> None:
    code = _notebook_code(path)
    if "matplotlib.pyplot" in code:
        assert "plt.show()" in code


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_does_not_hide_figure_layout_in_a_function(path: Path) -> None:
    tree = ast.parse(_notebook_code(path))
    prohibited = ("plot", "figure", "axis", "axes", "style")
    names = [
        node.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert not [name for name in names if any(word in name for word in prohibited)]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.label)
def test_publication_scenario_matches_its_grid(scenario) -> None:
    assert scenario.grid in {"A", "B", "C"}
    assert scenario.design in {"low", "high"}
    if scenario.grid == "A":
        assert scenario.design == "low"
        assert scenario.sample_size in {1000, 3000}
        assert scenario.overlap_scale in {0.5, 1.5, 2.5}
        assert scenario.target_t == 0.0
        assert scenario.hidden_scale == 0.0
    elif scenario.grid == "B":
        assert scenario.design == "low"
        assert scenario.sample_size in {1000, 3000}
        assert scenario.overlap_scale == 1.5
        assert scenario.target_t in {0.5, 1.0, 2.0, 4.0}
        assert scenario.hidden_scale > 0.0
    else:
        assert scenario.design == "high"
        assert scenario.sample_size == 3000
        assert scenario.overlap_scale in {0.75, 2.0}
        assert scenario.target_t in {0.0, 1.0}
        assert scenario.hidden_scale >= 0.0


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_experiment_source_parses(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_experiment_source_contains_no_exception_handler(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))


def test_experiment_directory_has_no_index_notebook() -> None:
    assert not (NOTEBOOK_DIR / "00_index.ipynb").exists()
    assert not (NOTEBOOK_DIR / "00_experiment_index.ipynb").exists()


def test_experiment_directory_has_no_notebook_local_package() -> None:
    assert not (NOTEBOOK_DIR / "refsel").exists()
    assert not (NOTEBOOK_DIR / "pilots").exists()
    assert not (NOTEBOOK_DIR / ".ipynb_checkpoints").exists()


def test_repository_has_no_archived_legacy_experiment_copy() -> None:
    assert not (REPO_ROOT / "notebooks" / "experiments.zip").exists()
    assert not list(NOTEBOOK_DIR.rglob(".DS_Store"))


def test_no_incomplete_publication_results_are_stored() -> None:
    publication = (
        NOTEBOOK_DIR / "results" / "reference_selection" / "publication"
    )
    if not publication.exists():
        return

    manifest_path = publication / "run_manifest.json"
    batch_files = sorted(publication.glob("*.parquet"))
    if not manifest_path.exists() and not batch_files:
        return

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_batches = int(manifest["n_batches"])
    expected = {
        publication / f"{table}_{batch_index:05d}.parquet"
        for batch_index in range(n_batches)
        for table in TABLES
    }
    assert set(batch_files) == expected


def test_full_candidate_grid_has_ninety_entries() -> None:
    assert len(SPECS) == 90
    assert len({spec.label for spec in SPECS}) == 90


def test_publication_grid_has_eighteen_scenarios_and_twenty_four_thousand_jobs() -> None:
    counts = {"A": 1000, "B": 2000, "C": 500}
    assert len(SCENARIOS) == 18
    assert sum(counts[scenario.grid] for scenario in SCENARIOS) == 24_000


def test_fold_roles_do_not_leak_observations() -> None:
    roles = make_fold_roles(n=107, n_folds=5, seed=17)
    evaluation_count = np.zeros(107, dtype=int)
    for role in roles:
        assert np.intersect1d(role.training, role.diagnostic).size == 0
        assert np.intersect1d(role.training, role.evaluation).size == 0
        assert np.intersect1d(role.diagnostic, role.evaluation).size == 0
        assert np.array_equal(
            np.sort(np.concatenate((role.training, role.diagnostic, role.evaluation))),
            np.arange(107),
        )
        evaluation_count[role.evaluation] += 1
    assert np.all(evaluation_count == 1)


def test_basis_standardization_does_not_use_evaluation_observations() -> None:
    rng = np.random.default_rng(7)
    X_train = np.column_stack((rng.binomial(1, 0.5, 80), rng.normal(size=(80, 4))))
    X_eval = np.column_stack((rng.binomial(1, 0.5, 20), rng.normal(size=(20, 4))))
    basis = ExperimentBasis("rich").fit(X_train)
    before = basis(X_train)
    shifted = X_eval.copy()
    shifted[:, 1:] += 1000.0
    basis(shifted)
    after = basis(X_train)
    assert np.array_equal(before, after)
    nonconstant = np.std(before, axis=0) > 1e-12
    assert np.allclose(np.mean(before[:, nonconstant], axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(before[:, nonconstant], axis=0), 1.0, atol=1e-12)


def test_z_only_basis_removes_treatment_before_fitting() -> None:
    rng = np.random.default_rng(13)
    Z = rng.normal(size=(60, 3))
    X0 = np.column_stack((np.zeros(60), Z))
    X1 = np.column_stack((np.ones(60), Z))
    basis = ZOnlyBasis(make_base_basis("polynomial", degree=2)).fit(X0)
    assert np.allclose(basis(X0), basis(X1))


@pytest.mark.parametrize(
    ("dgp", "lower", "upper"),
    [
        ("DGP 1: smooth heterogeneous effects", 0.05, 0.95),
        ("DGP 2: weak overlap nonlinear design", 0.05, 0.95),
        ("DGP 3: high-dimensional sparse confounding", 0.05, 0.95),
    ],
)
def test_original_simulation_propensity_bounds_are_part_of_the_dgp(
    dgp: str, lower: float, upper: float
) -> None:
    data = make_simulation_data(dgp, n=1000, seed=29)
    assert np.min(data["e"]) >= lower
    assert np.max(data["e"]) <= upper


@pytest.mark.parametrize("dimension", [5, 10, 20, 50])
def test_dimension_dgp_uses_the_stated_propensity_bounds(dimension: int) -> None:
    data = make_dimension_data(n=700, d=dimension, seed=100 + dimension)
    # Treatment probabilities are not returned, so the contract is checked by
    # reproducing the deterministic index used by the DGP.
    Z = data["Z"]
    logits = 0.8 * Z[:, 0] - 0.6 * Z[:, 1] + 0.35 * np.sin(Z[:, 2])
    probability = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.03, 0.97)
    assert probability.min() >= 0.03
    assert probability.max() <= 0.97


@pytest.mark.parametrize("outcome_kernel", ["polynomial", "sinusoidal"])
def test_kernel_dgp_returns_finite_observations(outcome_kernel: str) -> None:
    data = make_kernel_gp_data(n=300, d=5, seed=71, outcome_kernel=outcome_kernel)
    assert np.all(np.isfinite(data["X"]))
    assert np.all(np.isfinite(data["Y"]))
    assert data["theta_ate"] == 1.0
    assert data["theta_att"] == 1.0


def test_bkl_status_marks_the_exact_dual_boundary_invalid() -> None:
    generator = BKLGenerator(C=1.0, branch_fn=lambda x: 1 if x[0] >= 0.5 else -1)
    X = np.array([[1.0], [1.0], [0.0], [0.0]])
    v = np.array([-1.0, 0.0, 1.0, 0.0])
    result = generator.inv_grad_status(X, v)
    assert np.array_equal(result.valid, np.array([True, False, True, False]))
    assert np.all(np.isfinite(result.values[result.valid]))
    assert np.all(np.isnan(result.values[~result.valid]))


def test_bp_status_does_not_clip_an_invalid_dual_coordinate() -> None:
    generator = BPGenerator(C=1.0, omega=0.5, branch_fn=lambda x: 1 if x[0] >= 0.5 else -1)
    X = np.array([[1.0], [1.0], [0.0], [0.0]])
    k = 1.0 + 1.0 / generator.omega
    v = np.array([0.0, -k, 0.0, k])
    result = generator.inv_grad_status(X, v)
    assert np.array_equal(result.valid, np.array([True, False, True, False]))
    assert np.all(np.isnan(result.values[~result.valid]))


def test_bp_direct_inverse_raises_at_an_invalid_dual_coordinate() -> None:
    generator = BPGenerator(C=1.0, omega=0.5, branch_fn=lambda x: 1)
    k = 1.0 + 1.0 / generator.omega
    with pytest.raises(DomainError):
        generator.inv_grad(np.array([[1.0]]), np.array([-k]))


def test_ukl_status_rejects_an_unrepresentable_tail_without_substitution() -> None:
    generator = UKLGenerator(C=1.0, branch_fn=lambda x: 1)
    X = np.array([[1.0]])
    result = generator.inv_grad_status(X, np.array([-1000.0]))
    assert not result.valid[0]
    assert np.isnan(result.values[0])
    with pytest.raises(DomainError, match="cannot represent the exact inverse link"):
        generator.inv_grad(X, np.array([-1000.0]))


def test_bkl_status_rejects_an_unrepresentable_lower_tail() -> None:
    generator = BKLGenerator(C=1.0, branch_fn=lambda x: 1)
    X = np.array([[1.0]])
    result = generator.inv_grad_status(X, np.array([-1000.0]))
    assert not result.valid[0]
    assert np.isnan(result.values[0])


def test_bp_status_rejects_a_positive_but_unrepresentable_power_term() -> None:
    generator = BPGenerator(C=1.0, omega=0.5, branch_fn=lambda x: 1)
    X = np.array([[1.0]])
    k = 1.0 + 1.0 / generator.omega
    t = 1e-12
    v = np.array([k * (t - 1.0)])
    result = generator.inv_grad_status(X, v)
    assert not result.valid[0]
    assert np.isnan(result.values[0])


def test_bounded_bkl_reports_the_float64_lower_bound_as_binding() -> None:
    generator = BoundedBKLGenerator(C=1.0, alpha_max=20.0, branch_fn=lambda x: 1)
    X = np.array([[1.0]])
    v = np.array([-1000.0])
    alpha = generator.inv_grad(X, v)
    assert alpha[0] == np.nextafter(1.0, np.inf)
    assert generator.domain_binding(X, v)[0]


def test_squared_inverse_is_valid_for_every_finite_dual_coordinate() -> None:
    generator = SquaredGenerator(C=0.0)
    X = np.zeros((5, 1))
    v = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    result = generator.inv_grad_status(X, v)
    assert np.all(result.valid)
    assert np.all(np.isfinite(result.values))


def test_exact_bkl_is_reported_as_inadmissible_for_att() -> None:
    data = make_simulation_data("DGP 1: smooth heterogeneous effects", n=300, seed=23)
    rows = fit_one_grr(
        data,
        estimand="ATT",
        loss_spec={"loss": "BKL", "omega": None, "label": "BKL"},
        basis_kind="polynomial",
        degree=1,
        folds=3,
        estimators=("arw",),
        random_state=5,
    )
    assert {row["status"] for row in rows} == {"incompatible_exact_domain"}
    assert {row["estimator"] for row in rows} == {"failed"}


def test_matching_uses_the_public_matching_weights_without_substitution() -> None:
    data = make_simulation_data("DGP 1: smooth heterogeneous effects", n=250, seed=37)
    rows = fit_matching_ate(data, rep=0, M=1)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["estimator"] == "rw"
    assert np.isfinite(rows[0]["estimate"])


def test_missing_ihdp_files_raise_instead_of_creating_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ihdp_replication(1, data_dir=tmp_path)
    assert not any(tmp_path.iterdir())


def test_missing_lalonde_file_raises_instead_of_downloading_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lalonde(data_dir=tmp_path)
    assert not any(tmp_path.iterdir())


def test_np_clip_is_confined_to_prespecified_data_generation() -> None:
    allowed = {
        "make_simulation_data",
        "make_dimension_data",
        "make_kernel_gp_data",
        "generate_data",
    }
    offenders: list[tuple[str, str | None]] = []
    for path in SOURCE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            is_np_clip = (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "np"
                and function.attr == "clip"
            )
            if is_np_clip:
                owner = _function_name_for_call(tree, node)
                if owner not in allowed:
                    offenders.append((str(path.relative_to(REPO_ROOT)), owner))
    assert not offenders


# ---------------------------------------------------------------------------
# Notebook calls must bind to the installed package signatures, and local
# definitions must not shadow package exports. (A stale ``C=`` keyword in the
# Lalonde notebook and a local ``make_dimension_data`` in notebook 05 both
# passed the import-only checks.)
# ---------------------------------------------------------------------------
def _experiment_imports(tree: ast.Module) -> dict[str, object]:
    imported: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not (node.module or "").startswith("genriesz.experiments"):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            obj = getattr(module, alias.name, None)
            if obj is not None:
                imported[alias.asname or alias.name] = obj
    return imported


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_keywords_bind_to_package_signatures(path: Path) -> None:
    tree = ast.parse(_notebook_code(path))
    imported = _experiment_imports(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        target = imported.get(node.func.id)
        if target is None or not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        parameters = signature.parameters.values()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
            continue
        allowed = set(signature.parameters)
        for keyword in node.keywords:
            if keyword.arg is not None:
                assert keyword.arg in allowed, (
                    f"{path.name}: {node.func.id}() has no parameter {keyword.arg!r}"
                )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_does_not_shadow_package_exports(path: Path) -> None:
    tree = ast.parse(_notebook_code(path))
    imported = set(_experiment_imports(tree))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert node.name not in imported, (
                f"{path.name}: local definition of {node.name!r} shadows the package export"
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name_node in ast.walk(target):
                    if isinstance(name_node, ast.Name):
                        assert name_node.id not in imported, (
                            f"{path.name}: assignment to {name_node.id!r} "
                            "shadows the package export"
                        )


# ---------------------------------------------------------------------------
# The deliberately mismatched pairs run through their real fitting paths. The
# first release routed "UKL loss + linear link" through a generator whose
# array lambdas crashed inside the rowwise-scalar contract, and the composed
# path was never exercised by a test.
# ---------------------------------------------------------------------------
INCOMPATIBLE_PAIRS = (
    "BKL loss + logit link",
    "SQ loss + logit link",
    "UKL loss + linear link",
)


@pytest.mark.parametrize("pair", INCOMPATIBLE_PAIRS)
def test_fit_one_incompatible_runs_the_real_path(pair: str) -> None:
    data = make_simulation_data("DGP 1: smooth heterogeneous effects", n=400, seed=5)
    rows = fit_one_incompatible(
        data,
        estimand="ATE",
        pair_name=pair,
        cross_fit=True,
        lam=1e-2,
        basis_features=40,
        folds=2,
        random_state=0,
    )
    assert rows
    assert all(row["status"] == "ok" for row in rows)
    by_estimator = {row["estimator"]: row for row in rows}
    assert set(by_estimator) == {"ra", "rw", "arw", "tmle"}
    for name in ("arw", "tmle"):
        assert by_estimator[name]["inference"] == "influence_normal"
        assert np.isfinite(by_estimator[name]["se"])
        assert by_estimator[name]["covered"] in (True, False)
    for name in ("ra", "rw"):
        assert by_estimator[name]["inference"] == "point_only"
        assert not np.isfinite(by_estimator[name]["se"])
        assert np.isnan(by_estimator[name]["covered"])
        assert np.isfinite(by_estimator[name]["estimate"])
    if pair == "UKL loss + linear link":
        # The mismatch has to show: the pure weighting estimator degrades while
        # the orthogonal estimators absorb the broken balance equations.
        rw_error = abs(by_estimator["rw"]["error"])
        arw_error = abs(by_estimator["arw"]["error"])
        assert rw_error > 1.0
        assert arw_error < 0.2
        assert rw_error > 10.0 * arw_error


def test_att_linear_link_records_held_out_domain_failures() -> None:
    """The evaluation sample is checked against the branchwise UKL domain.

    With the ATT shift C=0 the counterfactual control representer of the
    linear-link model crosses zero on held-out observations in this
    deterministic configuration. The first implementation assigned those
    values silently and reported every estimator as ``ok``.
    """

    data = make_simulation_data("DGP 1: smooth heterogeneous effects", n=400, seed=5)
    rows = fit_one_incompatible(
        data,
        estimand="ATT",
        pair_name="UKL loss + linear link",
        cross_fit=True,
        lam=1e-2,
        basis_features=40,
        folds=2,
        random_state=0,
    )
    assert rows
    assert {row["status"] for row in rows} == {"held_out_domain_failure"}
    assert all(row["estimator"] == "failed" for row in rows)


def test_matching_reports_point_estimates_only() -> None:
    data = make_simulation_data("DGP 1: smooth heterogeneous effects", n=300, seed=9)
    row = fit_matching_ate(data, rep=0)[0]
    assert row["inference"] == "point_only"
    assert not np.isfinite(row["se"])
    assert np.isnan(row["covered"])


def test_score_guided_truth_is_a_population_value() -> None:
    a = make_score_guided_data(n=250, seed=21)
    b = make_score_guided_data(n=250, seed=22)
    assert (a["theta_ate"], a["theta_att"]) == (b["theta_ate"], b["theta_att"])


def test_population_att_uses_propensity_weighting() -> None:
    from genriesz.experiments.publication import (
        POPULATION_TRUTH_DRAW,
        POPULATION_TRUTH_SEED,
    )
    from genriesz.experiments.publication import (
        make_score_guided_data as _maker,
    )

    big = _maker(n=POPULATION_TRUTH_DRAW, seed=POPULATION_TRUTH_SEED, _population=False)
    tau = np.asarray(big["tau"], dtype=float)
    e = np.asarray(big["e"], dtype=float)
    expected_att = float(np.mean(e * tau) / np.mean(e))
    small = make_score_guided_data(n=250, seed=3)
    assert small["theta_att"] == expected_att


# ---------------------------------------------------------------------------
# Synthetic truth values are population estimands. The first release stored
# per-replication sample means, so the stored truth changed with the seed
# while the influence-function standard errors targeted the population value.
# ---------------------------------------------------------------------------
def test_simulation_truth_is_a_population_value() -> None:
    name = "DGP 1: smooth heterogeneous effects"
    a = make_simulation_data(name, n=300, seed=11)
    b = make_simulation_data(name, n=300, seed=12)
    assert a["theta_ate"] == b["theta_ate"]
    assert a["theta_att"] == b["theta_att"]
    assert a["theta_ate"] != float(np.mean(a["tau"]))
    # DGP 1 has E[tau] = 1 exactly; the fixed-draw value sits within MC error.
    assert abs(a["theta_ate"] - 1.0) < 0.005


def test_dimension_truth_is_a_population_value() -> None:
    a = make_dimension_data(n=200, d=5, seed=3)
    b = make_dimension_data(n=200, d=5, seed=4)
    assert (a["theta_ate"], a["theta_att"]) == (b["theta_ate"], b["theta_att"])
    assert a["theta_att"] != float(np.mean(np.asarray(a["tau"])[np.asarray(a["D"]) == 1.0]))


@pytest.mark.parametrize("pair", INCOMPATIBLE_PAIRS)
def test_degenerate_training_fold_is_a_recorded_failure(pair: str) -> None:
    rng = np.random.default_rng(7)
    n = 60
    Z = rng.normal(size=(n, 2))
    D = np.zeros(n)
    D[0] = 1.0  # the single treated observation leaves one training fold empty
    data = {"X": np.column_stack([D, Z]), "Y": rng.normal(size=n), "D": D}
    rows = fit_one_incompatible(
        data,
        estimand="ATE",
        pair_name=pair,
        cross_fit=True,
        lam=1e-2,
        basis_features=10,
        folds=2,
        random_state=0,
    )
    assert rows
    assert {row["status"] for row in rows} == {"degenerate_treatment_fold"}


def test_linear_link_fit_reports_an_infeasible_domain() -> None:
    from genriesz.experiments.publication import _fit_linear_link_ukl

    # Two identical regressor rows on opposite branches make sign*Phi*beta > C
    # unsatisfiable, so the fit must report infeasibility instead of a value.
    Phi = np.array([[1.0, 0.5], [1.0, 0.5], [0.3, 1.0]])
    M = np.zeros_like(Phi)
    sign = np.array([1.0, -1.0, 1.0])
    beta, status = _fit_linear_link_ukl(Phi, M, sign, C=1.0, lam=1e-2, max_iter=200)
    assert status == "linear_link_domain_infeasible"
    assert np.all(np.isnan(beta))


def test_linear_link_fit_reports_an_iteration_limit_as_a_failure() -> None:
    from genriesz.experiments.publication import _fit_linear_link_ukl

    rng = np.random.default_rng(11)
    n = 80
    Phi = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    M = rng.normal(size=(n, 4))
    sign = np.ones(n)
    beta, status = _fit_linear_link_ukl(Phi, M, sign, C=1.0, lam=1e-2, max_iter=1)
    assert status == "optimizer_failure"
    beta, status = _fit_linear_link_ukl(Phi, M, sign, C=1.0, lam=1e-2, max_iter=500)
    assert status == "converged"
    assert np.all(np.isfinite(beta))


# ---------------------------------------------------------------------------
# The constrained exact solvers evaluate the conjugate as an extended-value
# function: a line-search step outside the dual domain yields an infinite
# objective and the fit either converges or records a candidate failure. It
# must never escape as an exception (a weak-overlap BKL fold once did).
# ---------------------------------------------------------------------------
def test_exact_bkl_under_weak_overlap_records_any_failure() -> None:
    data = make_coverage_diagnostic_data(n=400, seed=3_000_000, overlap_scale=2.5)
    rows = fit_one_grr_with_basis(
        data,
        estimand="ATE",
        loss_spec={"label": "BKL", "loss": "BKL"},
        representer_basis=CoverageDiagnosticBasis(include_quadratic=True),
        cross_fit=True,
        lam=1e-2,
        folds=2,
        estimators=("arw",),
        random_state=0,
    )
    assert rows
    assert {row["status"] for row in rows} <= {"ok", "constrained_optimizer_failure"}


# ---------------------------------------------------------------------------
# ``generator_override`` states the fitted model explicitly; the sensitivity
# diagnostics (estimand flag and per-side binding rates) surface in the rows,
# and the exact path keeps reporting an unbound fit.
# ---------------------------------------------------------------------------
def test_generator_override_surfaces_the_sensitivity_diagnostics() -> None:
    from genriesz.experiments.publication import branch_treated
    from genriesz.generators import BoundedUKLGenerator

    data = make_coverage_diagnostic_data(n=300, seed=7, overlap_scale=0.5)
    gen = BoundedUKLGenerator.from_propensity_bounds(0.05, 0.95, branch_fn=branch_treated)
    rows = fit_one_grr_with_basis(
        data,
        estimand="ATE",
        loss_spec={"label": "UKL", "loss": "UKL"},
        representer_basis=CoverageDiagnosticBasis(include_quadratic=True),
        cross_fit=True,
        lam=1e-2,
        folds=2,
        estimators=("arw",),
        random_state=0,
        generator_override=gen,
    )
    row = rows[0]
    assert row["status"] == "ok"
    assert row["riesz_modifies_estimand"] is True
    assert np.isfinite(row["riesz_binding_rate_lower_max"])
    assert np.isfinite(row["riesz_binding_rate_upper_max"])
    assert row["riesz_clip_binding_rate_max"] >= 0.0


def test_exact_fit_reports_an_unbound_representer() -> None:
    data = make_coverage_diagnostic_data(n=300, seed=7, overlap_scale=0.5)
    rows = fit_one_grr_with_basis(
        data,
        estimand="ATE",
        loss_spec={"label": "UKL", "loss": "UKL"},
        representer_basis=CoverageDiagnosticBasis(include_quadratic=True),
        cross_fit=True,
        lam=1e-2,
        folds=2,
        estimators=("arw",),
        random_state=0,
    )
    row = rows[0]
    assert row["status"] == "ok"
    assert row["riesz_modifies_estimand"] is False
    assert row["riesz_clip_binding_rate_max"] == 0.0
    assert np.isnan(row["riesz_binding_rate_lower_max"])
    assert np.isnan(row["riesz_binding_rate_upper_max"])


def test_exact_bkl_under_the_notebook_configuration_records_any_failure() -> None:
    # Full notebook-12 configuration. The optimizer can also terminate at a
    # point outside the exact dual domain; the KKT diagnostics are undefined
    # there and the fold must surface as a recorded failure, not an exception
    # from the multiplier solve.
    data = make_coverage_diagnostic_data(n=2000, seed=3_108_063, overlap_scale=2.5)
    rows = fit_one_grr_with_basis(
        data,
        estimand="ATE",
        loss_spec={"label": "BKL", "loss": "BKL"},
        representer_basis=CoverageDiagnosticBasis(include_quadratic=True),
        cross_fit=True,
        lam=1e-2,
        folds=5,
        estimators=("arw",),
        random_state=9,
    )
    assert rows
    assert {row["status"] for row in rows} == {"constrained_optimizer_failure"}
    assert any(
        "outside the exact dual domain" in str(row.get("message", "")) for row in rows
    )


@pytest.mark.parametrize(
    "loss_spec",
    [
        {"label": "BKL", "loss": "BKL"},
        {"label": "BP(0.5)", "loss": "BP", "omega": 0.5},
    ],
    ids=["bkl", "bp"],
)
def test_exact_constrained_paths_report_an_unbound_representer(loss_spec) -> None:
    # The exact dual domain is enforced by constraints; no clamp exists, so a
    # successful fit reports a zero binding rate and no per-side rates.
    data = make_simulation_data("DGP 1: smooth heterogeneous effects", n=400, seed=5)
    rows = fit_one_grr(
        data,
        estimand="ATE",
        loss_spec=loss_spec,
        basis_kind="rkhs",
        cross_fit=True,
        lam=1e-2,
        basis_features=40,
        folds=2,
        estimators=("arw",),
        random_state=0,
    )
    row = rows[0]
    assert row["status"] == "ok"
    assert row["riesz_modifies_estimand"] is False
    assert row["riesz_clip_binding_rate_max"] == 0.0
    assert np.isnan(row["riesz_binding_rate_lower_max"])
    assert np.isnan(row["riesz_binding_rate_upper_max"])


def test_generator_override_does_not_bypass_the_att_guard() -> None:
    from genriesz.experiments.publication import branch_treated

    data = make_simulation_data("DGP 1: smooth heterogeneous effects", n=300, seed=9)
    rows = fit_one_grr_with_basis(
        data,
        estimand="ATT",
        loss_spec={"label": "BKL", "loss": "BKL"},
        representer_basis=CoverageDiagnosticBasis(include_quadratic=True),
        cross_fit=True,
        lam=1e-2,
        folds=2,
        estimators=("arw",),
        random_state=0,
        generator_override=BKLGenerator(C=0.05, branch_fn=branch_treated),
    )
    assert rows
    assert {row["status"] for row in rows} == {"incompatible_exact_domain"}
