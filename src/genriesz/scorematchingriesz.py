"""Core ScoreMatchingRiesz primitives.

This module contains reusable algorithmic building blocks for ScoreMatchingRiesz:

* denoising score matching for data-score estimation;
* DRE-infinity time-score matching for density-ratio estimation;
* joint data-score and time-score training;
* score integration utilities for density ratios and shift-policy ratios;
* generic neural baselines used by ScoreMatchingRiesz workflows.

Experiment designs, data-generating processes, paper-specific tables, plotting code,
and dataset-specific wrappers intentionally live outside ``src``. Put those in notebooks or
experiment scripts.
"""

from __future__ import annotations

import functools
import math
import random
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .utils import kfold_splits

try:  # optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except Exception as exc:  # pragma: no cover - exercised only without optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment,misc]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _torch_no_grad():
    if torch is None:
        def decorator(func):
            return func
        return decorator
    return torch.no_grad()


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ImportError(
            "genriesz.scorematchingriesz requires PyTorch. "
            "Install it with `pip install -e .[scorematchingriesz]`."
        ) from _TORCH_IMPORT_ERROR


def _as_2d_float_array(x: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got shape {arr.shape}.")
    return arr


def _as_1d_float_array(y: np.ndarray | Sequence[float], *, n: int, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got shape {arr.shape}.")
    return arr


def get_device(prefer_gpu: bool = True):
    """Return a PyTorch device for ScoreMatchingRiesz computations."""

    _require_torch()
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Set Python, NumPy, and PyTorch random seeds.

    This is a user-facing convenience for scripting. Library fit functions do
    NOT call it; each seeds the PyTorch RNG internally for reproducibility
    (:func:`_seed_torch`) but restores the global RNG state before returning
    (:func:`_keeps_global_torch_rng`), so fitting a model never mutates the
    caller's global PyTorch, NumPy, or Python random state.
    """

    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _seed_torch(seed: int) -> None:
    """Seed the CPU (and CUDA, if present) PyTorch RNGs used by the fit functions.

    ``torch.manual_seed`` is avoided on purpose: it *also* seeds MPS, XPU and any
    registered accelerator, none of which :func:`_keeps_global_torch_rng` saves
    and restores, and none of which the fits draw from (the device comes from
    :func:`get_device`, i.e. CUDA or CPU). Seeding the CPU default generator
    directly leaves those other device RNGs untouched while producing the
    identical CPU stream, so the pair (seed here, restore there) covers exactly
    the RNGs the fits actually use.
    """

    if torch is not None:
        torch.default_generator.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _keeps_global_torch_rng(fit_fn):
    """Run a fit with the CPU and CUDA PyTorch RNGs saved on entry, restored on exit.

    The fit functions seed the torch RNG (via :func:`_seed_torch`) so their
    output is reproducible from ``seed`` alone. Without this wrapper that seeding
    would also leave the caller's *global* torch RNG mutated on return: every fit
    would reset it to ``seed`` and then advance it by however many draws training
    consumed. Saving the state before the fit and restoring it afterwards keeps
    the seeding local -- the fit's weight init and every batch draw are
    byte-for-byte identical, but the caller's global torch RNG is untouched, just
    as the fit functions already leave the global NumPy and Python RNGs untouched.

    Only the CPU and CUDA RNGs are saved and restored; that is sufficient because
    :func:`_seed_torch` seeds only those (never MPS/XPU) and the fits only ever
    run on CPU or CUDA (:func:`get_device`). If ``_seed_torch`` is ever changed to
    seed another accelerator, that device's state must be saved and restored here
    too, or the fit would leak it.
    """

    @functools.wraps(fit_fn)
    def wrapper(*args, **kwargs):
        if torch is None:
            return fit_fn(*args, **kwargs)
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            return fit_fn(*args, **kwargs)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return wrapper


if torch is not None:

    class MLP(nn.Module):
        """Small fully connected network used by the score models."""

        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 256),
            *,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            dims = [int(in_dim), *[int(h) for h in hidden_dims], int(out_dim)]
            layers: list[nn.Module] = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)


    class TimeEmbedding(nn.Module):
        """Fourier embedding for scalar bridge time ``t``."""

        def __init__(self, emb_dim: int = 32, max_freq: int = 10) -> None:
            super().__init__()
            if emb_dim % 2 != 0:
                raise ValueError("emb_dim must be even.")
            freqs = torch.linspace(1.0, float(max_freq), emb_dim // 2)
            self.register_buffer("freqs", freqs)

        def forward(self, t: Tensor) -> Tensor:
            tt = t * self.freqs.view(1, -1) * 2.0 * math.pi
            return torch.cat([torch.sin(tt), torch.cos(tt)], dim=1)


    class TimeScoreNet(nn.Module):
        """Time-score network ``s_t(x,t)`` for DRE-infinity."""

        def __init__(
            self,
            x_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 256),
            *,
            t_emb_dim: int = 32,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.loss_history: list[float] = []
            self.t_embed = TimeEmbedding(emb_dim=t_emb_dim)
            self.net = MLP(
                in_dim=int(x_dim) + int(t_emb_dim),
                out_dim=1,
                hidden_dims=hidden_dims,
                layer_norm=layer_norm,
            )

        def forward(self, x: Tensor, t: Tensor) -> Tensor:
            return self.net(torch.cat([x, self.t_embed(t)], dim=1))


    class JointScoreNet(nn.Module):
        """Joint data-score and time-score network.

        ``forward`` returns ``(s_x(x,t), s_t(x,t))``.
        """

        def __init__(
            self,
            x_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 256),
            *,
            t_emb_dim: int = 32,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.loss_history: list[float] = []
            self.t_embed = TimeEmbedding(emb_dim=t_emb_dim)
            in_dim = int(x_dim) + int(t_emb_dim)
            self.data_net = MLP(
                in_dim=in_dim, out_dim=int(x_dim), hidden_dims=hidden_dims, layer_norm=layer_norm
            )
            self.time_net = MLP(
                in_dim=in_dim, out_dim=1, hidden_dims=hidden_dims, layer_norm=layer_norm
            )

        def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
            inp = torch.cat([x, self.t_embed(t)], dim=1)
            return self.data_net(inp), self.time_net(inp)


    class DataScoreDNet(nn.Module):
        """Denoising score matching network for one coordinate score."""

        def __init__(
            self,
            x_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 256),
            *,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.loss_history: list[float] = []
            self.treatment_index: int = 0
            self.net = MLP(
                in_dim=int(x_dim) + 1, out_dim=1, hidden_dims=hidden_dims, layer_norm=layer_norm
            )

        def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
            return self.net(torch.cat([x, sigma], dim=1))


    class ScalarNet(nn.Module):
        """Scalar network used for AME Riesz-regression baselines."""

        def __init__(
            self,
            x_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 256),
            *,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.loss_history: list[float] = []
            self.net = MLP(
                in_dim=int(x_dim), out_dim=1, hidden_dims=hidden_dims, layer_norm=layer_norm
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)


    class RatioNet(nn.Module):
        """Scalar network used for direct density-ratio baselines."""

        def __init__(
            self,
            x_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 256),
            *,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.loss_history: list[float] = []
            self.objective: str = ""
            self.net = MLP(
                in_dim=int(x_dim), out_dim=1, hidden_dims=hidden_dims, layer_norm=layer_norm
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)


    class OutcomeNet(nn.Module):
        """Outcome regression network."""

        def __init__(
            self,
            x_dim: int,
            hidden_dims: Sequence[int] = (256, 256, 256),
            *,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.loss_history: list[float] = []
            self.net = MLP(
                in_dim=int(x_dim), out_dim=1, hidden_dims=hidden_dims, layer_norm=layer_norm
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

else:  # pragma: no cover
    MLP = TimeEmbedding = TimeScoreNet = JointScoreNet = None  # type: ignore[misc,assignment]
    DataScoreDNet = ScalarNet = RatioNet = OutcomeNet = None  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class PointEstimate:
    """Simple estimate container for orthogonal-score calculations."""

    estimate: float
    se: float
    ci_low: float
    ci_high: float


def lambda_fn(t: Tensor, kind: str = "const") -> Tensor:
    """Time-score objective weight ``lambda(t)``."""

    if kind == "const":
        return torch.ones_like(t)
    if kind == "inv":
        eps = 1e-3
        return 1.0 / (t * (1.0 - t) + eps)
    if kind == "bump":
        return 4.0 * t * (1.0 - t)
    raise ValueError(f"Unknown lambda kind: {kind}")


def lambda_prime_fn(t: Tensor, kind: str = "const") -> Tensor:
    """Derivative of ``lambda_fn`` with respect to ``t``."""

    if kind == "const":
        return torch.zeros_like(t)
    if kind == "inv":
        eps = 1e-3
        return -(1.0 - 2.0 * t) / (t * (1.0 - t) + eps) ** 2
    if kind == "bump":
        return 4.0 * (1.0 - 2.0 * t)
    raise ValueError(f"Unknown lambda kind: {kind}")


def _warn_if_unstable_lambda_kind(kind: str) -> None:
    """Warn when a caller opts into the known-unstable ``"inv"`` time weight."""

    if kind == "inv":
        warnings.warn(
            "lambda_kind='inv' becomes very large near t=0 and t=1, so its mini-batch "
            "estimate of the DRE-infinity objective has catastrophic variance and training "
            "diverges under gradient clipping; use 'bump' or 'const'.",
            RuntimeWarning,
            # From this helper the caller's line is 4 frames up: helper -> fit
            # function body -> @_keeps_global_torch_rng wrapper -> caller.
            stacklevel=4,
        )


def interpolate_linear(x_q: Tensor, x_p: Tensor, t: Tensor) -> Tensor:
    """Linear bridge sample ``x_t = (1-t) x_q + t x_p``."""

    return (1.0 - t) * x_q + t * x_p


def time_smr_loss(
    model: TimeScoreNet,
    x_q: Tensor,
    x_p: Tensor,
    *,
    lambda_kind: str = "const",
    create_graph: bool = True,
) -> Tensor:
    """DRE-infinity time-score matching objective.

    Here ``x_q`` is a mini-batch from the numerator distribution ``q`` and ``x_p``
    is a mini-batch from the denominator distribution ``p``. The fitted time score
    integrates to ``log(q/p)``.
    """

    _require_torch()
    batch = x_q.shape[0]
    eps = 1e-4
    t = torch.rand(batch, 1, device=x_q.device, dtype=x_q.dtype) * (1.0 - 2.0 * eps) + eps
    t_req = t.detach().clone().requires_grad_(True)
    x_t = interpolate_linear(x_q, x_p, t).detach()
    s_t = model(x_t, t_req)
    ds_dt = torch.autograd.grad(s_t.sum(), t_req, create_graph=create_graph, retain_graph=True)[0]
    lam = lambda_fn(t, kind=lambda_kind)
    lam_p = lambda_prime_fn(t, kind=lambda_kind)
    interior = 2.0 * lam * ds_dt + 2.0 * lam_p * s_t + lam * s_t.square()

    # Integration-by-parts boundary at the interior truncation endpoints [eps, 1 - eps]:
    # evaluate lambda AND the bridge sample x_s = (1 - s) x_q + s x_p at those times.
    # Using lambda(1), or the raw q/p endpoints for x_s, leaves a linear-in-s residual
    # that makes the objective unbounded below -- the failure this function guards against.
    t0 = torch.zeros(batch, 1, device=x_q.device, dtype=x_q.dtype) + eps
    t1 = torch.ones(batch, 1, device=x_q.device, dtype=x_q.dtype) - eps
    s0 = model(interpolate_linear(x_q, x_p, t0).detach(), t0)
    s1 = model(interpolate_linear(x_q, x_p, t1).detach(), t1)
    boundary = (
        2.0 * lambda_fn(t0, kind=lambda_kind) * s0 - 2.0 * lambda_fn(t1, kind=lambda_kind) * s1
    )
    # t ~ U[eps, 1 - eps], so interior.mean() estimates the average over the interval;
    # scale by its width to recover the integral the boundary term (evaluated at the
    # fixed endpoints, not averaged over t) is stated against.
    return (1.0 - 2.0 * eps) * interior.mean() + boundary.mean()


def joint_smr_loss(
    model: JointScoreNet,
    x_q: Tensor,
    x_p: Tensor,
    *,
    lambda_kind: str = "const",
    data_weight: float = 1.0,
    create_graph: bool = True,
) -> Tensor:
    """Joint time-score and Hyvarinen data-score objective."""

    _require_torch()
    batch = x_q.shape[0]
    eps = 1e-4
    t = torch.rand(batch, 1, device=x_q.device, dtype=x_q.dtype) * (1.0 - 2.0 * eps) + eps
    t_req = t.detach().clone().requires_grad_(True)
    x_t = interpolate_linear(x_q, x_p, t).detach()
    x_t.requires_grad_(True)
    s_x, s_t = model(x_t, t_req)

    ds_dt = torch.autograd.grad(s_t.sum(), t_req, create_graph=create_graph, retain_graph=True)[0]
    lam = lambda_fn(t, kind=lambda_kind)
    lam_p = lambda_prime_fn(t, kind=lambda_kind)
    interior_time = 2.0 * lam * ds_dt + 2.0 * lam_p * s_t + lam * s_t.square()

    # Integration-by-parts boundary at the interior truncation endpoints [eps, 1 - eps]:
    # evaluate lambda AND the bridge sample x_s = (1 - s) x_q + s x_p at those times.
    # Using lambda(1), or the raw q/p endpoints for x_s, leaves a linear-in-s residual
    # that makes the objective unbounded below -- the failure this function guards against.
    t0 = torch.zeros(batch, 1, device=x_q.device, dtype=x_q.dtype) + eps
    t1 = torch.ones(batch, 1, device=x_q.device, dtype=x_q.dtype) - eps
    _, s0 = model(interpolate_linear(x_q, x_p, t0).detach(), t0)
    _, s1 = model(interpolate_linear(x_q, x_p, t1).detach(), t1)
    boundary_time = (
        2.0 * lambda_fn(t0, kind=lambda_kind) * s0 - 2.0 * lambda_fn(t1, kind=lambda_kind) * s1
    )
    # t ~ U[eps, 1 - eps]; scale the averaged interior by the interval width to recover
    # the integral the (un-averaged) boundary term is stated against.
    loss_time = (1.0 - 2.0 * eps) * interior_time.mean() + boundary_time.mean()

    v = torch.randn_like(x_t)
    dot = (s_x * v).sum(dim=1, keepdim=True)
    grad_x = torch.autograd.grad(dot.sum(), x_t, create_graph=create_graph, retain_graph=True)[0]
    trace_est = (grad_x * v).sum(dim=1, keepdim=True)
    hyvarinen = 0.5 * s_x.square().sum(dim=1, keepdim=True) + trace_est
    loss_data = (lam * hyvarinen).mean()
    return loss_time + float(data_weight) * loss_data


@_torch_no_grad()
def _integrate_time_score(
    s_eval: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    *,
    steps: int = 200,
    chunk_size: int = 2048,
) -> Tensor:
    """Compute ``- integral_0^1 s_eval(x,t) dt`` for each row of ``x``."""

    if int(steps) < 2:
        raise ValueError("steps must be at least 2.")
    device = x.device
    t_grid = torch.linspace(0.0, 1.0, int(steps), device=device, dtype=x.dtype)
    dt = float(t_grid[1] - t_grid[0])
    out = torch.empty((x.shape[0],), device=device, dtype=x.dtype)
    for start in range(0, x.shape[0], int(chunk_size)):
        end = min(x.shape[0], start + int(chunk_size))
        xb = x[start:end]
        vals = []
        for ti in t_grid:
            tb = torch.full((xb.shape[0], 1), float(ti), device=device, dtype=x.dtype)
            vals.append(s_eval(xb, tb).view(-1))
        s = torch.stack(vals, dim=0)
        out[start:end] = -dt * (0.5 * s[0] + s[1:-1].sum(dim=0) + 0.5 * s[-1])
    return out


@_torch_no_grad()
def log_ratio_from_time_score(
    model: TimeScoreNet,
    x: Tensor | np.ndarray,
    *,
    steps: int = 200,
    normalize: bool = True,
    x_p_for_norm: Tensor | np.ndarray | None = None,
    chunk_size: int = 2048,
    device=None,
) -> Tensor:
    """Compute ``log(q/p)(x)`` by integrating a trained time-score model."""

    _require_torch()
    if isinstance(x, np.ndarray):
        if device is None:
            device = next(model.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x_t = x
        if device is None:
            device = x_t.device
    log_r = _integrate_time_score(
        lambda xb, tb: model(xb, tb), x_t, steps=steps, chunk_size=chunk_size
    )
    if normalize:
        if x_p_for_norm is None:
            raise ValueError("normalize=True requires x_p_for_norm.")
        if isinstance(x_p_for_norm, np.ndarray):
            xp = torch.tensor(x_p_for_norm, dtype=x_t.dtype, device=x_t.device)
        else:
            xp = x_p_for_norm.to(x_t.device)
        log_r_p = _integrate_time_score(
            lambda xb, tb: model(xb, tb), xp, steps=steps, chunk_size=chunk_size
        )
        log_z = torch.logsumexp(log_r_p, dim=0) - math.log(log_r_p.numel())
        log_r = log_r - log_z
    return log_r.view(-1, 1)


@_torch_no_grad()
def log_ratio_from_joint_time_head(
    model: JointScoreNet,
    x: Tensor | np.ndarray,
    *,
    steps: int = 200,
    normalize: bool = True,
    x_p_for_norm: Tensor | np.ndarray | None = None,
    chunk_size: int = 2048,
    device=None,
) -> Tensor:
    """Compute ``log(q/p)(x)`` using the time head of a JointScoreNet."""

    _require_torch()
    if isinstance(x, np.ndarray):
        if device is None:
            device = next(model.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x_t = x
    def s_eval(xb: Tensor, tb: Tensor) -> Tensor:
        _, st = model(xb, tb)
        return st
    log_r = _integrate_time_score(s_eval, x_t, steps=steps, chunk_size=chunk_size)
    if normalize:
        if x_p_for_norm is None:
            raise ValueError("normalize=True requires x_p_for_norm.")
        if isinstance(x_p_for_norm, np.ndarray):
            xp = torch.tensor(x_p_for_norm, dtype=x_t.dtype, device=x_t.device)
        else:
            xp = x_p_for_norm.to(x_t.device)
        log_r_p = _integrate_time_score(s_eval, xp, steps=steps, chunk_size=chunk_size)
        log_z = torch.logsumexp(log_r_p, dim=0) - math.log(log_r_p.numel())
        log_r = log_r - log_z
    return log_r.view(-1, 1)


def _rand_batch(x: Tensor, batch_size: int) -> Tensor:
    idx = torch.randint(0, x.shape[0], (int(batch_size),), device=x.device)
    return x[idx]


@_keeps_global_torch_rng
def fit_time_smr_dre_infinity(
    x_q: np.ndarray,
    x_p: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
    t_emb_dim: int = 32,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 256,
    n_steps: int = 4000,
    grad_clip: float | None = 1.0,
    lambda_kind: str = "bump",
    layer_norm: bool = False,
    seed: int = 0,
    device=None,
) -> TimeScoreNet:
    """Fit Time-ScoreMatchingRiesz for a density ratio ``q/p``.

    ``lambda_kind`` sets the time weight ``lambda(t)``. ``"bump"`` (default,
    ``4 t (1 - t)``) and ``"const"`` stay small, so their mini-batch estimates are well
    behaved. ``"inv"`` (``1 / (t (1 - t) + 1e-3)``) reaches ~1e3 near ``t = 0`` and
    ``t = 1``: the objective itself is still bounded below, but its mini-batch estimate
    has catastrophic variance and training diverges under gradient clipping. It is kept
    only for experimentation and emits a warning.
    """

    _require_torch()
    _warn_if_unstable_lambda_kind(lambda_kind)
    if device is None:
        device = get_device()
    _seed_torch(seed)
    xq = torch.tensor(_as_2d_float_array(x_q, name="x_q"), device=device)
    xp = torch.tensor(_as_2d_float_array(x_p, name="x_p"), device=device)
    if xq.shape[1] != xp.shape[1]:
        raise ValueError("x_q and x_p must have the same number of columns.")
    model = TimeScoreNet(
        x_dim=xq.shape[1], hidden_dims=hidden_dims, t_emb_dim=t_emb_dim, layer_norm=layer_norm
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    history: list[float] = []
    for _ in range(int(n_steps)):
        loss = time_smr_loss(
            model,
            _rand_batch(xq, batch_size),
            _rand_batch(xp, batch_size),
            lambda_kind=lambda_kind,
            create_graph=True,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        history.append(float(loss.item()))
    model.eval()
    model.loss_history = history
    return model


@_keeps_global_torch_rng
def fit_joint_smr_dre_infinity(
    x_q: np.ndarray,
    x_p: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
    t_emb_dim: int = 32,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 256,
    n_steps: int = 4000,
    grad_clip: float | None = 1.0,
    lambda_kind: str = "bump",
    data_weight: float = 1.0,
    layer_norm: bool = False,
    seed: int = 0,
    device=None,
) -> JointScoreNet:
    """Fit Joint-ScoreMatchingRiesz for a density ratio ``q/p``.

    See :func:`fit_time_smr_dre_infinity` for ``lambda_kind`` (default ``"bump"``;
    ``"inv"`` is unstable and emits a warning).
    """

    _require_torch()
    _warn_if_unstable_lambda_kind(lambda_kind)
    if device is None:
        device = get_device()
    _seed_torch(seed)
    xq = torch.tensor(_as_2d_float_array(x_q, name="x_q"), device=device)
    xp = torch.tensor(_as_2d_float_array(x_p, name="x_p"), device=device)
    if xq.shape[1] != xp.shape[1]:
        raise ValueError("x_q and x_p must have the same number of columns.")
    model = JointScoreNet(
        x_dim=xq.shape[1], hidden_dims=hidden_dims, t_emb_dim=t_emb_dim, layer_norm=layer_norm
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    history: list[float] = []
    for _ in range(int(n_steps)):
        loss = joint_smr_loss(
            model,
            _rand_batch(xq, batch_size),
            _rand_batch(xp, batch_size),
            lambda_kind=lambda_kind,
            data_weight=data_weight,
            create_graph=True,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        history.append(float(loss.item()))
    model.eval()
    model.loss_history = history
    return model


@_keeps_global_torch_rng
def fit_data_smr_score_dsm(
    x: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 256,
    n_steps: int = 4000,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    treatment_index: int = 0,
    grad_clip: float | None = 1.0,
    layer_norm: bool = False,
    seed: int = 0,
    device=None,
) -> DataScoreDNet:
    """Fit a DSM data-score model for one coordinate of ``x``.

    The returned network estimates ``partial_{coordinate} log p_sigma(x)``.
    """

    _require_torch()
    if device is None:
        device = get_device()
    _seed_torch(seed)
    x_np = _as_2d_float_array(x, name="x")
    t_idx = int(treatment_index)
    if t_idx < 0 or t_idx >= x_np.shape[1]:
        raise ValueError("treatment_index out of bounds.")
    xt = torch.tensor(x_np, dtype=torch.float32, device=device)
    model = DataScoreDNet(
        x_dim=xt.shape[1], hidden_dims=hidden_dims, layer_norm=layer_norm
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    history: list[float] = []
    for _ in range(int(n_steps)):
        xb = _rand_batch(xt, batch_size)
        coord = xb[:, t_idx : t_idx + 1]
        before = xb[:, :t_idx]
        after = xb[:, t_idx + 1 :]
        u = torch.rand(int(batch_size), 1, device=device)
        sigma = float(sigma_min) * (float(sigma_max) / float(sigma_min)) ** u
        eps = torch.randn_like(coord)
        coord_tilde = coord + sigma * eps
        x_tilde = torch.cat([before, coord_tilde, after], dim=1)
        s_hat = model(x_tilde, sigma)
        loss = ((sigma * s_hat + eps) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        history.append(float(loss.item()))
    model.eval()
    model.loss_history = history
    model.treatment_index = t_idx
    return model


@_torch_no_grad()
def eval_data_score_d(
    model: DataScoreDNet,
    x: np.ndarray,
    *,
    sigma_eval: float = 0.01,
    device=None,
) -> np.ndarray:
    """Evaluate a fitted one-coordinate data-score model."""

    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    xt = torch.tensor(_as_2d_float_array(x, name="x"), device=device)
    sigma = torch.full((xt.shape[0], 1), float(sigma_eval), device=device, dtype=torch.float32)
    return model(xt, sigma).detach().cpu().numpy()


@_torch_no_grad()
def log_ratio_from_data_score_shift(
    model: DataScoreDNet,
    x: np.ndarray,
    delta: float,
    *,
    steps: int = 64,
    sigma_eval: float = 0.01,
    direction: str = "+",
    treatment_index: int | None = None,
    normalize: bool = False,
    x_p_for_norm: np.ndarray | None = None,
    norm_subsample: int | None = 4096,
    device=None,
) -> np.ndarray:
    """Construct a shift-policy log ratio from a data-score model.

    ``direction='+'`` returns ``log p_{+delta}(x)/p0(x)`` and ``direction='-'``
    returns ``log p_{-delta}(x)/p0(x)`` for a translation in ``treatment_index``.
    """

    _require_torch()
    if direction not in {"+", "-"}:
        raise ValueError("direction must be '+' or '-'.")
    if device is None:
        device = next(model.parameters()).device
    x_np = _as_2d_float_array(x, name="x")
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_idx = int(
        getattr(model, "treatment_index", 0) if treatment_index is None else treatment_index
    )
    if int(steps) <= 1 or abs(float(delta)) < 1e-15:
        log_r = torch.zeros((x_tensor.shape[0],), device=device, dtype=torch.float32)
    else:
        grid = torch.linspace(0.0, float(delta), int(steps), device=device, dtype=torch.float32)
        x_rep = (
            x_tensor.unsqueeze(0).expand(int(steps), x_tensor.shape[0], x_tensor.shape[1]).clone()
        )
        if direction == "+":
            x_rep[:, :, t_idx] = x_rep[:, :, t_idx] - grid[:, None]
            sign = -1.0
        else:
            x_rep[:, :, t_idx] = x_rep[:, :, t_idx] + grid[:, None]
            sign = +1.0
        flat = x_rep.reshape(int(steps) * x_tensor.shape[0], x_tensor.shape[1])
        sigma = torch.full(
            (flat.shape[0], 1), float(sigma_eval), device=device, dtype=torch.float32
        )
        scores = model(flat, sigma).reshape(int(steps), x_tensor.shape[0])
        log_r = torch.trapezoid(sign * scores, grid, dim=0)
    if normalize:
        if x_p_for_norm is None:
            raise ValueError("normalize=True requires x_p_for_norm.")
        xp = x_p_for_norm
        if norm_subsample is not None and xp.shape[0] > int(norm_subsample):
            # Evenly strided subsample: deterministic and unbiased for sorted
            # inputs (taking the first N rows is biased when x_p is ordered).
            idx = np.linspace(0, xp.shape[0] - 1, int(norm_subsample)).astype(int)
            xp = xp[idx]
        log_r_p = log_ratio_from_data_score_shift(
            model,
            xp,
            delta,
            steps=steps,
            sigma_eval=sigma_eval,
            direction=direction,
            treatment_index=t_idx,
            normalize=False,
            device=device,
        ).reshape(-1)
        m = float(np.max(log_r_p))
        log_z = m + math.log(float(np.mean(np.exp(log_r_p - m))))
        log_r = log_r - float(log_z)
    return log_r.detach().cpu().numpy().reshape(-1, 1)


@_keeps_global_torch_rng
def fit_sq_riesz_ame(
    x: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 256,
    n_steps: int = 4000,
    treatment_index: int = 0,
    grad_clip: float | None = 1.0,
    layer_norm: bool = False,
    seed: int = 0,
    device=None,
) -> ScalarNet:
    """Fit the squared-loss Riesz-regression baseline for AME."""

    _require_torch()
    if device is None:
        device = get_device()
    _seed_torch(seed)
    xt = torch.tensor(_as_2d_float_array(x, name="x"), dtype=torch.float32, device=device)
    model = ScalarNet(x_dim=xt.shape[1], hidden_dims=hidden_dims, layer_norm=layer_norm).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    t_idx = int(treatment_index)
    history: list[float] = []
    for _ in range(int(n_steps)):
        xb = _rand_batch(xt, batch_size).detach().clone().requires_grad_(True)
        alpha = model(xb)
        grad_x = torch.autograd.grad(alpha.sum(), xb, create_graph=True, retain_graph=True)[0]
        d_alpha = grad_x[:, t_idx : t_idx + 1]
        loss = (alpha.square() - 2.0 * d_alpha).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        history.append(float(loss.item()))
    model.eval()
    model.loss_history = history
    return model


@_torch_no_grad()
def eval_scalar_net(model: nn.Module, x: np.ndarray, *, device=None) -> np.ndarray:
    """Evaluate a scalar PyTorch model on NumPy inputs."""

    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    xt = torch.tensor(_as_2d_float_array(x, name="x"), dtype=torch.float32, device=device)
    return model(xt).detach().cpu().numpy()


@_keeps_global_torch_rng
def _fit_ratio_template(
    x_q: np.ndarray,
    x_p: np.ndarray,
    objective: str,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 256,
    n_steps: int = 4000,
    grad_clip: float | None = 1.0,
    layer_norm: bool = False,
    seed: int = 0,
    device=None,
) -> RatioNet:
    _require_torch()
    if device is None:
        device = get_device()
    _seed_torch(seed)
    xq = torch.tensor(_as_2d_float_array(x_q, name="x_q"), device=device)
    xp = torch.tensor(_as_2d_float_array(x_p, name="x_p"), device=device)
    if xq.shape[1] != xp.shape[1]:
        raise ValueError("x_q and x_p must have the same number of columns.")
    model = RatioNet(x_dim=xq.shape[1], hidden_dims=hidden_dims, layer_norm=layer_norm).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    history: list[float] = []
    for _ in range(int(n_steps)):
        bq = _rand_batch(xq, batch_size)
        bp = _rand_batch(xp, batch_size)
        if objective == "sq":
            r_q = F.softplus(model(bq))
            r_p = F.softplus(model(bp))
            loss = 0.5 * r_p.square().mean() - r_q.mean()
        elif objective == "ukl":
            f_q = model(bq)
            f_p = model(bp)
            log_z = torch.logsumexp(f_p, dim=0) - math.log(f_p.shape[0])
            loss = -(f_q - log_z).mean()
        elif objective == "bkl":
            f_q = model(bq)
            f_p = model(bp)
            loss = F.softplus(-f_q).mean() + F.softplus(f_p).mean()
        else:  # pragma: no cover
            raise ValueError(objective)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        history.append(float(loss.item()))
    model.eval()
    model.loss_history = history
    model.objective = objective
    return model


def fit_sq_riesz_ratio(x_q: np.ndarray, x_p: np.ndarray, **kwargs) -> RatioNet:
    """Fit a nonnegative squared-loss density-ratio model for ``q/p``."""

    return _fit_ratio_template(x_q, x_p, "sq", **kwargs)


def fit_ukl_riesz_ratio(x_q: np.ndarray, x_p: np.ndarray, **kwargs) -> RatioNet:
    """Fit a KLIEP-style UKL density-ratio model for ``q/p``."""

    return _fit_ratio_template(x_q, x_p, "ukl", **kwargs)


def fit_bkl_riesz_ratio(x_q: np.ndarray, x_p: np.ndarray, **kwargs) -> RatioNet:
    """Fit a logistic-classification density-ratio model for ``q/p``."""

    return _fit_ratio_template(x_q, x_p, "bkl", **kwargs)


@_torch_no_grad()
def eval_ratio_sq(
    model: RatioNet,
    x: np.ndarray,
    *,
    normalize: bool = True,
    x_p_for_norm: np.ndarray | None = None,
    device=None,
) -> np.ndarray:
    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    xt = torch.tensor(_as_2d_float_array(x, name="x"), dtype=torch.float32, device=device)
    r = F.softplus(model(xt))
    if normalize:
        if x_p_for_norm is None:
            raise ValueError("normalize=True requires x_p_for_norm.")
        xp = torch.tensor(
            _as_2d_float_array(x_p_for_norm, name="x_p_for_norm"),
            dtype=torch.float32,
            device=device,
        )
        r = r / (F.softplus(model(xp)).mean() + 1e-8)
    return r.detach().cpu().numpy()


@_torch_no_grad()
def eval_ratio_ukl(
    model: RatioNet, x: np.ndarray, x_p_for_norm: np.ndarray, *, device=None
) -> np.ndarray:
    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    xt = torch.tensor(_as_2d_float_array(x, name="x"), dtype=torch.float32, device=device)
    xp = torch.tensor(
        _as_2d_float_array(x_p_for_norm, name="x_p_for_norm"), dtype=torch.float32, device=device
    )
    f = model(xt)
    fp = model(xp)
    log_z = torch.logsumexp(fp, dim=0) - math.log(fp.shape[0])
    return torch.exp(f - log_z).detach().cpu().numpy()


@_torch_no_grad()
def eval_ratio_bkl(
    model: RatioNet,
    x: np.ndarray,
    *,
    normalize: bool = True,
    x_p_for_norm: np.ndarray | None = None,
    device=None,
) -> np.ndarray:
    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    xt = torch.tensor(_as_2d_float_array(x, name="x"), dtype=torch.float32, device=device)
    r = torch.exp(torch.clamp(model(xt), min=-30.0, max=30.0))
    if normalize:
        if x_p_for_norm is None:
            raise ValueError("normalize=True requires x_p_for_norm.")
        xp = torch.tensor(
            _as_2d_float_array(x_p_for_norm, name="x_p_for_norm"),
            dtype=torch.float32,
            device=device,
        )
        rp = torch.exp(torch.clamp(model(xp), min=-30.0, max=30.0))
        r = r / (rp.mean() + 1e-8)
    return r.detach().cpu().numpy()


@_keeps_global_torch_rng
def fit_outcome_net(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 256,
    n_epochs: int = 200,
    grad_clip: float | None = 1.0,
    layer_norm: bool = False,
    seed: int = 0,
    device=None,
) -> OutcomeNet:
    """Fit a neural outcome regression model by squared loss."""

    _require_torch()
    if device is None:
        device = get_device()
    _seed_torch(seed)
    x_np = _as_2d_float_array(x, name="x")
    y_np = _as_1d_float_array(y, n=x_np.shape[0], name="y")
    xt = torch.tensor(x_np, dtype=torch.float32, device=device)
    yt = torch.tensor(y_np, dtype=torch.float32, device=device).view(-1, 1)
    model = OutcomeNet(x_dim=xt.shape[1], hidden_dims=hidden_dims, layer_norm=layer_norm).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    steps_per_epoch = max(1, xt.shape[0] // int(batch_size))
    history: list[float] = []
    for _ in range(int(n_epochs)):
        perm = torch.randperm(xt.shape[0], device=device)
        for step in range(steps_per_epoch):
            idx = perm[step * int(batch_size) : (step + 1) * int(batch_size)]
            pred = model(xt[idx])
            loss = F.mse_loss(pred, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            history.append(float(loss.item()))
    model.eval()
    model.loss_history = history
    return model


@_torch_no_grad()
def predict_outcome(model: OutcomeNet, x: np.ndarray, *, device=None) -> np.ndarray:
    """Evaluate an outcome model on NumPy inputs."""

    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    xt = torch.tensor(_as_2d_float_array(x, name="x"), dtype=torch.float32, device=device)
    return model(xt).detach().cpu().numpy()


def partial_d_outcome(
    model: OutcomeNet, x: np.ndarray, *, coordinate: int = 0, device=None
) -> np.ndarray:
    """Compute a partial derivative of an outcome model by autograd."""

    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    xt = torch.tensor(_as_2d_float_array(x, name="x"), dtype=torch.float32, device=device)
    xt.requires_grad_(True)
    y = model(xt)
    grad = torch.autograd.grad(y.sum(), xt, create_graph=False, retain_graph=False)[0]
    return grad[:, int(coordinate) : int(coordinate) + 1].detach().cpu().numpy()


def wald_interval(score_values: np.ndarray, *, alpha: float = 0.05) -> PointEstimate:
    """Return mean, standard error, and a normal Wald interval for score values."""

    vals = np.asarray(score_values, dtype=float).reshape(-1)
    if vals.size < 2:
        raise ValueError("At least two score values are required.")
    estimate = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(vals.size))
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    return PointEstimate(
        estimate=estimate, se=se, ci_low=estimate - z * se, ci_high=estimate + z * se
    )


def crossfit_splits(
    n: int,
    *,
    n_folds: int = 2,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test index pairs for cross-fitting.

    A thin adapter over :func:`genriesz.utils.kfold_splits` that keeps this
    module's historical ``list[(train, test)]`` shape; the splits themselves are
    identical fold for fold.

    The bounds are re-checked here rather than deferred to ``kfold_splits`` so
    that the message names ``n_folds``, the argument the caller actually passed.
    ``shuffle`` is passed explicitly for the same reason: this function has
    always shuffled, and that must not silently hinge on the delegate's default.
    """

    n = int(n)
    n_folds = int(n_folds)
    if n < 2:
        raise ValueError("n must be at least 2.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > n:
        raise ValueError("n_folds must be at most n.")
    return [
        (f.train, f.test)
        for f in kfold_splits(n, folds=n_folds, random_state=seed, shuffle=True)
    ]


__all__ = [
    "PointEstimate",
    "get_device",
    "set_seed",
    "MLP",
    "TimeEmbedding",
    "TimeScoreNet",
    "JointScoreNet",
    "DataScoreDNet",
    "ScalarNet",
    "RatioNet",
    "OutcomeNet",
    "lambda_fn",
    "lambda_prime_fn",
    "interpolate_linear",
    "time_smr_loss",
    "joint_smr_loss",
    "log_ratio_from_time_score",
    "log_ratio_from_joint_time_head",
    "fit_time_smr_dre_infinity",
    "fit_joint_smr_dre_infinity",
    "fit_data_smr_score_dsm",
    "eval_data_score_d",
    "log_ratio_from_data_score_shift",
    "fit_sq_riesz_ame",
    "eval_scalar_net",
    "fit_sq_riesz_ratio",
    "fit_ukl_riesz_ratio",
    "fit_bkl_riesz_ratio",
    "eval_ratio_sq",
    "eval_ratio_ukl",
    "eval_ratio_bkl",
    "fit_outcome_net",
    "predict_outcome",
    "partial_d_outcome",
    "wald_interval",
    "crossfit_splits",
]
