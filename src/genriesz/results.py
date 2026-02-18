"""Result containers.

We keep results as small dataclasses so they can be printed nicely and also
inspected programmatically from notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingleEstimate:
    """A single point estimate with asymptotic (normal) inference."""

    name: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class FunctionalEstimate:
    """Container for estimates for the same estimand."""

    n: int
    alpha: float
    null: float
    estimand: str
    estimates: dict[str, SingleEstimate]
    diagnostics: dict[str, object]

    def love_plot_data(self, *, as_pandas: bool | None = None):
        """Return covariate-balance diagnostics used by :meth:`love_plot`.

        This method is available only when the underlying estimand is of
        treatment-effect type (ATE / ATT / DID). The returned object contains
        standardized mean differences (SMDs) *before* and *after* weighting.

        Parameters
        ----------
        as_pandas:
            If True, return a ``pandas.DataFrame``. If False, return a list of
            dictionaries. If None, return a DataFrame when pandas is available.
        """

        love = self.diagnostics.get("love_plot")
        if love is None:
            raise RuntimeError(
                "Love-plot diagnostics are not available for this result. "
                "They are computed only for treatment-effect functionals (ATE/ATT/DID)."
            )

        names = list(love["covariate_names"])
        smd_u = np.asarray(love["smd_unweighted"], dtype=float)
        smd_w = np.asarray(love["smd_weighted"], dtype=float)

        rows = []
        for nm, u, w in zip(names, smd_u, smd_w):
            rows.append(
                {
                    "covariate": str(nm),
                    "smd_unweighted": float(u),
                    "smd_weighted": float(w),
                    "abs_smd_unweighted": float(abs(u)),
                    "abs_smd_weighted": float(abs(w)),
                }
            )

        if as_pandas is False:
            return rows

        if as_pandas is True:
            import pandas as pd  # type: ignore

            return pd.DataFrame(rows)

        # as_pandas is None: prefer pandas when installed.
        try:
            import pandas as pd  # type: ignore

            return pd.DataFrame(rows)
        except Exception:
            return rows

    def love_plot(
        self,
        *,
        threshold: float = 0.1,
        max_covariates: int | None = 30,
        sort_by: str = "weighted",
        absolute: bool = True,
        ax=None,
    ):
        """Create a Love plot (covariate-balance plot).

        The plot compares standardized mean differences (SMDs) before vs. after
        weighting induced by the estimated Riesz representer.

        Notes
        -----
        - Requires ``matplotlib`` (not a hard dependency).
        - Only available for ATE / ATT / DID.
        """

        love = self.diagnostics.get("love_plot")
        if love is None:
            raise RuntimeError(
                "Love-plot diagnostics are not available for this result. "
                "They are computed only for treatment-effect functionals (ATE/ATT/DID)."
            )

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as err:  # pragma: no cover
            raise ImportError(
                "matplotlib is required for love_plot(). Install it via `pip install matplotlib`."
            ) from err

        names = list(love["covariate_names"])
        if absolute:
            x_u = np.asarray(love["abs_smd_unweighted"], dtype=float)
            x_w = np.asarray(love["abs_smd_weighted"], dtype=float)
            x_label = "Absolute standardized mean difference"
        else:
            x_u = np.asarray(love["smd_unweighted"], dtype=float)
            x_w = np.asarray(love["smd_weighted"], dtype=float)
            x_label = "Standardized mean difference"

        if str(sort_by).lower() == "unweighted":
            order = np.argsort(np.nan_to_num(np.abs(x_u), nan=-np.inf))[::-1]
        else:
            order = np.argsort(np.nan_to_num(np.abs(x_w), nan=-np.inf))[::-1]

        if max_covariates is not None:
            order = order[: int(max_covariates)]

        y = np.arange(len(order))
        if ax is None:
            fig_h = max(2.0, 0.25 * len(order) + 1.0)
            fig, ax = plt.subplots(figsize=(7.0, fig_h))
        else:
            fig = ax.figure

        ax.scatter(x_u[order], y, label="Unweighted")
        ax.scatter(x_w[order], y, label="Weighted")

        if absolute:
            ax.axvline(float(threshold), linestyle="--", linewidth=1)
        else:
            ax.axvline(float(threshold), linestyle="--", linewidth=1)
            ax.axvline(-float(threshold), linestyle="--", linewidth=1)
            ax.axvline(0.0, linestyle=":", linewidth=1)

        ax.set_yticks(y)
        ax.set_yticklabels([names[i] for i in order])
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_title(f"Love plot ({self.estimand})")
        ax.legend(loc="best")
        return fig, ax

    def summary_text(self) -> str:
        lines: list[str] = []
        lines.append(f"{self.estimand} estimates (n={self.n})")
        lines.append(f"alpha={self.alpha} | null={self.null}")
        if self.diagnostics:
            # Keep the summary readable: print only scalar diagnostics.
            diag_parts: list[str] = []
            for k, v in self.diagnostics.items():
                if k == "love_plot":
                    continue
                if isinstance(v, (int, float, str, bool)) or v is None:
                    diag_parts.append(f"{k}={v}")
            if diag_parts:
                lines.append("diagnostics: " + ", ".join(diag_parts))
        lines.append("")

        header = f"{'Estimator':<12}  {'Estimate':>12}  {'SE':>12}  {'CI':>27}  {'p-value':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        # Preferred display order
        order = ["ra", "rw", "arw", "tmle"]
        for key in order:
            if key not in self.estimates:
                continue
            e = self.estimates[key]
            ci = f"[{e.ci_low: .6g}, {e.ci_high: .6g}]"
            lines.append(
                f"{e.name:<12}  {e.estimate:>12.6g}  {e.se:>12.6g}  {ci:>27}  {e.p_value:>10.3g}"
            )

        # Any extra keys (e.g., arw_shared/arw_separate) come last
        for key in sorted(self.estimates.keys()):
            if key in order:
                continue
            e = self.estimates[key]
            ci = f"[{e.ci_low: .6g}, {e.ci_high: .6g}]"
            lines.append(
                f"{e.name:<12}  {e.estimate:>12.6g}  {e.se:>12.6g}  {ci:>27}  {e.p_value:>10.3g}"
            )

        return "\n".join(lines)
