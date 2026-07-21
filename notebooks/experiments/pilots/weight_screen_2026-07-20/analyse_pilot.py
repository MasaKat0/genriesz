# ruff: noqa -- 実行当時のまま保存した実測スクリプト（README.md 参照）
"""Read the weight-screen pilot and answer Sections 18.7 and 18.8."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path("notebooks/experiments/results/weight_screen_pilot")
PRIMARY = dict(rule="proposed", reference="correct", allowance_scale=1.0)


def label(threshold: float) -> str:
    return "none" if threshold < 0 else f"{threshold:g}"


def main() -> None:
    pd.set_option("display.width", 200)
    cand = pd.read_parquet(OUT / "candidate.parquet")
    rep = pd.read_parquet(OUT / "repetition.parquet")
    sel = pd.read_parquet(OUT / "selection.parquet")
    orc = pd.read_parquet(OUT / "oracle.parquet")
    for frame in (cand, rep, sel, orc):
        frame["screen"] = frame.min_ess_ratio.map(label)

    print("=" * 78)
    print("A. Admissible set and simultaneous radius (diagnostic fold)")
    print("=" * 78)
    adm = cand[cand.admissible]
    per_fold = (
        adm.groupby(["screen", "target_t", "repetition", "fold"])
        .agg(n_adm=("candidate", "size"), q_med=("diagnostic_radius", "median"))
        .reset_index()
    )
    print(
        per_fold.groupby(["target_t", "screen"])
        .agg(n_admissible=("n_adm", "mean"), median_radius=("q_med", "mean"))
        .round(4)
        .to_string()
    )

    print()
    print("=" * 78)
    print("B. Section 18.7: bias-aware conservatism (proposed / correct / rho=1)")
    print("=" * 78)
    r = rep.merge(pd.DataFrame([PRIMARY]), on=list(PRIMARY), how="inner")
    r = r[r.split_available]
    r["u_over_se"] = r.split_bias_bound / r.split_se
    r["ba_over_wald"] = r.bias_aware_split_length / r.wald_split_length
    summary = (
        r.groupby(["target_t", "screen"])
        .agg(
            n=("split_theta", "size"),
            wald_cov=("wald_split_covers", "mean"),
            ba_cov=("bias_aware_split_covers", "mean"),
            u_over_se=("u_over_se", "median"),
            ba_over_wald=("ba_over_wald", "median"),
            ba_length=("bias_aware_split_length", "median"),
        )
        .round(4)
    )
    print(summary.to_string())

    print()
    print("cross-fit intervals (unconditional coverage over all replications):")
    rc = rep.merge(pd.DataFrame([PRIMARY]), on=list(PRIMARY), how="inner")
    print(
        rc.groupby(["target_t", "screen"])
        .agg(
            wald_cf=("wald_cf_covers", "mean"),
            cons_cf=("conservative_cf_covers", "mean"),
            cons_len=("conservative_cf_length", "median"),
            wald_len=("wald_cf_length", "median"),
        )
        .round(4)
        .to_string()
    )

    print()
    print("=" * 78)
    print("C. Section 18.8: oracle regret by rule (fold level)")
    print("=" * 78)
    o = orc[(orc.reference == "correct") & (orc.allowance_scale == 1.0)]
    o = o[o.rule.isin(["proposed", "score_var", "bregman_cv", "lsif_cv"])]
    stat = (
        o.groupby(["target_t", "rule", "screen"])
        .oracle_regret.agg(
            n="size",
            median="median",
            p90=lambda x: x.quantile(0.90),
            p99=lambda x: x.quantile(0.99),
            max="max",
        )
        .round(4)
    )
    print(stat.to_string())

    print()
    print("=" * 78)
    print("D. What the screen removed, and whether it ever bit the good candidates")
    print("=" * 78)
    s = sel.merge(pd.DataFrame([PRIMARY]), on=list(PRIMARY), how="inner")
    print(
        s.groupby(["target_t", "screen"])
        .agg(n_ranked=("n_ranked", "mean"), available=("available", "mean"))
        .round(3)
        .to_string()
    )
    print()
    print("candidates chosen by `proposed`, by screen (t=4, top 6):")
    top = (
        s[s.target_t == 4.0]
        .groupby(["screen", "candidate"])
        .size()
        .rename("folds")
        .reset_index()
        .sort_values(["screen", "folds"], ascending=[True, False])
    )
    for screen, group in top.groupby("screen"):
        print(f"  screen={screen}: " + ", ".join(
            f"{row.candidate}({row.folds})" for row in group.head(6).itertuples()
        ))

    print()
    print("=" * 78)
    print("E. Did any fold lose every candidate?")
    print("=" * 78)
    counts = per_fold.groupby(["screen", "target_t"]).n_adm.min()
    print(counts.to_string())
    print()
    print("ESS ratio of the candidate `proposed` selected, by screen (t=4):")
    merged = s[s.target_t == 4.0].merge(
        cand[["screen", "target_t", "repetition", "fold", "candidate", "ess_ratio", "audit_risk"]],
        on=["screen", "target_t", "repetition", "fold", "candidate"],
        how="left",
        suffixes=("", "_c"),
    )
    print(
        merged.groupby("screen")
        .agg(
            ess_min=("ess_ratio", "min"),
            ess_p05=("ess_ratio", lambda x: np.nanquantile(x, 0.05)),
            ess_med=("ess_ratio", "median"),
        )
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
