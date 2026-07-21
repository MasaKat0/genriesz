# ruff: noqa -- 実測スクリプトの保存版（README.md 参照）
import sys, pandas as pd, numpy as np, pathlib
sys.path.insert(0, str(pathlib.Path("notebooks/experiments").resolve()))
sys.path.insert(0, "src")
OUT = pathlib.Path("notebooks/experiments/results/weight_screen_pilot")
orc = pd.read_parquet(OUT/"oracle.parquet")
cand = pd.read_parquet(OUT/"candidate.parquet")
rep = pd.read_parquet(OUT/"repetition.parquet")
lab = lambda v: "none" if v < 0 else f"{v:g}"
for f in (orc, cand, rep): f["screen"] = f.min_ess_ratio.map(lab)

print("=== 1. Did the screen raise the attainable oracle risk itself? (t=4, fold-level mean/max) ===")
o = orc[(orc.rule=="proposed")&(orc.reference=="correct")&(orc.allowance_scale==1.0)&(orc.target_t==4.0)]
print(o.groupby("screen").oracle_audit_risk.agg(["mean","median","max"]).round(4).to_string())

print()
print("=== 2. Fragility: folds with regret > 1 at t=4, by screen (which candidate, its ESS) ===")
big = o[o.oracle_regret > 1.0]
sel = pd.read_parquet(OUT/"selection.parquet"); sel["screen"] = sel.min_ess_ratio.map(lab)
s4 = sel[(sel.rule=="proposed")&(sel.reference=="correct")&(sel.allowance_scale==1.0)&(sel.target_t==4.0)]
m = big.merge(s4[["screen","repetition","fold","candidate"]], on=["screen","repetition","fold"], how="left")
m = m.merge(cand[cand.target_t==4.0][["screen","repetition","fold","candidate","ess_ratio"]],
            on=["screen","repetition","fold","candidate"], how="left")
print(m[["screen","repetition","fold","candidate","ess_ratio","oracle_regret"]]
      .sort_values(["screen","oracle_regret"]).to_string(index=False))

print()
print("=== 3. Other rules in oracle table (what reference label do they carry?) ===")
print(orc[orc.target_t==4.0].groupby(["rule","reference"]).size().head(20).to_string())

print()
print("=== 4. Excess length decomposition (proposed/correct/rho=1, split interval) ===")
# The floor is the same bounded-normal interval class with the bias bound known
# exactly (U = t*se, q_a = b_r = 0), at the interval's own nominal level
# 1-(tau-delta) = 0.96. For large t the binding tail is one-sided, so
# cv(t) < t + z_{0.975}: a "1 + t/z" floor would overstate the irreducible part.
from refsel.inference import bias_aware_critical_value
r = rep[(rep.rule=="proposed")&(rep.reference=="correct")&(rep.allowance_scale==1.0)&(rep.split_available)]
r = r.copy(); r["ratio"] = r.bias_aware_split_length/r.wald_split_length
z = 1.959963984540054
for t in (0.0, 4.0):
    floor = bias_aware_critical_value(t, 0.96) / z
    med = r[r.target_t==t].groupby("screen").ratio.median()
    print(f"t={t}: floor cv_0.96(t)/z_0.975={floor:.3f} | measured BA/Wald median: "
          + ", ".join(f"{k}={v:.2f}(premium {v/floor:.2f}x)" for k,v in med.items()))
