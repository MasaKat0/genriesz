# ruff: noqa -- 実行当時のまま保存した実測スクリプト（README.md 参照）
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("notebooks/experiments").resolve()))
sys.path.insert(0, "src")
import numpy as np
from refsel.dgp import generate_data
from refsel.selection import effective_sample_ratio

# ESS ratio of the TRUE representer alpha0 on diagnostic-fold-sized samples.
# If a screen threshold exceeds this, the screen excludes correct candidates.
print(f"{'design':<6} {'s':>5} {'t/hs':>5} {'n_diag':>7}  ESS(alpha0): p05 / median / p95")
for design, scales, n in [("low", (0.5, 1.5, 2.5), 3000), ("high", (0.75, 2.0), 3000)]:
    for s in scales:
        vals = []
        for seed in range(40):
            d = generate_data(n=n // 5, design=design, overlap_scale=s,
                              hidden_scale=0.0, seed=1000 + seed)
            vals.append(effective_sample_ratio(d.alpha0[:, None])[0])
        vals = np.array(vals)
        print(f"{design:<6} {s:>5} {0.0:>5} {n//5:>7}  "
              f"{np.quantile(vals, 0.05):.3f} / {np.median(vals):.3f} / {np.quantile(vals, 0.95):.3f}")
