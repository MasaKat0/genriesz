# ruff: noqa -- 実行当時のまま保存した実測スクリプト（README.md 参照）
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("notebooks/experiments").resolve()))
sys.path.insert(0, "src")
import numpy as np
from refsel.dgp import generate_data
from refsel.selection import effective_sample_ratio
from refsel.calibration import load_calibration, calibration_key

# Does the hidden (bias-calibration) term move alpha0's ESS? (grid B, s=1.5, t=4)
hs = load_calibration()[calibration_key("low", 3000, 1.5, 4.0)]
for label, h in [("hs=0", 0.0), (f"hs={hs:.3f} (t=4)", hs)]:
    vals = [effective_sample_ratio(
        generate_data(n=600, design="low", overlap_scale=1.5, hidden_scale=h,
                      seed=2000+s).alpha0[:, None])[0] for s in range(40)]
    v = np.array(vals)
    print(f"{label:<22} ESS(alpha0): p05={np.quantile(v,0.05):.3f} med={np.median(v):.3f}")
