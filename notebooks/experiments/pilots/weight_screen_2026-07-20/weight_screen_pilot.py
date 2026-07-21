# ruff: noqa -- 実行当時のまま保存した実測スクリプト（README.md 参照）
"""Pilot: does a pre-specified weight screen fix Sections 18.7 and 18.8?

Runs the same folds under four values of ``min_ess_ratio`` so that the
simultaneous radius, the bias-aware length, and the oracle regret can be read
off one shared design. The candidate fits are repeated across conditions, which
is wasteful but keeps the comparison on the committed code path.

Not part of the package: this measures which threshold to adopt, and the answer
belongs in the plan rather than in the repository as a script.
"""

from __future__ import annotations

import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path("notebooks/experiments").resolve()))
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from refsel.calibration import calibration_key, load_calibration  # noqa: E402
from refsel.grids import GRID_B_OVERLAP_SCALE  # noqa: E402
from refsel.runner import (  # noqa: E402
    ExperimentConfig,
    Numerics,
    Scenario,
    run_repetition,
)

REPLICATIONS = 120
THRESHOLDS: tuple[float | None, ...] = (None, 0.25, 0.40, 0.55)
OUTPUT = pathlib.Path("notebooks/experiments/results/weight_screen_pilot")
TABLES = ("candidate", "selection", "repetition", "oracle", "bound")


def scenarios() -> tuple[Scenario, ...]:
    table = load_calibration()
    hidden = table[calibration_key("low", 3000, GRID_B_OVERLAP_SCALE, 4.0)]
    return (
        Scenario(
            grid="B",
            design="low",
            sample_size=3000,
            overlap_scale=GRID_B_OVERLAP_SCALE,
            target_t=0.0,
            hidden_scale=0.0,
        ),
        Scenario(
            grid="B",
            design="low",
            sample_size=3000,
            overlap_scale=GRID_B_OVERLAP_SCALE,
            target_t=4.0,
            hidden_scale=hidden,
        ),
    )


def configuration(threshold: float | None, scenario: Scenario) -> ExperimentConfig:
    return ExperimentConfig(
        name="weight_screen_pilot",
        tier="smoke",  # unused: replications are driven by the loop below
        scenarios=(scenario,),
        numerics=Numerics(min_ess_ratio=threshold),
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for threshold in THRESHOLDS:
        for scenario in scenarios():
            config = configuration(threshold, scenario)
            for repetition in range(REPLICATIONS):
                jobs.append((threshold, (config, scenario, repetition)))
    print(f"{len(jobs)} jobs", flush=True)

    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    started = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(run_repetition, job): threshold for threshold, job in jobs}
        for future in as_completed(futures):
            threshold = futures[future]
            result = future.result()
            for name in TABLES:
                frame = result[name].copy()
                frame["min_ess_ratio"] = -1.0 if threshold is None else threshold
                collected[name].append(frame)
            done += 1
            if done % 40 == 0:
                rate = (time.monotonic() - started) / done
                print(
                    f"{done}/{len(jobs)}  {rate * (len(jobs) - done) / 60:.1f} min left",
                    flush=True,
                )

    for name in TABLES:
        out = pd.concat(collected[name], ignore_index=True)
        out.to_parquet(OUTPUT / f"{name}.parquet", index=False)
        print(f"wrote {name}: {out.shape}", flush=True)
    print(f"total {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
