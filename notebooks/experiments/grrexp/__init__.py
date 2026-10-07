"""Shared infrastructure for the registered experiments E-12..E-24.

Registration: doc/2026-10-07_experiment_registration.md (parent repository), §1.

- ``seeds``: random-number streams (§1.2)
- ``parallel``: process-level parallelism and the bit-identity check (§1.2)
- ``metrics``: table metrics and their MCSEs (§1.5)
- ``inference``: interval-null tests, Holm, predicted coverage (§1.8, §1.9)
- ``bootstrap``: bootstrap families on stream 4 (§1.9)
- ``env``, ``outputs``: provenance, data checks, outputs and manifest (§1.7)

Estimators and DGPs live in genriesz ``src/`` and in each notebook (§1.3).
"""

from .seeds import (
    CONFIRMATORY_ENTROPY,
    PILOT_ENTROPY,
    STREAMS,
    Seeds,
    fold_ids,
    rng,
    seed_sequence,
)

__all__ = [
    "CONFIRMATORY_ENTROPY",
    "PILOT_ENTROPY",
    "STREAMS",
    "Seeds",
    "fold_ids",
    "rng",
    "seed_sequence",
]
