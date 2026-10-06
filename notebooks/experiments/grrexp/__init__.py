"""Shared infrastructure for the registered experiments E-12..E-24.

Registration: doc/2026-10-07_experiment_registration.md (parent repository), §1.
Only the seed layer (§1.2) is implemented so far; see README.md in this folder's parent.
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
