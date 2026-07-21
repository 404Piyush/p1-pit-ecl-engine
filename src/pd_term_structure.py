"""PD term-structure utilities.

Given marginal PDs q_1, ..., q_T (default probability conditional on surviving
to year t-1), the cumulative (unconditional) PD up to year T is:

    0q_T = 1 - prod_{t=1}^{T} (1 - t-1q_t)

The sample cohort table from the brief is unit-tested in `tests/`.
"""

from __future__ import annotations

import numpy as np


def cumulative_from_marginal(marginal: np.ndarray) -> np.ndarray:
    """Convert a vector of marginal PDs into a cumulative term structure.

    Surv(t) = prod_{k<=t} (1 - q_k),  CumPD(t) = 1 - Surv(t).
    """
    marginal = np.asarray(marginal, dtype=float)
    survival = np.cumprod(1.0 - marginal)
    return 1.0 - survival


def marginal_from_cumulative(cumulative: np.ndarray) -> np.ndarray:
    cumulative = np.asarray(cumulative, dtype=float)
    out = np.empty_like(cumulative)
    prev_surv = 1.0
    for i, c in enumerate(cumulative):
        surv = 1.0 - c
        out[i] = (prev_surv - surv) / prev_surv if prev_surv > 0 else 0.0
        prev_surv = surv
    return out


def discount_factors(eir: float, n_steps: int, dt: float = 1.0) -> np.ndarray:
    """Effective-interest discount factors D_t = (1 + eir)^{-t}."""
    t = np.arange(1, n_steps + 1) * dt
    return 1.0 / np.power(1.0 + eir, t)


def term_structure_diagnostic() -> dict:
    """Reproduce the sample cohort table from the brief:

        Year 1: marginal 8.50%, cumulative 8.50%
        Year 2: marginal 13.78%, cumulative 21.11%
        Year 3: marginal 9.86%, cumulative 28.89%
    """
    marginal = np.array([0.0850, 0.1378, 0.0986])
    cumulative = cumulative_from_marginal(marginal)
    return {
        "marginal": marginal,
        "cumulative": np.round(cumulative, 6),
        "expected_cumulative": np.array([0.0850, 0.2111, 0.2889]),
    }
