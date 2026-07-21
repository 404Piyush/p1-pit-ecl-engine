"""Expected Credit Loss (ECL) calculator.

Per Ind AS 109:

    ECL = sum_{t=1}^{T} PD_t * LGD_t * EAD_t * D_t

For Stage 1 assets, T=1 year (12-month ECL); for Stages 2 and 3, T is the
remaining lifetime. The PD curves are *annual marginal* default probabilities
generated from the Vasicek + OU forward simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .staging import STAGE1
from .pd_term_structure import discount_factors


GRADE_LGD = {
    "A": 0.30,
    "B": 0.35,
    "C": 0.45,
    "D": 0.55,
    "E": 0.65,
    "F": 0.75,
    "G": 0.85,
}


def default_lgd_by_grade(grade: str) -> float:
    return GRADE_LGD.get(str(grade).upper(), 0.60)


@dataclass
class ECLConfig:
    horizon_12m: int = 1
    horizon_lifetime: int = 5
    dt: float = 1.0
    stressed_lgd_uplift: float = 0.10


def ead_for_loan(outstanding_principal: np.ndarray, accrued_interest: float = 0.02) -> np.ndarray:
    """EAD_t equals principal outstanding plus accrued interest at reporting date."""
    return np.asarray(outstanding_principal, dtype=float) * (1.0 + accrued_interest)


def ecl_per_loan(
    ead: np.ndarray,
    pd_curve_12m: np.ndarray,
    pd_curve_lifetime: np.ndarray,
    lgd: np.ndarray,
    eir: np.ndarray,
    stage: np.ndarray,
    cfg: ECLConfig | None = None,
) -> np.ndarray:
    """Compute per-loan ECL. `pd_curve_12m` must be length-1 (annual),
    `pd_curve_lifetime` must be length 5 (annual marginal PDs)."""
    cfg = cfg or ECLConfig()
    ead = np.asarray(ead, dtype=float)
    lgd = np.asarray(lgd, dtype=float)
    eir = np.asarray(eir, dtype=float)
    stage = np.asarray(stage, dtype=int)
    pd_curve_12m = np.asarray(pd_curve_12m, dtype=float)
    pd_curve_lifetime = np.asarray(pd_curve_lifetime, dtype=float)

    ecl = np.zeros(len(ead), dtype=float)
    for i in range(len(ead)):
        s = stage[i]
        eir_i = eir[i]
        lgd_i = lgd[i]

        if s == STAGE1:
            curve = pd_curve_12m
            horizon = cfg.horizon_12m
            if s == 1 and curve.size < 1:
                curve = np.array([0.05])
                horizon = 1
        else:
            curve = pd_curve_lifetime
            horizon = cfg.horizon_lifetime
            if s == 3:
                lgd_i = min(1.0, lgd_i + cfg.stressed_lgd_uplift)

        disc = discount_factors(eir_i, horizon, dt=cfg.dt)
        ecl[i] = float(np.sum(curve * lgd_i * ead[i] * disc))

    return ecl


def ecl_by_stage(stage: np.ndarray, ecl: np.ndarray) -> dict:
    return {
        "stage1_ecl": float(ecl[stage == STAGE1].sum()),
        "stage2_ecl": float(ecl[stage == 2].sum()),
        "stage3_ecl": float(ecl[stage == 3].sum()),
        "total_ecl": float(ecl.sum()),
        "n_stage1": int((stage == STAGE1).sum()),
        "n_stage2": int((stage == 2).sum()),
        "n_stage3": int((stage == 3).sum()),
    }
