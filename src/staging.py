"""Ind AS 109 staging engine with SICR detection.

Stage 1: Performing (12-month ECL).
Stage 2: Significant Increase in Credit Risk (lifetime ECL).
Stage 3: Defaulted / credit-impaired (lifetime ECL, LGD typically stressed).

The SICR criterion replaces a static DPD threshold with a relative
point-in-time movement in the lifetime default probability:

    SICR iff  PiT_LifetimePD / OriginationPD > threshold
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_SICR_THRESHOLD = 1.50
STAGE1, STAGE2, STAGE3 = 1, 2, 3


@dataclass
class StagingDecision:
    stages: np.ndarray
    sicr_flag: np.ndarray
    origination_pd: np.ndarray
    pit_lifetime_pd: np.ndarray


def compute_origination_pd(default_12m_arr: np.ndarray) -> np.ndarray:
    """Bookkeeping origination PD: empirical 12-month default rate at vintage.

    On a real book this is a calibrated per-grade PD. Here we use grade-level
    averages to stay leakage-free at origination time.
    """
    return np.asarray(default_12m_arr, dtype=float)


def assign_stages(
    origination_pd: np.ndarray,
    pit_lifetime_pd: np.ndarray,
    dpd_current: np.ndarray,
    threshold: float = DEFAULT_SICR_THRESHOLD,
) -> StagingDecision:
    origination_pd = np.asarray(origination_pd, dtype=float)
    pit_lifetime_pd = np.asarray(pit_lifetime_pd, dtype=float)
    dpd_current = np.asarray(dpd_current, dtype=float)

    ratio = np.divide(
        pit_lifetime_pd,
        np.where(origination_pd > 1e-6, origination_pd, 1e-6),
        out=np.ones_like(pit_lifetime_pd),
        where=origination_pd > 1e-6,
    )

    sicr = ratio > threshold

    stages = np.full(len(origination_pd), STAGE1, dtype=int)
    stages[sicr] = STAGE2
    stages[dpd_current > 90.0] = STAGE3
    stages = np.where(dpd_current > 30.0, np.maximum(stages, STAGE2), stages)

    return StagingDecision(
        stages=stages,
        sicr_flag=sicr.astype(int),
        origination_pd=origination_pd,
        pit_lifetime_pd=pit_lifetime_pd,
    )


def stage_breakdown(stages: np.ndarray) -> dict:
    n = len(stages)
    if n == 0:
        return {"stage1_pct": 0.0, "stage2_pct": 0.0, "stage3_pct": 0.0, "n": 0}
    s1 = float(np.mean(stages == STAGE1))
    s2 = float(np.mean(stages == STAGE2))
    s3 = float(np.mean(stages == STAGE3))
    return {"stage1_pct": s1, "stage2_pct": s2, "stage3_pct": s3, "n": int(n)}
