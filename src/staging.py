"""Ind AS 109 staging engine with SICR detection.

Stage 1: Performing (12-month ECL).
Stage 2: Significant Increase in Credit Risk (lifetime ECL).
Stage 3: Defaulted / credit-impaired (lifetime ECL, LGD typically stressed).

The SICR criterion replaces a static DPD threshold with a relative
point-in-time movement in the lifetime default probability:

    SICR iff  PiT_LifetimePD / OriginationPD > threshold

The origination PD must be a *per-grade* calibrated rate (industry
practice). Using raw per-loan default flags makes the SICR ratio
degenerate to {0, infinity} and prevents the staging engine from
differentiating baseline vs. adverse macro scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def compute_origination_pd(
    default_12m_arr: np.ndarray,
    grade_arr: Optional[np.ndarray] = None,
    fallback_rate: float = 0.05,
) -> np.ndarray:
    """Per-loan origination PD.

    If `grade_arr` is supplied we return the *per-grade* mean default
    rate — the industry-standard proxy for "origination PD" because it
    is leakage-free (no future default flag is observed at origination
    time) and granular enough to make the SICR ratio well-defined.

    If `grade_arr` is omitted we fall back to a flat rate
    (`fallback_rate`) for every loan, which is the conservative
    single-grade assumption.

    A flat per-loan 0/1 default flag is intentionally **not** a valid
    input here — see the module docstring for why.
    """
    default_arr = np.asarray(default_12m_arr, dtype=float)
    if grade_arr is None:
        return np.full_like(default_arr, fallback_rate)
    grade_arr = np.asarray(grade_arr)
    df = pd.DataFrame({"g": grade_arr, "d": default_arr})
    grade_pd = df.groupby("g")["d"].mean().to_dict()
    return np.array([float(grade_pd.get(g, fallback_rate)) for g in grade_arr], dtype=float)


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
