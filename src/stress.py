"""Stress-testing utilities for the PiT-ECL engine.

We measure the **forecast-error reduction** of the PiT engine relative to a
static Through-the-Cycle (TTC) point estimate. TTC has zero forecast variance
by construction but is biased when the macro is in any non-average state.
PiT is path-dependent and unbiased in expectation; the question is whether
its dispersion is well-calibrated to the realised losses.

Path-wise quantities:
- ECL_b_path, ECL_a_path : aggregate portfolio provisions per OU simulation
- LGD * EAD * PD_realised : "true" 12-month portfolio losses on each path

The variance reduction metric is then:

    coverage_pit  = E[ECL_pit] / E[true_losses]
    coverage_ttc  = E[TTC_constant] / E[true_losses]
    coverage_gap  = |1 - coverage_pit|  vs  |1 - coverage_ttc|

A lower coverage gap means lower forecast-error variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class StressResult:
    baseline_provisions_mean: float
    adverse_provisions_mean: float
    static_ttc_provisions_total: float
    realised_adverse_losses_mean: float
    adverse_coverage_ratio_pit: float
    adverse_coverage_ratio_ttc: float
    forecast_error_pit: float
    forecast_error_ttc: float
    forecast_error_reduction_pct: float
    n_paths: int
    baseline_std: float
    adverse_std: float
    stage_breakdown_baseline: dict
    stage_breakdown_adverse: dict


def path_wise_provisions(
    ecl_totals_per_path: np.ndarray,
    static_ttc_ecl: float,
    ecl_b: dict,
    ecl_a: dict,
    realised_losses_per_path: np.ndarray,
) -> StressResult:
    """Compare PiT vs TTC forecast accuracy under path-wise stress."""
    base = np.asarray(ecl_totals_per_path[:, 0], dtype=float)
    adv = np.asarray(ecl_totals_per_path[:, 1], dtype=float)
    realised = np.asarray(realised_losses_per_path, dtype=float)

    pit_mean = float(adv.mean())
    realised_mean = float(realised.mean())
    ttc_const = float(static_ttc_ecl)

    cov_pit = pit_mean / realised_mean if realised_mean > 0 else 1.0
    cov_ttc = ttc_const / realised_mean if realised_mean > 0 else 1.0

    err_pit = abs(1.0 - cov_pit)
    err_ttc = abs(1.0 - cov_ttc)

    if err_ttc > 0:
        err_red = max(0.0, (err_ttc - err_pit) / err_ttc) * 100.0
    else:
        err_red = 0.0

    return StressResult(
        baseline_provisions_mean=float(base.mean()),
        adverse_provisions_mean=pit_mean,
        static_ttc_provisions_total=ttc_const,
        realised_adverse_losses_mean=realised_mean,
        adverse_coverage_ratio_pit=float(cov_pit),
        adverse_coverage_ratio_ttc=float(cov_ttc),
        forecast_error_pit=float(err_pit),
        forecast_error_ttc=float(err_ttc),
        forecast_error_reduction_pct=float(err_red),
        n_paths=int(len(base)),
        baseline_std=float(base.std(ddof=1)),
        adverse_std=float(adv.std(ddof=1)),
        stage_breakdown_baseline=ecl_b,
        stage_breakdown_adverse=ecl_a,
    )
