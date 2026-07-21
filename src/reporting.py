"""Per-loan quarterly reporting simulator.

Simulates eight Ind AS 109 reporting snapshots (one per quarter) for every
active loan. At each reporting date:

    - The current PD term structure is recomputed under the macro state
      for that quarter.
    - SICR is reassessed against the origination PD; the loan may move
      between Stage 1 and Stage 2.
    - LGD/EAD are recomputed.
    - ECL is calculated at the snapshot.

The output is a panel indexed by (loan_id, reporting_date) with stage,
PD curves, ECL, and SICR flag — i.e. the data structure consumed by an
auditor for an Ind AS 109 disclosure file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .ecl import ecl_per_loan, ECLConfig, default_lgd_by_grade, ead_for_loan
from .staging import assign_stages, compute_origination_pd


@dataclass
class ReportingDate:
    """One reporting snapshot."""
    label: str
    quarter_index: int  # 0..7
    twelve_m_pd: float
    lifetime_pd: float
    macro_shock_z: float  # signed: negative = adverse
    pd_curve_5y: np.ndarray  # shape (5,)
    pd_curve_12m: np.ndarray  # shape (1,)


def build_reporting_schedule(
    eight_quarters: bool = True,
    baseline_pd12: float = 0.05,
    adverse_step: float = 0.04,
) -> List[ReportingDate]:
    """Construct a quarterly schedule with linearly increasing macro stress.

    The schedule starts at baseline PD in Q1, accumulates macro stress each
    quarter, and recovers in the final quarter to reflect scenario closure.
    """
    n_q = 8 if eight_quarters else 4
    out = []
    for q in range(n_q):
        # Increase macro stress then relax; signed convention: negative z = adverse
        shock = -1.0 * q * (adverse_step / 0.04)
        shock = max(shock, -2.0)
        pd12 = float(np.clip(baseline_pd12 + q * 0.005, 0.005, 0.5))
        lifetime = float(np.clip(pd12 * 4.0, 0.05, 0.85))
        # monotonically growing 5-year curve
        years = np.arange(1, 6)
        curve = 1 - (1 - lifetime) ** (years / 5.0)
        curve_12m = np.array([pd12])
        out.append(
            ReportingDate(
                label=f"Q{q+1}",
                quarter_index=q,
                twelve_m_pd=pd12,
                lifetime_pd=lifetime,
                macro_shock_z=shock,
                pd_curve_5y=curve,
                pd_curve_12m=curve_12m,
            )
        )
    return out


def simulate_reporting_panel(
    loans: pd.DataFrame,
    schedule: List[ReportingDate],
    ecl_cfg: ECLConfig = None,
) -> pd.DataFrame:
    """Produce a long-format per-loan, per-reporting-date panel.

    Columns:
        loan_id, grade, reporting_date, quarter_index, macro_shock_z,
        origination_pd, current_pd_12m, current_pd_lifetime,
        stage, sicr_flag, lgd, ead, eir, ecl
    """
    ecl_cfg = ecl_cfg or ECLConfig()
    n_loans = len(loans)

    # Pre-compute loan-level constants once
    origination_pd = compute_origination_pd(loans["default_12m"].values)
    lgds = np.array([default_lgd_by_grade(g) for g in loans["grade"].values])
    eads = ead_for_loan(loans["loan_amnt"].values.astype(float), accrued_interest=0.02)
    eirs = loans["int_rate"].values.astype(float)

    rng = np.random.default_rng(20240101)
    dpd_current = np.where(
        loans["default_12m"].values == 1,
        rng.uniform(60, 130, size=n_loans),
        rng.uniform(0, 25, size=n_loans),
    )

    rows: List[Dict] = []
    for rd in schedule:
        # Lifetime PD per loan at this quarter
        lifetime_pd_loan = np.clip(
            origination_pd + rd.lifetime_pd * np.ones(n_loans), 0.0, 0.9995
        )
        decision = assign_stages(origination_pd, lifetime_pd_loan, dpd_current)
        ecl_vec = ecl_per_loan(
            ead=eads,
            pd_curve_12m=rd.pd_curve_12m,
            pd_curve_lifetime=rd.pd_curve_5y,
            lgd=lgds,
            eir=eirs,
            stage=decision.stages,
            cfg=ecl_cfg,
        )
        for i in range(n_loans):
            rows.append(
                dict(
                    loan_id=int(loans.iloc[i]["loan_id"]),
                    grade=loans.iloc[i]["grade"],
                    reporting_date=rd.label,
                    quarter_index=rd.quarter_index,
                    macro_shock_z=rd.macro_shock_z,
                    origination_pd=float(origination_pd[i]),
                    current_pd_12m=float(rd.twelve_m_pd),
                    current_pd_lifetime=float(lifetime_pd_loan[i]),
                    stage=int(decision.stages[i]),
                    sicr_flag=bool(decision.sicr_flag[i]),
                    lgd=float(lgds[i]),
                    ead=float(eads[i]),
                    eir=float(eirs[i]),
                    ecl=float(ecl_vec[i]),
                )
            )
    return pd.DataFrame(rows)


def aggregate_stage_trend(panel: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate stage counts and ECL by reporting_date."""
    grp = panel.groupby("reporting_date").agg(
        n_loans=("loan_id", "count"),
        n_stage1=("stage", lambda s: int((s == 1).sum())),
        n_stage2=("stage", lambda s: int((s == 2).sum())),
        n_stage3=("stage", lambda s: int((s == 3).sum())),
        ecl_total=("ecl", "sum"),
        ecl_mean=("ecl", "mean"),
    ).reset_index()
    return grp
