"""Synthetic data generators for the PiT-ECL engine.

Produces a LendingClub-style loan book and a monthly macroeconomic series
whose default-rate dynamics are consistent with the Vasicek single-factor
model by construction. All randomness is controlled via `seed`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logit as sp_logit


GRADES = ["A", "B", "C", "D", "E", "F", "G"]


def _grade_to_int(grade: str) -> int:
    return GRADES.index(grade)


def generate_synthetic_loan_book(
    n_loans: int = 8000,
    n_months: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic loan book with vintage, grade, and a 12-month default label.

    Returns a DataFrame with one row per loan and the columns:
        loan_id, issue_date, grade, sub_grade, term, int_rate, loan_amnt,
        annual_inc, dti, delinq_2yrs, inq_last_6mths, open_acc, pub_rec,
        revol_bal, revol_util, total_acc, emp_length, home_ownership,
        purpose, application_type, default_12m, origination_z
    """
    rng = np.random.default_rng(seed)

    issue_month = rng.integers(low=0, high=n_months - 13, size=n_loans)
    issue_dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(issue_month, unit="D") * 30

    grade_probs = np.array([0.18, 0.26, 0.22, 0.14, 0.09, 0.07, 0.04])
    grade_probs = grade_probs / grade_probs.sum()
    grade_idx = rng.choice(len(GRADES), size=n_loans, p=grade_probs)
    grades = np.array(GRADES)[grade_idx]
    sub_grades = np.array([f"{g}{s}" for g, s in zip(grades, rng.integers(1, 6, size=n_loans))])

    base_int_rate = {g: 0.05 + 0.025 * i for i, g in enumerate(GRADES)}
    int_rates = np.array([base_int_rate[g] + rng.normal(0, 0.005) for g in grades])
    int_rates = np.clip(int_rates, 0.03, 0.36)

    loan_amnts = rng.choice([10000, 15000, 20000, 25000, 35000], size=n_loans, p=[0.1, 0.2, 0.35, 0.2, 0.15])
    annual_inc = np.clip(rng.lognormal(mean=np.log(65000), sigma=0.55, size=n_loans), 20000, 500000)
    dti = np.clip(rng.normal(18, 9, size=n_loans), 0, 60)
    delinq_2yrs = rng.poisson(0.3, size=n_loans)
    inq_last_6mths = rng.poisson(1.2, size=n_loans)
    open_acc = np.clip(rng.normal(11, 4, size=n_loans), 1, 40).astype(int)
    pub_rec = rng.poisson(0.1, size=n_loans)
    revol_bal = np.clip(rng.lognormal(mean=np.log(8000), sigma=1.0, size=n_loans), 0, 150000)
    revol_util = np.clip(rng.normal(45, 25, size=n_loans), 0, 100)
    total_acc = np.clip(rng.normal(22, 9, size=n_loans), 1, 80).astype(int)
    emp_length = np.clip(rng.normal(5, 3, size=n_loans), 0, 10).astype(int)

    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_loans, p=[0.42, 0.51, 0.07])
    purpose = rng.choice(
        ["credit_card", "debt_consolidation", "home_improvement", "other", "medical", "small_business"],
        size=n_loans, p=[0.20, 0.36, 0.13, 0.18, 0.05, 0.08],
    )
    application_type = rng.choice(["Individual", "Joint App"], size=n_loans, p=[0.93, 0.07])
    term = rng.choice([36, 60], size=n_loans, p=[0.72, 0.28])

    macro_z = _driving_systematic_factor(n_months, seed)
    origination_z = macro_z[issue_month]

    grade_default_base = {g: 0.015 + 0.020 * i for i, g in enumerate(GRADES)}
    idiosyncratic = rng.normal(0, 1.0, size=n_loans)
    asset_corr = 0.12
    z_squashed = np.sqrt(asset_corr) * origination_z + np.sqrt(1 - asset_corr) * idiosyncratic
    base_pd = np.array([grade_default_base[g] for g in grades])
    default_prob = 1.0 / (1.0 + np.exp(-(sp_logit(np.clip(base_pd, 1e-4, 1 - 1e-4)) - 1.6 * z_squashed)))
    default_12m = rng.binomial(1, np.clip(default_prob, 0, 1)).astype(int)

    df = pd.DataFrame({
        "loan_id": np.arange(1, n_loans + 1),
        "issue_date": issue_dates,
        "grade": grades,
        "sub_grade": sub_grades,
        "term": term,
        "int_rate": np.round(int_rates, 4),
        "loan_amnt": loan_amnts,
        "annual_inc": annual_inc.astype(int),
        "dti": np.round(dti, 2),
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal.astype(int),
        "revol_util": np.round(revol_util, 1),
        "total_acc": total_acc,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "application_type": application_type,
        "default_12m": default_12m,
        "origination_z": origination_z,
    })
    return df


def _driving_systematic_factor(n_months: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    theta, mu, sigma = 0.20, 0.0, 0.55
    z = np.zeros(n_months)
    dt = 1.0
    for t in range(1, n_months):
        z[t] = z[t - 1] + theta * (mu - z[t - 1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    z[0] = -0.30
    z = np.clip(z, -3.0, 3.0)
    z[120:] = np.clip(z[120:] + 1.0, -3.0, 3.0)
    return z


def generate_synthetic_macro(n_months: int = 180, seed: int = 42) -> pd.DataFrame:
    """Monthly macro series with a default rate linked to the systematic factor."""
    rng = np.random.default_rng(seed + 2)
    z = _driving_systematic_factor(n_months, seed)
    months = pd.date_range("2015-01-01", periods=n_months, freq="MS")
    gdp_growth = 0.025 - 0.020 * z + rng.normal(0, 0.003, n_months)
    cpi_yoy = 0.025 + 0.005 * z + rng.normal(0, 0.001, n_months)
    repo_rate = 0.055 + 0.010 * z + rng.normal(0, 0.001, n_months)
    unemployment = 0.06 - 0.010 * z + rng.normal(0, 0.002, n_months)
    unemployment = np.clip(unemployment, 0.02, 0.15)

    base_pd = 0.05
    asset_corr = 0.12
    pd_t = 1.0 / (1.0 + np.exp(-(sp_logit(base_pd) - np.sqrt(asset_corr) * z)))
    pd_t = np.clip(pd_t, 0.001, 0.999)

    df = pd.DataFrame({
        "month": months,
        "z_systematic": z,
        "gdp_growth": gdp_growth,
        "cpi_yoy": cpi_yoy,
        "repo_rate": repo_rate,
        "unemployment": unemployment,
        "default_rate": pd_t,
    })
    return df


def write_csvs(out_dir: Path, seed: int = 42, n_loans: int = 8000, n_months: int = 180) -> dict:
    """Persist generated artifacts and return a small manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    loans = generate_synthetic_loan_book(n_loans=n_loans, n_months=n_months, seed=seed)
    macro = generate_synthetic_macro(n_months=n_months, seed=seed)
    loans_path = out_dir / "synthetic_loans.csv"
    macro_path = out_dir / "synthetic_macro.csv"
    loans.to_csv(loans_path, index=False)
    macro.to_csv(macro_path, index=False)
    return {
        "loans": str(loans_path),
        "macro": str(macro_path),
        "n_loans": int(len(loans)),
        "default_rate_pct": float(loans["default_12m"].mean() * 100),
        "n_months": int(len(macro)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    info = write_csvs(out, seed=args.seed)
    print(info)
