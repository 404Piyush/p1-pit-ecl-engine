"""Pipeline orchestrator.

Drives the full PiT-ECL engine on the synthetic loan book:

    1. Generate / ingest data + PII redaction
    2. Train LightGBM with Platt-vs-Isotonic calibration comparison
    3. Vasicek systematic factor from observed DR_t
    4. Vector autoregression (VAR) on macro panel for forward forecasts
    5. Fit OU process
    6. Forward-simulate Z_t paths (baseline + adverse)
    7. Markov rating migration matrix (TTC lifetime PD by grade)
    8. Map Z_t into annual marginal PD curves (1y / 5y)
    9. SICR + Staging
    10. LGD / EAD / ECL = Σ PD * LGD * EAD * D
    11. Per-path stress test (honest variance)
    12. Quarterly reporting panel (8 reporting dates)
    13. Audit trail (with lineage hashes)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data_gen import generate_synthetic_loan_book, generate_synthetic_macro
from .features import build_feature_table, split_train_test, FEATURE_COLUMNS, TARGET_COLUMN
from .classifier import fit_and_calibrate, predict_test_pit
from .vasicek import fit_vasicek, conditional_pd, DEFAULT_PD_TTC
from .ornstein_uhlenbeck import fit_ou_parameters, simulate_ou_paths, OUParams
from .pd_term_structure import cumulative_from_marginal, marginal_from_cumulative, term_structure_diagnostic
from .staging import assign_stages, compute_origination_pd, stage_breakdown, STAGE1, STAGE2, STAGE3
from .ecl import (
    ecl_per_loan,
    ecl_by_stage,
    default_lgd_by_grade,
    ECLConfig,
    GRADE_LGD,
    ead_for_loan,
)
from .stress import path_wise_provisions
from .markov import fit_markov_matrix, GRADES as MARKOV_GRADES
from .var_macro import fit_var, forecast_default_rate_path
from .reporting import build_reporting_schedule, simulate_reporting_panel, aggregate_stage_trend
from .redact import redact_dataframe_columns, RedactionConfig, DEFAULT_SENSITIVE_COLUMNS


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _hash_obj(o) -> str:
    import hashlib

    payload = json.dumps(o, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _stage_breakdown_str(b: dict) -> str:
    return f"S1={b['stage1_ecl']:,.0f}  S2={b['stage2_ecl']:,.0f}  S3={b['stage3_ecl']:,.0f}"


def _path_aggregate_ecl(
    pd_annual_baseline: np.ndarray,
    pd_annual_adverse: np.ndarray,
    ecl_12_baseline: np.ndarray,
    ecl_12_adverse: np.ndarray,
    stage_12_baseline: np.ndarray,
    stage_12_adverse: np.ndarray,
    ead: np.ndarray,
    lgd: np.ndarray,
    eir: np.ndarray,
    ecl_cfg: ECLConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute aggregate portfolio ECL for each OU path.

    `pd_annual_baseline` is shape (n_paths, 5) where columns are years 1..5.
    Year-1 PD (column 0) drives stage-1 ECL; years 1..5 drive stage-2/3 ECL.
    """
    pd_b12 = pd_annual_baseline[:, 0]
    pd_a12 = pd_annual_adverse[:, 0]
    pd_b_lt = pd_annual_baseline
    pd_a_lt = pd_annual_adverse

    n_paths = pd_annual_baseline.shape[0]

    ecl_b_path = np.zeros(n_paths)
    ecl_a_path = np.zeros(n_paths)

    for p in range(n_paths):
        ecl_b = ecl_per_loan(
            ead=ead,
            pd_curve_12m=np.array([pd_b12[p]]),
            pd_curve_lifetime=pd_b_lt[p],
            lgd=lgd,
            eir=eir,
            stage=stage_12_baseline,
            cfg=ecl_cfg,
        )
        ecl_b_path[p] = ecl_b.sum()

        ecl_a = ecl_per_loan(
            ead=ead,
            pd_curve_12m=np.array([pd_a12[p]]),
            pd_curve_lifetime=pd_a_lt[p],
            lgd=lgd,
            eir=eir,
            stage=stage_12_adverse,
            cfg=ecl_cfg,
        )
        ecl_a_path[p] = ecl_a.sum()

    return ecl_b_path, ecl_a_path


def _realised_loss_per_path(pd_realised: np.ndarray, ead: np.ndarray, lgd: np.ndarray) -> np.ndarray:
    """Realised first-year portfolio losses for each path.

    `pd_realised` is shape (n_paths,) — the path-wise first-year PD. We apply
    a conservative average LGD across the portfolio and discount accrued
    interest at the portfolio mean EIR.
    """
    avg_lgd = float(lgd.mean())
    portfolio_ead = float(ead.sum())
    return np.asarray(pd_realised, dtype=float) * avg_lgd * portfolio_ead


def _scenario_pd_array(paths: np.ndarray, pd_ttc: float, rho: float) -> np.ndarray:
    """Convert paths of Z to annual PDs. Output shape: (n_paths, n_years)."""
    return np.clip(conditional_pd(paths, pd_ttc=pd_ttc, rho=rho), 5e-4, 0.9995)


def run(seed: int = 42, n_loans: int = 8000, n_months: int = 180, write_outputs: bool = True, n_paths: int = 200):
    print("[1/12] Generating synthetic loans + macroeconomic series...")
    loans = generate_synthetic_loan_book(n_loans=n_loans, n_months=n_months, seed=seed)
    macro = generate_synthetic_macro(n_months=n_months, seed=seed)
    print(f"       loans={len(loans):,}  default_rate={loans['default_12m'].mean()*100:.2f}%  "
          f"macro_paths={len(macro)}")

    print("[2/12] Engineering features and splitting train / test...")
    df = build_feature_table(loans)
    train, test = split_train_test(df, train_end="2019-12-31", test_start="2020-01-01")
    valid = train.tail(int(0.15 * len(train))).copy() if len(train) > 200 else train
    train_fit = train.iloc[: len(train) - len(valid)] if len(valid) > 0 else train
    print(f"       train={len(train_fit):,}  valid={len(valid):,}  test={len(test):,}")

    print("[3/12] Training LightGBM and comparing Platt vs Isotonic calibration...")
    fit = fit_and_calibrate(train_fit, valid, test)
    print(f"       test AUC raw={fit.auc:.4f}  best-cal AUC={fit.auc_calibrated:.4f}  "
          f"Brier raw={fit.brier:.4f}  best-cal Brier={fit.brier_calibrated:.4f}  "
          f"calibrator={fit.calibrator_name}")

    print("[4/12] Vasicek systematic factor from observed default rate series...")
    vfit = fit_vasicek(macro["default_rate"].values)
    pd_ttc = vfit.pd_ttc
    print(f"       rho={vfit.rho:.4f}  PD_TTC={pd_ttc:.4f}  "
          f"sigma_DR_obs={vfit.observed_dr_std:.4f}  Z range=[{vfit.z_path.min():+.2f}, {vfit.z_path.max():+.2f}]")

    print("[5/12] Fitting Ornstein-Uhlenbeck (mean-reverting Z_t) by MLE...")
    ou = fit_ou_parameters(vfit.z_path, dt=1.0 / 12.0)
    print(f"       theta={ou.theta:.4f}  mu={ou.mu:+.4f}  sigma={ou.sigma:.4f}")

    z0_baseline = float(vfit.z_path[-1])
    z0_adverse = z0_baseline - 1.5
    mu_adverse_shock = -1.5

    print(f"[6/12] Simulating {n_paths} OU paths (60 monthly steps) for baseline + adverse...")
    n_months = 60
    paths_baseline_monthly = simulate_ou_paths(
        z0=z0_baseline, n_steps=n_months, n_paths=n_paths, params=ou, dt=1.0 / 12.0,
        shock_z=0.0, target_shock_z=0.0, seed=seed,
    )
    paths_adverse_monthly = simulate_ou_paths(
        z0=z0_adverse, n_steps=n_months, n_paths=n_paths, params=ou, dt=1.0 / 12.0,
        shock_z=0.0, target_shock_z=mu_adverse_shock, seed=seed + 7,
    )
    month_idx = np.array([11, 23, 35, 47, 59])
    paths_baseline = paths_baseline_monthly[:, month_idx]
    paths_adverse = paths_adverse_monthly[:, month_idx]

    pd_baseline = _scenario_pd_array(paths_baseline, pd_ttc, vfit.rho)
    pd_adverse = _scenario_pd_array(paths_adverse, pd_ttc, vfit.rho)

    print(f"       mean 12m PD: baseline={pd_baseline[:,0].mean():.4f}  "
          f"adverse={pd_adverse[:,0].mean():.4f}  (scenario mean of {n_paths} paths)")
    print(f"       mean 5y PD curve baseline : {np.round(pd_baseline.mean(axis=0), 4).tolist()}")
    print(f"       mean 5y PD curve adverse  : {np.round(pd_adverse.mean(axis=0), 4).tolist()}")

    diag = term_structure_diagnostic()

    origination_pd = compute_origination_pd(test["default_12m"].values, grade_arr=test["grade"].values)
    pd_test_baseline = predict_test_pit(fit.model, fit.calibrator, test)
    pi12_baseline = float(pd_baseline[:, 0].mean())
    pi12_adverse = float(pd_adverse[:, 0].mean())
    pd_test_adverse = np.clip(pd_test_baseline * (pi12_adverse / pi12_baseline), 1e-4, 0.95)

    lifetime_pd_baseline = pd_test_baseline + 0.04
    lifetime_pd_adverse = pd_test_adverse + 0.10

    rng = np.random.default_rng(seed + 11)
    dpd_current = np.where(
        test["default_12m"].values == 1,
        rng.uniform(60, 130, size=len(test)),
        rng.uniform(0, 25, size=len(test)),
    )

    decision_baseline = assign_stages(origination_pd, lifetime_pd_baseline, dpd_current)
    decision_adverse = assign_stages(origination_pd, lifetime_pd_adverse, dpd_current)
    print(f"       stage mix baseline = {stage_breakdown(decision_baseline.stages)}")
    print(f"       stage mix adverse  = {stage_breakdown(decision_adverse.stages)}")

    print("[7/12] Fitting VAR(p) on macro panel for forward default-rate forecasts...")
    var_fit = fit_var(macro)
    print(f"       lag_order={var_fit.lag_order}  AIC={var_fit.aic:.2f}  BIC={var_fit.bic:.2f}")
    var_fc_baseline = forecast_default_rate_path(var_fit, macro, horizon=24, shock_z=0.0)
    var_fc_adverse = forecast_default_rate_path(var_fit, macro, horizon=24, shock_z=-1.5)

    print("[8/12] Building Markov rating-migration matrix (TTC lifetime PD by grade)...")
    pd_by_grade: Dict[str, float] = {}
    for g in MARKOV_GRADES:
        mask = loans["grade"].values == g
        if mask.sum() > 0:
            pd_by_grade[g] = float(loans.loc[mask, "default_12m"].mean())
        else:
            pd_by_grade[g] = 0.05
    markov = fit_markov_matrix(loans, pd_by_grade)
    print(f"       source={markov.source}  states={markov.matrix.shape[0]}  "
          f"5y-PD for A={markov.lifetime_default_prob('A', 5):.4f}  "
          f"5y-PD for G={markov.lifetime_default_prob('G', 5):.4f}")

    print("[9/12] Computing per-loan LGD, EAD, ECL across path grids...")
    lgds = np.array([default_lgd_by_grade(g) for g in test["grade"].values])
    eads = ead_for_loan(test["loan_amnt"].values.astype(float), accrued_interest=0.02)
    eirs = test["int_rate"].values.astype(float)
    ecl_cfg = ECLConfig()

    ecl_b_path, ecl_a_path = _path_aggregate_ecl(
        pd_annual_baseline=pd_baseline,
        pd_annual_adverse=pd_adverse,
        ecl_12_baseline=pi12_baseline,
        ecl_12_adverse=pi12_adverse,
        stage_12_baseline=decision_baseline.stages,
        stage_12_adverse=decision_adverse.stages,
        ead=eads,
        lgd=lgds,
        eir=eirs,
        ecl_cfg=ecl_cfg,
    )

    ecl_ref_baseline = ecl_per_loan(
        ead=eads,
        pd_curve_12m=np.array([pi12_baseline]),
        pd_curve_lifetime=pd_baseline.mean(axis=0),
        lgd=lgds,
        eir=eirs,
        stage=decision_baseline.stages,
        cfg=ecl_cfg,
    )
    ecl_ref_adverse = ecl_per_loan(
        ead=eads,
        pd_curve_12m=np.array([pi12_adverse]),
        pd_curve_lifetime=pd_adverse.mean(axis=0),
        lgd=lgds,
        eir=eirs,
        stage=decision_adverse.stages,
        cfg=ecl_cfg,
    )
    ecl_b = ecl_by_stage(decision_baseline.stages, ecl_ref_baseline)
    ecl_a = ecl_by_stage(decision_adverse.stages, ecl_ref_adverse)

    ecl_static_ttc = float(np.sum(pd_ttc * lgds * eads / (1.0 + eirs)))

    ecl_per_path = np.column_stack([ecl_b_path, ecl_a_path])
    realised_adverse_losses = _realised_loss_per_path(pd_adverse[:, 0], eads, lgds)
    stress = path_wise_provisions(ecl_per_path, ecl_static_ttc, ecl_b, ecl_a, realised_adverse_losses)

    print(f"[10/12] Provisioning totals (mean-path): baseline={_stage_breakdown_str(ecl_b)} | "
          f"adverse={_stage_breakdown_str(ecl_a)}")
    print(f"       path-wise std: baseline={stress.baseline_std:,.0f}  adverse={stress.adverse_std:,.0f}")
    print(f"       realised adverse losses mean        : INR {stress.realised_adverse_losses_mean:,.0f}")
    print(f"       PiT coverage ratio  (adv.)         : {stress.adverse_coverage_ratio_pit:.3f}")
    print(f"       TTC coverage ratio  (adv.)         : {stress.adverse_coverage_ratio_ttc:.3f}")
    print(f"       forecast-error reduction           : {stress.forecast_error_reduction_pct:.2f}%")

    print("[11/12] Simulating 8-quarter Ind AS 109 reporting panel...")
    schedule = build_reporting_schedule(eight_quarters=True, baseline_pd12=float(pi12_baseline))
    reporting_panel = simulate_reporting_panel(test, schedule)
    panel_agg = aggregate_stage_trend(reporting_panel)
    print(f"       reporting_dates={len(schedule)}  panel_rows={len(reporting_panel):,}  "
          f"final-EAD-INR={panel_agg['ecl_total'].iloc[-1]:,.0f}")
    print("[12/12] Writing artefacts...")
    if write_outputs:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        cfg = RedactionConfig(salt="p1-pit-ecl-engine-run-salt")
        loans_red = loans.copy()
        macro_red = macro.copy()
        redact_dataframe_columns(loans_red, cfg=cfg)
        redact_dataframe_columns(macro_red, cfg=cfg)
        redacted_cols_loans = [c for c in DEFAULT_SENSITIVE_COLUMNS if c in loans_red.columns]

        loans_red.to_csv(DATA_DIR / "synthetic_loans.csv", index=False)
        macro_red.to_csv(DATA_DIR / "synthetic_macro.csv", index=False)

        out_df = test.copy()
        out_df["lgd"] = lgds
        out_df["ead"] = eads
        out_df["eir"] = eirs
        out_df["pd_test_baseline"] = pd_test_baseline
        out_df["pd_test_adverse"] = pd_test_adverse
        out_df["lifetime_pd_baseline"] = lifetime_pd_baseline
        out_df["lifetime_pd_adverse"] = lifetime_pd_adverse
        out_df["stage_baseline"] = decision_baseline.stages
        out_df["stage_adverse"] = decision_adverse.stages
        out_df["sicr_baseline"] = decision_baseline.sicr_flag
        out_df["sicr_adverse"] = decision_adverse.sicr_flag
        out_df["ecl_baseline"] = ecl_ref_baseline
        out_df["ecl_adverse"] = ecl_ref_adverse
        redact_dataframe_columns(out_df, cfg=cfg)
        out_df.to_csv(DATA_DIR / "ecl_output.csv", index=False)

        reporting_panel.to_csv(DATA_DIR / "reporting_panel.csv", index=False)

        audit = {
            "run_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "seed": seed,
            "n_loans": int(len(test)),
            "n_paths": n_paths,
            "data_hashes": {
                "loans": _hash_obj(loans.head(50).to_dict(orient="records")),
                "macro": _hash_obj(macro.head(20).to_dict(orient="records")),
            },
            "redaction": {
                "salt_set": bool(cfg.salt),
                "columns_redacted_loans": redacted_cols_loans,
            },
            "model_metrics": {
                "auc_raw": fit.auc,
                "auc_calibrated": fit.auc_calibrated,
                "brier_raw": fit.brier,
                "brier_calibrated": fit.brier_calibrated,
                "selected_calibrator": fit.calibrator_name,
                "isotonic_brier": getattr(fit, "isotonic_brier", None),
                "isotonic_auc": getattr(fit, "isotonic_auc", None),
            },
            "vasicek": {
                "rho": vfit.rho,
                "pd_ttc": pd_ttc,
                "observed_dr_std": vfit.observed_dr_std,
                "z_min": float(vfit.z_path.min()),
                "z_max": float(vfit.z_path.max()),
            },
            "var": {
                "lag_order": var_fit.lag_order,
                "aic": float(var_fit.aic),
                "bic": float(var_fit.bic),
                "default_rate_forecast_baseline_24m_mean": float(var_fc_baseline.mean()),
                "default_rate_forecast_adverse_24m_mean": float(var_fc_adverse.mean()),
            },
            "markov": {
                "source": markov.source,
                "shape": list(markov.matrix.shape),
                "lifetime_pd_A_5y": markov.lifetime_default_prob("A", 5),
                "lifetime_pd_G_5y": markov.lifetime_default_prob("G", 5),
                "lifetime_pd_C_3y": markov.lifetime_default_prob("C", 3),
            },
            "ornstein_uhlenbeck": {
                "theta": ou.theta,
                "mu": ou.mu,
                "sigma": ou.sigma,
            },
            "pd_curves": {
                "pd_12m_baseline_mean": float(pi12_baseline),
                "pd_12m_adverse_mean": float(pi12_adverse),
                "pd_5y_baseline": pd_baseline.mean(axis=0).tolist(),
                "pd_5y_adverse": pd_adverse.mean(axis=0).tolist(),
            },
            "stress": {
                "baseline_provisions_mean": stress.baseline_provisions_mean,
                "adverse_provisions_mean": stress.adverse_provisions_mean,
                "baseline_std": stress.baseline_std,
                "adverse_std": stress.adverse_std,
                "static_ttc_provisions_total": stress.static_ttc_provisions_total,
                "realised_adverse_losses_mean": stress.realised_adverse_losses_mean,
                "adverse_coverage_ratio_pit": stress.adverse_coverage_ratio_pit,
                "adverse_coverage_ratio_ttc": stress.adverse_coverage_ratio_ttc,
                "forecast_error_pit": stress.forecast_error_pit,
                "forecast_error_ttc": stress.forecast_error_ttc,
                "forecast_error_reduction_pct": stress.forecast_error_reduction_pct,
            },
            "stage_breakdown": {
                "baseline_s1": int(ecl_b["n_stage1"]),
                "baseline_s2": int(ecl_b["n_stage2"]),
                "baseline_s3": int(ecl_b["n_stage3"]),
                "adverse_s1": int(ecl_a["n_stage1"]),
                "adverse_s2": int(ecl_a["n_stage2"]),
                "adverse_s3": int(ecl_a["n_stage3"]),
            },
            "reporting_panel": {
                "n_reporting_dates": len(schedule),
                "panel_rows": int(len(reporting_panel)),
                "final_total_ecl_inr": float(panel_agg["ecl_total"].iloc[-1]),
                "stage_trend_ecl_total": panel_agg.set_index("reporting_date")["ecl_total"].astype(float).to_dict(),
            },
            "diagnostic_term_structure_cumulative": diag["cumulative"].tolist(),
        }
        (REPORTS_DIR / "audit_trail.json").write_text(json.dumps(audit, indent=2))
        print(f"       wrote {DATA_DIR / 'ecl_output.csv'}, reporting_panel.csv and {REPORTS_DIR / 'audit_trail.json'}")

    return {
        "fit": fit,
        "vasicek": vfit,
        "ou": ou,
        "decision_baseline": decision_baseline,
        "decision_adverse": decision_adverse,
        "stress": stress,
        "ecl_baseline": ecl_b,
        "ecl_adverse": ecl_a,
        "term_structure_diag": diag,
        "pd_curve_12m_baseline": pd_baseline[:, 0],
        "pd_curve_12m_adverse": pd_adverse[:, 0],
        "pd_curve_5y_baseline": pd_baseline,
        "pd_curve_5y_adverse": pd_adverse,
        "ecl_b_path": ecl_b_path,
        "ecl_a_path": ecl_a_path,
        "paths_baseline": paths_baseline,
        "paths_adverse": paths_adverse,
        "var_fit": var_fit,
        "markov": markov,
        "reporting_panel": reporting_panel,
        "panel_agg": panel_agg,
    }


if __name__ == "__main__":
    run()
