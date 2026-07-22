"""Integration smoke tests for the canonical pipeline and reporting helpers.

These tests exercise the top-level entry points of the production pipeline
(`pipeline.run`, `reporting.build_reporting_schedule`, etc.) so that
end-to-end wiring is covered in addition to the per-module unit tests
in `tests/test_all.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import run
from src.reporting import (
    build_reporting_schedule,
    simulate_reporting_panel,
    aggregate_stage_trend,
    ReportingDate,
)
from src.data_gen import generate_synthetic_loan_book, generate_synthetic_macro
from src.markov import GRADES as MARKOV_GRADES


def _small_loan_book(n: int = 80, seed: int = 7) -> pd.DataFrame:
    """Generate a small loan book large enough to be representative.

    Uses n_months=120 so loans span 2015-2024 and split functions can produce
    non-empty train and test sets.
    """
    return generate_synthetic_loan_book(n_loans=n, n_months=120, seed=seed)


class TestPipelineRunSmoke:
    """End-to-end smoke tests with reduced dimensions to keep CI fast."""

    def test_run_returns_full_output_dict(self, tmp_path):
        out = run(seed=11, n_loans=400, n_months=120, write_outputs=False, n_paths=20)
        expected_keys = {
            "fit",
            "vasicek",
            "ou",
            "decision_baseline",
            "decision_adverse",
            "stress",
            "ecl_baseline",
            "ecl_adverse",
            "term_structure_diag",
            "pd_curve_12m_baseline",
            "pd_curve_12m_adverse",
            "pd_curve_5y_baseline",
            "pd_curve_5y_adverse",
            "ecl_b_path",
            "ecl_a_path",
            "paths_baseline",
            "paths_adverse",
            "var_fit",
            "markov",
            "reporting_panel",
            "panel_agg",
        }
        missing = expected_keys - set(out.keys())
        assert not missing, f"pipeline.run output missing keys: {missing}"

    def test_run_produces_paths_of_expected_shape(self):
        out = run(seed=11, n_loans=400, n_months=120, write_outputs=False, n_paths=20)
        assert out["paths_baseline"].shape == (20, 5)
        assert out["paths_adverse"].shape == (20, 5)
        assert out["ecl_b_path"].shape == (20,)
        assert out["ecl_a_path"].shape == (20,)

    def test_run_var_fit_is_aic_selected(self):
        out = run(seed=11, n_loans=400, n_months=120, write_outputs=False, n_paths=20)
        assert out["var_fit"].lag_order >= 1
        assert np.isfinite(out["var_fit"].aic)
        assert np.isfinite(out["var_fit"].bic)
        assert "default_rate" in out["var_fit"].columns

    def test_run_markov_has_default_state(self):
        out = run(seed=11, n_loans=400, n_months=120, write_outputs=False, n_paths=20)
        assert out["markov"].matrix.shape == (8, 8)
        # Last row must be absorbing: P(default | default) == 1
        last_row = out["markov"].matrix[-1, :]
        assert np.isclose(last_row.sum(), 1.0)
        assert np.isclose(last_row[-1], 1.0)

    def test_run_ecl_by_stage_keys(self):
        out = run(seed=11, n_loans=400, n_months=120, write_outputs=False, n_paths=20)
        for label, stage_breakdown in [("baseline", out["ecl_baseline"]), ("adverse", out["ecl_adverse"])]:
            assert {"stage1_ecl", "stage2_ecl", "stage3_ecl", "total_ecl",
                    "n_stage1", "n_stage2", "n_stage3"} <= set(stage_breakdown.keys())
            total_loans = stage_breakdown["n_stage1"] + stage_breakdown["n_stage2"] + stage_breakdown["n_stage3"]
            assert total_loans > 0, f"{label} loan counts must be positive"

    def test_run_writes_artefacts_when_enabled(self, tmp_path, monkeypatch):
        # Redirect DATA_DIR/REPORTS_DIR for this test
        from src import pipeline as pmod

        monkeypatch.setattr(pmod, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(pmod, "REPORTS_DIR", tmp_path / "reports")
        # Write minimal run
        out = run(seed=11, n_loans=300, n_months=120, write_outputs=True, n_paths=10)
        assert (tmp_path / "data" / "synthetic_loans.csv").exists()
        assert (tmp_path / "data" / "synthetic_macro.csv").exists()
        assert (tmp_path / "data" / "ecl_output.csv").exists()
        assert (tmp_path / "data" / "reporting_panel.csv").exists()
        assert (tmp_path / "reports" / "audit_trail.json").exists()


class TestReportingHelpers:
    """Unit tests for the reporting simulator."""

    def test_build_reporting_schedule_eight_quarters(self):
        sched = build_reporting_schedule(eight_quarters=True)
        assert len(sched) == 8
        assert all(isinstance(r, ReportingDate) for r in sched)
        assert [r.label for r in sched] == [f"Q{i+1}" for i in range(8)]
        assert all(r.pd_curve_5y.shape == (5,) for r in sched)
        assert all(r.pd_curve_12m.shape == (1,) for r in sched)

    def test_build_reporting_schedule_four_quarters(self):
        sched = build_reporting_schedule(eight_quarters=False)
        assert len(sched) == 4

    def test_schedule_pd_curves_monotone_increasing(self):
        sched = build_reporting_schedule(eight_quarters=True)
        lifetime_pds = [r.lifetime_pd for r in sched]
        assert lifetime_pds == sorted(lifetime_pds), "lifetime PDs must be monotone non-decreasing"

    def test_simulate_reporting_panel_shape(self):
        loans = _small_loan_book(n=40, seed=3)
        sched = build_reporting_schedule(eight_quarters=True)
        panel = simulate_reporting_panel(loans, sched)
        assert len(panel) == 40 * 8
        expected_cols = {
            "loan_id", "grade", "reporting_date", "quarter_index", "macro_shock_z",
            "origination_pd", "current_pd_12m", "current_pd_lifetime",
            "stage", "sicr_flag", "lgd", "ead", "eir", "ecl",
        }
        assert expected_cols <= set(panel.columns)

    def test_simulate_reporting_panel_ecl_non_negative(self):
        loans = _small_loan_book(n=40, seed=3)
        sched = build_reporting_schedule(eight_quarters=True)
        panel = simulate_reporting_panel(loans, sched)
        assert (panel["ecl"] >= 0).all()

    def test_aggregate_stage_trend_sums_to_n(self):
        loans = _small_loan_book(n=40, seed=3)
        sched = build_reporting_schedule(eight_quarters=True)
        panel = simulate_reporting_panel(loans, sched)
        agg = aggregate_stage_trend(panel)
        assert len(agg) == 8
        for _, row in agg.iterrows():
            assert row["n_loans"] == 40
            assert row["n_stage1"] + row["n_stage2"] + row["n_stage3"] == 40

    def test_aggregate_stage_trend_ecl_total_non_negative(self):
        loans = _small_loan_book(n=40, seed=3)
        sched = build_reporting_schedule(eight_quarters=True)
        panel = simulate_reporting_panel(loans, sched)
        agg = aggregate_stage_trend(panel)
        assert (agg["ecl_total"] >= 0).all()


class TestDataGenIntegration:
    """Tests for the synthetic data generator."""

    def test_loan_book_schema(self):
        loans = generate_synthetic_loan_book(n_loans=100, n_months=120, seed=1)
        expected = {
            "loan_id", "issue_date", "grade", "sub_grade", "term", "int_rate",
            "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
            "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            "emp_length", "home_ownership", "purpose", "application_type",
            "default_12m", "origination_z",
        }
        assert expected <= set(loans.columns)
        assert len(loans) == 100

    def test_loan_book_grades_in_valid_set(self):
        loans = generate_synthetic_loan_book(n_loans=100, n_months=120, seed=1)
        assert set(loans["grade"].unique()) <= set(MARKOV_GRADES)

    def test_macro_schema(self):
        macro = generate_synthetic_macro(n_months=60, seed=1)
        assert set(["month", "z_systematic", "gdp_growth", "cpi_yoy",
                    "repo_rate", "unemployment", "default_rate"]) <= set(macro.columns)
        assert len(macro) == 60
        assert (macro["default_rate"] > 0).all() and (macro["default_rate"] < 1).all()
