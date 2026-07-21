"""Test suite for p1-pit-ecl-engine.

Run with:
    py -3.11 -m pytest tests/ -v

Covers:
- pd_term_structure (marginal/cumulative identity)
- vasicek (systematic factor, conditional PD, rho calibration)
- ornstein_uhlenbeck (MLE fit + simulation shape sanity)
- staging (SICR, per-grade origination PD, DPD rule precedence)
- markov (row-stochasticity, absorbing state, lifetime PD monotonicity)
- redact (PAN, Aadhaar, mobile, email, account; determinism)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pd_term_structure import (
    cumulative_from_marginal,
    marginal_from_cumulative,
    term_structure_diagnostic,
    discount_factors,
)
from src.vasicek import (
    vasicek_systematic_factor,
    conditional_pd,
    fit_vasicek,
    calibrate_rho,
)
from src.ornstein_uhlenbeck import (
    fit_ou_parameters,
    simulate_ou_paths,
)
from src.staging import (
    assign_stages,
    compute_origination_pd,
    stage_breakdown,
    DEFAULT_SICR_THRESHOLD,
    STAGE1,
    STAGE2,
    STAGE3,
)
from src.markov import (
    fit_markov_matrix,
    GRADES,
    ABSORBING_DEFAULT,
    _synthetic_matrix_from_ttc,
)
from src.redact import (
    redact_pan,
    redact_aadhaar,
    redact_mobile,
    redact_email,
    redact_account,
    redact_dataframe_columns,
    RedactionConfig,
    sensitive_columns,
)


# ---------------------------------------------------------------- pd_term_structure

class TestPDTermStructure:
    """Verifies the marginal-to-cumulative PD identity from the brief."""

    def test_marginal_to_cumulative_matches_brief(self):
        """Reproduces the brief's 8.50/13.78/9.86 cohort identity."""
        diag = term_structure_diagnostic()
        cum = diag["cumulative"]
        assert np.allclose(cum[0], 0.0850, atol=1e-4)
        assert np.allclose(cum[1], 0.2111, atol=1e-4)
        assert np.allclose(cum[2], 0.2889, atol=1e-4)

    def test_survival_form_identity(self):
        """Product of (1 - marginal) equals (1 - cumulative)."""
        marginals = np.array([0.10, 0.15, 0.08, 0.05, 0.04])
        cum = cumulative_from_marginal(marginals)
        survival_product = float(np.prod(1.0 - marginals))
        assert np.isclose(1.0 - cum[-1], survival_product, atol=1e-9)

    def test_roundtrip_marginal_cumulative(self):
        m = np.array([0.05, 0.10, 0.07, 0.04, 0.03])
        c = cumulative_from_marginal(m)
        m_back = marginal_from_cumulative(c)
        assert np.allclose(m, m_back, atol=1e-9)

    def test_discount_factors_monotone_decreasing(self):
        d = discount_factors(eir=0.12, n_steps=5)
        assert d.shape == (5,)
        assert np.all(np.diff(d) <= 0)
        assert np.isclose(d[0], 1.0 / (1.0 + 0.12))


# ---------------------------------------------------------------- vasicek

class TestVasicek:
    """Tests for the Vasicek single-factor systematic model."""

    def test_systematic_factor_finite(self):
        dr = np.array([0.04, 0.05, 0.06, 0.05, 0.04])
        z = vasicek_systematic_factor(dr, pd_ttc=0.05, rho=0.10)
        assert z.shape == (5,)
        assert np.all(np.isfinite(z))

    def test_conditional_pd_in_unit_interval(self):
        z = np.linspace(-3.0, 3.0, 20)
        p = conditional_pd(z, pd_ttc=0.05, rho=0.10)
        assert np.all(p > 0)
        assert np.all(p < 1)

    def test_higher_systematic_factor_lowers_pd(self):
        """Under the Z>0 = good-economy convention, higher z implies lower PD."""
        p_low_z = conditional_pd(z=np.array([-2.0]), pd_ttc=0.05, rho=0.10)
        p_high_z = conditional_pd(z=np.array([2.0]), pd_ttc=0.05, rho=0.10)
        assert p_high_z[0] < p_low_z[0]

    def test_fit_vasicek_returns_finite_z(self):
        rng = np.random.default_rng(42)
        dr = np.clip(rng.normal(0.05, 0.01, size=120), 0.001, 0.5)
        vfit = fit_vasicek(dr)
        assert np.all(np.isfinite(vfit.z_path))
        assert 0.0 < vfit.rho <= 0.5
        assert 0.0 < vfit.pd_ttc < 0.5

    def test_calibrate_rho_in_unit_interval(self):
        rng = np.random.default_rng(7)
        dr = np.clip(rng.normal(0.05, 0.01, size=120), 0.001, 0.5)
        rho = calibrate_rho(dr, pd_ttc=0.05)
        assert 0.0 <= rho <= 0.999


# ---------------------------------------------------------------- ornstein_uhlenbeck

class TestOrnsteinUhlenbeck:
    def test_fit_ou_recovers_sane_parameters(self):
        rng = np.random.default_rng(0)
        true_theta, true_mu, true_sigma = 2.0, 0.5, 1.0
        z = np.zeros(120)
        z[0] = true_mu
        for t in range(1, len(z)):
            z[t] = z[t-1] + true_theta * (true_mu - z[t-1]) * (1/12) + true_sigma * np.sqrt(1/12) * rng.normal()
        ou = fit_ou_parameters(z, dt=1/12)
        assert ou.theta > 0
        assert ou.sigma > 0

    def test_simulate_ou_shape(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, size=120)
        ou = fit_ou_parameters(z, dt=1/12)
        paths = simulate_ou_paths(z0=0.0, n_steps=12, n_paths=50, params=ou, dt=1/12, seed=1)
        # simulate_ou_paths returns (n_paths, n_steps + 1) — initial z0 included
        assert paths.shape == (50, 13)
        assert np.all(np.isfinite(paths))


# ---------------------------------------------------------------- staging

class TestStaging:
    def test_per_grade_origination_pd(self):
        """Origination PD must equal per-grade mean default rate."""
        default_arr = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        grade_arr = np.array(["A","A","A","A","B","B","B","B"])
        opd = compute_origination_pd(default_arr, grade_arr=grade_arr)
        assert np.isclose(opd[grade_arr == "A"].mean(), 0.5)
        assert np.isclose(opd[grade_arr == "B"].mean(), 0.25)

    def test_origination_pd_fallback(self):
        opd = compute_origination_pd(np.array([0, 1]), grade_arr=None)
        assert np.all(opd == 0.05)

    def test_sicr_threshold_triggers_stage2(self):
        origination_pd = np.array([0.05, 0.05])
        pit_lifetime = np.array([0.05 * 1.4, 0.05 * 1.6])  # second triggers SICR
        dpd = np.zeros(2)
        decision = assign_stages(origination_pd, pit_lifetime, dpd)
        assert decision.stages[0] == STAGE1
        assert decision.stages[1] == STAGE2

    def test_dpd_over_90_forces_stage3(self):
        origination_pd = np.array([0.05])
        pit_lifetime = np.array([0.04])
        dpd = np.array([100.0])
        decision = assign_stages(origination_pd, pit_lifetime, dpd)
        assert decision.stages[0] == STAGE3

    def test_dpd_30_to_90_forces_stage2(self):
        origination_pd = np.array([0.05])
        pit_lifetime = np.array([0.04])
        dpd = np.array([60.0])
        decision = assign_stages(origination_pd, pit_lifetime, dpd)
        assert decision.stages[0] == STAGE2

    def test_breakdown_sums_to_one(self):
        stages = np.array([1, 1, 2, 3, 1, 2])
        b = stage_breakdown(stages)
        assert np.isclose(b["stage1_pct"] + b["stage2_pct"] + b["stage3_pct"], 1.0)

    def test_stage_mix_differentiates_baseline_vs_adverse(self):
        """After fix: per-grade origination PDs must produce different stage
        mixes for different PiT lifetime PD vectors."""
        rng = np.random.default_rng(0)
        n = 1000
        grade_arr = rng.choice(GRADES, size=n)
        default_arr = (rng.uniform(0, 1, size=n) < 0.08).astype(int)
        opd = compute_origination_pd(default_arr, grade_arr=grade_arr)
        dpd = np.zeros(n)
        decision_baseline = assign_stages(opd, opd * 1.2, dpd)
        decision_adverse = assign_stages(opd, opd * 3.0, dpd)
        b_b = stage_breakdown(decision_baseline.stages)
        b_a = stage_breakdown(decision_adverse.stages)
        assert b_a["stage2_pct"] > b_b["stage2_pct"]


# ---------------------------------------------------------------- markov

class TestMarkov:
    def test_synthetic_matrix_is_row_stochastic(self):
        pd = {"A": 0.02, "B": 0.05, "C": 0.08, "D": 0.12, "E": 0.18, "F": 0.25, "G": 0.35}
        M = _synthetic_matrix_from_ttc(pd)
        assert M.shape == (8, 8)
        assert np.allclose(M.sum(axis=1), 1.0, atol=1e-9)
        # default row is absorbing
        d_idx = 7
        assert M[d_idx, d_idx] == 1.0

    def test_lifetime_pd_increases_with_horizon(self):
        pd_by_g = {g: 0.05 + i * 0.04 for i, g in enumerate(GRADES)}
        m = fit_markov_matrix(loans=pd.DataFrame({"grade": [], "next_grade": []}), pd_by_grade=pd_by_g)
        p1 = m.lifetime_default_prob("C", 1)
        p3 = m.lifetime_default_prob("C", 3)
        p5 = m.lifetime_default_prob("C", 5)
        assert p1 <= p3 <= p5
        assert 0.0 < p1 < 1.0

    def test_default_state_is_absorbing(self):
        pd_by_g = {g: 0.05 for g in GRADES}
        m = fit_markov_matrix(loans=pd.DataFrame(), pd_by_grade=pd_by_g)
        # default absorbing row is always the last row of the matrix
        d_row = m.matrix.shape[0] - 1
        assert m.matrix[d_row, d_row] == 1.0
        assert np.sum(m.matrix[d_row, :d_row]) == 0.0


# ---------------------------------------------------------------- redact

class TestRedact:
    def test_redact_pan_mask_format(self):
        cfg = RedactionConfig(salt="test-salt")
        out = redact_pan("ABCDE1234F", cfg)
        assert out.startswith("XXXXX")
        assert out.endswith("F")
        assert out != "ABCDE1234F"

    def test_redact_aadhaar_keeps_last_four(self):
        cfg = RedactionConfig(salt="x")
        assert redact_aadhaar("234123412346", cfg) == "XXXX-XXXX-2346"

    def test_redact_mobile_keeps_last_four(self):
        cfg = RedactionConfig(salt="x")
        assert redact_mobile("9876543210", cfg) == "XXXXXX3210"

    def test_redact_email_keeps_domain(self):
        cfg = RedactionConfig(salt="x")
        out = redact_email("alice@example.com", cfg)
        assert "@example.com" in out
        assert not out.startswith("alice")

    def test_redact_email_mask_domain(self):
        cfg = RedactionConfig(salt="x", mask_email_domain_visible=False)
        out = redact_email("bob@example.com", cfg)
        assert out.endswith("@example.invalid")

    def test_redact_account_keeps_last_four(self):
        cfg = RedactionConfig(salt="x")
        assert redact_account("1234567890123", cfg) == "XXXXXXXXX0123"

    def test_redact_determinism_same_salt_same_output(self):
        cfg1 = RedactionConfig(salt="alpha")
        cfg2 = RedactionConfig(salt="alpha")
        a = redact_pan("ABCDE1234F", cfg1)
        b = redact_pan("ABCDE1234F", cfg2)
        assert a == b

    def test_redact_different_salt_different_output(self):
        a = redact_pan("ABCDE1234F", RedactionConfig(salt="salt-1"))
        b = redact_pan("ABCDE1234F", RedactionConfig(salt="salt-2"))
        assert a != b

    def test_sensitive_columns_detection(self):
        cols = ["loan_id", "pan", "aadhaar", "email", "grade", "phone"]
        assert set(sensitive_columns(cols)) == {"pan", "aadhaar", "email", "phone"}

    def test_redact_dataframe_columns_inplace(self):
        df = pd.DataFrame({
            "loan_id": [1, 2],
            "pan": ["ABCDE1234F", "PQRST5678K"],
            "grade": ["A", "B"],
        })
        cols = redact_dataframe_columns(df, cfg=RedactionConfig(salt="x"))
        assert cols == ["pan"]
        assert df["loan_id"].tolist() == [1, 2]
        assert df["grade"].tolist() == ["A", "B"]
        assert df["pan"].iloc[0] != "ABCDE1234F"

    def test_redact_invalid_inputs_pass_through(self):
        cfg = RedactionConfig(salt="x")
        assert redact_pan("not-a-pan", cfg) == "not-a-pan"
        assert redact_aadhaar("123", cfg) == "123"
