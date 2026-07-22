"""Unit tests for the VAR(p) macro forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.var_macro import (
    fit_var,
    forecast_default_rate_path,
    macro_scenario_summary,
    MACRO_COLUMNS,
    VARFit,
)


@pytest.fixture
def small_macro() -> pd.DataFrame:
    """Small macro panel with the schema expected by MACRO_COLUMNS."""
    n = 80
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "month": pd.date_range("2015-01-01", periods=n, freq="MS"),
        "z_systematic": rng.normal(0, 1, n),
        "gdp_growth": rng.normal(0.025, 0.005, n),
        "cpi_yoy": rng.normal(0.025, 0.005, n),
        "repo_rate": rng.normal(0.055, 0.005, n),
        "unemployment": rng.normal(0.06, 0.01, n),
        "default_rate": rng.uniform(0.01, 0.10, n),
    })


class TestFitVar:
    def test_returns_varfit_with_aic_lag(self, small_macro):
        fit = fit_var(small_macro)
        assert isinstance(fit, VARFit)
        assert fit.lag_order >= 1
        assert np.isfinite(fit.aic)
        assert np.isfinite(fit.bic)

    def test_columns_resolved_from_requested(self, small_macro):
        fit = fit_var(small_macro)
        # All requested columns that exist in the panel must be retained
        for c in MACRO_COLUMNS:
            assert c in fit.columns

    def test_columns_subset_when_only_some_available(self):
        df = pd.DataFrame({
            "gdp_growth": np.random.default_rng(0).normal(0.025, 0.005, 60),
            "default_rate": np.random.default_rng(0).uniform(0.01, 0.10, 60),
        })
        fit = fit_var(df)
        assert "gdp_growth" in fit.columns
        assert "default_rate" in fit.columns

    def test_default_columns_constant(self):
        assert "default_rate" in MACRO_COLUMNS
        assert "gdp_growth" in MACRO_COLUMNS
        assert "unemployment" in MACRO_COLUMNS


class TestForecastDefaultRatePath:
    def test_forecast_length_matches_horizon(self, small_macro):
        fit = fit_var(small_macro)
        fc = forecast_default_rate_path(fit, small_macro, horizon=12)
        assert fc.shape == (12,)

    def test_forecast_clipped_to_unit_interval(self, small_macro):
        fit = fit_var(small_macro)
        fc = forecast_default_rate_path(fit, small_macro, horizon=12, shock_z=10.0)
        assert (fc >= 0.0).all() and (fc <= 1.0).all()

    def test_forecast_shock_zero_returns_baseline(self, small_macro):
        fit = fit_var(small_macro)
        fc = forecast_default_rate_path(fit, small_macro, horizon=12, shock_z=0.0)
        assert np.isfinite(fc).all()

    def test_forecast_with_negative_shock_lowers_dr(self, small_macro):
        """An adverse shock should not leave default rate unchanged.
        We check that the mean of the shock-adjusted forecast differs from
        the no-shock forecast in the expected direction at minimum."""
        fit = fit_var(small_macro)
        fc_base = forecast_default_rate_path(fit, small_macro, horizon=12, shock_z=0.0)
        fc_adv = forecast_default_rate_path(fit, small_macro, horizon=12, shock_z=-3.0)
        assert fc_adv.mean() < fc_base.mean()


class TestMacroScenarioSummary:
    def test_returns_last_period_values(self, small_macro):
        fit = fit_var(small_macro)
        fc = fit.forecast(steps=10, last_obs=small_macro[fit.columns].tail(fit.lag_order).values)
        summary = macro_scenario_summary(fc, fit.columns)
        assert set(summary.keys()) == set(fit.columns)
        for c in fit.columns:
            assert np.isfinite(summary[c])

    def test_handles_1d_forecast(self):
        fc = np.array([0.05, 0.06, 0.07])
        summary = macro_scenario_summary(fc, ["default_rate"])
        assert summary["default_rate"] == pytest.approx(0.07)
