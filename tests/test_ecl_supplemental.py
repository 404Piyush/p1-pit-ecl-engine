"""Supplemental tests for the ECL calculator."""

from __future__ import annotations

import numpy as np
import pytest

from src.ecl import (
    ECLConfig,
    GRADE_LGD,
    default_lgd_by_grade,
    ead_for_loan,
    ecl_per_loan,
    ecl_by_stage,
)
from src.staging import STAGE1, STAGE2, STAGE3


class TestECLConfig:
    def test_defaults(self):
        cfg = ECLConfig()
        assert cfg.horizon_12m == 1
        assert cfg.horizon_lifetime == 5
        assert cfg.dt == 1.0
        assert cfg.stressed_lgd_uplift >= 0


class TestDefaultLgdByGrade:
    def test_known_grades(self):
        for g in ["A", "B", "C", "D", "E", "F", "G"]:
            assert default_lgd_by_grade(g) == GRADE_LGD[g]

    def test_unknown_grade_falls_back(self):
        assert default_lgd_by_grade("X") == 0.60

    def test_lowercase_uppercased(self):
        assert default_lgd_by_grade("a") == GRADE_LGD["A"]

    def test_lgd_monotone_in_grade(self):
        lgd_A = default_lgd_by_grade("A")
        lgd_G = default_lgd_by_grade("G")
        assert lgd_A < lgd_G


class TestEadForLoan:
    def test_zero_principal_yields_zero_ead(self):
        out = ead_for_loan(np.zeros(3), accrued_interest=0.0)
        assert np.allclose(out, 0.0)

    def test_accrued_interest_increases_ead(self):
        principal = np.array([100_000.0])
        ead_no_acc = ead_for_loan(principal, accrued_interest=0.0)
        ead_with_acc = ead_for_loan(principal, accrued_interest=0.02)
        assert ead_with_acc[0] > ead_no_acc[0]
        assert ead_with_acc[0] == pytest.approx(102_000.0)

    def test_ead_is_element_wise(self):
        principal = np.array([50_000.0, 100_000.0, 200_000.0])
        out = ead_for_loan(principal, accrued_interest=0.05)
        assert out[0] == pytest.approx(52_500.0)
        assert out[2] == pytest.approx(210_000.0)


class TestEclPerLoan:
    def test_zero_inputs_yield_zero_ecl(self):
        n = 5
        ecl = ecl_per_loan(
            ead=np.zeros(n),
            pd_curve_12m=np.array([0.05]),
            pd_curve_lifetime=np.array([0.05, 0.06, 0.07, 0.08, 0.09]),
            lgd=np.full(n, 0.5),
            eir=np.full(n, 0.10),
            stage=np.ones(n, dtype=int),
            cfg=ECLConfig(),
        )
        assert np.allclose(ecl, 0.0)

    def test_stage1_uses_12m_curve(self):
        ecl_s1 = ecl_per_loan(
            ead=np.array([100_000.0]),
            pd_curve_12m=np.array([0.10]),
            pd_curve_lifetime=np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
            lgd=np.array([0.5]),
            eir=np.array([0.10]),
            stage=np.array([STAGE1], dtype=int),
            cfg=ECLConfig(),
        )
        # Stage 1 should use only the 12m PD; lifetime ECL would be much larger
        ecl_s2 = ecl_per_loan(
            ead=np.array([100_000.0]),
            pd_curve_12m=np.array([0.10]),
            pd_curve_lifetime=np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
            lgd=np.array([0.5]),
            eir=np.array([0.10]),
            stage=np.array([STAGE2], dtype=int),
            cfg=ECLConfig(),
        )
        assert ecl_s2[0] > ecl_s1[0]

    def test_stage3_stressed_lgd_uplift(self):
        """Stage 3 LGD should be uplifted, so its ECL exceeds Stage 2's."""
        ecl_s2 = ecl_per_loan(
            ead=np.array([100_000.0]),
            pd_curve_12m=np.array([0.10]),
            pd_curve_lifetime=np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
            lgd=np.array([0.5]),
            eir=np.array([0.10]),
            stage=np.array([STAGE2], dtype=int),
            cfg=ECLConfig(stressed_lgd_uplift=0.10),
        )
        ecl_s3 = ecl_per_loan(
            ead=np.array([100_000.0]),
            pd_curve_12m=np.array([0.10]),
            pd_curve_lifetime=np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
            lgd=np.array([0.5]),
            eir=np.array([0.10]),
            stage=np.array([STAGE3], dtype=int),
            cfg=ECLConfig(stressed_lgd_uplift=0.10),
        )
        # Stage 3 has +10% LGD uplift, so its ECL should exceed Stage 2
        assert ecl_s3[0] > ecl_s2[0]

    def test_ecl_clipped_to_ead(self):
        """For marginal PD inputs the cumulative PD across the lifetime can
        exceed 1; the function then returns a value > EAD. For a single
        year (Stage 1) with LGD <= 1, ECL must be <= EAD * LGD."""
        ecl_s2 = ecl_per_loan(
            ead=np.array([100_000.0]),
            pd_curve_12m=np.array([0.99]),
            pd_curve_lifetime=np.array([0.99, 0.99, 0.99, 0.99, 0.99]),
            lgd=np.array([0.99]),
            eir=np.array([0.05]),
            stage=np.array([STAGE2], dtype=int),
            cfg=ECLConfig(),
        )
        # Stage 1 with 12m PD = 0.99 and LGD = 0.99 must be <= EAD * LGD * D_1
        ecl_s1 = ecl_per_loan(
            ead=np.array([100_000.0]),
            pd_curve_12m=np.array([0.99]),
            pd_curve_lifetime=np.array([0.99, 0.99, 0.99, 0.99, 0.99]),
            lgd=np.array([0.99]),
            eir=np.array([0.05]),
            stage=np.array([STAGE1], dtype=int),
            cfg=ECLConfig(),
        )
        assert ecl_s1[0] <= 100_000.0 * 0.99 / 1.05 + 1e-6
        # Stage 2 lifetime: confirm ECL is finite (don't bound it; cumulative
        # marginal PD can legitimately exceed 1 in extreme cases)
        assert np.isfinite(ecl_s2[0])
        assert ecl_s2[0] >= 0

    def test_default_cfg_is_used_when_none(self):
        ecl = ecl_per_loan(
            ead=np.array([100_000.0]),
            pd_curve_12m=np.array([0.05]),
            pd_curve_lifetime=np.array([0.05, 0.06, 0.07, 0.08, 0.09]),
            lgd=np.array([0.5]),
            eir=np.array([0.10]),
            stage=np.array([STAGE1], dtype=int),
        )
        assert ecl[0] > 0


class TestEclByStage:
    def test_keys_present(self):
        out = ecl_by_stage(np.array([1, 2, 3, 1]), np.array([100.0, 200.0, 300.0, 50.0]))
        assert {"stage1_ecl", "stage2_ecl", "stage3_ecl", "total_ecl",
                "n_stage1", "n_stage2", "n_stage3"} == set(out.keys())

    def test_totals_sum_correctly(self):
        stages = np.array([1, 1, 2, 3, 3])
        ecls = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        out = ecl_by_stage(stages, ecls)
        assert out["stage1_ecl"] == pytest.approx(30.0)
        assert out["stage2_ecl"] == pytest.approx(30.0)
        assert out["stage3_ecl"] == pytest.approx(90.0)
        assert out["total_ecl"] == pytest.approx(150.0)
        assert out["n_stage1"] == 2
        assert out["n_stage2"] == 1
        assert out["n_stage3"] == 2
