"""Vasicek single-factor model.

For each time t, the systematic factor Z_t is extracted from observed default
rate DR_t and the long-run through-the-cycle PD_TTC:

    Z_t = ( Phi^{-1}(PD_TTC)  -  sqrt(1 - rho) * Phi^{-1}(DR_t) ) / sqrt(rho)

Asset correlation rho is calibrated empirically by moment-matching the observed
default-rate volatility to the Vasicek-implied volatility:

    sigma_DR = Phi( (Phi^{-1}(PD_TTC) - sqrt(rho) Z_t) / sqrt(1 - rho) )
    Var(DR) ~ ( PD_TTC*(1-PD_TTC) * rho )     (first-order approx., Vasicek 2002)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


DEFAULT_RHO = 0.12
DEFAULT_PD_TTC = 0.05

EPS = 1e-6


@dataclass
class VasicekFit:
    rho: float
    pd_ttc: float
    z_path: np.ndarray
    observed_dr_std: float


def _clip_dr(dr: np.ndarray) -> np.ndarray:
    return np.clip(dr, EPS, 1 - EPS)


def calibrate_rho(dr_series: pd.Series | np.ndarray, pd_ttc: float = DEFAULT_PD_TTC) -> float:
    dr = np.asarray(_clip_dr(dr_series), dtype=float)
    sigma_obs = dr.std(ddof=1)
    var_target = sigma_obs ** 2

    def objective(rho: float) -> float:
        rho = float(np.clip(rho, 1e-4, 0.95))
        var_vasicek = pd_ttc * (1 - pd_ttc) * rho
        return var_vasicek - var_target

    if objective(0.05) > 0:
        return 0.05
    if objective(0.90) < 0:
        return 0.90
    return float(brentq(objective, 0.05, 0.90, xtol=1e-5))


def vasicek_systematic_factor(
    dr_series: pd.Series | np.ndarray,
    pd_ttc: float = DEFAULT_PD_TTC,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Closed-form inversion of the Vasicek (2002) single-factor model.

    Returns Z_t such that Phi( (Phi^{-1}(PD_TTC) - sqrt(rho) Z_t) / sqrt(1-rho) )
    equals the observed default rate.
    """
    dr = _clip_dr(np.asarray(dr_series, dtype=float))
    z = (norm.ppf(pd_ttc) - np.sqrt(1 - rho) * norm.ppf(dr)) / np.sqrt(rho)
    return z


def fit_vasicek(dr_series: pd.Series | np.ndarray, pd_ttc: float = DEFAULT_PD_TTC) -> VasicekFit:
    dr = np.asarray(dr_series, dtype=float)
    rho = calibrate_rho(dr, pd_ttc=pd_ttc)
    z = vasicek_systematic_factor(dr, pd_ttc=pd_ttc, rho=rho)
    return VasicekFit(rho=rho, pd_ttc=pd_ttc, z_path=z, observed_dr_std=float(dr.std(ddof=1)))


def conditional_pd(z: np.ndarray, pd_ttc: float, rho: float) -> np.ndarray:
    """Vasicek-implied conditional default probability given systematic factor Z_t."""
    return norm.cdf((norm.ppf(pd_ttc) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def reconstruct_dr(z: np.ndarray, pd_ttc: float, rho: float) -> np.ndarray:
    return conditional_pd(z, pd_ttc=pd_ttc, rho=rho)
