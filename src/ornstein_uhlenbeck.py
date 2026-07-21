"""Ornstein-Uhlenbeck (Vasicek) process for the systematic credit factor.

    Z_{t+dt} = Z_t + theta * (mu - Z_t) * dt + sigma * sqrt(dt) * eps_t

Two knobs:
- shock_z: applied to z0 (initial-condition shock) - represents a one-time
  economic shock pushing the starting state.
- target_shock_z: applied to mu (the long-run mean) - represents a lasting
  regime shift; used to keep Z elevated throughout the forecast horizon under
  adverse scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float
    a: float
    b: float


def fit_ou_parameters(z: np.ndarray, dt: float = 1.0 / 12.0) -> OUParams:
    z = np.asarray(z, dtype=float)
    z_t = z[:-1]
    z_tp1 = z[1:]

    b, a = np.polyfit(z_t, z_tp1, 1)
    residuals = z_tp1 - (a + b * z_t)
    s2 = float(np.var(residuals, ddof=1))

    if not (-1 < b < 1):
        b = float(np.clip(b, -0.999, 0.999))

    theta = -np.log(max(b, 1e-6)) / dt
    mu = a / max(1 - b, 1e-6)
    sigma2 = s2 * (-2.0 * np.log(max(b, 1e-6))) / (dt * max(1 - b ** 2, 1e-6))
    sigma = float(np.sqrt(max(sigma2, 1e-8)))

    if not np.isfinite([theta, mu, sigma]).all():
        theta, mu, sigma = 0.20, float(np.mean(z)), 0.55

    return OUParams(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        a=float(a),
        b=float(b),
    )


def simulate_ou_paths(
    z0: float,
    n_steps: int,
    n_paths: int,
    params: OUParams,
    dt: float = 1.0,
    shock_z: float = 0.0,
    target_shock_z: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Forward-simulate Z_t paths.

    Args:
        z0: starting state of the systematic factor.
        n_steps: number of time steps.
        n_paths: number of Monte Carlo paths.
        params: fitted OUParams (theta, mu, sigma).
        dt: time step (default = 1 year for annual resolution).
        shock_z: additive shock applied to z0 at t=0.
        target_shock_z: additive shock applied to mu (long-run mean);
            positive value keeps Z elevated under adverse scenarios.
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = z0 + shock_z
    eps = rng.standard_normal((n_paths, n_steps))
    theta = params.theta
    mu_eff = params.mu + target_shock_z
    sigma = params.sigma
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] + theta * (mu_eff - paths[:, t]) * dt + sigma * np.sqrt(dt) * eps[:, t]
    return paths
