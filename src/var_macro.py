"""Vector AutoRegression (VAR) module for macro variables.

Implements a small VAR(p) on the macro panel (GDP growth, CPI y/y, repo rate,
unemployment, default rate) to obtain joint forecast distributions. These
forecasts inform the OU-process shock points used by the simulation.

References:
- Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
- Stock & Watson (2001). Vector Autoregressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as StatsmodelsVAR


MACRO_COLUMNS: List[str] = [
    "gdp_growth",
    "cpi_yoy",
    "repo_rate",
    "unemployment",
    "default_rate",
]


def _resolve_columns(macro: pd.DataFrame, requested: List[str]) -> List[str]:
    """Map requested columns to whatever is actually in the macro frame.

    Allows the data generator's naming to vary (e.g. `unemployment` vs
    `unemployment_rate`) without breaking the pipeline.
    """
    out = []
    for c in requested:
        if c in macro.columns:
            out.append(c)
        else:
            if c == "unemployment" and "unemployment_rate" in macro.columns:
                out.append("unemployment_rate")
                continue
            out.append(c)  # will raise downstream if truly missing
    return out


@dataclass
class VARFit:
    """Container for a fitted VAR model."""
    var_model: StatsmodelsVAR
    fitted_results: object
    lag_order: int
    aic: float
    bic: float
    columns: List[str]

    def forecast(self, steps: int, last_obs: np.ndarray) -> np.ndarray:
        """Forecast `steps` periods ahead, conditional on `last_obs`."""
        fc = self.fitted_results.forecast(y=last_obs, steps=steps)
        return np.asarray(fc)


def fit_var(macro: pd.DataFrame, columns: List[str] = None, max_lag: int = 4) -> VARFit:
    """Fit a VAR on the macro panel using AIC-selected lag order."""
    requested = columns or MACRO_COLUMNS
    cols = [c for c in requested if c in macro.columns]
    data = macro[cols].astype(float).copy()
    # drop first rows with NaN after differencing if needed
    data = data.dropna()
    model = StatsmodelsVAR(data)
    try:
        sel = model.select_order(maxlags=max_lag)
        lag_order = int(sel.aic if sel.aic is not None else (sel.bic or 1))
        lag_order = max(1, min(lag_order, max_lag))
    except Exception:
        lag_order = 1
    res = model.fit(lag_order)
    return VARFit(
        var_model=model,
        fitted_results=res,
        lag_order=lag_order,
        aic=float(res.aic),
        bic=float(res.bic),
        columns=cols,
    )


def forecast_default_rate_path(
    fit: VARFit,
    macro: pd.DataFrame,
    horizon: int,
    shock_z: float = 0.0,
) -> np.ndarray:
    """Forecast the next `horizon` months of default_rate from the VAR.

    `shock_z` is an additive shock on the systematic factor scale applied to
    the default_rate column at every step (used to construct adverse
    scenarios).
    """
    last_obs = macro[fit.columns].tail(fit.lag_order).values
    fc = fit.forecast(horizon, last_obs)
    if shock_z != 0.0:
        # Translate z-shock into a default-rate shock.
        # Empirical rule: 1 std-dev of z ~ 0.01 default-rate.
        fc[:, fit.columns.index("default_rate")] += 0.01 * shock_z
        fc[:, fit.columns.index("default_rate")] = np.clip(fc[:, fit.columns.index("default_rate")], 0.0, 1.0)
    return fc[:, fit.columns.index("default_rate")]


def macro_scenario_summary(fc: np.ndarray, columns: List[str]) -> Dict[str, float]:
    """Return last-period forecast for each macro column."""
    if fc.ndim == 1:
        return {col: float(fc[-1]) for col in columns}
    return {col: float(fc[-1, i]) for i, col in enumerate(columns)}
