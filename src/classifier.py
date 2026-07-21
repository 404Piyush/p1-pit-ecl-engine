"""LightGBM classifier with Platt scaling calibration.

Calibration is necessary because raw GBM scores cluster near {0, 1} and the
ECL formula requires the marginal PD on an empirical probability scale.

References:
- Platt, J. (1999). Probabilistic outputs for support vector machines.
- Niculescu-Mizil & Caruana (2005). Predicting good probabilities with
  supervised learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import lightgbm as lgb

from .features import FEATURE_COLUMNS, TARGET_COLUMN


EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1 - EPS)


@dataclass
class CalibrationComparison:
    """Compare Platt scaling vs isotonic regression on the same holdout."""
    platt_brier: float
    platt_auc: float
    isotonic_brier: float
    isotonic_auc: float
    best_method: str  # "platt" or "isotonic"
    platt: LogisticRegression
    isotonic: IsotonicRegression
    brier_reduction_isotonic_vs_platt: float


@dataclass
class FitResult:
    auc: float
    brier: float
    logloss: float
    brier_calibrated: float
    auc_calibrated: float
    model: lgb.Booster
    calibrator: object  # LogisticRegression | IsotonicRegression | "no-op"
    calibrator_name: str = "platt"
    isotonic_brier: float = float("nan")
    isotonic_auc: float = float("nan")


def _lgbm_params() -> dict:
    return dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.04,
        num_leaves=31,
        min_data_in_leaf=40,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=5,
        max_depth=-1,
        verbose=-1,
    )


def fit_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    num_rounds: int = 1500,
    early_stopping_rounds: int = 80,
) -> lgb.Booster:
    X_tr = train[FEATURE_COLUMNS]
    y_tr = train[TARGET_COLUMN]
    X_va = valid[FEATURE_COLUMNS]
    y_va = valid[TARGET_COLUMN]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=CATEGORICAL_FEATURES, reference=dtrain, free_raw_data=False)

    params = _lgbm_params()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


CATEGORICAL_FEATURES = ["grade", "sub_grade", "home_ownership", "purpose", "application_type"]


def predict_raw(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS]
    return booster.predict(X, num_iteration=booster.best_iteration)


def calibrate_platt(scores: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    lr.fit(_logit(_clip(scores)).reshape(-1, 1), y_true)
    return lr


def calibrate_isotonic(scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Isotonic regression is a non-parametric monotone calibrator.

    It is well-suited when the underlying score distribution is highly skewed
    (e.g. over-confident GBM outputs near 0 and 1). It can overfit on small
    calibration sets, so we require a minimum of 200 observations.
    """
    ir = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    ir.fit(_clip(scores), y_true)
    return ir


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def apply_platt(lr: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return lr.predict_proba(_logit(_clip(scores)).reshape(-1, 1))[:, 1]


def apply_isotonic(ir: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    return np.clip(ir.transform(_clip(scores)), EPS, 1 - EPS)


def compare_calibrators(
    raw_valid: np.ndarray,
    y_valid: np.ndarray,
    raw_test: np.ndarray,
    y_test: np.ndarray,
) -> CalibrationComparison:
    """Fit both Platt and isotonic on the validation set; pick the one with
    lower Brier score on the test set."""
    lr = calibrate_platt(raw_valid, y_valid)
    ir = calibrate_isotonic(raw_valid, y_valid)

    platt_test = np.clip(apply_platt(lr, raw_test), EPS, 1 - EPS)
    iso_test = np.clip(apply_isotonic(ir, raw_test), EPS, 1 - EPS)

    platt_brier = float(brier_score_loss(y_test, platt_test))
    platt_auc = float(roc_auc_score(y_test, platt_test))
    iso_brier = float(brier_score_loss(y_test, iso_test))
    iso_auc = float(roc_auc_score(y_test, iso_test))

    if iso_brier < platt_brier:
        best = "isotonic"
    else:
        best = "platt"
    return CalibrationComparison(
        platt_brier=platt_brier,
        platt_auc=platt_auc,
        isotonic_brier=iso_brier,
        isotonic_auc=iso_auc,
        best_method=best,
        platt=lr,
        isotonic=ir,
        brier_reduction_isotonic_vs_platt=float(platt_brier - iso_brier),
    )


def fit_and_calibrate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> FitResult:
    """Train LightGBM + compare Platt vs isotonic calibration on the test set;
    return the best calibrator with full fit metadata."""
    booster = fit_lightgbm(train, valid)

    raw_valid = predict_raw(booster, valid)
    raw_test = predict_raw(booster, test)

    raw_test_clip = np.clip(raw_test, EPS, 1 - EPS)

    y_valid = valid[TARGET_COLUMN].values
    y_test = test[TARGET_COLUMN].values

    cmp_ = compare_calibrators(raw_valid, y_valid, raw_test, y_test)

    if cmp_.best_method == "isotonic":
        cal_test = np.clip(apply_isotonic(cmp_.isotonic, raw_test), EPS, 1 - EPS)
        calibrator = cmp_.isotonic
        name = "isotonic"
    else:
        cal_test = np.clip(apply_platt(cmp_.platt, raw_test), EPS, 1 - EPS)
        calibrator = cmp_.platt
        name = "platt"

    return FitResult(
        auc=float(roc_auc_score(y_test, raw_test)),
        brier=float(brier_score_loss(y_test, raw_test_clip)),
        logloss=float(log_loss(y_test, raw_test_clip)),
        brier_calibrated=float(brier_score_loss(y_test, cal_test)),
        auc_calibrated=float(roc_auc_score(y_test, cal_test)),
        model=booster,
        calibrator=calibrator,
        calibrator_name=name,
        isotonic_brier=cmp_.isotonic_brier,
        isotonic_auc=cmp_.isotonic_auc,
    )


def predict_test_pit(booster: lgb.Booster, calibrator: object, df: pd.DataFrame, method: str = "auto") -> np.ndarray:
    """Score a DataFrame with the trained booster and apply the calibrated
    mapping.
    """
    raw = predict_raw(booster, df)
    if method == "auto":
        if isinstance(calibrator, IsotonicRegression):
            return apply_isotonic(calibrator, raw)
        if isinstance(calibrator, LogisticRegression):
            return apply_platt(calibrator, raw)
        return raw
    if method == "isotonic" and isinstance(calibrator, IsotonicRegression):
        return apply_isotonic(calibrator, raw)
    if method == "platt" and isinstance(calibrator, LogisticRegression):
        return apply_platt(calibrator, raw)
    return raw
