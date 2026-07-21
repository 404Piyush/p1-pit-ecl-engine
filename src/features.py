"""Feature configuration and leakage-safe train/test splits."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit


TARGET_COLUMN = "default_12m"

LEAKAGE_COLUMNS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "debt_settlement_flag",
    "settlement_status",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]

CATEGORICAL_COLUMNS = ["grade", "sub_grade", "home_ownership", "purpose", "application_type"]

NUMERIC_COLUMNS = [
    "term",
    "int_rate",
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "emp_length",
    "origination_z",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def build_feature_table(loans: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage columns and produce an ML-ready frame."""
    df = loans.copy()
    drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    df["issue_year"] = df["issue_date"].dt.year
    return df


def split_train_test(
    loans: pd.DataFrame,
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = loans[loans["issue_date"] <= train_end].copy()
    test = loans[(loans["issue_date"] >= test_start)].copy()
    return train, test


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)
