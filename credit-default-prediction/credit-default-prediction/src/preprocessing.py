"""Preprocessing shared by the training notebook and the Streamlit app.

Mirrors the notebook steps (drop columns -> missing values -> anomalies ->
feature engineering) for a *raw* dataframe that has no TARGET column.
"""
import numpy as np
import pandas as pd

DROP_COLS = ["SK_ID_CURR", "FLAG_MOBIL", "AMT_GOODS_PRICE"]
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
DAYS_EMPLOYED_ANOMALY = 365243
DEFAULT_FAM_MEMBERS = 2  # mode of CNT_FAM_MEMBERS in the training set


def full_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Drop unnecessary columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2) Missing values
    for col in [c for c in df.columns if "AMT_REQ_CREDIT_BUREAU" in c]:
        df[col] = df[col].fillna(0)
    if "OWN_CAR_AGE" in df.columns:
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(-1)
    if "CNT_FAM_MEMBERS" in df.columns:
        df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].fillna(DEFAULT_FAM_MEMBERS)

    # 3) Anomalies
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({DAYS_EMPLOYED_ANOMALY: np.nan})
    df["CODE_GENDER"] = df["CODE_GENDER"].replace({"XNA": np.nan})

    # 4) Feature engineering
    df["AGE_YEARS"] = df["DAYS_BIRTH"].abs() / 365
    df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].abs() / 365
    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    ext = [c for c in EXT_COLS if c in df.columns]
    if ext:
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)

    return df
