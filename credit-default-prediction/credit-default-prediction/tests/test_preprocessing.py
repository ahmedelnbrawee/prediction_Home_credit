import numpy as np
import pandas as pd

from src.preprocessing import full_preprocess


def sample_row(**overrides):
    row = {
        "SK_ID_CURR": 1, "FLAG_MOBIL": 1, "AMT_GOODS_PRICE": 1000.0,
        "AMT_CREDIT": 500000.0, "AMT_INCOME_TOTAL": 100000.0, "AMT_ANNUITY": 25000.0,
        "DAYS_BIRTH": -12775, "DAYS_EMPLOYED": -1825, "CODE_GENDER": "F",
        "OWN_CAR_AGE": np.nan, "CNT_FAM_MEMBERS": np.nan,
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.nan,
        "EXT_SOURCE_1": 0.3, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.7,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_features_and_drops():
    out = full_preprocess(sample_row())
    assert {"SK_ID_CURR", "FLAG_MOBIL", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED"}.isdisjoint(out.columns)
    assert out.loc[0, "AGE_YEARS"] == 35
    assert out.loc[0, "YEARS_EMPLOYED"] == 5
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == 5
    assert out.loc[0, "CREDIT_TERM"] == 20
    assert np.isclose(out.loc[0, "EXT_SOURCE_MEAN"], 0.5)


def test_missing_values_filled():
    out = full_preprocess(sample_row())
    assert out.loc[0, "OWN_CAR_AGE"] == -1
    assert out.loc[0, "AMT_REQ_CREDIT_BUREAU_YEAR"] == 0
    assert out.loc[0, "CNT_FAM_MEMBERS"] == 2


def test_employment_anomaly_flagged():
    out = full_preprocess(sample_row(DAYS_EMPLOYED=365243))
    assert out.loc[0, "DAYS_EMPLOYED_ANOM"] == 1
    assert np.isnan(out.loc[0, "YEARS_EMPLOYED"])


def test_input_not_mutated():
    df = sample_row()
    before = df.copy()
    full_preprocess(df)
    pd.testing.assert_frame_equal(df, before)
