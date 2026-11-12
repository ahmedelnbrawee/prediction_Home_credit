"""Credit Default Prediction - Streamlit app."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import full_preprocess

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "credit_model.joblib"
ARTIFACTS_PATH = BASE_DIR / "models" / "app_artifacts.joblib"

st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    return joblib.load(MODEL_PATH), joblib.load(ARTIFACTS_PATH)


if not (MODEL_PATH.exists() and ARTIFACTS_PATH.exists()):
    st.error(
        "Model files not found. Place `credit_model.joblib` and "
        "`app_artifacts.joblib` inside the `models/` folder "
        "(see README for how to export them from the notebook)."
    )
    st.stop()

model, art = load_assets()
defaults = art["defaults"]
opts = art["cat_options"]
model_columns = art["model_columns"]
raw_columns = art["raw_columns"]


def predict_proba(raw_df: pd.DataFrame):
    X = full_preprocess(raw_df).reindex(columns=model_columns)
    return model.predict_proba(X)[:, 1]


def risk_banner(p: float, thr: float):
    if p >= thr:
        st.error(f"High risk: probability is above the decision threshold ({thr:.2f}).")
    elif p >= thr * 0.6:
        st.warning("Medium risk: close to the decision threshold.")
    else:
        st.success("Low risk: probability is well below the decision threshold.")


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("💳 Credit Risk")
    st.caption("Default probability estimation using a soft-voting ensemble "
               "(Logistic Regression, Random Forest, LightGBM, XGBoost) trained with SMOTE.")
    st.divider()
    threshold = st.slider(
        "Decision threshold", 0.05, 0.90, float(art.get("threshold", 0.25)), 0.05,
        help="Clients with probability >= threshold are flagged as high risk.",
    )
    if art.get("val_auc") is not None:
        st.metric("Validation ROC-AUC", f"{art['val_auc']:.4f}")
    st.divider()
    st.caption("This tool is for educational purposes and should not be the only "
               "basis for real lending decisions.")

st.title("Credit Default Prediction")
tab_single, tab_batch = st.tabs(["Single client", "Batch (CSV)"])

# ------------------------------------------------------------ Single client
with tab_single:
    with st.form("client_form"):
        st.subheader("Personal information")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", opts["CODE_GENDER"])
        age = c2.slider("Age (years)", 18, 70, 35)
        children = c3.number_input("Children", 0, 20, 0)
        family = c1.selectbox("Family status", opts["NAME_FAMILY_STATUS"])
        education = c2.selectbox("Education", opts["NAME_EDUCATION_TYPE"])
        housing = c3.selectbox("Housing type", opts["NAME_HOUSING_TYPE"])
        own_car = c1.selectbox("Owns a car", opts["FLAG_OWN_CAR"])
        own_realty = c2.selectbox("Owns real estate", opts["FLAG_OWN_REALTY"])

        st.subheader("Employment & income")
        c4, c5, c6 = st.columns(3)
        income_type = c4.selectbox("Income type", opts["NAME_INCOME_TYPE"])
        years_emp = c5.slider("Years employed", 0.0, 45.0, 5.0, 0.5)
        income = c6.number_input("Total yearly income", 10_000.0, 10_000_000.0, 150_000.0, 5_000.0)

        st.subheader("Loan")
        c7, c8 = st.columns(2)
        credit = c7.number_input("Loan amount", 10_000.0, 5_000_000.0, 500_000.0, 10_000.0)
        annuity = c8.number_input("Yearly annuity", 1_000.0, 300_000.0, 25_000.0, 1_000.0)

        st.subheader("External scores (0 to 1)")
        e1, e2, e3 = st.columns(3)
        ext1 = e1.slider("EXT_SOURCE_1", 0.0, 1.0, float(round(defaults["EXT_SOURCE_1"], 2)), 0.01)
        ext2 = e2.slider("EXT_SOURCE_2", 0.0, 1.0, float(round(defaults["EXT_SOURCE_2"], 2)), 0.01)
        ext3 = e3.slider("EXT_SOURCE_3", 0.0, 1.0, float(round(defaults["EXT_SOURCE_3"], 2)), 0.01)

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        row = dict(defaults)  # every column starts from the training median / mode
        row.update({
            "CODE_GENDER": gender, "FLAG_OWN_CAR": own_car, "FLAG_OWN_REALTY": own_realty,
            "CNT_CHILDREN": children, "NAME_FAMILY_STATUS": family,
            "NAME_EDUCATION_TYPE": education, "NAME_HOUSING_TYPE": housing,
            "NAME_INCOME_TYPE": income_type, "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": -int(age * 365), "DAYS_EMPLOYED": -int(years_emp * 365),
            "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        })
        proba = float(predict_proba(pd.DataFrame([row]))[0])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Default probability", f"{proba:.1%}")
        m2.metric("Credit / income", f"{credit / income:.2f}")
        m3.metric("Loan term (years)", f"{credit / annuity:.1f}")
        st.progress(min(proba, 1.0))
        risk_banner(proba, threshold)

# -------------------------------------------------------------------- Batch
with tab_batch:
    st.write("Upload a CSV with the same columns as the original dataset "
             "(e.g. `test.csv`). Missing optional columns are filled with training defaults.")
    file = st.file_uploader("CSV file", type=["csv"])

    if file is not None:
        raw = pd.read_csv(file)
        required = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
                    "DAYS_BIRTH", "DAYS_EMPLOYED", "CODE_GENDER"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            data = raw.drop(columns=["TARGET"], errors="ignore").copy()
            for col in raw_columns:
                if col not in data.columns:
                    data[col] = defaults[col]

            with st.spinner("Scoring clients..."):
                proba = predict_proba(data)

            ids = raw["SK_ID_CURR"] if "SK_ID_CURR" in raw.columns else pd.Series(raw.index)
            out = pd.DataFrame({
                "SK_ID_CURR": ids,
                "DEFAULT_PROBABILITY": proba.round(6),
                "HIGH_RISK": (proba >= threshold).astype(int),
            })

            k1, k2, k3 = st.columns(3)
            k1.metric("Clients", f"{len(out):,}")
            k2.metric("Flagged high risk", f"{int(out['HIGH_RISK'].sum()):,}")
            k3.metric("Average probability", f"{proba.mean():.1%}")

            st.dataframe(out.head(100), use_container_width=True)
            st.download_button(
                "Download predictions (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
