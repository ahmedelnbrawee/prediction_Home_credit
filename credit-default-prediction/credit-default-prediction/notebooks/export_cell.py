# Run this cell at the END of the training notebook (after the model is trained).
# It saves the model + app artifacts into ../models and writes a pinned requirements.txt
# so the deployed environment matches the training environment.
import os
import joblib
from importlib.metadata import version

os.makedirs("models", exist_ok=True)
raw = df.drop(columns=["TARGET"])

defaults = {c: (raw[c].mode()[0] if raw[c].dtype == "object" else raw[c].median())
            for c in raw.columns}

cat_cols = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_EDUCATION_TYPE",
            "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]
cat_options = {c: sorted(x for x in raw[c].dropna().unique().tolist() if x != "XNA")
               for c in cat_cols}

joblib.dump(pipeline_voting, "models/credit_model.joblib", compress=3)
joblib.dump({
    "raw_columns": list(raw.columns),
    "defaults": defaults,
    "cat_options": cat_options,
    "model_columns": list(X_train.columns),
    "threshold": custom_threshold,
    "val_auc": float(auc_score_voting),
}, "models/app_artifacts.joblib", compress=3)

pkgs = ["streamlit", "pandas", "numpy", "scikit-learn", "imbalanced-learn",
        "lightgbm", "xgboost", "joblib"]
with open("requirements.txt", "w") as f:
    for p in pkgs:
        try:
            f.write(f"{p}=={version(p)}\n")
        except Exception:
            f.write(f"{p}\n")  # e.g. streamlit is not installed in Colab by default

print("Saved models/credit_model.joblib, models/app_artifacts.joblib and requirements.txt")
print(open("requirements.txt").read())
