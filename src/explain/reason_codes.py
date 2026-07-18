import numpy as np
import pandas as pd
import shap
import joblib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

REASON_MAP = {
    "EXT_SOURCE": "Limited or unfavourable external credit bureau score",
    "INST_IS_LATE": "History of late payments on previous instalments",
    "INST_DAYS_LATE": "Severity of past payment delays",
    "INST_PAYMENT_DIFF": "Shortfall between amounts due and amounts paid",
    "INST_AMT_PAYMENT": "Pattern of payments on previous instalments",
    "INST_AMT_INSTALMENT": "Size of previous instalment obligations",
    "INST_": "Payment behaviour on previous instalments",
    "CREDIT_TERM": "Repayment term relative to loan size",
    "ANNUITY_INCOME_RATIO": "High debt-service burden relative to income",
    "CREDIT_INCOME_RATIO": "Requested credit is high relative to income",
    "INCOME_PER_PERSON": "Household income relative to dependants",
    "AMT_INCOME": "Reported income level",
    "AMT_ANNUITY": "Size of the repayment obligation",
    "AMT_CREDIT": "Amount of credit requested",
    "AMT_GOODS_PRICE": "Value of goods financed relative to credit",
    "DAYS_EMPLOYED": "Short or unverified employment history",
    "YEARS_EMPLOYED": "Short or unverified employment history",
    "BUREAU_OVERDUE": "Overdue balances recorded at other institutions",
    "BUREAU_ACTIVE": "Number of active credit lines elsewhere",
    "BUREAU_AMT_CREDIT_SUM_DEBT": "Outstanding debt reported at other institutions",
    "BUREAU_AMT_CREDIT_SUM": "Total credit extended by other institutions",
    "BUREAU_DAYS_CREDIT": "Age and recency of credit history elsewhere",
    "BUREAU_": "Credit history reported by other institutions",
    "BB_": "Monthly repayment status on external credit lines",
    "PREV_REFUSAL_RATE": "Previous credit applications were declined",
    "PREV_REFUSED": "Previous credit applications were declined",
    "PREV_APPROVED": "History of previously approved credit",
    "PREV_AMT": "Size of previous credit applications",
    "PREV_": "History of previous applications with this institution",
    "POS_": "Repayment status on previous point-of-sale credit",
    "CC_": "Usage and repayment of revolving credit",
    "AGE_YEARS": "Age-related risk profile of the applicant segment",
    "YEARS_BIRTH": "Age-related risk profile of the applicant segment",
    "DAYS_BIRTH": "Age-related risk profile of the applicant segment",
    "YEARS_REGISTRATION": "Stability of registered personal details",
    "YEARS_ID_PUBLISH": "Recency of identity document update",
    "REGION_RATING": "Risk rating of the applicant's region",
    "REGION_": "Regional characteristics of the applicant's address",
    "ORGANIZATION_TYPE": "Employment sector of the applicant",
    "OCCUPATION_TYPE": "Occupation category of the applicant",
    "NAME_EDUCATION": "Education level recorded on the application",
    "NAME_FAMILY_STATUS": "Family status recorded on the application",
    "NAME_INCOME_TYPE": "Source of the applicant's income",
    "NAME_CONTRACT": "Type of credit product requested",
    "FLAG_DOCUMENT": "Completeness of supporting documentation",
    "FLAG_OWN": "Asset ownership recorded on the application",
    "CNT_FAM_MEMBERS": "Number of dependants in the household",
    "CNT_CHILDREN": "Number of children in the household",
    "OWN_CAR_AGE": "Age of vehicle owned by the applicant",
}


def map_reason(feature: str) -> str:
    for key, reason in REASON_MAP.items():
        if key in feature:
            return reason
    return f"Model factor: {feature}"


class ApplicantExplainer:
    def __init__(self):
        self.model = joblib.load(MODELS / "lgbm_model.pkl")
        self.feature_names = joblib.load(MODELS / "feature_names.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, row: pd.DataFrame, top_n: int = 5) -> dict:
        row = row[self.feature_names]
        prob = float(self.model.predict_proba(row)[:, 1][0])

        sv = self.explainer.shap_values(row)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)

        contrib = pd.DataFrame({
            "feature": self.feature_names,
            "shap": sv,
            "value": row.iloc[0].values,
        })

        risk_up = contrib.sort_values("shap", ascending=False).head(top_n)
        risk_down = contrib.sort_values("shap").head(top_n)

        return {
            "default_probability": round(prob, 4),
            "risk_band": "high" if prob > 0.15 else "medium" if prob > 0.08 else "low",
            "top_risk_drivers": [
                {
                    "feature": r.feature,
                    "shap_value": round(float(r.shap), 4),
                    "feature_value": None if pd.isna(r.value) else round(float(r.value), 4),
                    "reason_code": map_reason(r.feature),
                }
                for r in risk_up.itertuples()
            ],
            "top_protective_factors": [
                {
                    "feature": r.feature,
                    "shap_value": round(float(r.shap), 4),
                    "reason_code": map_reason(r.feature),
                }
                for r in risk_down.itertuples()
            ],
        }


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED / "features.parquet")
    X = df.drop(columns=[c for c in ["TARGET", "SK_ID_CURR"] if c in df.columns])
    X.columns = [c.replace(" ", "_").replace(":", "_").replace(",", "_") for c in X.columns]

    ex = ApplicantExplainer()
    for i in [0, 1, 2]:
        result = ex.explain(X.iloc[[i]])
        print(f"\n--- Applicant {df['SK_ID_CURR'].iloc[i]} ---")
        print(json.dumps(result, indent=2))