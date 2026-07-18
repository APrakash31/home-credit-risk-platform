import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SAMPLE_SIZE = 5000


def load():
    model = joblib.load(MODELS / "lgbm_model.pkl")
    feature_names = joblib.load(MODELS / "feature_names.pkl")
    df = pd.read_parquet(PROCESSED / "features.parquet")
    ids = df["SK_ID_CURR"]
    X = df.drop(columns=[c for c in ["TARGET", "SK_ID_CURR"] if c in df.columns])
    X.columns = [c.replace(" ", "_").replace(":", "_").replace(",", "_") for c in X.columns]
    X = X[feature_names]
    return model, X, ids


def main():
    model, X, ids = load()

    sample = X.sample(n=min(SAMPLE_SIZE, len(X)), random_state=42)
    print(f"Computing SHAP values on {len(sample):,} rows...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs.to_csv(REPORTS / "shap_global_importance.csv", index=False)

    joblib.dump(explainer, MODELS / "shap_explainer.pkl")

    print("\nTop 15 by mean |SHAP|:")
    print(mean_abs.head(15).to_string(index=False))
    print(f"\nSaved plots to {REPORTS}")


if __name__ == "__main__":
    main()