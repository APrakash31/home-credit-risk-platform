import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 34,
    "max_depth": 8,
    "min_child_samples": 100,
    "subsample": 0.87,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.04,
    "reg_lambda": 0.07,
    "is_unbalance": True,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def load_data():
    df = pd.read_parquet(PROCESSED / "features.parquet")
    y = df["TARGET"]

    # Protected characteristics — excluded per credit policy section 6
    protected = [c for c in df.columns if c.startswith("CODE_GENDER")]

    drop = ["TARGET", "SK_ID_CURR"] + protected
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X.columns = [c.replace(" ", "_").replace(":", "_").replace(",", "_") for c in X.columns]
    print(f"Excluded protected features: {protected}")
    return X, y, df["SK_ID_CURR"]


def train():
    X, y, ids = load_data()
    print(f"Training on {X.shape[0]:,} rows x {X.shape[1]:,} features")
    print(f"Default rate: {y.mean():.2%}")

    mlflow.set_experiment("home-credit-risk")

    with mlflow.start_run():
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_features", X.shape[1])

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(X))
        scores = []
        models = []

        for fold, (tr_idx, va_idx) in enumerate(folds.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            model = lgb.LGBMClassifier(n_estimators=3000, **PARAMS)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )

            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            score = roc_auc_score(y_va, oof[va_idx])
            scores.append(score)
            models.append(model)
            print(f"  Fold {fold}: AUC = {score:.5f}  (best iter {model.best_iteration_})")

        overall = roc_auc_score(y, oof)
        print(f"\nOOF AUC: {overall:.5f}   mean {np.mean(scores):.5f} +/- {np.std(scores):.5f}")

        mlflow.log_metric("oof_auc", overall)
        mlflow.log_metric("cv_auc_mean", np.mean(scores))
        mlflow.log_metric("cv_auc_std", np.std(scores))

        best = models[int(np.argmax(scores))]
        joblib.dump(best, MODELS / "lgbm_model.pkl")
        joblib.dump(list(X.columns), MODELS / "feature_names.pkl")
        mlflow.lightgbm.log_model(best.booster_, "model")

        pd.DataFrame({"SK_ID_CURR": ids, "pred": oof, "actual": y}).to_parquet(
            PROCESSED / "oof_predictions.parquet", index=False
        )

        importance = pd.DataFrame({
            "feature": X.columns,
            "gain": best.booster_.feature_importance("gain"),
        }).sort_values("gain", ascending=False)
        importance.to_csv(MODELS / "feature_importance.csv", index=False)

        print("\nTop 15 features:")
        print(importance.head(15).to_string(index=False))

    return overall


if __name__ == "__main__":
    train()