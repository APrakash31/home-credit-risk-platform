import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

MIN_AUC = 0.75


def main():
    oof_path = PROCESSED / "oof_predictions.parquet"
    if not oof_path.exists():
        print("No OOF predictions found — skipping validation.")
        return

    df = pd.read_parquet(oof_path)
    auc = roc_auc_score(df["actual"], df["pred"])

    passed = auc >= MIN_AUC
    report = {"auc": round(auc, 5), "threshold": MIN_AUC, "passed": passed}
    (REPORTS / "validation_report.json").write_text(json.dumps(report, indent=2))

    print(f"Model AUC: {auc:.5f}  (threshold {MIN_AUC})")
    if not passed:
        raise SystemExit("Model performance below threshold — pipeline failed.")
    print("Validation passed.")


if __name__ == "__main__":
    main()