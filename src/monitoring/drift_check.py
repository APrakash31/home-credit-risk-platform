import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TOP_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "INST_IS_LATE_MEAN", "CREDIT_TERM", "ANNUITY_INCOME_RATIO",
    "CREDIT_INCOME_RATIO", "YEARS_EMPLOYED", "YEARS_BIRTH",
    "BUREAU_ACTIVE_COUNT",
]

PSI_THRESHOLD = 0.25


def psi(expected, actual, buckets=10):
    """Population Stability Index between two distributions."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if len(expected) < 100 or len(actual) < 100:
        return np.nan

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    a_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)

    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))


def main():
    df = pd.read_parquet(PROCESSED / "features.parquet")

    # Simulate a reference vs current population split.
    # In production these would be the training window and recent applications.
    split = len(df) // 2
    reference = df.iloc[:split]
    current = df.iloc[split:]

    results = []
    for feat in TOP_FEATURES:
        if feat not in df.columns:
            continue
        score = psi(reference[feat], current[feat])
        results.append({
            "feature": feat,
            "psi": round(score, 4) if not np.isnan(score) else None,
            "status": "ALERT" if (score and score > PSI_THRESHOLD) else "stable",
        })

    alerts = [r for r in results if r["status"] == "ALERT"]

    report = {
        "psi_threshold": PSI_THRESHOLD,
        "features_checked": len(results),
        "alerts": len(alerts),
        "results": results,
    }

    out = REPORTS / "drift_report.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"Drift check complete — {len(alerts)} alert(s) of {len(results)} features")
    for r in results:
        print(f"  {r['feature']:28s} PSI={r['psi']}  {r['status']}")

    if alerts:
        print("\nPSI above threshold — investigate and consider retraining.")

    return len(alerts)


if __name__ == "__main__":
    main()