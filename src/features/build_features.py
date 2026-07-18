import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_application():
    df = pd.read_csv(RAW / "application_train.csv")

    # Sentinel value found in EDA: 365243 means "not employed"
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Convert negative day counts to positive years
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]:
        df[col.replace("DAYS_", "YEARS_")] = -df[col] / 365

    # Domain ratios — these consistently rank among the strongest features
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    return df


def aggregate_numeric(df, group_col, prefix):
    """Collapse a child table to one row per applicant."""
    numeric = df.select_dtypes(include=["number"]).drop(
        columns=[c for c in df.columns if c.startswith("SK_ID")], errors="ignore"
    )
    numeric[group_col] = df[group_col]

    agg = numeric.groupby(group_col).agg(["mean", "max", "min", "sum", "count"])
    agg.columns = [f"{prefix}_{col}_{stat.upper()}" for col, stat in agg.columns]
    return agg.reset_index()


def build_bureau_features():
    bureau = pd.read_csv(RAW / "bureau.csv")
    bb = pd.read_csv(RAW / "bureau_balance.csv")

    bb_agg = aggregate_numeric(bb, "SK_ID_BUREAU", "BB")
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    agg = aggregate_numeric(bureau, "SK_ID_CURR", "BUREAU")

    # Behavioral flags carry real signal
    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").size()
    agg["BUREAU_ACTIVE_COUNT"] = agg["SK_ID_CURR"].map(active).fillna(0)

    overdue = bureau[bureau["AMT_CREDIT_SUM_OVERDUE"] > 0].groupby("SK_ID_CURR").size()
    agg["BUREAU_OVERDUE_COUNT"] = agg["SK_ID_CURR"].map(overdue).fillna(0)

    return agg


def build_previous_features():
    prev = pd.read_csv(RAW / "previous_application.csv")
    agg = aggregate_numeric(prev, "SK_ID_CURR", "PREV")

    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR").size()
    refused = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size()

    agg["PREV_APPROVED_COUNT"] = agg["SK_ID_CURR"].map(approved).fillna(0)
    agg["PREV_REFUSED_COUNT"] = agg["SK_ID_CURR"].map(refused).fillna(0)
    agg["PREV_REFUSAL_RATE"] = agg["PREV_REFUSED_COUNT"] / (
        agg["PREV_APPROVED_COUNT"] + agg["PREV_REFUSED_COUNT"]
    ).replace(0, np.nan)

    return agg


def build_installment_features():
    inst = pd.read_csv(RAW / "installments_payments.csv")

    # Payment behavior — the strongest behavioral signal in the dataset
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(int)

    agg = aggregate_numeric(inst, "SK_ID_CURR", "INST")

    late_rate = inst.groupby("SK_ID_CURR")["IS_LATE"].mean()
    agg["INST_LATE_RATE"] = agg["SK_ID_CURR"].map(late_rate)

    return agg


def main():
    print("Loading application data...")
    df = load_application()

    for name, builder in [
        ("bureau", build_bureau_features),
        ("previous", build_previous_features),
        ("installments", build_installment_features),
    ]:
        print(f"Building {name} features...")
        feats = builder()
        df = df.merge(feats, on="SK_ID_CURR", how="left")
        print(f"  shape now: {df.shape}")

    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=list(cat_cols), dummy_na=True)

    # Clean up infinities from ratio division
    df = df.replace([np.inf, -np.inf], np.nan)

    out = PROCESSED / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {df.shape[0]:,} rows x {df.shape[1]:,} columns to {out}")


if __name__ == "__main__":
    main()