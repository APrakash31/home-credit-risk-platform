import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DEMO = ROOT / "app" / "demo_data"
DEMO.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PROCESSED / "features.parquet")

# Stratified sample preserving the default rate
sample = df.groupby("TARGET", group_keys=False).apply(
    lambda g: g.sample(n=int(5000 * len(g) / len(df)), random_state=42)
)

sample.to_parquet(DEMO / "features_sample.parquet", index=False)
print(f"Sample: {len(sample):,} rows, default rate {sample['TARGET'].mean():.2%}")
print(f"Size: {(DEMO / 'features_sample.parquet').stat().st_size / 1_048_576:.1f} MB")