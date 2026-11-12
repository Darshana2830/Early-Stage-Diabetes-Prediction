
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data/early_stage_diabetes.csv")
df = df.drop_duplicates().reset_index(drop=True)

mapping = {
    "Yes": 1, "No": 0,
    "Male": 1, "Female": 0,
    "Positive": 1, "Negative": 0
}

for col in df.select_dtypes(include=["str", "object"]).columns:
    df[col] = df[col].map(mapping)

df.to_csv(ROOT / "data/diabetes_clean.csv", index=False)

print("Cleaned dataset saved:", df.shape)
