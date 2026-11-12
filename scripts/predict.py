
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(ROOT / "models/best_diabetes_model.joblib")
features = joblib.load(ROOT / "models/feature_names.joblib")

sample = [40, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1]

data = pd.DataFrame([sample], columns=features)
prediction = model.predict(data)[0]

print("Prediction:", "Positive" if prediction == 1 else "Negative")
