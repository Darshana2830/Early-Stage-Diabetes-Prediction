
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/diabetes_clean.csv")

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "logistic_regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "svm": make_pipeline(StandardScaler(), SVC()),
    "knn": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = round(
        accuracy_score(y_test, predictions) * 100, 2
    )
    joblib.dump(model, ROOT / "models" / (name + ".joblib"))

best_name = max(results, key=results.get)
joblib.dump(
    models[best_name],
    ROOT / "models/best_diabetes_model.joblib"
)
joblib.dump(list(X.columns), ROOT / "models/feature_names.joblib")

print("Training completed!")
print("Accuracy:", results)
print("Best model:", best_name)
