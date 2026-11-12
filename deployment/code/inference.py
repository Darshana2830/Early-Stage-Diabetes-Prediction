
import os
import joblib
import numpy as np

def model_fn(model_dir):
    model = joblib.load(
        os.path.join(model_dir, "best_diabetes_model.joblib")
    )
    features = joblib.load(
        os.path.join(model_dir, "feature_names.joblib")
    )
    return model, features

def input_fn(request_body, request_content_type):
    import json
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["instances"])
    raise ValueError("Unsupported content type")

def predict_fn(input_data, model_and_features):
    model, features = model_and_features
    return model.predict(input_data)

def output_fn(prediction, accept):
    import json
    results = [
        "Positive" if int(value) == 1 else "Negative"
        for value in prediction
    ]
    return json.dumps({"predictions": results}), "application/json"
