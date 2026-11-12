
import os
import boto3
from flask import Flask, render_template, request

app = Flask(__name__)

REGION = "ap-southeast-2"
ENDPOINT = "early-diabetes-prediction-endpoint"

FEATURES = [
    "age", "gender", "polyuria", "polydipsia",
    "sudden_weight_loss", "weakness", "polyphagia",
    "genital_thrush", "visual_blurring", "itching",
    "irritability", "delayed_healing", "partial_paresis",
    "muscle_stiffness", "alopecia", "obesity"
]

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            values = []

            for feature in FEATURES:
                value = request.form.get(feature)

                if feature == "age":
                    values.append(float(value))
                else:
                    values.append(int(value))

            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT,
                ContentType="application/json",
                Body=__import__("json").dumps({
                    "instances": [values]
                })
            )

            import json
            prediction = json.loads(
                response["Body"].read().decode("utf-8")
            )

            result = prediction["predictions"][0]

        except Exception as exc:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Please check your inputs and try again."

    return render_template(
        "index.html", result=result, error=error
    )

@app.route("/health")
def health():
    return {"status": "running"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
