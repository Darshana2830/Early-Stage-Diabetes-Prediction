# Early Stage Diabetes Risk Prediction

An educational machine-learning project using the UCI Early Stage
Diabetes Risk Prediction dataset.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest and other ML classifiers
- AWS S3
- Amazon SageMaker
- Flask
- HTML and CSS

## Dataset
- 520 original records
- 16 input features
- Target: Positive or Negative
- 269 exact duplicate records removed
- 251 records after deduplication

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Gradient Boosting

## Best Model
Random Forest

Accuracy on the recorded test split: 94.12%

## AWS Deployment
Region: ap-southeast-2

S3 bucket: early-diabetes-ml-56b2351db2

SageMaker model: early-diabetes-prediction-model

Endpoint: early-diabetes-prediction-endpoint

Execution role: EarlyDiabetesSageMakerRole

## Project Structure
- data/: original and cleaned datasets
- models/: saved machine-learning models
- scripts/: preparation and training scripts
- deployment/code/: SageMaker inference script
- templates/: Flask HTML page
- app.py: Flask application
- deployment_info.json: AWS configuration
- artifacts/: model comparison results

## Setup
Install dependencies:

    pip install -r requirements-ml.txt
    pip install -r requirements.txt

Run data preparation:

    python scripts/prepare_data.py

Train models:

    python scripts/train.py

Run web application:

    python app.py

Configure AWS credentials securely before using the live endpoint.

## Disclaimer
This is an educational machine-learning demonstration, not a
medical diagnostic device. Predictions must not guide healthcare decisions.