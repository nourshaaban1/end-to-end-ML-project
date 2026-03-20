from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route("/")
def home():
    """Render the main prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    error = None
    result = None

    try:
        # Collect form data
        features = CustomData(
            gender=request.form.get("gender"),
            SeniorCitizen=request.form.get("SeniorCitizen"),
            Partner=request.form.get("Partner"),
            Dependents=request.form.get("Dependents"),
            tenure=int(request.form.get("tenure")),
            PhoneService=request.form.get("PhoneService"),
            MultipleLines=request.form.get("MultipleLines"),
            InternetService=request.form.get("InternetService"),
            OnlineSecurity=request.form.get("OnlineSecurity"),
            OnlineBackup=request.form.get("OnlineBackup"),
            DeviceProtection=request.form.get("DeviceProtection"),
            TechSupport=request.form.get("TechSupport"),
            StreamingTV=request.form.get("StreamingTV"),
            StreamingMovies=request.form.get("StreamingMovies"),
            Contract=request.form.get("Contract"),
            PaperlessBilling=request.form.get("PaperlessBilling"),
            PaymentMethod=request.form.get("PaymentMethod"),
            MonthlyCharges=float(request.form.get("MonthlyCharges")),
            TotalCharges=float(request.form.get("TotalCharges")),
        )

        # Convert to DataFrame
        features_df = features.to_dataframe()

        # Make prediction
        pipeline = PredictPipeline()
        prediction, label_encoder = pipeline.predict(features_df)

        # Get probability scores
        preprocessor, _, model = pipeline.load_all_objects()
        features_transformed = preprocessor.transform(features_df)
        proba = model.predict_proba(features_transformed)[0]

        # Prepare result
        predicted_class = label_encoder.inverse_transform([int(prediction[0])])[0]
        churn_probability = proba[1] if len(proba) > 1 else proba[0]

        result = {
            "prediction": predicted_class,
            "churn_probability": churn_probability,
            "confidence": max(proba),
        }

    except Exception as e:
        error = str(e)

    return render_template("result.html", result=result, error=error)


@app.route("/predict-api", methods=["POST"])
def predict_api():
    """API endpoint for prediction (JSON input)."""
    try:
        data = request.get_json()

        features = CustomData(
            gender=data.get("gender"),
            SeniorCitizen=str(data.get("SeniorCitizen")),
            Partner=data.get("Partner"),
            Dependents=data.get("Dependents"),
            tenure=int(data.get("tenure")),
            PhoneService=data.get("PhoneService"),
            MultipleLines=data.get("MultipleLines"),
            InternetService=data.get("InternetService"),
            OnlineSecurity=data.get("OnlineSecurity"),
            OnlineBackup=data.get("OnlineBackup"),
            DeviceProtection=data.get("DeviceProtection"),
            TechSupport=data.get("TechSupport"),
            StreamingTV=data.get("StreamingTV"),
            StreamingMovies=data.get("StreamingMovies"),
            Contract=data.get("Contract"),
            PaperlessBilling=data.get("PaperlessBilling"),
            PaymentMethod=data.get("PaymentMethod"),
            MonthlyCharges=float(data.get("MonthlyCharges")),
            TotalCharges=float(data.get("TotalCharges")),
        )

        features_df = features.to_dataframe()
        pipeline = PredictPipeline()
        prediction, label_encoder = pipeline.predict(features_df)

        preprocessor, _, model = pipeline.load_all_objects()
        features_transformed = preprocessor.transform(features_df)
        proba = model.predict_proba(features_transformed)[0]

        predicted_class = label_encoder.inverse_transform([int(prediction[0])])[0]
        churn_probability = proba[1] if len(proba) > 1 else proba[0]

        result = {
            "prediction": predicted_class,
            "churn_probability": round(churn_probability, 4),
            "confidence": round(max(proba), 4),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Ensure artifacts exist
    artifacts_dir = Path(os.path.join(os.getcwd(), "artifacts"))
    model_path = artifacts_dir / "model.pkl"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run the training pipeline first: python -m src.pipeline.train_pipeline")
        sys.exit(1)

    app.run(host="0.0.0.0", port=5000)
