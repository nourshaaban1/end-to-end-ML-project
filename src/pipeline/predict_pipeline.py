import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.artifacts_dir = Path(os.path.join(os.getcwd(), "artifacts"))
        self.model_path = self.artifacts_dir / "model.pkl"
        self.preprocessor_path = self.artifacts_dir / "preprocessor.pkl"
        self.label_encoder_path = self.artifacts_dir / "label_encoder.pkl"

    def load_all_objects(self):
        """Load the preprocessor, label encoder, and model from artifacts."""
        try:
            logging.info("Loading preprocessor, label encoder, and model from artifacts")

            preprocessor = load_object(self.preprocessor_path)
            label_encoder = load_object(self.label_encoder_path)
            model = load_object(self.model_path)

            logging.info("Successfully loaded preprocessor, label encoder, and model")
            return preprocessor, label_encoder, model

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features_df):
        """
        Predict churn for new customer data.

        Args:
            features_df: DataFrame with customer features

        Returns:
            Predicted churn labels (encoded) and the label encoder
        """
        try:
            logging.info("Starting prediction pipeline")

            # Load all objects
            preprocessor, label_encoder, model = self.load_all_objects()

            # Apply feature engineering (mirror the training process)
            features_df = self._apply_feature_engineering(features_df)

            # Drop id column if present
            if "id" in features_df.columns:
                features_df = features_df.drop(columns=["id"])

            num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'ServiceCount']

            for col in num_features:
                if col in features_df.columns:
                    features_df[col] = features_df[col].astype('float64')

            # Transform features using the preprocessor
            features_transformed = preprocessor.transform(features_df)

            # Make predictions
            predictions = model.predict(features_transformed).astype(int)


            logging.info(f"Predictions completed. Shape: {predictions.shape}")
            return predictions, label_encoder

        except Exception as e:
            raise CustomException(e, sys)

    def _apply_feature_engineering(self, df: pd.DataFrame):
        """Apply the same feature engineering as in training."""
        try:

            if 'SeniorCitizen' in df.columns:
                df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            # Feature Engineering: Average monthly charge per year of tenure
            if 'TotalCharges' in df.columns and 'tenure' in df.columns:
                df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)

            service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

            df['ServiceCount'] = 0.0
            for col in service_cols:
                if col in df.columns:
                    if col == 'InternetService':
                        df['ServiceCount'] = df['ServiceCount'] + (df[col] != 'No').astype(float)
                    elif col == 'PhoneService':
                        df['ServiceCount'] = df['ServiceCount'] + (df[col] == 'Yes').astype(float)
                    else:
                        df['ServiceCount'] = df['ServiceCount'] + (df[col] == 'Yes').astype(float)

            return df

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Class to hold customer data for prediction."""

    def __init__(
        self,
        gender: str,
        SeniorCitizen: str,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float,
    ):
        self.gender = gender
        self.SeniorCitizen = str(SeniorCitizen)  # FIX Bug 3: enforce str at construction
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def to_dataframe(self):
        """Convert custom data to DataFrame."""
        try:
            data_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Example: Make a prediction
        sample_data = CustomData(
            gender="Male",
            SeniorCitizen="0",
            Partner="No",
            Dependents="No",
            tenure=36,
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="DSL",
            OnlineSecurity="Yes",
            OnlineBackup="No",
            DeviceProtection="Yes",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="One year",
            PaperlessBilling="Yes",
            PaymentMethod="Credit card (automatic)",
            MonthlyCharges=75.0,
            TotalCharges=2700.0,
        )

        features_df = sample_data.to_dataframe()
        print("Sample features:")
        print(features_df)

        pipeline = PredictPipeline()
        predictions, label_encoder = pipeline.predict(features_df)

        print(f"\nPredicted label: {predictions[0]}")
        print(f"Predicted class: {label_encoder.inverse_transform([predictions[0]])[0]}")

    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Prediction failed: {e}")