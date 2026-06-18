import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.neural_network import ChurnClassifier


class PredictPipeline:
    def __init__(self):
        self.artifacts_dir = Path(os.path.join(os.getcwd(), "artifacts"))
        self.model_path = self.artifacts_dir / "model.pkl"
        self.nn_model_path = self.artifacts_dir / "model_nn.pth"
        self.preprocessor_path = self.artifacts_dir / "preprocessor.pkl"
        self.label_encoder_path = self.artifacts_dir / "label_encoder.pkl"
        self.input_features = None  # Will be determined after loading preprocessor

    def load_all_objects(self):
        """Load the preprocessor, label encoder, and models from artifacts."""
        try:
            logging.info("Loading preprocessor, label encoder, and models from artifacts")

            preprocessor = load_object(self.preprocessor_path)
            label_encoder = load_object(self.label_encoder_path)
            model = load_object(self.model_path)

            # Determine input features from preprocessor
            # The preprocessor.transform() returns a numpy array, we need to get the shape
            dummy_input = pd.DataFrame({'tenure': [1], 'MonthlyCharges': [50], 'TotalCharges': [50],
                                        'AvgMonthlyCharge': [50], 'ServiceCount': [1]})
            # Add all categorical columns with dummy values
            cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
            for col in cat_cols:
                dummy_input[col] = 'No'

            transformed = preprocessor.transform(dummy_input)
            self.input_features = transformed.shape[1]

            # Load NN model if available
            nn_model = None
            if self.nn_model_path.exists():
                try:
                    nn_model = ChurnClassifier.load(self.nn_model_path, self.input_features)
                    logging.info("Neural Network model loaded successfully")
                except Exception as e:
                    logging.warning(f"Could not load Neural Network model: {e}")

            logging.info("Successfully loaded preprocessor, label encoder, and models")
            return preprocessor, label_encoder, model, nn_model

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features_df, use_nn: bool = False):
        """
        Predict churn for new customer data.

        Args:
            features_df: DataFrame with customer features
            use_nn: If True, use the Neural Network model; otherwise use sklearn model

        Returns:
            Predicted churn labels (encoded), label_encoder, and prediction probabilities
        """
        try:
            logging.info("Starting prediction pipeline")

            # Load all objects
            preprocessor, label_encoder, model, nn_model = self.load_all_objects()

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
            if use_nn and nn_model is not None:
                logging.info("Using Neural Network model for prediction")
                nn_model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features_transformed)
                    predictions_proba = nn_model(features_tensor).numpy().flatten()
                    predictions = (predictions_proba > 0.5).astype(int)
            else:
                logging.info("Using sklearn model for prediction")
                predictions = model.predict(features_transformed).astype(int)
                predictions_proba = None
                if hasattr(model, "predict_proba"):
                    predictions_proba = model.predict_proba(features_transformed)[:, 1]
                elif hasattr(model, "decision_function"):
                    from sklearn.utils import check_scalar
                    decisions = model.decision_function(features_transformed)
                    from scipy.special import expit
                    predictions_proba = expit(decisions)

            logging.info(f"Predictions completed. Shape: {predictions.shape}")
            return predictions, label_encoder, predictions_proba

        except Exception as e:
            raise CustomException(e, sys)

    def predict_with_all_models(self, features_df):
        """
        Predict churn using all available models and return consensus.

        Args:
            features_df: DataFrame with customer features

        Returns:
            dict with predictions from each model
        """
        try:
            logging.info("Starting prediction pipeline with all models")

            # Load all objects
            preprocessor, label_encoder, model, nn_model = self.load_all_objects()

            # Apply feature engineering
            features_df = self._apply_feature_engineering(features_df)

            # Drop id column if present
            if "id" in features_df.columns:
                features_df = features_df.drop(columns=["id"])

            num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'ServiceCount']

            for col in num_features:
                if col in features_df.columns:
                    features_df[col] = features_df[col].astype('float64')

            # Transform features
            features_transformed = preprocessor.transform(features_df)

            results = {}

            # Sklearn model prediction
            results['sklearn'] = {
                'predictions': model.predict(features_transformed).astype(int),
            }
            if hasattr(model, "predict_proba"):
                results['sklearn']['probabilities'] = model.predict_proba(features_transformed)[:, 1]
            else:
                from scipy.special import expit
                results['sklearn']['probabilities'] = expit(model.decision_function(features_transformed))

            # NN model prediction if available
            if nn_model is not None:
                nn_model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features_transformed)
                    nn_proba = nn_model(features_tensor).numpy().flatten()
                    results['neural_network'] = {
                        'predictions': (nn_proba > 0.5).astype(int),
                        'probabilities': nn_proba
                    }

            logging.info(f"Predictions completed. Models used: {list(results.keys())}")
            return results, label_encoder

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
        self.SeniorCitizen = str(SeniorCitizen)
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

        # Predict using sklearn model
        predictions, label_encoder, proba = pipeline.predict(features_df, use_nn=False)
        print(f"\n[Sklearn] Predicted label: {predictions[0]}")
        print(f"[Sklearn] Predicted class: {label_encoder.inverse_transform([predictions[0]])[0]}")
        if proba is not None:
            print(f"[Sklearn] Prediction probability: {proba[0]:.4f}")

        # Predict using NN model if available
        try:
            predictions_nn, label_encoder_nn, proba_nn = pipeline.predict(features_df, use_nn=True)
            print(f"\n[Neural Network] Predicted label: {predictions_nn[0]}")
            print(f"[Neural Network] Predicted class: {label_encoder_nn.inverse_transform([predictions_nn[0]])[0]}")
            if proba_nn is not None:
                print(f"[Neural Network] Prediction probability: {proba_nn[0]:.4f}")
        except Exception as e:
            print(f"\nNeural Network prediction not available: {e}")

        # Predict with all models
        try:
            all_results, le = pipeline.predict_with_all_models(features_df)
            print(f"\nAll model predictions: {list(all_results.keys())}")
            for model_name, result in all_results.items():
                print(f"  {model_name}: pred={result['predictions'][0]}, proba={result['probabilities'][0]:.4f}")
        except Exception as e:
            print(f"Multi-model prediction failed: {e}")

    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Prediction failed: {e}")
