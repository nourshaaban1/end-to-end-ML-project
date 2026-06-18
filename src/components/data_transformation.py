import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: Path = Path(os.path.join("artifacts", "preprocessor.pkl"))
    label_encoder_file_path: Path = Path(os.path.join("artifacts", "label_encoder.pkl"))


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_feature_engineered_df(self, df):
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)

        service_cols = [
            'PhoneService','InternetService','OnlineSecurity','OnlineBackup',
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
        ]

        df['ServiceCount'] = 0.0
        for col in service_cols:
            if col in df.columns:
                if col == 'InternetService':
                    df['ServiceCount'] += (df[col] != 'No').astype(float)
                else:
                    df['ServiceCount'] += (df[col] == 'Yes').astype(float)

        return df

    def get_data_transformer_object(self):
        """
        This function is responsible for creating the preprocessor object
        """
        try:
            logging.info("Creating the preprocessor pipeline object")

            num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'ServiceCount']
            cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
            
            num_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, num_features),
                    ("cat", cat_transformer, cat_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, validation_path, test_path):
        try:
            logging.info("Data transformation started")

            train_df = pd.read_csv(train_path)
            validation_df = pd.read_csv(validation_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train, validation and test data completed")

            # Apply Feature Engineering
            train_df = self.get_feature_engineered_df(train_df)
            validation_df = self.get_feature_engineered_df(validation_df)
            test_df = self.get_feature_engineered_df(test_df)

            # Drop unnecessary "id" columns
            for df in [train_df, validation_df, test_df]:
                if "id" in df.columns:
                    df.drop(columns=["id"], inplace=True)

            target_column_name = "Churn"

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_validation_df = validation_df.drop(columns=[target_column_name])
            target_feature_validation_df = validation_df[target_column_name]

            if target_column_name in test_df.columns:
                input_feature_test_df = test_df.drop(columns=[target_column_name])
                target_feature_test_df = test_df[target_column_name]
            else:
                input_feature_test_df = test_df
                target_feature_test_df = None

            # --- Label Encoding: fit on train target only, then transform all splits ---
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_validation_df = label_encoder.transform(target_feature_validation_df)
            if target_feature_test_df is not None:
                target_feature_test_df = label_encoder.transform(target_feature_test_df)

            logging.info(f"Label encoding applied. Classes: {list(label_encoder.classes_)}")

            # --- SMOTENC: must run BEFORE preprocessing (data is still mixed-type) ---
            # Identify categorical column indices in the raw feature dataframe.
            # These must match the columns defined in get_data_transformer_object().
            cat_features = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
            all_columns = list(input_feature_train_df.columns)
            cat_indices = [all_columns.index(col) for col in cat_features if col in all_columns]

            logging.info(f"Applying SMOTENC on training data with {len(cat_indices)} categorical features")

            smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
            input_feature_train_resampled, target_feature_train_resampled = smotenc.fit_resample(
                input_feature_train_df, target_feature_train_df
            )

            logging.info("Applying preprocessing on training and testing datasets")

            preprocessor = self.get_data_transformer_object()

            # Fit on the resampled training data, then transform all splits
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_resampled)
            input_feature_validation_arr = preprocessor.transform(input_feature_validation_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_resampled)]
            validation_arr = np.c_[input_feature_validation_arr, np.array(target_feature_validation_df)]
            if target_feature_test_df is not None:
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            else:
                test_arr = input_feature_test_arr

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)
            save_object(file_path=self.data_transformation_config.label_encoder_file_path, obj=label_encoder)

            logging.info("Preprocessor and label encoder objects saved successfully")

            return (
                train_arr,
                validation_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)