import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: Path = Path(os.path.join("artifacts", "preprocessor.pkl"))


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_feature_engineered_df(self, df: pd.DataFrame):
        try:
            # Convert SeniorCitizen to string (categorical)
            if 'SeniorCitizen' in df.columns:
                df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

            # Feature Engineering: Average monthly charge per year of tenure
            if 'TotalCharges' in df.columns and 'tenure' in df.columns:
                df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)

            # Feature Engineering: Service density
            service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            df['ServiceCount'] = 0
            for col in service_cols:
                if col in df.columns:
                    if col == 'InternetService':
                        df['ServiceCount'] += (df[col] != 'No').astype(int)
                    elif col == 'PhoneService':
                        df['ServiceCount'] += (df[col] == 'Yes').astype(int)
                    else:
                        df['ServiceCount'] += (df[col] == 'Yes').astype(int)

            return df
        except Exception as e:
            raise CustomException(e, sys)
    
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

            logging.info("Applying preprocessing on training and testing datasets")

            preprocessor = self.get_data_transformer_object()
            
            # Fit and transform
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_validation_arr = preprocessor.transform(input_feature_validation_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            validation_arr = np.c_[input_feature_validation_arr, np.array(target_feature_validation_df)]
            if target_feature_test_df is not None:
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            else:
                test_arr = input_feature_test_arr

            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)

            logging.info("Preprocessor object saved successfully")

            return (
                train_arr,
                validation_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)