from dataclasses import dataclass
import os
import sys

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: Path = Path(os.path.join('artifacts','model.pkl'))

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, validation_array):
        try:
            logging.info("Splitting training and validation data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val,   y_val   = validation_array[:, :-1], validation_array[:, -1]

            models = {
                'RandomForestClassifier'         : RandomForestClassifier(),
                'HistGradientBoostingClassifier' : HistGradientBoostingClassifier(),
            }

            logging.info("Evaluating models via cross-validation and validation set scoring")
            results_df, trained_models = evaluate_models(X_train, y_train, X_val, y_val, models)

            logging.info(f"Model evaluation results:\n{results_df.to_string(index=False)}")

            # --- Pick the best model (top row is already sorted by Val ROC-AUC) ---
            best_row        = results_df.iloc[0]
            best_model_name = best_row['Model']
            best_model      = trained_models[best_model_name]

            best_val_roc_auc = best_row['Val ROC-AUC']
            best_val_f1      = best_row['Val F1-Score']
            best_val_acc     = best_row['Val Accuracy']

            logging.info(
                f"Best model: {best_model_name} — "
                f"Val Accuracy: {best_val_acc:.4f}, "
                f"Val F1: {best_val_f1:.4f}, "
                f"Val ROC-AUC: {best_val_roc_auc:.4f}"
            )

            # --- Sanity-check: reject if best model is below acceptable threshold ---
            if best_val_roc_auc < 0.6:
                raise CustomException(
                    f"No model met the minimum ROC-AUC threshold of 0.6. "
                    f"Best was {best_model_name} with {best_val_roc_auc:.4f}.",
                    sys
                )

            # --- Save the best model ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model '{best_model_name}' saved to {self.model_trainer_config.trained_model_file_path}")

            return {
                'best_model_name' : best_model_name,
                'Val Accuracy'    : best_val_acc,
                'Val F1-Score'    : best_val_f1,
                'Val ROC-AUC'     : best_val_roc_auc,
                'all_results'     : results_df
            }

        except Exception as e:
            raise CustomException(e, sys)