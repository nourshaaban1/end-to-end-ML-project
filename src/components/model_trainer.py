from dataclasses import dataclass
import os
import sys

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object, train_neural_network, save_torch_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: Path = Path(os.path.join('artifacts','model.pkl'))
    nn_model_file_path: Path = Path(os.path.join('artifacts','model_nn.pth'))


class ModelTrainer:
    def __init__(self, include_nn: bool = True):
        self.model_trainer_config = ModelTrainerConfig()
        self.include_nn = include_nn

    def initiate_model_trainer(self, train_array, validation_array):
        try:
            logging.info("Splitting training and validation data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val,   y_val   = validation_array[:, :-1], validation_array[:, -1]

            models = {
                'RandomForestClassifier'         : RandomForestClassifier(),
                'HistGradientBoostingClassifier' : HistGradientBoostingClassifier(),
            }

            logging.info("Evaluating sklearn models via cross-validation and validation set scoring")
            results_df, trained_models = evaluate_models(X_train, y_train, X_val, y_val, models)

            # Add Neural Network if enabled
            if self.include_nn:
                logging.info("Training Neural Network model...")
                nn_input_features = X_train.shape[1]

                # Get baseline from sklearn models for comparison
                baseline_roc_auc = max(results_df['Val ROC-AUC'].max() if len(results_df) > 0 else 0.5, 0.5)

                # Train NN with shorter epochs for faster iteration
                nn_model, nn_history = train_neural_network(
                    X_train, y_train.reshape(-1, 1),
                    X_val, y_val.reshape(-1, 1),
                    input_features=nn_input_features,
                    epochs=30,
                    batch_size=32,
                    learning_rate=0.001,
                    patience=5
                )

                # Evaluate NN on validation set
                from src.utils import evaluate_neural_network
                nn_metrics = evaluate_neural_network(nn_model, X_val, y_val)

                logging.info(f"Neural Network Results - Acc: {nn_metrics['accuracy']:.4f}, F1: {nn_metrics['f1']:.4f}, ROC-AUC: {nn_metrics['roc_auc']:.4f}")

                # Add NN to results
                results_df = pd.concat([
                    results_df,
                    pd.DataFrame([{
                        'Model': 'NeuralNetwork',
                        'CV Accuracy': nn_metrics['accuracy'],
                        'CV F1-Score': nn_metrics['f1'],
                        'CV ROC-AUC': nn_metrics['roc_auc'],
                        'Val Accuracy': nn_metrics['accuracy'],
                        'Val F1-Score': nn_metrics['f1'],
                        'Val ROC-AUC': nn_metrics['roc_auc']
                    }])
                ], ignore_index=True)

                trained_models['NeuralNetwork'] = nn_model

                # Save NN model separately
                save_torch_model(nn_model, self.model_trainer_config.nn_model_file_path)
                logging.info(f"Neural Network model saved to {self.model_trainer_config.nn_model_file_path}")

            logging.info(f"Model evaluation results:\n{results_df.to_string(index=False)}")

            # --- Pick the best model (top row is already sorted by Val ROC-AUC) ---
            best_row        = results_df.iloc[0]
            best_model_name = best_row['Model']
            best_model      = trained_models[best_model_name]

            best_val_roc_auc = best_row['Val ROC-AUC']
            best_val_f1      = best_row['Val F1-Score']
            best_val_acc     = best_row['Val Accuracy']

            logging.info(
                f"Best model: {best_model_name} "
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

            # Hyperparameter Tuning for the chosen best model
            if best_model_name in ['HistGradientBoostingClassifier', 'RandomForestClassifier']:
                logging.info(f"Initiating Hyperparameter Tuning for {best_model_name}...")

                param_grids = {
                    'HistGradientBoostingClassifier': {
                        'max_iter': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [None, 7, 10],
                        'max_leaf_nodes': [31, 50]
                    },
                    'RandomForestClassifier': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 7, 10],
                        'min_samples_split': [2, 5]
                    }
                }
                grid_search = GridSearchCV(
                    estimator=best_model,
                    param_grid=param_grids[best_model_name],
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )

                logging.info(f"Grid searching {best_model_name}...")
                grid_search.fit(X_train, y_train)

                best_tuned_model = grid_search.best_estimator_
                logging.info(f"GridSearch completed! Best CV ROC-AUC = {grid_search.best_score_:.4f}")
                for p, v in grid_search.best_params_.items():
                    logging.info(f"   - {p}: {v}")

                # Re-evaluate the tuned model on the validation set
                y_pred = best_tuned_model.predict(X_val)
                y_pred_proba = best_tuned_model.predict_proba(X_val)[:, 1] if hasattr(best_tuned_model, "predict_proba") \
                               else (best_tuned_model.decision_function(X_val) if hasattr(best_tuned_model, "decision_function") else y_pred)

                tuned_val_roc_auc = roc_auc_score(y_val, y_pred_proba)
                tuned_val_f1 = f1_score(y_val, y_pred, zero_division=0)
                tuned_val_acc = accuracy_score(y_val, y_pred)

                logging.info(
                    f"Tuned Model Performance on Validation Set: "
                    f"Val Accuracy: {tuned_val_acc:.4f}, "
                    f"Val F1: {tuned_val_f1:.4f}, "
                    f"Val ROC-AUC: {tuned_val_roc_auc:.4f}"
                )

                # Replace properties with tuned properties
                best_model = best_tuned_model
                best_val_roc_auc = tuned_val_roc_auc
                best_val_f1 = tuned_val_f1
                best_val_acc = tuned_val_acc
            else:
                logging.info(f"No parameter grid defined for {best_model_name}. Skipping tuning.")

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
                'all_results'     : results_df.reset_index(drop=True)
            }

        except Exception as e:
            raise CustomException(e, sys)
