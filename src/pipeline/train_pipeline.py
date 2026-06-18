import sys
from pathlib import Path

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self, include_nn=True):
        self.include_nn = include_nn

    def run(self):
        try:
            logging.info("========== TRAINING PIPELINE STARTED ==========")

            # 1. Ingestion
            ingestion = DataIngestion()
            train_path, val_path, test_path = ingestion.ingest_data()

            # 2. Transformation
            transformation = DataTransformation()

            train_arr, val_arr, _, preprocessor_path, label_encoder_path = (
                transformation.initiate_data_transformation(
                    train_path,
                    val_path,
                    test_path
                )
            )

            # 3. Training
            trainer = ModelTrainer(include_nn=self.include_nn)

            results = trainer.initiate_model_trainer(
                train_arr,
                val_arr
            )

            logging.info("========== TRAINING COMPLETED ==========")

            return {
                "best_model": results['best_model_name'],
                "metrics": {
                    "accuracy": results['Val Accuracy'],
                    "f1": results['Val F1-Score'],
                    "roc_auc": results['Val ROC-AUC'],
                },
                "artifacts": {
                    "model": "artifacts/model.pkl",
                    "nn_model": "artifacts/model_nn.pth",
                    "preprocessor": str(preprocessor_path),
                    "label_encoder": str(label_encoder_path),
                }
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline(include_nn=True)
    result = pipeline.run()

    print("\n===== TRAINING SUMMARY =====")
    print(result)