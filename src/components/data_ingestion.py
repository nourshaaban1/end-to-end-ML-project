import sys
import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path: Path = Path("artifacts/train.csv")
    validation_data_path: Path = Path("artifacts/validation.csv")
    test_data_path: Path = Path("artifacts/test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def ingest_data(self):
        try:
            logging.info("Data ingestion started")

            df = pd.read_csv("data/train.csv")

            os.makedirs("artifacts", exist_ok=True)

            train_set, val_set = train_test_split(
                df,
                test_size=0.2,
                stratify=df["Churn"],
                random_state=42
            )

            train_set.to_csv(self.config.train_data_path, index=False)
            val_set.to_csv(self.config.validation_data_path, index=False)

            if os.path.exists("data/test.csv"):
                test_df = pd.read_csv("data/test.csv")
                test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return (
                self.config.train_data_path,
                self.config.validation_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)