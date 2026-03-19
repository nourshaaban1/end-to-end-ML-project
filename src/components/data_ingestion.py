import sys
import os 
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: Path = Path(os.path.join("artifacts", "train.csv"))
    validation_data_path: Path = Path(os.path.join("artifacts", "validation.csv"))
    test_data_path: Path = Path(os.path.join("artifacts", "test.csv"))

class DataIngestion:

    def __init__(self):
        self.config = DataIngestionConfig()
    
    def ingest_data(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv("data/train.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            
            # Splitting the data into train and validation sets
            train_set, validation_set = train_test_split(df, test_size=0.2, random_state=42)
            
            logging.info("Train validation split initiated")

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            validation_set.to_csv(self.config.validation_data_path, index=False, header=True)

            # Copy or process test data if it exists in data folder
            if os.path.exists("data/test.csv"):
                test_df = pd.read_csv("data/test.csv")
                test_df.to_csv(self.config.test_data_path, index=False, header=True)
                logging.info("Test set also saved to artifacts")

            logging.info("Data ingestion completed successfully")

            return (
                self.config.train_data_path,
                self.config.validation_data_path,
                self.config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.ingest_data()