import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    real_users_path: str = os.path.join("data", "real_users.csv")
    fake_users_path: str = os.path.join("data", "fake_users.csv")
    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _load_and_label_dataset(self, file_path, label):
        """Load a single dataset CSV and assign the target label."""
        try:
            logging.info(f"Loading dataset: {file_path}")
            df = pd.read_csv(file_path)
            df[self.ingestion_config.target_column] = label
            logging.info(
                f"Loaded {len(df)} rows from {file_path} with target label {label}"
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        """
        Load real and fake user datasets, combine them with target labels,
        and split into train and test sets saved under the artifacts folder.
        """
        try:
            logging.info("Data ingestion started")

            real_df = self._load_and_label_dataset(
                self.ingestion_config.real_users_path, label=0
            )
            fake_df = self._load_and_label_dataset(
                self.ingestion_config.fake_users_path, label=1
            )

            combined_df = pd.concat([real_df, fake_df], ignore_index=True)
            combined_df = combined_df.sample(
                frac=1, random_state=self.ingestion_config.random_state
            ).reset_index(drop=True)

            logging.info(
                f"Combined dataset shape: {combined_df.shape}, "
                f"class distribution: {combined_df[self.ingestion_config.target_column].value_counts().to_dict()}"
            )

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            combined_df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw combined data saved at {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(
                combined_df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=combined_df[self.ingestion_config.target_column],
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(
                f"Train data saved at {self.ingestion_config.train_data_path} "
                f"with shape {train_set.shape}"
            )
            logging.info(
                f"Test data saved at {self.ingestion_config.test_data_path} "
                f"with shape {test_set.shape}"
            )
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
