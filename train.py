import sys

from src.components.model_training import ModelTrainer
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:

        train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()

        train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
            train_path=train_data_path, test_path=test_data_path
        )

        best_model_name, best_model_score = ModelTrainer().initiate_model_trainer(
            train_array=train_arr, test_array=test_arr
        )
        print(f"Best model: {best_model_name} (accuracy: {best_model_score:.4f})")
    except Exception as e:
        raise CustomException(e, sys)
