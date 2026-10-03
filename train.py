import argparse
import sys

from src.components.model_training import ModelTrainer
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.predict_pipeline import predict_account

DUMMY_INPUTS = []


def run_training():
    train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()

    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
        train_path=train_data_path, test_path=test_data_path
    )

    best_model_name, best_model_score = ModelTrainer().initiate_model_trainer(
        train_array=train_arr, test_array=test_arr
    )
    print(f"Best model: {best_model_name} (accuracy: {best_model_score:.4f})")


def run_prediction():
    print("Running prediction pipeline on dummy input data...\n")
    for index, dummy_input in enumerate(DUMMY_INPUTS, start=1):
        result = predict_account(dummy_input)
        print(f"Test account #{index}: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake account detection pipeline")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Skip training and predict on dummy input data using the saved model",
    )
    args = parser.parse_args()

    try:
        if args.predict:
            run_prediction()
        else:
            run_training()
    except Exception as e:
        raise CustomException(e, sys)
