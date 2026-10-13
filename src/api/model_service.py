import os
import sys
from typing import Literal, cast

import pandas as pd

from src.api.schemas import AccountData, PredictionResponse
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import LABELS, prepare_input_data
from src.utils import load_object


class ModelService:
    """
    Loads the saved preprocessor and trained model once and serves
    predictions for the API. Reuses `prepare_input_data` from the offline
    prediction pipeline so inference behavior stays identical.
    """

    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor = None
        self.model = None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def preprocessor_loaded(self) -> bool:
        return self.preprocessor is not None

    def load(self) -> None:
        try:
            logging.info("Loading preprocessor and trained model for API")
            self.preprocessor = load_object(file_path=self.preprocessor_path)
            self.model = load_object(file_path=self.model_path)
            logging.info("Model artifacts loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def _predict_frame(self, input_df: pd.DataFrame) -> list[PredictionResponse]:
        try:
            if self.preprocessor is None:
                raise RuntimeError("Preprocessor is not loaded")
            if self.model is None:
                raise RuntimeError("Model is not loaded")

            features = self.preprocessor.transform(input_df)
            predictions = self.model.predict(features)

            probabilities = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)

            results: list[PredictionResponse] = []
            for index, prediction in enumerate(predictions):
                fake_probability = None
                real_probability = None
                if probabilities is not None:
                    real_probability = round(float(probabilities[index, 0]), 4)
                    fake_probability = round(float(probabilities[index, 1]), 4)
                results.append(
                    PredictionResponse(
                        prediction=cast(
                            Literal["Real Account", "Fake Account"],
                            LABELS.get(int(prediction), str(prediction)),
                        ),
                        fake_probability=fake_probability,
                        real_probability=real_probability,
                    )
                )
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def predict_single(self, account: AccountData) -> PredictionResponse:
        try:
            input_df = prepare_input_data(account.model_dump())
            return self._predict_frame(input_df)[0]
        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, accounts: list[AccountData]) -> list[PredictionResponse]:
        try:
            if not accounts:
                return []
            rows = [
                prepare_input_data(account.model_dump()).iloc[0] for account in accounts
            ]
            input_df = pd.DataFrame(rows)
            return self._predict_frame(input_df)
        except Exception as e:
            raise CustomException(e, sys)


model_service = ModelService()
