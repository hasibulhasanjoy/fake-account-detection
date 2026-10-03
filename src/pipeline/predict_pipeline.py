import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# Feature columns expected by the saved preprocessor. The value of each
# entry is only used to infer the column type: str -> categorical/text
# column, anything else -> numerical column. Missing values are kept as
# NaN so the imputers fitted during training can fill them with the exact
# same statistics used while the model was trained.
DEFAULT_VALUES = {
    # numerical / boolean flag columns
    "statuses_count": 0,
    "followers_count": 0,
    "friends_count": 0,
    "favourites_count": 0,
    "listed_count": 0,
    "default_profile": 0,
    "default_profile_image": 0,
    "geo_enabled": 0,
    "profile_use_background_image": 0,
    "profile_background_tile": 0,
    "protected": 0,
    "verified": 0,
    "utc_offset": 0,
    # categorical columns
    "lang": "en",
    "time_zone": "",
    "profile_text_color": "",
    "profile_sidebar_border_color": "",
    "profile_sidebar_fill_color": "",
    "profile_background_color": "",
    "profile_link_color": "",
    # free-text columns
    "name": "",
    "screen_name": "",
    "description": "",
    "location": "",
}

LABELS = {0: "Real Account", 1: "Fake Account"}


def prepare_input_data(input_data: dict) -> pd.DataFrame:
    """
    Convert a dict of raw account attributes into a single-row DataFrame
    that matches the schema expected by the saved preprocessor.

    Missing values are kept as NaN so that the imputers fitted during
    training apply the same statistics they learned on the training data.
    Treating missing categorical values as "" instead would produce
    unseen categories that get one-hot encoded to all zeros, which pushes
    the input far outside the training distribution.

    Empty or whitespace-only strings are treated as missing for the same
    reason; present values are coerced to the column's expected type.
    """
    try:
        row = {}
        for column, type_hint in DEFAULT_VALUES.items():
            value = input_data.get(column)

            if value is None or (isinstance(value, float) and np.isnan(value)):
                row[column] = np.nan
            elif isinstance(value, str) and not value.strip():
                row[column] = np.nan
            elif isinstance(type_hint, str):
                row[column] = str(value)
            else:
                try:
                    row[column] = float(value)
                except (TypeError, ValueError):
                    logging.warning(
                        f"Invalid value for '{column}': {value!r}; treating as missing"
                    )
                    row[column] = np.nan

        return pd.DataFrame([row])
    except Exception as e:
        raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict(self, input_df: pd.DataFrame) -> list[dict]:
        """
        Load the saved preprocessor and trained model, transform the
        incoming input data and return one result dict per row with the
        prediction label (and probability when the model supports it).
        """
        try:
            logging.info("Loading preprocessor and trained model for prediction")
            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)

            features = preprocessor.transform(input_df)
            predictions = model.predict(features)
            logging.info(f"Prediction completed: {predictions.tolist()}")

            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[:, 1]

            results: list[dict] = []
            for index, prediction in enumerate(predictions):
                result: dict[str, object] = {
                    "prediction": LABELS.get(int(prediction), str(prediction))
                }
                if probabilities is not None:
                    result["fake_probability"] = round(
                        float(probabilities[index]), 4
                    )
                results.append(result)

            return results
        except Exception as e:
            raise CustomException(e, sys)


def predict_account(input_data: dict) -> dict:
    """
    Receive custom input data for a single account, fill missing values
    with defaults, and predict whether the account is fake or not.

    Returns a dict with the prediction label (and probability when the
    trained model supports it).
    """
    try:
        input_df = prepare_input_data(input_data)
        results = PredictPipeline().predict(input_df)
        return results[0]
    except Exception as e:
        raise CustomException(e, sys)
