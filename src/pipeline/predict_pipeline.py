import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# Feature columns expected by the saved preprocessor, with a suitable
# default for each one. Any missing key in the incoming input data is
# replaced by its default value before prediction.
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
    that matches the schema expected by the saved preprocessor. Missing
    or invalid values are replaced with suitable defaults.
    """
    try:
        row = {}
        for column, default in DEFAULT_VALUES.items():
            value = input_data.get(column, default)

            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = default

            if isinstance(default, str):
                row[column] = str(value)
            else:
                try:
                    row[column] = float(value)
                except (TypeError, ValueError):
                    logging.warning(
                        f"Invalid value for '{column}': {value!r}; "
                        f"using default {default!r}"
                    )
                    row[column] = float(default)

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


DUMMY_INPUTS = [
    {
        # Clearly genuine-looking account
        "name": "Alice Johnson",
        "screen_name": "alice_j",
        "description": "Software engineer. Coffee lover. Opinions are my own.",
        "location": "New York, USA",
        "statuses_count": 5400,
        "followers_count": 1200,
        "friends_count": 800,
        "favourites_count": 3100,
        "listed_count": 12,
        "default_profile": 0,
        "default_profile_image": 0,
        "geo_enabled": 1,
        "profile_use_background_image": 1,
        "profile_background_tile": 0,
        "protected": 0,
        "verified": 1,
        "utc_offset": -18000,
        "lang": "en",
        "time_zone": "Eastern Time (US & Canada)",
        "profile_text_color": "333333",
        "profile_sidebar_border_color": "C0DEED",
        "profile_sidebar_fill_color": "DDEEF6",
        "profile_background_color": "C0DEED",
        "profile_link_color": "0084B4",
    },
    {
        # Clearly fake-looking account (missing most optional fields on
        # purpose to exercise the default-filling logic)
        "screen_name": "user_8827361",
        "statuses_count": 12,
        "followers_count": 3,
        "friends_count": 2500,
        "default_profile": 1,
        "default_profile_image": 1,
    },
    {
        # Almost entirely empty input -> all defaults applied
        "name": "Unknown User",
    },
]
