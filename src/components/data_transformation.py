import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

    # Columns that are identifiers, timestamps or leak the target label
    drop_columns: list[str] | None = None

    # Columns containing continuous/count/flag values
    numerical_columns: list[str] | None = None

    # Columns containing low-cardinality discrete values
    categorical_columns: list[str] | None = None

    # Columns containing free text
    text_columns: list[str] | None = None

    def __post_init__(self):
        if self.drop_columns is None:
            self.drop_columns = [
                "id",
                "created_at",
                "updated",
                "url",
                "profile_image_url",
                "profile_banner_url",
                "profile_background_image_url_https",
                "dataset",
                "target",
            ]
        if self.numerical_columns is None:
            self.numerical_columns = [
                "statuses_count",
                "followers_count",
                "friends_count",
                "favourites_count",
                "listed_count",
                "default_profile",
                "default_profile_image",
                "geo_enabled",
                "profile_use_background_image",
                "profile_background_tile",
                "protected",
                "verified",
                "utc_offset",
            ]
        if self.categorical_columns is None:
            self.categorical_columns = [
                "lang",
                "time_zone",
                "profile_text_color",
                "profile_sidebar_border_color",
                "profile_sidebar_fill_color",
                "profile_background_color",
                "profile_link_color",
            ]
        if self.text_columns is None:
            self.text_columns = ["name", "screen_name", "description", "location"]


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        """
        Build the preprocessing pipeline:
        - numerical columns: median imputation + StandardScaler
        - categorical columns: most-frequent imputation + OneHotEncoder
        - text columns: constant imputation + TF-IDF vectorization
        """
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            # TF-IDF is used for free-text columns; a separate pipeline is
            # created per text column so each gets its own vocabulary.
            text_pipelines = []
            for text_column in self.transformation_config.text_columns or []:
                max_features = 300 if text_column == "description" else 100
                text_pipeline = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value=""),
                        ),
                        (
                            "flatten",
                            FunctionTransformer(np.ravel, validate=False),
                        ),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=max_features,
                                stop_words="english",
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                )
                text_pipelines.append(
                    (f"text_{text_column}", text_pipeline, [text_column])
                )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numerical_pipeline,
                        self.transformation_config.numerical_columns,
                    ),
                    (
                        "cat",
                        categorical_pipeline,
                        self.transformation_config.categorical_columns,
                    ),
                    *text_pipelines,
                ],
                remainder="drop",
                sparse_threshold=0,
            )

            logging.info("Preprocessing pipeline created")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Load the train and test CSVs, apply the preprocessing pipeline and
        save the fitted preprocessor for later use during prediction.
        """
        try:
            logging.info("Data transformation started")
            logging.info(f"Reading train data from {train_path}")
            train_df = pd.read_csv(train_path)
            logging.info(f"Reading test data from {test_path}")
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")
            preprocessor = self.get_transformer_object()

            target_column = "target"
            drop_columns = [
                col
                for col in self.transformation_config.drop_columns or []
                if col in train_df.columns and col != target_column
            ]

            input_feature_train_df = train_df.drop(
                columns=drop_columns + [target_column]
            )
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns + [target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing pipeline on train and test dataframes")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(
                f"Saved preprocessing object at {self.transformation_config.preprocessor_obj_file_path}"
            )
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info(
                f"Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}"
            )
            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
