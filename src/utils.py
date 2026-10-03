import os
import sys
import dill
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, model, param_grid):
    """
    Train a single model with GridSearchCV hyperparameter tuning and
    evaluate it on the test data.

    Returns a dict of evaluation metrics: accuracy, precision, recall and f1.
    """
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring="accuracy",
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        model.set_params(**grid_search.best_params_)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "best_params": grid_search.best_params_,
        }
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple models using GridSearchCV hyperparameter tuning.

    Returns a dict mapping model name to its test accuracy.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training and tuning model: {model_name}")
            metrics = evaluate_model(
                X_train,
                y_train,
                X_test,
                y_test,
                model,
                params.get(model_name, {}),
            )
            report[model_name] = metrics["accuracy"]
            logging.info(
                f"{model_name} - accuracy: {metrics['accuracy']:.4f}, "
                f"precision: {metrics['precision']:.4f}, "
                f"recall: {metrics['recall']:.4f}, "
                f"f1: {metrics['f1']:.4f}"
            )

        return report
    except Exception as e:
        raise CustomException(e, sys)
