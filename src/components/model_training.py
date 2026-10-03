import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _get_models_and_params(self):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
            "SVM": SVC(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
        }

        params = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear"],
            },
            "Decision Tree": {
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
            },
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 2],
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
            "CatBoost": {
                "iterations": [100, 200],
                "learning_rate": [0.05, 0.1],
                "depth": [4, 6],
            },
            "SVM": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
            },
            "KNN": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
            },
            "Naive Bayes": {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        }

        return models, params

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train all supervised models with GridSearchCV hyperparameter tuning,
        select the best performing model based on test accuracy and save it
        to model.pkl for later use during prediction.
        """
        try:
            logging.info("Model training started")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models, params = self._get_models_and_params()

            logging.info("Evaluating models with GridSearchCV hyperparameter tuning")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            logging.info(f"Model evaluation report: {model_report}")

            best_model_name = max(
                model_report, key=lambda model_name: model_report[model_name]
            )
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    f"No acceptable model found. Best accuracy: {best_model_score}",
                    sys,
                )

            logging.info(
                f"Best performing model: {best_model_name} with "
                f"test accuracy: {best_model_score:.4f}"
            )

            logging.info(
                f"Saving best model ({best_model_name}) at "
                f"{self.model_trainer_config.trained_model_file_path}"
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info("Model training completed successfully")

            return best_model_name, best_model_score
        except Exception as e:
            raise CustomException(e, sys)
