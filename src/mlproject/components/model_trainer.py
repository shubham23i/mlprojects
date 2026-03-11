import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.mlproject.exception import custom_exception
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logging.info("Splitting training and testing data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {

                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },

                "CatBoostRegressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },

                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            logging.info("Evaluating models")

            model_report = evaluate_models(
            X_train, y_train, X_test, y_test, models, params
            )

            logging.info(f"Model report: {model_report}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            best_params = params[best_model_name]

            logging.info(f"Best model found: {best_model_name}")

            best_model.fit(X_train, y_train)

            predicted = best_model.predict(X_test)

            rmse, mae, r2 = self.eval_metrics(y_test, predicted)

            dagshub.init(repo_owner="shubham23i", repo_name="mlprojects", mlflow=True)
            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()
            ).scheme

            with mlflow.start_run():

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                if tracking_url_type_store != "file":

                    mlflow.sklearn.log_model(
                        best_model,
                        "model",
                        registered_model_name=best_model_name,
                    )

                else:

                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise custom_exception("No good model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model,
            )

            r2_final = r2_score(y_test, predicted)

            return r2_final

        except Exception as e:
            raise custom_exception(e, sys)