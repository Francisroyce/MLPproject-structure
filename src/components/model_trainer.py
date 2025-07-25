import os
import sys
import json
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    model_report_path = os.path.join('artifacts', 'model_report.json')
    catboost_info_dir: str = os.path.join('artifacts', 'catboost_info')  # <- Added

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def tune_model(self, model, param_grid, x_train, y_train):
        """Perform GridSearchCV on the given model."""
        try:
            gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            gs.fit(x_train, y_train)
            logging.info(f"Tuned {model.__class__.__name__} with best params: {gs.best_params_}")
            return gs.best_estimator_
        except Exception as e:
            logging.warning(f"Hyperparameter tuning failed for {model.__class__.__name__}: {e}")
            return model  # fallback to default if tuning fails

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Ensure catboost_info directory exists
            os.makedirs(self.model_trainer_config.catboost_info_dir, exist_ok=True)

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, train_dir=self.model_trainer_config.catboost_info_dir),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Optional hyperparameter tuning for selected models
            tuned_models = {
                "Random Forest Regressor": self.tune_model(RandomForestRegressor(), {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5, None]
                }, x_train, y_train),

                "XGBRegressor": self.tune_model(XGBRegressor(), {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1]
                }, x_train, y_train),

                "CatBoosting Regressor": self.tune_model(
                    CatBoostRegressor(verbose=False, train_dir=self.model_trainer_config.catboost_info_dir),
                    {
                        "depth": [4, 6],
                        "learning_rate": [0.05, 0.1]
                    },
                    x_train,
                    y_train
                )
            }

            # Replace untuned models with tuned ones
            for name in tuned_models:
                models[name] = tuned_models[name]

            model_report: dict = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models
            )

            # Save model report to JSON
            os.makedirs(os.path.dirname(self.model_trainer_config.model_report_path), exist_ok=True)
            with open(self.model_trainer_config.model_report_path, 'w') as f:
                json.dump(model_report, f, indent=4)

            logging.info("Model performance report:")
            for name, score in model_report.items():
                if score < 0.5:
                    logging.warning(f"{name}: Low R2 = {score:.4f}")
                else:
                    logging.info(f"{name}: R2 Score = {score:.4f}")

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found (score < 0.6)", sys)

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Overfitting/Underfitting check
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(f"{best_model_name} - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

            return best_model_name, test_score

        except Exception as e:
            raise CustomException(e, sys)
