import os
import sys

import numpy as np
import pandas as pd

import pickle
from src.logger import logging
from src.exception import CustomException

import dill

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error("Failed to save object", exc_info=True)
        raise CustomException(e, sys)

#model evaluation
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = models[model_name]

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
