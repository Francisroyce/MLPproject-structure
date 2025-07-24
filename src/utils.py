import os
import sys

import numpy as np
import pandas as pd

import pickle
from src.logger import logging
from src.exception import CustomException

import dill

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error("Failed to save object", exc_info=True)
        raise CustomException(e, sys)

