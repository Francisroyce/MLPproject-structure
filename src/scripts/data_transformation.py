# src/scripts/data_transformation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    transformer.initiate_data_transformation(train_data, test_data)
