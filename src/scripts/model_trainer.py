# src/scripts/model_trainer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_array, test_array, _ = transformation.initiate_data_transformation(train_data, test_data)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_array, test_array)
