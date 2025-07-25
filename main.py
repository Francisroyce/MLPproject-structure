from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.data_ingestion import DataIngestion

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

if __name__ == "__main__":
    # Data Ingestion
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    # Data Transformation
    transformation = DataTransformation()
    train_array, test_array, _ = transformation.initiate_data_transformation(train_data, test_data)

    # Model Training
    trainer = ModelTrainer()
    best_model_name, test_score = trainer.initiate_model_trainer(train_array, test_array)

    print(f"\n Best model: {best_model_name} with Test R2 Score: {test_score:.4f}")
