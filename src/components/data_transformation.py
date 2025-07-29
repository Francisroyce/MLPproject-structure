import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    train_array_path = os.path.join('artifacts', 'train_array.npy')
    test_array_path = os.path.join('artifacts', 'test_array.npy')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, input_df):
        try:
            # Dynamically identify numerical and categorical columns
            numerical_columns = input_df.select_dtypes(exclude="object").columns.tolist()
            categorical_columns = input_df.select_dtypes(include="object").columns.tolist()

            logging.info(f"Numerical columns detected: {numerical_columns}")
            logging.info(f"Categorical columns detected: {categorical_columns}")

            # Define pipelines
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # ColumnTransformer to apply the above pipelines
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logging.info("Preprocessor pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error in creating the preprocessor object", exc_info=True)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data successfully.")

            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            preprocessor_obj = self.get_data_transformer_object(input_feature_train_df)

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            # Save preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)
            logging.info("Preprocessor object saved successfully.")

            # ✅ Save arrays to disk for DVC output tracking
            np.save(self.data_transformation_config.train_array_path, train_arr)
            np.save(self.data_transformation_config.test_array_path, test_arr)
            logging.info("Train and test arrays saved successfully.")

            logging.info("Data transformation completed.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error in initiate_data_transformation", exc_info=True)
            raise CustomException(e, sys)
