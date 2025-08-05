import sys
import os
import pandas as pd

# Add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

if __name__ == "__main__":
    # Use your exact data structure
    input_data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=78,
        writing_score=72
    )

    df = input_data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    predictions = pipeline.predict(df)

    # Save predictions
    pd.DataFrame({"prediction": predictions}).to_csv("artifacts/predictions.csv", index=False)
