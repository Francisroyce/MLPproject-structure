import pandas as pd
import matplotlib.pyplot as plt
import json
from src.utils import load_object
from src.logger import logging  # make sure your logger is configured

def show_feature_importance(
    model_path="artifacts/model.pkl",
    preprocessor_path="artifacts/preprocessor.pkl",
    train_csv_path="artifacts/train.csv",
    output_json_path="artifacts/feature_contributions.json"
):
    try:
        # Load model and preprocessor
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)

        # Load training data
        train_df = pd.read_csv(train_csv_path)
        x = train_df.drop(columns=["math_score"])

        # Get transformed feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(len(model.coef_))]  # fallback

        # Feature Importance or Coefficients
        data_to_save = {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            # Logging
            logging.info("Feature importances:")
            for _, row in importance_df.iterrows():
                logging.info(f"{row['Feature']}: {row['Importance']:.4f}")
                data_to_save[row['Feature']] = row['Importance']

            # Save to JSON
            with open(output_json_path, "w") as f:
                json.dump(data_to_save, f, indent=4)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df["Feature"], importance_df["Importance"])
            plt.xlabel("Feature Importance")
            plt.title(f"Feature Importance for {model.__class__.__name__}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        elif hasattr(model, "coef_"):
            coefficients = model.coef_
            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": coefficients
            }).sort_values(by="Coefficient", key=abs, ascending=False)

            # Logging
            logging.info("Model coefficients:")
            for _, row in coef_df.iterrows():
                logging.info(f"{row['Feature']}: {row['Coefficient']:.4f}")
                data_to_save[row['Feature']] = row['Coefficient']

            # Save to JSON
            with open(output_json_path, "w") as f:
                json.dump(data_to_save, f, indent=4)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(coef_df["Feature"], coef_df["Coefficient"])
            plt.xlabel("Coefficient Value")
            plt.title(f"Feature Coefficients for {model.__class__.__name__}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        else:
            logging.warning(f"The model type {model.__class__.__name__} does not support feature importance or coefficients.")
            print(f"The model type {model.__class__.__name__} does not support feature importance or coefficients.")

    except Exception as e:
        logging.error("An error occurred while displaying feature importance.", exc_info=True)
        raise e
