import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def main(data_path: str, model_path: str) -> None:
    """
    Main function to predict the total sales.

    Args:
        data_path (str): Path to the data. Must be a CSV or Parquet file.
        model_path (str): Path to the model. Must be a pickle file.
    """
    try:
        data = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
    except Exception as e:
        logging.error("An error occurred: %s", e)
    else:
        prediction_path = data_path.split("/")[-1].replace("predict-", "predict-done-")
        if not prediction_path.startswith("predict-done-"):
            prediction_path = f"predict-done-{prediction_path}"
        predict(data, model, prediction_path)

def configure_logging() -> None:
    """
    Configure logging.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)-18s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename="data/logs/predict.log",
        filemode="a",
    )

def predict(data: pd.DataFrame, model: RandomForestRegressor, prediction_path: str) -> None:
    """
    Predict the total sales.

    Args:
        data (pd.DataFrame): Data to be used for prediction.
        model: Model to be used for prediction.
        prediction_path (str): Path to save the predictions.
    """
    predictions = model.predict(data)
    data["prediction_total_sales"] = predictions
    data_dir = os.path.relpath("data", os.getcwd())
    file_path = os.path.join(data_dir, prediction_path)
    data.to_parquet(file_path, index=False)
    logging.info("Predictions saved to %s.", file_path)

if __name__ == "__main__":
    configure_logging()
    if len(sys.argv) != 3:
        logging.error("Invalid number of arguments.")
        logging.error("USAGE: python predict.py <path to model> <path to data>")
        sys.exit(1)
    else:
        DATA_PATH = sys.argv[-1]
        MODEL_PATH = sys.argv[-2]
        main(DATA_PATH, MODEL_PATH)
        sys.exit(0)
