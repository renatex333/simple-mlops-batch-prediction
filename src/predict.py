import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def main(data_path: str, model_path: str) -> None:
    """
    Main function to predict the total sales.

    Args:
        data_path (str): Path to the data.
        model_path (str): Path to the model.
    """
    data = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    prediction_path = data_path.split("/")[-1].replace("predict-", "predict-done-")
    if not prediction_path.startswith("predict-done-"):
        prediction_path = f"predict-done-{prediction_path}"
    predict(data, model, prediction_path)

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
    print(f"Predictions saved to {file_path}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python predict.py <path to model> <path to data>")
        sys.exit(1)
    else:
        data_path = sys.argv[-1]
        model_path = sys.argv[-2]
        main(data_path, model_path)
        sys.exit(0)
