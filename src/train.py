import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def main(data_path: str, target: str) -> None:
    """
    Main function to train the model.

    Args:
        data_path (str): Path to the data. Must be a CSV or Parquet file.
        target (str): Target column to predict.
    """
    data = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    model_name = data_path.split("/")[-1].replace(".parquet", ".pkl").replace(".csv", ".pkl").replace("train-", "model-")
    train_model(data, target, model_name)

def configure_logging() -> None:
    """
    Configure logging.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)-18s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename="data/logs/train.log",
        filemode="a",
    )

def train_model(data: pd.DataFrame, target: str, model_name: str) -> None:
    """
    Train the model.

    Args:
        data (pd.DataFrame): Data to be used for training.
        target (str): Target column to predict.
        model_name (str): Name to save the model.
    """
    X = data.drop(columns=[target])
    y = data[target]

    logging.info("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=195)
    model.fit(X, y)
    logging.info("Model trained successfully!")

    save_model(model, model_name)

def save_model(model: RandomForestRegressor, model_name: str) -> None:
    """
    Save the model to disk.

    Args:
        model: Model to be saved.
        model_name: Name of the model.
    """
    model_dir = os.path.relpath("models", os.getcwd())
    model_path = os.path.join(model_dir, model_name)
    logging.info("Saving model to %s", model_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    configure_logging()
    if len(sys.argv) < 2:
        logging.error("No data path provided.")
        logging.error("USAGE: python train.py <path to training data>")
        sys.exit(1)
    else:
        DATA_PATH = sys.argv[-1]
        main(DATA_PATH, "total_sales")
        sys.exit(0)
