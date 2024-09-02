import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def main(data_path: str) -> None:
    """
    Main function to train the model.

    Args:
        data_path (str): Path to the data.
    """
    data = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    train_model(data, "total_sales", data_path)


def train_model(data: pd.DataFrame, target: str, data_path: str) -> None:
    X = data.drop(columns=[target])
    y = data[target]
    
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=195)
    model.fit(X, y)

    model_name = data_path.split("/")[-1].replace(".parquet", ".pkl").replace(".csv", ".pkl").replace("train-", "model-")
    if not model_name.startswith("model-"):
        model_name = f"model-{model_name}"
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
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: python train.py <path to training data>")
        sys.exit(1)
    else:
        DATA_PATH = sys.argv[-1]
        main(DATA_PATH)
        sys.exit(0)
