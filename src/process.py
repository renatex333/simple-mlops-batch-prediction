import os
import pandas as pd

def main(file_name: str) -> None:
    data_dir = os.path.relpath("data", os.getcwd())
    file_path = os.path.join(data_dir, file_name)
    save_path = os.path.join(data_dir, file_name.replace(".csv", ".parquet"))
    raw_data = pd.read_csv(file_path)
    processed_data = raw_to_weekday(raw_data)
    processed_data.to_parquet(save_path, index=False)
    print(f"Processed data saved to {save_path}.")

def raw_to_weekday(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data to:
    - Separate weekday - day - montyh - year;
    - Group sales by store_id, year, month, day, weekday;
    """
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    raw_data["year"] = raw_data["date"].dt.year
    raw_data["month"] = raw_data["date"].dt.month
    raw_data["day"] = raw_data["date"].dt.day
    raw_data["weekday"] = raw_data["date"].dt.weekday
    raw_data.drop(columns=["date", "client_id", "product_id"], inplace=True)
    data = raw_data.groupby(
        ["store_id", "year", "month", "day", "weekday"]
    ).agg(
        {"price": "sum"}
    ).reset_index()
    data.rename(columns={"price": "total_sales"}, inplace=True)
    return data

if __name__ == "__main__":
    FILE_NAME = "train-2023-08-01.csv"
    main(FILE_NAME)
