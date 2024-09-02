"""
Data collection script designed to simulate data ingestion for the batch pipeline.

Author: Prof. Dr. Maciel Calebe Vidal, INSPER, São Paulo - SP - Brazil, 2024.
"""

import sys
import os
import numpy as np
import random
import datetime
import calendar
import pandas as pd
import itertools


class Config:
    stores = {
        5000: {
            "avg_n": 100,
            "avg_price": 350.0,
            "std": 10.0,
            "boost_weekday": [6, 7],
            "boost_months": [5, 12],
        },
        5001: {
            "avg_n": 10,
            "avg_price": 500.0,
            "std": 20.0,
            "boost_weekday": [7],
            "boost_months": [5, 12],
        },
        5002: {
            "avg_n": 25,
            "avg_price": 400.0,
            "std": 10.0,
            "boost_weekday": [7],
            "boost_months": [4, 10, 12],
        },
        5003: {
            "avg_n": 200,
            "avg_price": 220.0,
            "std": 12.0,
            "boost_weekday": [1, 3, 7],
            "boost_months": [],
        },
        5004: {
            "avg_n": 140,
            "avg_price": 415.0,
            "std": 17.0,
            "boost_weekday": [4, 6, 7],
            "boost_months": [4, 10, 12],
        },
        5005: {
            "avg_n": 50,
            "avg_price": 890.0,
            "std": 15.0,
            "boost_weekday": [6, 7],
            "boost_months": [5, 12],
        },
    }
    product_ids = np.random.randint(1000, 3000, size=30)


def generate_day_sales(store_id, date):
    config = Config.stores[store_id]
    year, month, day = date.year, date.month, date.day
    n_sales = np.random.poisson(lam=config["avg_n"])

    if date.weekday() in config["boost_weekday"]:
        n_sales = int(n_sales * random.uniform(1.6, 1.7))

    if month in config["boost_months"]:
        n_sales = int(n_sales * random.uniform(1.45, 1.50))

    stores = np.full(n_sales, store_id)
    products = np.random.choice(Config.product_ids, size=n_sales)
    prices = np.random.normal(
        loc=config["avg_price"], scale=config["std"], size=n_sales
    )
    dates = np.full(n_sales, date.strftime("%Y-%m-%d"))
    client_ids = np.random.randint(100000, 400000, size=n_sales)

    return pd.DataFrame(
        {
            "store_id": stores,
            "date": dates,
            "client_id": client_ids,
            "product_id": products,
            "price": prices,
        }
    )


def generate_predict_register(store_id, date):
    return pd.DataFrame(
        {
            "store_id": [store_id],
            "year": [date.year],
            "month": [date.month],
            "day": [date.day],
            "weekday": [date.weekday()],
        }
    )


def generate_data(year_from, month_from, day_from, year_to, month_to, day_to, type_):
    dates = pd.date_range(
        start=f"{year_from}-{month_from:02d}-{day_from:02d}",
        end=f"{year_to}-{month_to:02d}-{day_to:02d}",
    )
    store_ids = list(Config.stores.keys())
    combinations = itertools.product(store_ids, dates)

    dfs = []
    for store_id, date in combinations:
        if type_ == "train":
            dfs.append(generate_day_sales(store_id, date))
        else:
            dfs.append(generate_predict_register(store_id, date))

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    print("Simulate data ingestion!")

    out_type = sys.argv[-1]

    if len(sys.argv) != 8 or out_type not in ["train", "predict"]:
        print("USAGE: python get_data.py <year_from> <month_from> <day_from> <year_to> <month_to> <day_to> <train/predict>")
    else:
        date_args = sys.argv[1:-1]
        date_args = [int(x) for x in date_args]
        df = generate_data(*date_args, out_type)
        st_date = "-".join(sys.argv[4:-1])
        if out_type == "train":
            file_name = f"{out_type}-{st_date}.csv"
        else:
            file_name = f"{out_type}-{st_date}.parquet"

        data_dir = os.path.relpath("data", os.getcwd())
        file_path = os.path.join(data_dir, file_name)
        print(f"Saving to {file_path} file...")

        if out_type == "train":
            df.to_csv(file_path, index=False)
        else:
            df.to_parquet(file_path.replace(".csv", ".parquet"), index=False)