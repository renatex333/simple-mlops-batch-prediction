# ML Batch Prediction Project

Welcome to this ML batch prediction project! This project focuses on performing batch predictions using machine learning models, trained on historical sales data, and deploying predictions on new data.

## Installing Dependencies

To install the project dependencies, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Project Structure

- **`data/`**: Contains the data files used by the model.
- **`models/`**: Contains trained machine learning models and encoders.
- **`src/`**: Contains the main source code for data processing, training models, and making predictions.

## Usage

### 1. Simulate Data Ingestion

To simulate data ingestion and generate data for training or prediction, run the following command:

```bash
python get_data.py <year_from> <month_from> <day_from> <year_to> <month_to> <day_to> <train/predict>
```

- Example for training data:
  ```bash
  python get_data.py 2022 01 01 2023 08 01 train
  ```

- Example for prediction data:
  ```bash
  python get_data.py 2023 08 02 2023 08 03 predict
  ```

This will generate data in the `data/` directory, saving the files as either `.csv` (for training) or `.parquet` (for prediction).

### 2. Processing Training Data

To process raw data (e.g., transforming it to weekdays and aggregating sales), you can use the data processing script:

```bash
python process.py <data_path>
```

- Example:
  ```bash
  python process.py train-2023-08-01.csv
  ```

The processed data will be saved in the `data/` directory as a `.parquet`.

### 2. Training a Machine Learning Model

Once the training data is generated, use the following command to train the machine learning model:

```bash
python train.py <path_to_training_data>
```

- Example:
  ```bash
  python train.py data/train-2023-08-01.parquet
  ```

This will train a RandomForestRegressor model and save it in the `models/` directory as a `.pkl` file.

### 3. Making Predictions

To perform predictions on new data using the trained model, run:

```bash
python predict.py <path_to_model> <path_to_data>
```

- Example:
  ```bash
  python predict.py models/model-2023-08-01.pkl data/predict-2023-08-03.parquet
  ```

This will generate predictions and save the results in the `data/` directory as a `.parquet` file with the prefix `predict-done`.
