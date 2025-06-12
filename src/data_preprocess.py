import logging
import os
import pandas as pd

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocess')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocess.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def split_dataset() -> None:
    """
    Splits the dataframe into training and testing datasets.
    """
    try:
        file_path = "data/processed/stock_data_new.csv"
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from: %s", file_path)

        train_size = int(len(df) * 0.8)

        X = df[["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "rolling_mean_3", "rolling_std_3", "rolling_mean_7", "rolling_std_7"]]
        y = df["Close"]

        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        X_train.columns = X_train.columns.str.strip()
        X_test.columns = X_test.columns.str.strip()

        X_train.to_csv("data/train/X_train.csv", index=False)
        y_train.to_csv("data/train/y_train.csv", index=False)
        X_test.to_csv("data/test/X_test.csv", index=False)
        y_test.to_csv("data/test/y_test.csv", index=False)
        logger.debug("Data splitted successfully.")
    except Exception as e:
        logger.error("Error occurred while splitting the dataset: %s", e)
        raise

if __name__ == "__main__":
    split_dataset()
