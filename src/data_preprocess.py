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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features for the training the model.
    """
    df["lag_1"] = df["Close"].shift(1)  #Closing price one day prior
    df["lag_2"] = df["Close"].shift(2)  #Closing price two days prior
    df["rolling_mean"] = df["Close"].rolling(3).mean()  #Average closing price for 3 consecutive days

    df.dropna(inplace=True) #Dropping NaN values

    df.to_csv('data/processed/stock_data_processed.csv')
    logger.debug("Data preprocessed and loaded to stock_data_processed.csv")

    return df
