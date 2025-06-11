import pandas as pd
import logging
import os

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def new_features() -> None:
    """
    Creates new features for the training the model.
    """
    try:
        file_path = "data/raw/stock_data.csv"
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from: %s", file_path)

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        df["lag_1"] = df["Close"].shift(1)  #Closing price one day prior
        df["lag_2"] = df["Close"].shift(2)  #Closing price two days prior
        df["rolling_mean_3"] = df["Close"].shift(1).rolling(window=3).mean()  #Average closing price from previous 3 days

        df.dropna(inplace=True) #Dropping NaN values
        
        save_to = 'data/processed/stock_data_new.csv'
        df.to_csv(save_to, index=False)
        logger.debug("Added new features to the data and loaded to: %s", save_to)
    except Exception as e:
        logger.error("Error occurred while creating new features: %s", e)
        raise

if __name__ == "__main__":
    new_features()