import pandas as pd
import logging
import os
from src.google.data_preprocess import split_dataset

#Ensures logs directory exists
log_dir = 'logs/google' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('xgb_features')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'xgb_features.log')
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
        file_path = "data/raw/google/stock_data.csv"
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from: %s", file_path)
        
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        df["lag_1"] = df["Close"].shift(1)  #Closing price 1 day prior
        df["lag_2"] = df["Close"].shift(2)  #Closing price 2 days prior
        df["lag_3"] = df["Close"].shift(3)  #Closing price 3 days prior
        df["lag_4"] = df["Close"].shift(4)  #Closing price 4 days prior
        df["lag_5"] = df["Close"].shift(5)  #Closing price 5 days prior
        # df["return_1"] = df["Close"].pct_change(1).shift(1)     # % change from previous day
        # df["return_3"] = df["Close"].pct_change(3).shift(1)     # % change from 3 days ago
        df["rolling_mean_3"] = df["Close"].shift(1).rolling(window=3).mean()  #Average closing price from previous 3 days
        # df["rolling_std_3"] = df["Close"].shift(1).rolling(window=3).std()
        df["rolling_mean_7"] = df["Close"].shift(1).rolling(window=7).mean()
        # df["rolling_std_7"] = df["Close"].shift(1).rolling(window=7).std()
        # df["Date"] = pd.to_datetime(df["Date"])
        # df["day_of_week"] = df["Date"].dt.dayofweek           # 0 = Monday, ..., 4 = Friday
        # df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
        # df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
        # df["volume_change"] = df["Volume"].pct_change().shift(1)
        df["rolling_vol_mean_5"] = df["Volume"].shift(1).rolling(5).mean()
        df["ema_10"] = df["Close"].ewm(span=10).mean().shift(1)
        df["momentum_3"] = df["Close"] - df["Close"].shift(3)
        df["lag_rolling_mean_3"] = df["rolling_mean_3"].shift(1)


        df.dropna(inplace=True) #Dropping NaN values
        
        save_to = 'data/feature_engineered/google/xgb/stock_data_new.csv'
        df.to_csv(save_to, index=False)
        logger.debug("Added new features to the data and loaded to: %s", save_to)
    except Exception as e:
        logger.error("Error occurred while creating new features: %s", e)
        raise

if __name__ == "__main__":
    new_features()
    split_dataset('xgb')