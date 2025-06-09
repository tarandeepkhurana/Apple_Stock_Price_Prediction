import yfinance as yf
import logging
import os
import pandas as pd

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_fetcher')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'data_fetcher.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    """
    It downloads the Apple stock data from yfinance library.
    """
    df = yf.download(ticker, start=start, end=None)
    df.to_csv('data/raw/stock_data.csv')
    logger.debug("Data loaded properly to stock_data.csv")
    return df