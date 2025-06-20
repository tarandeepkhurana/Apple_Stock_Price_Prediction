import logging
from src.google.data_fetcher import fetch_data
from src.google.recent_stock_predictions import predict_last_15_days, generate_last_15days_manual_features
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    """
    This function runs the pipeline to update the data and model's prediction.
    """
    logging.info("Fetching new stock data...")
    fetch_data()

    logging.info("Creating the input features for last 15 days...")
    generate_last_15days_manual_features()

    logging.info("Running predictions for the last 15 days...")
    predict_last_15_days()

    df = pd.read_csv("monitoring/predictions/google/predictions_log.csv")
    current_mae = df.loc[0, 'MAE']
    
    baseline_mae = 3.2
    if current_mae > baseline_mae:
        logging.info(f"MAE deteriorated: {current_mae} | Retrain the model.")
    else:
        logging.info("Model is still performing well.")

if __name__ == "__main__":
    main()