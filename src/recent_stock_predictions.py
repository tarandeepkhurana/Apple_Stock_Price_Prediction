import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('recent_stock_predictions')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'recent_stock_predictions.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def predict_last_15_days():
    """
    Predicts last 15 days closing stock prices to create the plots.
    """
    try:
        df = pd.read_csv("data/processed/stock_data_new.csv")

        df_last_15 = df.tail(15)

        model = joblib.load('models/best_model.pkl')
        logger.debug("Model loaded successfully")

        X = df_last_15[["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "rolling_mean_3", "rolling_std_3", "rolling_mean_7", "rolling_std_7"]]
        y_pred = model.predict(X)

        log_file = "monitoring/predictions_log.csv"

        log_df = pd.DataFrame(columns=["Date", "Predicted", "Actual", "MAE", "MSE"])

        log_df['Date'] = df_last_15['Date'].values
        log_df['Actual'] = df_last_15['Close'].values
        log_df['Predicted'] = y_pred
        
        mae = mean_absolute_error(log_df['Actual'], log_df['Predicted'])
        mse = mean_squared_error(log_df['Actual'], log_df['Predicted'])

        log_df['MAE'] = mae
        log_df['MSE'] = mse

        log_df.to_csv(log_file, index=False)
        logger.debug("Last 15 days prediction completed")
    except Exception as e:
        logger.error("Error occurred while predicting last 15 days stock prices: %s", e)
        raise

if __name__ == "__main__":
    predict_last_15_days()


