import pandas as pd
import os
import joblib
import logging

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('daily_predict')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'daily_predict.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def daily_predict():
    """
    Predicts the next days closing stock price, updates the predictions_log.csv
    """
    try:
        df = pd.read_csv('data/processed/stock_data_new.csv')
        last_date = df["Date"].iloc[-1]
        print("Last raw data date:", last_date)

        model = joblib.load("models/best_model.pkl")
        logger.debug("Model loaded successfully.")
        
        # Get last 3 actual closing prices (for 8th, 9th, 10th)
        lag_1 = float(df["Close"].iloc[-1])      # 10th
        lag_2 = float(df["Close"].iloc[-2])      # 9th
        lag_3 = float(df["Close"].iloc[-3])      # 8th

        rolling_mean_3 = (lag_1 + lag_2 + lag_3) / 3

        X_latest = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "rolling_mean_3": rolling_mean_3
        }])
        
        X_latest.columns = X_latest.columns.str.strip()

        prediction = model.predict(X_latest)[0]
        
        log_file = "monitoring/predictions_log.csv"
        log_df = pd.read_csv(log_file)
        
        prediction_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)

        log_df["Date"] = prediction_date
        log_df["Predicted"] = prediction
        log_df["Actual"] = None
        log_df["MAE"] = None
        log_df["MSE"] = None
        
        log_df.to_csv(log_file, index=False)
        print(f"Prediction for {prediction_date} logged: {prediction:.2f}")
    except Exception as e:
        logger.error("Error occurred while predicting next day's closing price: %s", e)
        raise
 
if __name__ == "__main__":
    daily_predict()
