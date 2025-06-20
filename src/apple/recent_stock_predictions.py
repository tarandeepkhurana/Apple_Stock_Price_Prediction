import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging

#Ensures logs directory exists
log_dir = 'logs/apple' 
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

import pandas as pd
import os

def generate_last_15days_manual_features(
    raw_csv_path="data/raw/apple/stock_data.csv",
    output_csv_path="monitoring/manual_features/apple/manual_last_15days_features.csv"
):
    """
    Manually computes features for the last 15 days using the same logic as get_next_day_features.
    Saves the final DataFrame to a CSV.
    """
    df = pd.read_csv(raw_csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # ⬇️ Force numeric type conversion
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    all_features = []

    # Ensure we have enough data to compute features
    if len(df) < 20:
        raise ValueError("Not enough data to compute 15-day features. Require at least 20 rows.")

    # Loop over the last 15 rows, compute features for each day using past values
    for i in range(len(df) - 15, len(df)):
        current_date = df.iloc[i]["Date"]
        close = df["Close"]
        volume = df["Volume"]

        try:
            lag_1 = close.iloc[i - 1]
            lag_2 = close.iloc[i - 2]
            lag_3 = close.iloc[i - 3]
            lag_4 = close.iloc[i - 4]
            lag_5 = close.iloc[i - 5]

            return_1 = (lag_1 - lag_2) / lag_2
            return_3 = (lag_1 - lag_4) / lag_4

            rolling_mean_3 = close.iloc[i - 3:i].mean()
            rolling_std_3 = close.iloc[i - 3:i].std()

            rolling_mean_7 = close.iloc[i - 7:i].mean()
            rolling_std_7 = close.iloc[i - 7:i].std()

            day_of_week = current_date.dayofweek
            is_month_start = int(current_date.is_month_start)
            is_month_end = int(current_date.is_month_end)

            volume_change = (volume.iloc[i - 1] - volume.iloc[i - 2]) / volume.iloc[i - 2]
            rolling_vol_mean_5 = volume.iloc[i - 5:i].mean()

            ema_10 = close.iloc[:i].ewm(span=10).mean().iloc[-1]
            momentum_3 = close.iloc[i - 1] - close.iloc[i - 4]
            lag_rolling_mean_3 = close.iloc[i - 4:i - 1].mean()

            features = {
                "Date": current_date.strftime("%Y-%m-%d"),  # Add date column for tracking
                "Close": close.iloc[i],
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_4": lag_4,
                "lag_5": lag_5,
                "return_1": return_1,
                "return_3": return_3,
                "rolling_mean_3": rolling_mean_3,
                "rolling_std_3": rolling_std_3,
                "rolling_mean_7": rolling_mean_7,
                "rolling_std_7": rolling_std_7,
                "day_of_week": day_of_week,
                "is_month_start": is_month_start,
                "is_month_end": is_month_end,
                "volume_change": volume_change,
                "rolling_vol_mean_5": rolling_vol_mean_5,
                "ema_10": ema_10,
                "momentum_3": momentum_3,
                "lag_rolling_mean_3": lag_rolling_mean_3,
            }

            all_features.append(features)

        except IndexError:
            print(f"Skipping date {current_date} due to insufficient history.")
            continue

    # Create DataFrame and save
    features_df = pd.DataFrame(all_features)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save to CSV
    features_df.to_csv(output_csv_path, index=False)
    print(f"Saved manual features for last 15 days to: {output_csv_path}")


def predict_last_15_days():
    """
    Predicts last 15 days closing stock prices to create the plots.
    """
    try:
        df = pd.read_csv("monitoring/manual_features/apple/manual_last_15days_features.csv")

        model = joblib.load('models/best_model.pkl')
        logger.debug("Model loaded successfully")

        X = df[["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "return_1", "return_3", "rolling_mean_3", "rolling_std_3", "rolling_mean_7", "rolling_std_7", "day_of_week", "is_month_start", "is_month_end", "volume_change", "rolling_vol_mean_5", "ema_10", "momentum_3", "lag_rolling_mean_3"]]
        y_pred = model.predict(X)

        log_file = "monitoring/predictions/apple/predictions_log.csv"

        log_df = pd.DataFrame(columns=["Date", "Predicted", "Actual", "MAE", "MSE"])

        log_df['Date'] = df['Date'].values
        log_df['Actual'] = df['Close'].values
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

def get_prediction_trend(predictions: list[float]) -> str:
    """
    Given a list of recent predictions, determine the trend.
    Returns: "Upward 📈", "Downward 📉", or "Stable ➖"
    """
    diffs = [predictions[i+1] - predictions[i] for i in range(len(predictions) - 1)]

    if all(d > 0 for d in diffs):
        return "up"
    elif all(d < 0 for d in diffs):
        return "down"
    else:
        return "neutral"

    
if __name__ == "__main__":
   predict_last_15_days()


