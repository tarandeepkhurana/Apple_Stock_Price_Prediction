import pandas as pd
import os
import joblib
import logging
import numpy as np
import xgboost as xgb
import ta

#Ensures logs directory exists
log_dir = 'logs/apple' 
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

def get_next_day_features() -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Generates the next day features for the model.
    """
    #Load scaler
    scaler = joblib.load('models/apple/best_model/scaler_xgb.pkl')

    df = pd.read_csv('data/raw/apple/stock_data.csv')
    last_date = df["Date"].iloc[-1]
    print("Last raw data date:", last_date)
    
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    
    df['RSI'] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
    macd_indicator = ta.trend.MACD(close=df["Close"])
    df['MACD'] = macd_indicator.macd()

    prediction_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)

    # Get recent values
    close = df["Close"]
    volume = df["Volume"]

    lag_1 = close.iloc[-1]
    lag_2 = close.iloc[-2]
    lag_3 = close.iloc[-3]
    lag_4 = close.iloc[-4]
    lag_5 = close.iloc[-5]

    return_1 = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
    return_3 = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]

    rolling_mean_3 = close.iloc[-3:].mean()
    rolling_std_3 = close.iloc[-3:].std()

    rolling_mean_7 = close.iloc[-7:].mean()
    rolling_std_7 = close.iloc[-7:].std()

    day_of_week = pd.to_datetime(prediction_date).dayofweek
    is_month_start = int(pd.to_datetime(prediction_date).is_month_start)
    is_month_end = int(pd.to_datetime(prediction_date).is_month_end)

    volume_change = (volume.iloc[-1] - volume.iloc[-2]) / volume.iloc[-2]
    rolling_vol_mean_5 = volume.iloc[-5:].mean()

    ema_10 = close.ewm(span=10).mean().iloc[-1]
    momentum_3 = close.iloc[-1] - close.iloc[-4]
    
    lag_rolling_mean_3 = close.iloc[-4:-1].mean()
    
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]

    X_pred = pd.DataFrame([{
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "rolling_mean_3": rolling_mean_3,
        "rolling_mean_7": rolling_mean_7,
        "ema_10": ema_10,
        "momentum_3": momentum_3,
        "lag_rolling_mean_3": lag_rolling_mean_3,
        "RSI": rsi,
        "MACD": macd
    }])
    
    X_pred_scaled = scaler.transform(X_pred)
    X_pred_scaled_df = pd.DataFrame(X_pred_scaled, columns=X_pred.columns)

    prediction_date = prediction_date.strftime("%Y-%m-%d")
    
    return X_pred_scaled_df, prediction_date


def daily_predict() -> tuple[float, str, float, float]:
    """
    Predicts the next days closing stock price, updates the predictions_log.csv
    """
    try:
        X_pred, prediction_date = get_next_day_features()

        model = joblib.load("models/apple/best_model/xgb_model.pkl")
        logger.debug("Model loaded successfully.")
        
        X_pred.columns = X_pred.columns.str.strip()
        
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(X_pred)

         # Get raw margin outputs at each boosting round
        margin_preds = np.array([
            booster.predict(dmatrix, iteration_range=(0, i), output_margin=True)[0]
            for i in range(1, model.n_estimators + 1)
        ])

        prediction = float(model.predict(X_pred)[0])

        # Estimate 95% CI from margin predictions
        lower = np.percentile(margin_preds, 5)
        upper = np.percentile(margin_preds, 95)
        
        print(f"Prediction for {prediction_date} logged: {prediction:.2f}")

        return prediction, prediction_date, round(lower, 2), round(upper, 2)
    except Exception as e:
        logger.error("Error occurred while predicting next day's closing price: %s", e)
        raise
 
if __name__ == "__main__":
    daily_predict()
