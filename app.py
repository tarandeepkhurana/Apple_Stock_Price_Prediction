from flask import Flask, jsonify, render_template
import pandas as pd
import json
from src.apple.daily_predict import daily_predict
from src.apple.recent_stock_predictions import get_prediction_trend
from src.google.recent_stock_predictions import get_prediction_trend_google
from src.google.daily_predict import daily_predict_google

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/apple")
def apple_dashboard():
    df = pd.read_csv("data/last_15_days/last_15_days_predictions/apple/predictions_log_apple.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    df["Absolute_Error"] = (df["Predicted"] - df["Actual"]).abs()

    prediction_value, prediction_date, lower_bound, upper_bound = daily_predict()
    recent_pred = df["Predicted"].tail(5).tolist()
    trend = get_prediction_trend(recent_pred)
    
    df_raw = pd.read_csv("data/raw/apple/stock_data.csv")
    last_15 = df_raw.tail(15)
    volume = last_15["Volume"].tolist()
    closing_prices = last_15["Close"].tolist()

    return render_template("apple.html",
                           labels=df["Date"].tolist(),
                           predicted=df["Predicted"].tolist(),
                           actual=df["Actual"].tolist(),
                           mae=df["MAE"].tolist(),
                           mse=df["MSE"].tolist(),
                           prediction_date=prediction_date,
                           prediction_value=round(prediction_value, 2),
                           lower_bound=lower_bound,
                           upper_bound=upper_bound,
                           trend=trend,
                           error_table=df.tail(15).to_dict(orient="records"),
                           volume=volume,
                           closing_prices=closing_prices)

@app.route("/google")
def google_dashboard():
    df = pd.read_csv("data/last_15_days/last_15_days_predictions/google/predictions_log_google.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    df["Absolute_Error"] = (df["Predicted"] - df["Actual"]).abs()

    prediction_value, prediction_date, lower_bound, upper_bound = daily_predict_google()
    recent_pred = df["Predicted"].tail(5).tolist()
    trend = get_prediction_trend_google(recent_pred)
    
    df_raw = pd.read_csv("data/raw/google/stock_data.csv")
    last_15 = df_raw.tail(15)
    volume = last_15["Volume"].tolist()
    closing_prices = last_15["Close"].tolist()

    return render_template("google.html",
                           labels=df["Date"].tolist(),
                           predicted=df["Predicted"].tolist(),
                           actual=df["Actual"].tolist(),
                           mae=df["MAE"].tolist(),
                           mse=df["MSE"].tolist(),
                           prediction_date=prediction_date,
                           prediction_value=round(prediction_value, 2),
                           lower_bound=lower_bound,
                           upper_bound=upper_bound,
                           trend=trend,
                           error_table=df.tail(15).to_dict(orient="records"),
                           volume=volume,
                           closing_prices=closing_prices)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
