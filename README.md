# Stock Price Prediction

This project aims to predict the **next day's closing prices of Apple (AAPL) and Google (GOOGL)** using historical stock data and machine learning. It provides both **quantitative evaluation (MAE, MSE)** and **visualizations** to understand the models' performances.

---

## 🚀 Features

- 📈 Predicts the **next day's closing prices** of AAPL and GOOGL stocks.
- 📊 Displays a **graph comparing actual vs. predicted closing prices** for the last 15 days.
- 🧮 Shows **MAE** (Mean Absolute Error) and **MSE** (Mean Squared Error) to assess prediction accuracy.
- 📉 Provides a **volume vs. closing price** graph to visualize stock trends.

---

## 🧠 Model Details

### Apple Stocks
- **Algorithm Used:** XGBoostRegressor
- **Features Used:**
  - `lag_1` to `lag_3`: Closing prices from the previous 1 to 3 days
  - `rolling_mean_3`: 3-day rolling average of the closing price (excluding today)
  - `rolling_mean_7`: 7-day rolling mean
  - `ema_10`: 10-day exponential moving average of closing price
  - `momentum_3`: Difference between today's close and close 3 days ago
  - `lag_rolling_mean_3`: Lagged version of the 3-day rolling mean
  - `RSI`: 14-day relative strength index measuring recent price momentum
  - `MACD`: Difference between 12-day and 26-day exponential moving averages


### Google Stocks
- **Algorithm Used:** XGBoostRegressor
- **Features Used:**
  - `lag_1` to `lag_5`: Closing prices from the previous 1 to 5 days
  - `rolling_mean_3`: 3-day rolling average of the closing price (excluding today)
  - `rolling_mean_7`: 7-day rolling average of the closing price
  - `rolling_vol_mean_5`: 5-day rolling average of trading volume
  - `ema_10`: 10-day exponential moving average of closing price
  - `momentum_3`: Difference between today's close and close 3 days ago
  - `lag_rolling_mean_3`: Lagged version of the 3-day rolling mean

---

## 📈 Evaluation Metrics

### Apple Stocks
| Metric | Value |
|--------|-------|
| MAE    | 2.28  |
| MSE    | 7.13  |

### Google Stocks
| Metric | Value |
|--------|-------|
| MAE    | 2.15  |
| MSE    | 7.16  |

*(These will update dynamically based on the most recent run.)*

---

## 🛠️ Tech Stack

- **Backend:** Python (Flask)
- **Frontend:** HTML, Bootstrap 5, Chart.js
- **Modeling & Data Processing:** XGBoostRegressor, RandomForestRegressor,Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Chart.js
- **Experiment Tracking:** MLflow
- **Logging:** Python logging module

