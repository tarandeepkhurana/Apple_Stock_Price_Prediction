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
  - `lag_1` to `lag_5`: Closing prices from the previous 1 to 5 days
  - `return_1`: % change in closing price from the previous day
  - `return_3`: % change in closing price from 3 days ago
  - `rolling_mean_3`: 3-day rolling average of the closing price (excluding today)
  - `rolling_std_3`: 3-day rolling standard deviation (volatility)
  - `rolling_mean_7`: 7-day rolling mean
  - `rolling_std_7`: 7-day rolling std deviation
  - `volume_change`: % change in volume from previous day
  - `rolling_vol_mean_5`: 5-day rolling average of trading volume
  - `ema_10`: 10-day exponential moving average of closing price
  - `momentum_3`: Difference between today's close and close 3 days ago
  - `lag_rolling_mean_3`: Lagged version of the 3-day rolling mean
  - `day_of_week`: Integer (0 = Monday, ..., 4 = Friday)
  - `is_month_start`: 1 if it's the first day of the month, else 0
  - `is_month_end`: 1 if it's the last day of the month, else 0

### Google Stocks
- **Algorithm Used:** RandomForestRegressor
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
| MAE    | 2.15  |
| MSE    | 5.85  |

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

