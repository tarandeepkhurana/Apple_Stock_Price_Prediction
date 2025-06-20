import mlflow
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

def monitoring():
    df = pd.read_csv("monitoring/manual_features/google/manual_last_15days_features.csv")

    xgboost = joblib.load("models/google/xgboost.pkl")
    lightgbm = joblib.load("models/google/lightgbm.pkl")
    rf = joblib.load("models/google/randomforest.pkl")

    X = df.iloc[:, 2:]
    y = df["Close"]
    
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment('Monitoring_last_15_days')

    with mlflow.start_run():
        y_pred_xgboost = xgboost.predict(X)
        y_pred_lightgbm = lightgbm.predict(X)
        y_pred_rf = rf.predict(X)

        mae_xgboost = mean_absolute_error(y, y_pred_xgboost)
        mse_xgboost = mean_squared_error(y, y_pred_xgboost)

        mae_lightgbm = mean_absolute_error(y, y_pred_lightgbm)
        mse_lightgbm = mean_squared_error(y, y_pred_lightgbm)

        mae_rf = mean_absolute_error(y, y_pred_rf)
        mse_rf = mean_squared_error(y, y_pred_rf)

        mlflow.log_metrics({
            "mae_xgboost": mae_xgboost,
            "mse_xgboost": mse_xgboost,
            "mae_lightgbm": mae_lightgbm,
            "mse_lightgbm": mse_lightgbm,
            "mae_rf": mae_rf,
            "mse_rf": mse_rf
        })

        mlflow.log_artifact(__file__)
        mlflow.set_tag("Run", "4")
        
if __name__ == "__main__":
    monitoring()
    