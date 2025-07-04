import mlflow
import pandas as pd
import logging
import os
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('eda')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'eda.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def feature_correlation():
    """
    Plots a heatmap showing feature correlation.
    """
    file_path = "data/processed/stock_data_new.csv"
    df = pd.read_csv(file_path)
    logger.debug("Data loaded successfully from: %s", file_path)

    X = df[["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "return_1", "return_3", "rolling_mean_3", "rolling_std_3", "rolling_mean_7", "rolling_std_7", "day_of_week", "is_month_start", "is_month_end", "volume_change", "rolling_vol_mean_5", "ema_10", "momentum_3", "lag_rolling_mean_3"]]

    # Compute correlation matrix
    corr_matrix = X.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("artifacts/correlation_heatmap.png")  # Save locally for MLflow logging
    logger.debug("Heatmap saved.")

    mlflow.set_experiment('EDA')

    with mlflow.start_run():
        # Log the image
        mlflow.log_artifact("artifacts/correlation_heatmap.png")
        mlflow.set_tag("Run", "8")
        mlflow.log_param("num_features", X.shape[1])

if __name__ == "__main__":
    feature_correlation()