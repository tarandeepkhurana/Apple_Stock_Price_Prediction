import mlflow
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

#Ensures logs directory exists
log_dir = 'logs/google' 
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

def feature_correlation(model: str):
    """
    Plots a heatmap showing feature correlation.
    """
    file_path = f"data/feature_engineered/google/{model}/stock_data_new.csv"
    df = pd.read_csv(file_path)
    logger.debug("Data loaded successfully from: %s", file_path)

    X = df.iloc[:, 6:]

    # Compute correlation matrix
    corr_matrix = X.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("artifacts/correlation_heatmap.png")  # Save locally for MLflow logging
    logger.debug("Heatmap saved.")
    
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(f'EDA_{model}')

    with mlflow.start_run():
        # Log the image
        mlflow.log_artifact("artifacts/correlation_heatmap.png")
        mlflow.log_param("num_features", X.shape[1])

if __name__ == "__main__":
    feature_correlation('xgb')
    # feature_correlation('rf')
    # feature_correlation('lgbm')

    