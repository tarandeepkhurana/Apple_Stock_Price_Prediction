import logging
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure the "logs" directory exists
log_dir = 'logs/google'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocess')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocess.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def split_dataset(model: str) -> None:
    """
    Splits the dataframe into training and testing datasets.
    """
    try:
        file_path = f"data/feature_engineered/google/{model}/stock_data_new.csv"
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from: %s", file_path)

        train_size = int(len(df) * 0.8)

        X = df.iloc[:, 6:]
        y = df["Close"]

        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        X_train.columns = X_train.columns.str.strip()
        X_test.columns = X_test.columns.str.strip()
        
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrames
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        X_train_scaled_df.to_csv(f"data/train/google/{model}/X_train.csv", index=False)
        y_train.to_csv(f"data/train/google/{model}/y_train.csv", index=False)
        X_test_scaled_df.to_csv(f"data/test/google/{model}/X_test.csv", index=False)
        y_test.to_csv(f"data/test/google/{model}/y_test.csv", index=False)
        logger.debug("Data splitted successfully.")

        joblib.dump(scaler, f'models/google/scaler_{model}.pkl')
        logger.debug("Scaler saved")
        
    except Exception as e:
        logger.error("Error occurred while splitting the dataset: %s", e)
        raise

if __name__ == "__main__":
    split_dataset('xgb')
    split_dataset('rf')
    split_dataset('lgbm')
