import mlflow
from lightgbm import LGBMRegressor
import pandas as pd
import logging
import os
import yaml
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import matplotlib.pyplot as plt

#Ensures logs directory exists
log_dir = 'logs/apple' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('lgbm_tuning')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'lgbm_tuning.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
    
def tune_model():
    """
    Trains the XGBRegressor model on the preprocessed stock price data
    """
    # Load params.yaml
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    X_train = pd.read_csv("data/train/apple/lgbm/X_train.csv")
    y_train = pd.read_csv("data/train/apple/lgbm/y_train.csv")
    X_test = pd.read_csv("data/test/apple/lgbm/X_test.csv")
    y_test = pd.read_csv("data/test/apple/lgbm/y_test.csv")
    logger.debug("Training and tesing data loaded successfully.")

    model = LGBMRegressor(random_state=42)

    # Define time series cross-validator
    tscv = TimeSeriesSplit(n_splits=5)

    scoring = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }

    # Applying GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params['param_grid_lightgbm'],
        scoring=scoring,
        refit='r2',  
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    mlflow.set_experiment('LGBM_Tune')

    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        #BEST PARAMETERS
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        
        #BEST MODEL
        model = grid_search.best_estimator_
        mlflow.sklearn.log_model(model, "lgbm_model")
        
        #TRAINING METRICS
        y_train_pred = model.predict(X_train)
        true_train_mse = mean_squared_error(y_train, y_train_pred)
        true_train_mae = mean_absolute_error(y_train, y_train_pred)
        true_train_r2 = r2_score(y_train, y_train_pred)
        n = X_train.shape[0]  # number of samples
        k = X_train.shape[1]  # number of features
        train_adjusted_r2 = 1 - ((1 - true_train_r2) * (n - 1)) / (n - k - 1)

        mlflow.log_metrics({
            "train_mse": true_train_mse,
            "train_mae": true_train_mae,
            "train_r2": true_train_r2,
            "train_adjusted_r2": train_adjusted_r2
        })
        
        #EVALUATION METRICS
        y_pred = model.predict(X_test)

        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        n = X_test.shape[0]  # number of samples
        k = X_test.shape[1]  # number of features
        test_adjusted_r2 = 1 - ((1 - test_r2) * (n - 1)) / (n - k - 1)

        mlflow.log_metrics({
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "test_adjusted_r2": test_adjusted_r2
        })

        #Feature importance
        features = X_train.columns.tolist()
        importances = model.feature_importances_
        df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        html_path = f"artifacts/LightGBM_feature_importance.html" 
        df.to_html(html_path, index=False)
        mlflow.log_artifact(html_path)
        
        #LAST 15 DAYS COMPARISON
        plt.figure(figsize=(10, 5))
        plt.plot(y_pred[-15:], label="Predicted")
        plt.plot(y_test.tail(15).to_numpy(), label="Actual", linestyle="--")
        plt.xlabel("Last 15 Days")
        plt.ylabel("Stock Price")
        plt.title("Predicted vs Actual Apple Stock Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("monitoring/pred_vs_actual.png")
        # plt.show()
        mlflow.log_artifact("monitoring/pred_vs_actual.png")

        #SAVING THE MODEL
        joblib.dump(model, 'models/apple/lgbm_model.pkl')
        logger.debug("Model saved successfully.")
        

if __name__ == "__main__":
    tune_model()