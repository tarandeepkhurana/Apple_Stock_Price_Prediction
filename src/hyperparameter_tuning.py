import mlflow
from xgboost import XGBRegressor
import pandas as pd
import logging
import os
import yaml
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('hypertuning')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'hypertuning.log')
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

    X_train = pd.read_csv("data/train/X_train.csv")
    y_train = pd.read_csv("data/train/y_train.csv")
    logger.debug("Training data loaded successfully.")

    model = XGBRegressor(random_state=42)

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
        param_grid=params['param_grid'],
        scoring=scoring,
        refit='r2',  
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    mlflow.set_experiment('Model_Tune')

    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        # Get best parameters
        best_params = grid_search.best_params_

        # Get index of best estimator
        best_index = grid_search.best_index_

        # Get best scores for each metric
        best_mse = -grid_search.cv_results_['mean_test_mse'][best_index]  # negate because we set greater_is_better=False
        best_mae = -grid_search.cv_results_['mean_test_mae'][best_index]
        best_r2 = grid_search.cv_results_['mean_test_r2'][best_index]

        # Log to MLflow
        mlflow.log_params(best_params)

        mlflow.log_metrics({
            "best_mse": best_mse,
            "best_mae": best_mae,
            "best_r2": best_r2
        })

        # Log training data
        train_df = X_train.copy()
        train_df['target'] = y_train

        train_df = mlflow.data.from_pandas(train_df)
        mlflow.log_input(train_df, "training_data")

        # Log source code
        mlflow.log_artifact(__file__)
        
        model = grid_search.best_estimator_
        joblib.dump(model, 'models/best_model.pkl')

        # Log the best model
        mlflow.sklearn.log_model(model, "best_model")
    
        mlflow.set_tag("Run", "7")

if __name__ == "__main__":
    tune_model()