import mlflow
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import logging
import os
import yaml
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensures logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('randomforest')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'randomforest.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def tune_model():
    """
    Trains the RandomForestRegressor model on the preprocessed stock price data
    """
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    X_train = pd.read_csv("data/train/X_train.csv")
    y_train = pd.read_csv("data/train/y_train.csv")
    logger.debug("Training data loaded successfully.")

    model = RandomForestRegressor(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)

    scoring = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params['param_grid_randomforest'],
        scoring=scoring,
        refit='r2',
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    mlflow.set_experiment('Model_Tune_RandomForest')

    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        mlflow.log_params(best_params)

        model = grid_search.best_estimator_
        
        # Predict on training set
        y_train_pred = model.predict(X_train)
        true_train_mse = mean_squared_error(y_train, y_train_pred)
        true_train_mae = mean_absolute_error(y_train, y_train_pred)
        true_train_r2 = r2_score(y_train, y_train_pred)

        mlflow.log_metrics({
            "train_mse": true_train_mse,
            "train_mae": true_train_mae,
            "train_r2": true_train_r2
        })

        train_df = X_train.copy()
        train_df['target'] = y_train

        train_df = mlflow.data.from_pandas(train_df)
        mlflow.log_input(train_df, "training_data")

        mlflow.log_artifact(__file__)

        model = grid_search.best_estimator_
        joblib.dump(model, 'models/randomforest/best_model.pkl')

        mlflow.sklearn.log_model(model, "best_model")
        mlflow.set_tag("Run", "1")

if __name__ == "__main__":
    tune_model()
