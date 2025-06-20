import mlflow
from xgboost import XGBRegressor
import pandas as pd
import logging
import os
import yaml
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib

#Ensures logs directory exists
log_dir = 'logs/google' 
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

    X_train = pd.read_csv("data/train/google/X_train.csv")
    y_train = pd.read_csv("data/train/google/y_train.csv")
    X_test = pd.read_csv("data/test/google/X_test.csv")
    y_test = pd.read_csv("data/test/google/y_test.csv")
    logger.debug("Training and testing data loaded successfully.")

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
    
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment('XGBoost')
    
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        # Get best parameters
        best_params = grid_search.best_params_

        # Log to MLflow
        mlflow.log_params(best_params)
        
        model = grid_search.best_estimator_

        # Predict on training set
        y_train_pred = model.predict(X_train)
        true_train_mse = mean_squared_error(y_train, y_train_pred)
        true_train_mae = mean_absolute_error(y_train, y_train_pred)
        true_train_r2 = r2_score(y_train, y_train_pred)

        # Adjusted R² calculation
        n = X_train.shape[0]  # number of samples
        k = X_train.shape[1]  # number of features
        adjusted_r2 = 1 - ((1 - true_train_r2) * (n - 1)) / (n - k - 1)

        mlflow.log_metrics({
            "train_mse": true_train_mse,
            "train_mae": true_train_mae,
            "train_r2": true_train_r2,
            "adjusted_r2": adjusted_r2
        })

        # Log training data
        train_df = X_train.copy()
        train_df['target'] = y_train

        train_df = mlflow.data.from_pandas(train_df)
        mlflow.log_input(train_df, "training_data")
        
        joblib.dump(model, 'models/google/xgboost.pkl')

        # Log the best model
        mlflow.sklearn.log_model(model, "xgboost")
        
        y_pred = model.predict(X_test)

        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
            
        mlflow.log_metric('test_mae', test_mae)
        mlflow.log_metric('test_mse', test_mse)
        mlflow.log_metric('test_r2', test_r2)
        
        importances = model.feature_importances_
        feature_names = X_train.columns

        # Log each feature importance as a metric
        for name, importance in zip(feature_names, importances):
            mlflow.log_metric(f"feat_imp_{name}", float(importance))

        mlflow.set_tag("Run", "6")

if __name__ == "__main__":
    tune_model()