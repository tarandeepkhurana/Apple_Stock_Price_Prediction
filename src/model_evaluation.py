import pandas as pd
import joblib
import os 
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def evaluate_model():
    """
    Evaluates the best model got from hyperparameter tuning.
    """
    try:
        X_test = pd.read_csv("data/test/X_test.csv")
        y_test = pd.read_csv("data/test/y_test.csv")
        logger.debug("Test dataset loaded successfully.")

        model = joblib.load('models/best_model.pkl')
        logger.debug("Model loaded successfully.")
        
        mlflow.set_experiment('Model_Evaluate')

        with mlflow.start_run():
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_metric('MAE', mae)
            mlflow.log_metric('MSE', mse)
            mlflow.log_metric('R2', r2)

            mlflow.set_tag("Run", "1")
    except Exception as e:
        logger.error("Error occurred while evaluating the model: %s", e)
        raise

if __name__ == "__main__":
    evaluate_model()


        

