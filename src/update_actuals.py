import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging

#Ensures logs directory exists
log_dir = 'logs' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_fetcher')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'data_fetcher.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def update_actuals():
    """
    This function updates the actual closing price of the stock in predictions_log.csv 
    """
    try:
        # Load logs and raw stock data
        log_df = pd.read_csv("monitoring/predictions_log.csv")
        raw_df = pd.read_csv("data/raw/stock_data.csv")

        updated = False

        for i in range(len(log_df)):
            if pd.isna(log_df.loc[i, "Actual"]):
                log_date = log_df.loc[i, "Date"]
                actual_row = raw_df[raw_df["Date"] == log_date]
                if not actual_row.empty:
                    actual_price = actual_row["Close"].values[0]
                    log_df.loc[i, "Actual"] = actual_price

                    # Compute MAE and MSE till this point
                    y_true = log_df["Actual"].dropna()
                    y_pred = log_df["Predicted"].loc[y_true.index]
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)

                    log_df.loc[i, "MAE"] = mae
                    log_df.loc[i, "MSE"] = mse

                    updated = True
                    print(f"Updated actual for {log_date}: {actual_price:.2f} | MAE: {mae:.2f}, MSE: {mse:.2f}")

        if updated:
            log_df.to_csv("monitoring/predictions_log.csv", index=False)
        else:
            print("No new actual values to update.")
    
    except Exception as e:
        logger.error("Unexpected error occured while updating the actual value: %s", e)
        raise    


update_actuals()
