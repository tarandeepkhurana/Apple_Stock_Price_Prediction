import pandas as pd
import matplotlib.pyplot as plt
import mlflow

def plot_predictions():
    df = pd.read_csv("monitoring/predictions_log.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Actual"])
    
    mlflow.set_experiment('Result_Visualize')
    with mlflow.start_run():
        plt.figure(figsize=(10, 5))
        plt.plot(df["Date"], df["Predicted"], label="Predicted")
        plt.plot(df["Date"], df["Actual"], label="Actual", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title("Predicted vs Actual Apple Stock Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("monitoring/pred_vs_actual.png")
        # plt.show()
        mlflow.log_artifact("monitoring/pred_vs_actual.png")
        mlflow.set_tag("Run", "6")

    

if __name__ == "__main__":
    plot_predictions()
