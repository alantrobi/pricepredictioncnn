import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

os.makedirs("outputs", exist_ok=True)

def predict_and_plot(company, csv_path):

    print(f"\nProcessing: {company}")

    model = load_model(f"{company}_img_model.keras")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    folder = f"dataset_img/{company}"

    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".png")],
        key=lambda x: int(x.split(".")[0])
    )

    y_true = []
    y_pred = []

    for file in files:

        sample_id = int(file.replace(".png", ""))

        # Load image
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Actual next price
        actual_price = df["price"].iloc[sample_id + 128]

        # Previous price
        prev_price = df["price"].iloc[sample_id + 128 - 1]

        # Predict % change
        pred_change = model.predict(img, verbose=0)[0][0]

        # 🔥 Convert to actual price
        pred_price = prev_price * (1 + pred_change)

        y_true.append(actual_price)
        y_pred.append(pred_price)

    # Plot
    plt.figure()

    plt.plot(y_true, label="Actual Price", linestyle='--')
    plt.plot(y_pred, label="Predicted Price")

    plt.title(f"{company.upper()} - Actual vs Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"outputs/{company}_prediction.png")
    plt.show()

    # MAE
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    print(f"{company} MAE: {mae:.2f}")


# Run
predict_and_plot("tcs", "data/processed/tcs.csv")
predict_and_plot("bajaj", "data/processed/bajaj.csv")
predict_and_plot("vedanta", "data/processed/vedanta.csv")