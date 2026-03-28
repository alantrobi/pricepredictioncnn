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

    dates = []
    y_true = []
    y_pred = []
    differences = []

    for file in files:

        sample_id = int(file.replace(".png", ""))

        # Load image
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)


        # Date
        actual_index = sample_id * 16
        prediction_index = actual_index + 128

        date = df.index[prediction_index]
        actual_price = df["price"].iloc[prediction_index]
        prev_price = df["price"].iloc[prediction_index - 1]

        # Predict % change
        pred_change = model.predict(img, verbose=0)[0][0]

        # Convert to actual price
        pred_price = prev_price * (1 + pred_change)

        difference = actual_price - pred_price

        dates.append(date.strftime("%Y-%m-%d"))
        y_true.append(actual_price)
        y_pred.append(pred_price)
        differences.append(difference)

    # ----------------------------
    # GRAPH (UNCHANGED)
    # ----------------------------
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

    # ----------------------------
    # TABLE IMAGE
    # ----------------------------
    table_df = pd.DataFrame({
        "Date": dates,
        "Actual": np.round(y_true, 2),
        "Predicted": np.round(y_pred, 2),
        "Difference": np.round(differences, 2)
    })

    rows = len(table_df)
    fig_height = max(6, rows * 0.35)


    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="upper center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(f"outputs/{company}_prediction_table.png",dpi=200,bbox_inches="tight")
    plt.close()

    # MAE
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    print(f"{company} MAE: {mae:.2f}")


# Run
predict_and_plot("tcs", "data/processed/tcs.csv")
predict_and_plot("bajaj", "data/processed/bajaj.csv")
predict_and_plot("vedanta", "data/processed/vedanta.csv")