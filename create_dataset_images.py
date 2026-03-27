import pandas as pd
import numpy as np
from scipy.signal import stft
import os
import cv2

L = 128
H = 16   # 🔥 more data
features = ["price", "revenue", "profit", "sensex", "usd_inr"]

def create_dataset_for_company(csv_path, company):

    print(f"\nProcessing: {company}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    base = f"dataset_img/{company}"
    os.makedirs(base, exist_ok=True)

    sample_id = 0

    for i in range(0, len(df) - L - 1, H):

        window = df.iloc[i:i+L]
        spectrograms = []

        for feature in features:

            signal = window[feature].values

            # 🔥 FULL NORMALIZATION
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            _, _, Zxx = stft(signal, nperseg=64)

            mag = np.log1p(np.abs(Zxx))

            # normalize spectrogram
            mag = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
            mag = mag[:64, :64]

            spectrograms.append(mag)

        combined = np.vstack(spectrograms)

        # resize
        combined = cv2.resize(combined, (128, 128))

        # scale to image
        combined = (combined - combined.min()) / (combined.max() + 1e-8)
        combined = (combined * 255).astype(np.uint8)

        cv2.imwrite(f"{base}/{sample_id}.png", combined)

        # 🔥 TARGET = % CHANGE
        next_price = df["price"].iloc[i+L]
        current_price = df["price"].iloc[i+L-1]

        target = (next_price - current_price) / current_price

        np.save(f"{base}/{sample_id}_y.npy", target)

        sample_id += 1

    print(f"✅ Done: {company} | Samples: {sample_id}")


create_dataset_for_company("data/processed/tcs.csv", "tcs")
create_dataset_for_company("data/processed/bajaj.csv", "bajaj")
create_dataset_for_company("data/processed/vedanta.csv", "vedanta")