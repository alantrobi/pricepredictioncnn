import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

# ----------------------------
# Companies list
# ----------------------------
companies = ["tcs", "bajaj", "vedanta"]

# ----------------------------
# Features
# ----------------------------
features = ["price", "revenue", "profit", "sensex", "usd_inr"]

# ----------------------------
# Loop through companies
# ----------------------------
for company in companies:

    print(f"Processing: {company}")

    # Create folder per company
    os.makedirs(f"outputs/spectrograms/{company}", exist_ok=True)

    # Load data
    df = pd.read_csv(f"data/processed/{company}.csv", index_col=0, parse_dates=True)

    # ----------------------------
    # STFT for each feature
    # ----------------------------
    for feature in features:

        signal = df[feature].values

        # Remove DC component
        signal = signal - np.mean(signal)

        # STFT
        f, t, Zxx = stft(signal, nperseg=64)

        magnitude = np.abs(Zxx)

        # ----------------------------
        # Plot
        # ----------------------------
        plt.figure()

        plt.pcolormesh(t, f, magnitude, shading='gouraud')
        plt.title(f"{company.upper()} - {feature.upper()}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label="Magnitude")

        # Save
        plt.savefig(f"outputs/spectrograms/{company}/{feature}.png")

        plt.close()

    print(f"✅ Done: {company}")