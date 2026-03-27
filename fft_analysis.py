import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Companies + Paths
# ----------------------------
companies = {
    "tcs": "data/processed/tcs.csv",
    "bajaj": "data/processed/bajaj.csv",
    "vedanta": "data/processed/vedanta.csv"
}

# ----------------------------
# Features
# ----------------------------
features = ["price", "revenue", "profit", "sensex", "usd_inr"]

# ----------------------------
# Process Each Company
# ----------------------------
for company, path in companies.items():

    print(f"Processing: {company}")

    # Create output folder
    output_dir = f"outputs/frequency/{company}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # ----------------------------
    # FFT for each feature
    # ----------------------------
    for feature in features:

        signal = df[feature].values

        # 🔥 Remove DC component
        signal = signal - np.mean(signal)

        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        magnitude = np.abs(fft_vals)

        # ----------------------------
        # Plot
        # ----------------------------
        plt.figure()

        plt.semilogy(
            freqs[:len(freqs)//2],
            magnitude[:len(magnitude)//2]
        )

        plt.title(f"{company.upper()} - FFT - {feature.upper()}")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude (log scale)")

        plt.tight_layout()

        # Save
        plt.savefig(f"{output_dir}/fft_{feature}.png")
        plt.close()

    print(f"Saved FFTs for {company}\n")