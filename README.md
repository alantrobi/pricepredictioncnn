# 📊 Stock Price Prediction using CNN on Spectrogram Images

---

## 🔍 Overview

This project builds a full pipeline to predict stock price movement using:

* Signal processing (FFT, STFT)
* Spectrogram image generation
* CNN-based regression

The system converts financial time-series data into **images**, allowing a CNN to learn patterns and predict future price changes.

---

# ⚙️ Initial Setup (IMPORTANT)

Before running the project:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

# 📁 Project Structure (Generated Automatically)

> ⚠️ Note: Most folders are **NOT present initially**.
> They are **created automatically by the Python scripts**.

```text
data/
    processed/
        bajaj.csv
        tcs.csv
        vedanta.csv

dataset_img/
    bajaj/
    tcs/
    vedanta/

outputs/
    frequency/
    spectrograms/
    bajaj_prediction.png
    tcs_prediction.png
    vedanta_prediction.png
```

---

# 🚀 Pipeline (Execution Order)

```text
download_data.py
→ prepare_data.py
→ fft_analysis.py
→ stft_analysis.py
→ create_dataset_images.py
→ cnn_regression_images.py
→ predict_image.py
```

---

# 📁 1. download_data.py

## 🔹 What it does

Downloads **market-related time series data only**:

```text
✔ Stock prices (TCS, Bajaj, Vedanta)
✔ Sensex index
✔ USD/INR exchange rate
```

---

## 🔹 Important Clarification

```text
❌ Revenue and Profit are NOT downloaded
```

They are **manually defined inside**:

```text
prepare_data.py
```

---

## 🔹 Output

```text
data/
    bajaj.csv
    tcs.csv
    vedanta.csv
    sensex.csv
    usd_inr.csv
```

---

# 📁 2. prepare_data.py

## 🔹 What it does

This is a **critical data engineering step**.

It:

1. Combines all data sources
2. Adds financial data (revenue, profit)
3. Aligns dates across datasets
4. Performs interpolation and missing value handling

---

## 🔹 Features created

```text
price
revenue
profit
sensex
usd_inr
```

---

## 🔹 Interpolation & Data Filling

Because:

```text
Stock → daily data
Revenue/Profit → quarterly data
```

👉 There are missing values.

---

### ✔ What we did:

#### 1. Forward Fill

```text
Carry last known value forward
```

Used for:

* revenue
* profit

---

#### 2. Interpolation

```text
Smoothly estimate missing values between known points
```

Used for:

* aligning datasets
* filling gaps

---

## 🔹 Why this is necessary

```text
CNN requires continuous, aligned data
```

Without filling:

```text
❌ Missing values → model breaks
```

---

## 🔹 Output

```text
data/processed/
    tcs.csv
    bajaj.csv
    vedanta.csv
```

---

# 📁 3. fft_analysis.py

## 🔹 What it does

* Applies FFT to each feature
* Generates frequency-domain graphs

---

## 🔹 Output

```text
outputs/frequency/{company}/fft_*.png
```

---

## 🔹 Why log scale?

```text
Financial signals have large magnitude differences
```

Log scale helps:

```text
✔ visualize small + large frequencies
✔ reveal hidden patterns
```

---

## 🔹 Insight

```text
Most energy is in low frequency → long-term trends dominate
```

---

# 📁 4. stft_analysis.py

## 🔹 What it does

* Applies STFT (Short-Time Fourier Transform)

---

## 🔹 Output

```text
outputs/spectrograms/{company}/{feature}.png
```

---

## 🔹 What is a Spectrogram?

```text
Time vs Frequency vs Magnitude (color)
```

---

## 🔹 Purpose

```text
Shows how frequency changes over time
```

---

# 📁 5. create_dataset_images.py

## 🔹 What it does

Creates the **CNN training dataset**

---

## 🔹 Process

### 1. Windowing

```text
Take 128 days → 1 sample
```

---

### 2. STFT per feature

Each feature is converted into a spectrogram.

---

### 3. Stack 5 features into ONE image

```text
--------------------------------
PRICE
--------------------------------
REVENUE
--------------------------------
PROFIT
--------------------------------
SENSEX
--------------------------------
USD_INR
--------------------------------
```

---

### 4. Resize

```text
Final image = 128 × 128
```

---

### 5. Normalize

* Signal normalization
* Spectrogram normalization
* Image scaling (0–255)

---

## 🔹 Output

```text
dataset_img/{company}/

0.png
0_y.npy
1.png
1_y.npy
...
```

---

## 🔹 What is `.png`?

```text
Combined spectrogram image (input)
```

---

## 🔹 What is `.npy`?

```text
Target value (label):

(next_price - current_price) / current_price
```

👉 Percentage price change

---

# 📁 6. cnn_regression_images.py

## 🔹 What it does

* Loads dataset
* Trains CNN model

---

## 🔹 Input

```text
128 × 128 image
```

---

## 🔹 Output

```text
{company}_img_model.keras
```

---

## 🔹 What CNN learns

```text
Patterns in spectrogram → price movement
```

It does NOT know features explicitly.

It learns based on:

```text
Position + intensity + patterns
```

---

# 📁 7. predict_image.py

## 🔹 What it does

* Uses trained model
* Predicts future price

---

## 🔹 Output

```text
outputs/
    bajaj_prediction.png
    tcs_prediction.png
    vedanta_prediction.png
```

---

## 🔹 Graph Meaning

```text
Actual Price vs Predicted Price
```

---

# 🧠 Data Flow Summary

```text
Download Data
→ Add financial data
→ Clean & interpolate
→ FFT (frequency understanding)
→ STFT (time-frequency)
→ Spectrogram images
→ Dataset (image + label)
→ CNN training
→ Prediction
```

---

# 🏆 Key Concepts

## ✔ Why spectrograms?

Convert time-series → image patterns

## ✔ Why CNN?

Best for pattern recognition

## ✔ Why % change?

Stable, scale-independent target

## ✔ Why interpolation?

Align different time frequencies (daily vs quarterly)

---

# ▶️ How to Run

```bash
python main.py
```

---

# 👨‍💻 Author

Alan Robi
