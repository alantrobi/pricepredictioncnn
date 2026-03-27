# PRICE PREDICTION WITH CNN

**Alan T. Robi**
**TCR24CS009**

---

## Overview

This project presents a complete pipeline for predicting stock price movement using:

* Signal processing techniques (FFT, STFT)
* Spectrogram-based image representation
* Convolutional Neural Network (CNN)

The system transforms financial time-series data into image representations and uses a CNN to learn patterns and predict future price changes.

---

## Why Virtual Environment (venv)?

A virtual environment is used to:

* Ensure all required libraries are installed correctly
* Avoid conflicts with other Python projects
* Maintain reproducibility across systems

Without a virtual environment, different library versions may cause the code to fail. Using `venv` ensures consistent execution.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Execution

Run the complete pipeline:

```bash
python main.py
```

Each Python file executes sequentially and produces outputs required for the next stage.

---

## download_data.py

### Function

Downloads market-related time-series data:

* Stock prices (TCS, Bajaj, Vedanta)
* Sensex index
* USD/INR exchange rate

### Note

Revenue and profit data are not downloaded here.
They are manually defined in `prepare_data.py`.

### Output

```
data/
    tcs.csv
    bajaj.csv
    vedanta.csv
    sensex.csv
    usd_inr.csv
```

---

## prepare_data.py

### Function

* Combines all features into a single dataset:

  * price
  * revenue
  * profit
  * sensex
  * usd_inr
* Aligns all data to a daily time scale

---

### Interpolation and Data Filling

Since:

* Stock data is daily
* Revenue and profit are quarterly

Missing values occur. These are handled using:

* **Interpolation**: Smooth estimation between known values
* **Forward/Backward filling**: Ensures no missing values remain

This step is necessary because machine learning models require complete datasets.

---

### Output

```
data/processed/
    tcs.csv
    bajaj.csv
    vedanta.csv
```

This processed data is used for all further analysis.

---

## fft_analysis.py

### Function

Applies Fast Fourier Transform (FFT) to each feature.

### Output

```
outputs/frequency/{company}/fft_*.png
```

### Explanation

* X-axis: Frequency
* Y-axis: Magnitude (log scale)

Log scaling is used to handle large differences in magnitude and improve visibility.

### Insight

Most energy appears in low frequencies, indicating that stock data is dominated by long-term trends.

---

## stft_analysis.py

### Function

Applies Short-Time Fourier Transform (STFT) to each feature.

### Output

```
outputs/spectrograms/{company}/{feature}.png
```

### Spectrogram Explanation

* X-axis: Time
* Y-axis: Frequency
* Color: Magnitude

This representation shows how frequency components change over time.

---

## create_dataset_images.py

### Function

Creates the dataset used for CNN training.

---

### Process

1. **Windowing**

   * 128 days of data form one sample

2. **Spectrogram Generation**

   * Each feature is converted into a spectrogram

3. **Feature Stacking**

   * Five spectrograms are stacked vertically into a single image:

```
PRICE
REVENUE
PROFIT
SENSEX
USD_INR
```

4. **Resizing**

   * Final image size is 128 × 128

5. **Normalization**

   * Signal normalization
   * Spectrogram normalization
   * Image scaling

---

### Output

```
dataset_img/{company}/

0.png
0_y.npy
1.png
1_y.npy
...
```

---

### File Explanation

**.png file**

* Combined spectrogram image
* Represents 128 days of multivariate data

**.npy file**

Stores the target value:

```
(next_price - current_price) / current_price
```

This represents the percentage change in stock price.

---

## cnn_regression_images.py

### Function

* Loads dataset images and labels
* Trains a CNN regression model

---

### Input

```
128 × 128 grayscale image
```

---

### Output

```
{company}_img_model.keras
```

---

### Learning Mechanism

The CNN does not explicitly know feature names.
It learns patterns based on:

* Spatial structure
* Intensity variations
* Position of patterns within the image

---

## predict_image.py

### Function

* Uses trained CNN model
* Predicts future stock prices

---

### Output

```
outputs/

tcs_prediction.png
bajaj_prediction.png
vedanta_prediction.png
```

---

### Graph Explanation

Displays:

* Actual price
* Predicted price

This is used to evaluate model performance.

---

## Data Flow Summary

```
Download Data
→ Prepare and Interpolate
→ FFT Analysis
→ STFT (Spectrogram)
→ Dataset Creation (Images + Labels)
→ CNN Training
→ Prediction
```

---

## Key Concepts

* Spectrograms convert time-series data into image form
* CNN models learn patterns from images
* Normalization improves training stability
* Predicting percentage change simplifies the learning task

---
