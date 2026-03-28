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

Without a virtual environment, different library versions may cause the code to fail. Using `venv` ensures consistent execution.

<img width="121" height="44" alt="image" src="https://github.com/user-attachments/assets/f381b159-9a1a-46a2-92d1-0f005ceb7314" />

---

## Setup

Ensure python is installed in the device.
In terminal, change the working directory to the directory where every downloaded python files exist.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash python3 ``` is used on Linux/macOS to explicitly ensure Python 3 is selected, since ```bash python ``` may point to an older version on some systems.

---

## Execution

Run the complete pipeline:

```bash id="7h4mzp"
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

```text
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

This step ensures a continuous dataset suitable for machine learning.

---

### Output

```text
data/processed/
    tcs.csv
    bajaj.csv
    vedanta.csv
```

---

## fft_analysis.py

### Function

Applies Fast Fourier Transform (FFT) to each feature.

### Output

```text
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

```text
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

```text
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

```text
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

```text
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

```text
128 × 128 grayscale image
```

---

### Output

```text
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

This file uses the trained CNN model to generate predictions from the spectrogram image dataset.

For every dataset image:

* the corresponding 128-day input window is loaded
* the CNN predicts the **next price percentage change**
* the predicted percentage change is converted back into the **actual stock price**
* the result is compared with the real stock price

---

### Outputs Produced

This file produces **two outputs for each company** inside the `outputs/` folder:

```text
{company}_prediction.png
{company}_prediction_table.png
```

---

### 1) Prediction Graph

The graph shows:

* Actual stock price
* Predicted stock price

This is used to visually compare model performance over all prediction samples.

---

### 2) Prediction Table

A complete prediction table is also generated and saved as:

```text
{company}_prediction_table.png
```

The table contains:

* **Date** → exact date for which the prediction is made
* **Actual** → real stock price on that date
* **Predicted** → CNN predicted stock price
* **Difference** → actual − predicted

---

### What dates are predicted?

Predictions are **not made for every single day**.

The dataset was created using:

* **Window length = 128 days**
* **Hop size = 16**

This means:

* each image uses 128 days as input
* the next prediction corresponds to the **day immediately after that 128-day window**
* the next sample starts 16 rows later

---

## cleanup.py

### Function

Deletes generated data and model files to reset the project.

---

### What it removes

* All generated folders (dataset, outputs, processed data, etc.)
* All `.keras` model files

It keeps only the virtual environment (`venv`) intact.

---

### When to use it

* To rerun the entire pipeline from scratch
* To clear corrupted or outdated outputs

---

### Important Note

This script permanently deletes files. It should be used with caution.

---

## Data Flow Summary

```text
→ Download Data
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
