import subprocess
import sys

scripts = [
    "download_data.py",
    "prepare_data.py",
    "fft_analysis.py",
    "stft_analysis.py",
    "create_dataset_images.py",
    "cnn_regression_images.py",
    "predict_image.py"
]

print("\n🚀 PIPELINE STARTED\n")

for script in scripts:
    print(f"\n▶ Running: {script}")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\n❌ Error in {script}. Stopping.")
        break

print("\n✅ DONE")
