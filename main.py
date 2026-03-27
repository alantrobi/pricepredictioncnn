import subprocess

steps = [
    "download_data.py",
    "prepare_data.py",
    "fft_analysis.py",
    "stft_analysis.py",
    "create_dataset_images.py",
    "cnn_regression_images.py",
    "predict_image.py"
]

print("\n🚀 PIPELINE STARTED\n")

for step in steps:
    print(f"\n▶ Running: {step}")

    try:
        subprocess.run(["python", step], check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ Error in {step}. Stopping.")
        break

print("\n✅ DONE\n")