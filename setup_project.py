import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Define paths
project_root = os.getcwd()  # Root directory of the project
data_dir = os.path.join(project_root, 'data', 'raw_data')  # Directory where the data will be stored
requirements_file = os.path.join(project_root, 'requirements.txt')  # Requirements file path

# Kaggle competition details
competition_name = "booking-challenge"
kaggle_zip_path = os.path.join(data_dir, f"{competition_name}.zip")

# Step 1: Ensure the data directory exists
print(f"Ensuring data directory exists at: {data_dir}")
os.makedirs(data_dir, exist_ok=True)

# Step 2: Download the dataset from Kaggle
print(f"Downloading dataset from Kaggle competition: {competition_name}")
api = KaggleApi()
api.authenticate()  # Authenticate using the kaggle.json file
api.competition_download_files(competition_name, path=data_dir)

# Step 3: Extract the dataset
print(f"Extracting dataset to: {data_dir}")
with zipfile.ZipFile(kaggle_zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Step 4: Clean up the zip file
print(f"Cleaning up the zip file: {kaggle_zip_path}")
os.remove(kaggle_zip_path)

print(f"Dataset successfully downloaded and extracted to: {data_dir}")

# Step 5: Install dependencies
print("Installing Python dependencies...")
if os.path.exists(requirements_file):
    os.system(f"pip install -r {requirements_file}")
else:
    print(f"No requirements file found at {requirements_file}. Skipping dependency installation.")

# Step 6: Finalize setup
print("Setup complete! Your project is ready to use.")
