# Copied from https://www.kaggle.com/datasets/nih-chest-xrays/data
import kagglehub

# Download latest version
path = kagglehub.dataset_download("nih-chest-xrays/data", output_dir="./data")

print("Path to dataset files:", path)