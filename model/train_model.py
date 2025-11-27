import joblib

# Imports from other files
from model_utils import train_linear_regression, load_pulsebat_data
from preprocessing import mergeSort2D
from config import *

# Load PulseBat Voltage data
df = load_pulsebat_data(DATASET_FILEPATH)
if df is None:
    raise ValueError("Failed to train and save linear regression model")

voltage_cols = [f'U{i}' for i in range(1, 22)]      # Create a list of columns U1-U21
X = df[voltage_cols]                                # voltage values for each row
y = df['SOH']                                       # state of health value for each row

# Train and save linear regression model as an object
model, _, _ = train_linear_regression(X, y)
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"Trained model object saved to {MODEL_OUTPUT_PATH}")