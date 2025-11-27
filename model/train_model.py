import joblib

from model_utils import train_linear_regression, load_pulsebat_data
from preprocessing import mergeSort2D
from config import *

# Train and save model into a .pk1 file for chatbot use
# Load data
df = load_pulsebat_data(DATASET_FILEPATH)
if df is None:
    raise ValueError("Failed to train and save linear regression model")

voltage_cols = [f'U{i}' for i in range(1, 22)]      # Create a list of columns U1-U21
X = df[voltage_cols].to_numpy()                     # voltage values for each row
y = df['SOH']                                       # state of health value for each row

# Train and output saved model
model, _, _ = train_linear_regression(X, y)
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"Trained model saved to {MODEL_OUTPUT_PATH}")