import joblib

from model_utils import train_linear_regression, load_pulsebat_data
from preprocessing import mergeSort2D

# CONSTANTS
DATASET_FILEPATH = "../data/PulseBat Dataset.csv"
MODEL_OUTPUT_PATH = "../app/lr_model.pk1"
TEST_SIZE = 0.25 # reserve portion of data for testing
RANDOM_STATE = 15 # seed for random shuffle: make train/test split the same every run

# Train and save model into a .pk1 file for chatbot use
def main():

    # Load data
    df = load_pulsebat_data(DATASET_FILEPATH)
    if df is None:
        return
    
    voltage_cols = [f'U{i}' for i in range(1, 22)]      # Create a list of columns U1-U21
    X = df[voltage_cols].to_numpy()                     # voltage values for each row
    y = df['SOH']                                       # state of health value for each row
    X_merge = mergeSort2D(X)                            # perform merge sort preprocessing

    # Train and output saved model
    model, _, _ = train_linear_regression(X_merge, y)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Trained model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()