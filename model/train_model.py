import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

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
    X = df[voltage_cols]                                # voltage values for each row
    y = df['SOH']                                       # state of health value for each row

    # Train and output saved model
    model, _, _ = train_linear_regression(X, y)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Trained model saved to {MODEL_OUTPUT_PATH}")

# Retrieve PulseBat data from file
def load_pulsebat_data(DATASET_FILEPATH):
    try:
        df = pd.read_csv(DATASET_FILEPATH)
        print(f"Dataset loaded: {df.shape} samples, {df.shape[1]} columns")
        return df 
    except FileNotFoundError:
        print(f"ERROR: File '{DATASET_FILEPATH}' not found!")
        return None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

# Model training function: based on voltages from U1-21 predict SOH (state of health) 
def train_linear_regression(X, y):

    # Split model into test and training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Train linear regression model to fit training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained successfully!\n")

    return model, X_test, y_test

if __name__ == "__main__":
    main()