import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from config import *

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
