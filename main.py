import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# Constants for configuring model
DATASET_FILE = "PulseBat Dataset.csv" #path to PulseBat data
TEST_SIZE = 0.25 # reserve portion of data for testing
RANDOM_STATE = 15 # seed for random shuffle: make train/test split the same every run


# Retrieve PulseBat data from file
def load_pulsebat_data(file_path):
    global df
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape} samples, {df.shape[1]} columns")

    # TODO: exceptions for any errors 

load_pulsebat_data(DATASET_FILE)

# Train Model
# Based on voltages from U1-21, predict SOH (state of health) 

# names of columns 
voltage_cols = [f'U{i}' for i in range(1, 22)]
# voltage values for each row
X = df[voltage_cols]
# state of health value for each row
y = df['SOH']

# split model into test and training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression model trained successfully!")

# TODO: preprocess with sorting (multiple sorting methods for data)

# Evaluate Model
# TODO: Compare sorting methods
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mean_squared_error = mean_squared_error(y_test, y_pred)
root_mean_squared_error = np.sqrt(mean_squared_error)


print("\nMODEL EVALUATION RESULTS")
print(f"R² Score: {r2:.5f}")
print(f"Mean Squared Error (MSE): {mean_squared_error:.5f}")
print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error:.5f}")
# TODO: Maybe add more metrics

#TODO: Find best sorting method and use 