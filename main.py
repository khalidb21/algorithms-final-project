import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Constants for configuring model
DATASET_FILE = "PulseBat Dataset.csv" #path to PulseBat data
TEST_SIZE = 0.25 # reserve portion of data for testing
RANDOM_STATE = 15 # seed for random shuffle: make train/test split the same every run
NUMBER_OF_BATTERIES = 671

# Retrieve PulseBat data from file
def load_pulsebat_data(file_path):
    global df
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape} samples, {df.shape[1]} columns")
    # TODO: exceptions for any errors 

def mergeSort(array):
    if len(array) > 1: 
        left = []
        right = []
        mid = len(array) // 2

        for x in range(0, mid):
            left.append(array[x])
        for x in range(mid + 1, len(array)):
            right.append(array[x])
       
        mergeSort(left)
        mergeSort(right)

        i = 0
        j = 0
        k = 0
 
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                array[k] = left[i]                
                i += 1
            else:
                array[k] = right[j]
                j += 1
            k += 1
     
        while i < len(left):
            array[k] = left[i]
            i += 1
            k += 1
 
        while j < len(right):
            array[k] = right[j]
            j += 1
            k += 1

def mergeSort2D(array):
    array_copy = array.copy() #create a copy so original data is not changed

    for x in range(0, NUMBER_OF_BATTERIES - 1): #loop through all batteries
        mergeSort(array_copy[x])
    return array_copy

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
#train model on mergesorted data
x_unsorted = pd.DataFrame(X).to_numpy() #convert dataframe to numpy array to work with
x_mergesort = mergeSort2D(x_unsorted)

x_train_mergesort, x_test_mergesort, y_train_mergesort, y_test_mergesort = train_test_split(
    x_mergesort, y, test_size = TEST_SIZE, random_state = RANDOM_STATE
)
model_mergesort = LinearRegression()
model_mergesort.fit(x_train_mergesort, y_train_mergesort)

# Evaluate Model
# TODO: Compare sorting methods
y_pred = model.predict(X_test)
r2_unsorted = r2_score(y_test, y_pred)
mean_squared_error_unsorted = mean_squared_error(y_test, y_pred)
root_mean_squared_error_unsorted = np.sqrt(mean_squared_error_unsorted)
mean_absolute_error_unsorted = mean_absolute_error(y_test, y_pred)

print("\nMODEL EVALUATION RESULTS")
print(f"R² Score: {r2_unsorted:.5f}")
print(f"Mean Squared Error (MSE): {mean_squared_error_unsorted:.5f}")
print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error_unsorted:.5f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error_unsorted:.5f}")

#mergesort model evaluation
y_pred_mergesort = model_mergesort.predict(x_test_mergesort)
r2_mergesort = r2_score(y_test_mergesort, y_pred_mergesort)
mean_squared_error_mergesort = mean_squared_error(y_test_mergesort, y_pred_mergesort)
root_mean_squared_error_mergesort = np.sqrt(mean_squared_error_mergesort)
mean_absolute_error_mergesort = mean_absolute_error(y_test_mergesort, y_pred_mergesort)

print("\nMERGESORT MODEL EVALUATION RESULTS")
print(f"R² Score: {r2_mergesort:.5f}")
print(f"Mean Squared Error (MSE): {mean_squared_error_mergesort:.5f}")
print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error_mergesort:.5f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error_mergesort:.5f}")
# TODO: Maybe add more metrics

#TODO: Find best sorting method and use 
