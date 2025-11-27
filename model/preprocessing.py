import copy

from config import *
from model_utils import train_linear_regression, load_pulsebat_data
from evaluate_performance import evaluate_model

def main():
    df = load_pulsebat_data(DATASET_FILEPATH)
    if df is None:
        return
    
    voltage_cols = [f"U{i}" for i in range(1, 22)]
    X = df[voltage_cols].to_numpy()
    y = df["SOH"]

    # Run comparison of sorting strategies
    best_method = compare_sorting_methods(X, y)
    print(f"\nBEST METHOD SELECTED (Lowest MSE): {best_method.upper()}")
    print("However, results are similar enough to consider sorting to have a neglible effect.\n")
    print("UNSORTED DATA WILL BE USED TO TRAIN MODEL.\n")

def mergeSort2D(array):
    
    # Merge sort 
    def mergeSort(array):
        if len(array) > 1: 
            left = []
            right = []
            mid = len(array) // 2

            for x in range(0, mid):
                left.append(array[x])
            for x in range(mid, len(array)):
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

    array_copy = copy.deepcopy(array) #create a copy so original data is not changed

    for x in range(len(array_copy)): #loop through all batteries
        mergeSort(array_copy[x])
    return array_copy

# Selection Sort
def selectionSort2D(array_2d):

    def selectionSort(row):
        n = len(row)
        for i in range(n - 1):
            min = i
            for j in range(i + 1, n):
                if row[j] < row[min]:
                    min = j
            if min != i:
                row[i], row[min] = row[min], row[i]

    b = array_2d.copy()         
    for r in range(len(b)):     
        selectionSort(b[r])
    return b


def compare_sorting_methods(X, y):
    
    # Store results in a dictionary with name : mse pairs 
    results = {}

    # Unsorted data evaluation
    model_unsorted, X_test, y_test = train_linear_regression(X, y)
    print("\n====== UNSORTED DATA RESULTS ======")
    mse_unsorted = evaluate_model(model_unsorted, X_test, y_test)
    results["unsorted"] = float(mse_unsorted)

    # Merge sort evaluation
    X_merge = mergeSort2D(X)
    model_merge, X_test, y_test = train_linear_regression(X_merge, y)
    print("\n====== MERGESORT RESULTS ======")
    mse_merge = evaluate_model(model_merge, X_test, y_test)
    results["merge"] = float(mse_merge)

    # Selection sort evaluation
    X_sel = selectionSort2D(X)
    model_sel, X_test, y_test = train_linear_regression(X_sel, y)
    print("\n====== SELECTION SORT RESULTS ======")
    mse_sel = evaluate_model(model_sel, X_test, y_test)
    results["selection"] = float(mse_sel)

    # Pick the best preprocessing method
    best_method, _ = min(results.items(), key=lambda item: item[1])
    return best_method

if __name__ == "__main__":
    main()