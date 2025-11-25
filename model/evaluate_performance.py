from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from config import *
from model_utils import train_linear_regression, load_pulsebat_data

# Sample evaluation results using unsorted data
def main():
    # Load data
    df = load_pulsebat_data(DATASET_FILEPATH)
    if df is None:
        return
    
    voltage_cols = [f'U{i}' for i in range(1, 22)]      # Create a list of columns U1-U21
    X = df[voltage_cols]                                # voltage values for each row
    y = df['SOH']                                       # state of health value for each row

    # Train model and evaluate
    model, X_test, y_test = train_linear_regression(X, y)
    evaluate_model(model,X_test,y_test)

# Linear regression model evaluation function
def evaluate_model(model, X_test, y_test):
    
    # get performance metrics
    y_pred = model.predict(X_test)
    variance = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Print evaluation results
    print("\nMODEL EVALUATION RESULTS")
    print(f"R² Score: {variance:.5f}")
    print(f"Mean Squared Error (MSE): {mse:.5f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.5f}")
    print(f"Mean Absolute Error (MAE): {mae:.5f}")

    # Return mean square root
    return mse

if __name__ == "__main__":
    main()