import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load model from pk1 file
def load_model(model_path):

    # Load pretrained linear regression model
    try:
        model = joblib.load(model_path)
        return model
    
    # Error handling for loading model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Classify battery based on threshold
def classify_battery(soh, threshold):

    # Returns a tuple
    if soh >= threshold:
        return "Healthy", "green", "✅"
    else:
        return "Unhealthy", "red", "❌"

# Predict SOH based on entered voltage samples from user
def predict_soh(model, voltage_samples):
    
    # Predict SOH from voltage samples
    # Args:
    #     model: Trained Linear Regression model
    #     voltage_samples: Dictionary with voltage values {U1: value, U2: value, ...}
    # Returns:
    #     Predicted SOH value

    try:
        # Create a df for model input
        input_columns = [f'U{i}' for i in range(1, 22)]
        input_voltages = pd.DataFrame([[voltage_samples.get(f'U{i}') for i in range(1, 22)]], columns=input_columns)
        soh = model.predict(input_voltages)[0]      # Predict SOH using the model
        return np.clip(soh, 0.0, 1.0)               # Keep SOH range between 0-1
    
    # If model fails to create a prediction
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

def parse_voltage_input(user_input):

    # Parse user input for voltage sample values
    # Format: "3.85, 3.87 ..." as comma/space separated values
    # Returns:
    #     Dictionary {U1: value, U2: value, ...}

    try:
        # Store voltage samples in a dictionary
        voltage_samples = {}

        # Parse as comma or space separated values
        samples = [float(x.strip()) for x in user_input.replace(',', ' ').split()]
        if len(samples) != 21:
            st.error(f"❌ Expected 21 values, got {len(samples)}")
            return None
        
        # For every number in samples list, create an index and store inside dictionary
        for i, val in enumerate(samples, 1):
            voltage_samples[f'U{i}'] = val
        
        # Validate we have all 21 cells
        if len(voltage_samples) != 21:
            st.error(f"❌ Expected 21 values, got {len(voltage_samples)}")
            return None
        
        return voltage_samples
    
    # If user inputs invalid format for voltage measurements
    except ValueError as e:
        st.error(f"❌ Invalid input format: {e}")
        return None