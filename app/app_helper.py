import joblib
import streamlit as st
import numpy as np

# Load model from pk1 file
def load_model(model_path):
    """Load the pre-trained Linear Regression model"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Classify battery based on threshold
def classify_battery(soh, threshold):

    # Classify battery health based on SOH threshold
    # Args:
    #     soh: Predicted SOH value (0.0 to 1.0)
    #     threshold: Classification threshold (default 0.6)
    # Returns:
    #     Tuple of (status, color, emoji)

    if soh >= threshold:
        return "Healthy", "green", "✅"
    else:
        return "Unhealthy", "red", "❌"

# Predict soh based on entered values from user
def predict_soh(model, cell_values):
    
    # Predict SOH from cell voltage values
    # Args:
    #     model: Trained Linear Regression model
    #     cell_values: Dictionary with cell values {U1: value, U2: value, ...}
    # Returns:
    #     Predicted SOH value or None if error

    try:
        # Create feature array for U1-U21
        features = np.array([[cell_values.get(f'U{i}', 0.5) for i in range(1, 22)]])
        soh = model.predict(features)[0]
        return np.clip(soh, 0.0, 1.0)       #keep between 0-1
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

def parse_cell_input(user_input):

    # Parse user input for cell values
    # Format: "U1:0.85 U2:0.87 ..." or just comma/space separated values
    # Returns:
    #     Dictionary {U1: value, U2: value, ...} or None if error

    try:
        cell_values = {}
        
        # Try parsing "U1:0.85 U2:0.87" format
        if ':' in user_input:
            pairs = user_input.strip().split()
            for pair in pairs:
                cell, val = pair.split(':')
                cell_values[cell.strip()] = float(val)
        else:
            # Parse as comma or space separated values
            values = [float(x.strip()) for x in user_input.replace(',', ' ').split()]
            if len(values) != 21:
                st.error(f"❌ Expected 21 values, got {len(values)}")
                return None
            for i, val in enumerate(values, 1):
                cell_values[f'U{i}'] = val
        
        # Validate we have all 21 cells
        if len(cell_values) != 21:
            st.error(f"❌ Expected 21 cells, got {len(cell_values)}")
            return None
        
        return cell_values
    except ValueError as e:
        st.error(f"❌ Invalid input format: {e}")
        return None