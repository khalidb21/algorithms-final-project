# Battery Pack SOH Prediction

A machine learning application that predicts battery State of Health (SOH) using linear regression. This project combines predictive modeling with a conversational chatbot interface to provide battery health insights.

## Overview

This project analyzes PulseBat voltage features (U1–U21) to predict the overall State of Health of a battery pack. The application classifies battery health as either "Healthy" or "Unhealthy" based on the predicted SOH value:

- **SOH ≥ Threshold**: Healthy Battery
- **SOH < Threshold**: Unhealthy Battery

The project also implements sorting algorithms (Merge Sort and Selection Sort) for data processing and integrates a chatbot powered by the Gemini API to answer battery-related questions and communicate health status.

## Dataset Description

The linear regression model is based on the PulseBat dataset, which contains pulse-response measurements and SOH values for individual lithium-ion battery cells. A full battery pack includes 21 cells, labelled U1 through U21, and an associated SOH value for the battery. The SOH value represents how much usable capacity the cell still has compared to when it was new (ranging from 0-1 where 1 is brand new and 0 is completely degraded). 

## Features

- **Predictive Modeling**: Linear regression model trained on PulseBat voltage features
- **Algorithm Implementation**: Merge Sort and Selection Sort for comparing preprocessing sorts and their effects on the accuracy of the linear regression
- **Chatbot Integration**: AI-powered chatbot using Google Gemini API for user interaction
- **Web Interface**: Interactive Streamlit app for easy visualization and prediction
- **Performance Metrics**: Comprehensive evaluation with R² Score, MSE, MAE, and RMSE

## Project Structure

```
algorithms-final-project/
├── main.py                      # Main execution script
├── sorting.py                   # Sorting algorithms (Merge Sort, Selection Sort)
├── regression.py                # Linear regression helper functions
├── PulseBat Dataset.csv         # Training dataset with voltage features
├── model/
│   ├── config.py                # Constants used across the project (paths, filenames, settings)
│   ├── model_utils.py           # Shared utilities for loading, saving, and preparing models
│   ├── evaluate_performance.py  # Functions for evaluating regression performance (MSE, MAE, R²)
│   ├── preprocessing.py         # Data preprocessing steps such as mergeSort2D and feature preparation
│   └── train_model.py           # Model training script that generates and exports the .pk1 model
├── app/
│   ├── app_helper.py            # Helper functions for Streamlit: input parsing, prediction logic, model loading
│   └── app.py                   # Streamlit web application (UI + chatbot + prediction interface)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template (e.g., GEMINI_API_KEY)
└── README.md                    # This file
```

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Google Gemini API Key (free tier available)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/khalidb21/algorithms-final-project.git
cd algorithms-final-project
```

### 2. Create Virtual Environment

**For Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**For Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/api-keys)
2. Sign in with your Google account
3. Click **"Get API Key"** and then **"Create API Key"**
4. Create a `.env` file in the root directory (copy from `.env.example`):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Train Model

```bash
cd model
python train_model.py
cd ..
```

### Run Web Application

```bash
cd app
streamlit run app.py
```

This launches an interactive web interface at `http://localhost:8501` where you can:
- Enter your battery pack voltage samples
- View SOH predictions and get insights powered by AI
- Chat with the AI chatbot for battery-related questions

## Model Performance

The linear regression model is evaluated using standard metrics:

- **R² Score**: Coefficient of determination (0-1 scale)
- **Mean Squared Error (MSE)**: Average squared prediction errors
- **Mean Absolute Error (MAE)**: Average absolute prediction errors
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors

```
MODEL EVALUATION RESULTS
R² Score: 0.54909
Mean Squared Error (MSE): 0.00260
Root Mean Squared Error (RMSE): 0.05098
Mean Absolute Error (MAE): 0.04064
```

## Algorithms

### Sorting Algorithms

The project implements two sorting algorithms for data preprocessing:

- **Merge Sort**: O(n log n) time complexity, stable sorting
- **Selection Sort**: O(n²) time complexity, in-place sorting

Both are applied to rows of the dataset for comparative performance analysis to the unsorted data used in the web application.

### Linear Regression

A scikit-learn linear regression model maps the 21-dimensional voltage feature space to a single SOH prediction.

## API Integration

The project integrates with the **Google Gemini API** to power a conversational chatbot that:
- Answers battery-related questions
- Explains SOH predictions
- Provides maintenance recommendations
- Communicates health status

## Troubleshooting

**Issue**: Import errors or missing modules
- **Solution**: Ensure virtual environment is activated and run `pip install -r requirements.txt`

**Issue**: Gemini API Key not recognized
- **Solution**: Verify `.env` file exists in the root directory with correct API key format

**Issue**: Dataset file not found
- **Solution**: Ensure `PulseBat Dataset.csv` is in the project root directory

**Issue**: Streamlit app won't start
- **Solution**: Make sure you're in the `app/` directory when running `streamlit run app.py`

## Future Enhancements

- Advanced ML models
- Batch prediction functionality
- Export prediction reports to PDF
- Multi-language chatbot support

## Dependencies

See `requirements.txt` for complete list. Key packages:
- scikit-learn: Machine learning
- numpy: Math and data handling
- pandas: Data manipulation
- streamlit: Web interface
- python-dotenv: Environment variable management
- google-generativeai: Gemini API integration


