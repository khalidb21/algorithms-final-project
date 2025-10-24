# Description

**Current Phase: Train & Evaluate Linear Model**
- Reads PulseBat data
- Trains Model
- Evaluates linear regression model based on standard metrics
- Additional sorting methods to compare data prepocessing effects on model
- SOH Classification based user chosen threshold

# How to setup and run code

**Install necessary python libraries:** 

`pip install pandas numpy scikit-learn`

**Run main file:** 

`python main.py`

**Output Description:**

The program prints model training and evaluation results for predicting battery State of Health (SOH). 

It displays:
- Model Evaluation Metrics — R², MSE, RMSE, and MAE for three models (Unsorted, MergeSorted, and SelectionSorted data).
- Battery Classification Results — based on a user-defined SOH threshold, showing overall accuracy, classification statistics for healthy/unhealthy batteries, and a simplified confusion matrix.
