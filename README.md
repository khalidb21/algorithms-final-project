# Description

**Algorithms Final Project — PulseBat SOH Prediction** 

This repo predicts the *battery State of Health (SOH)* from PulseBat voltage features `U1...U21` using a simple linear regression model and row-wise sorting experiments with *Merge Sort* and *Selection Sort*. We will also design a simple chatbot model that will transmit the batter's health status (SOH) and answer some simple bettery related questions by linking our chatbot to a Chatbot API.

If SOH < 0.6 → Unhealthy Battery   

If SOH ≥ 0.6 → Healthy Battery

# How to setup and run code

**in Git Bash**  
*git clone https://github.com/khalidb21/algorithms-final-project.git*  

*cd algorithms-final-project*

`python main.py`

## Install dependencies

### Step 1

create virtual env
`python -m venv .venv`

### Step 2

activate virtual env (windows)
`.venv\Scripts\activate`

activate virtual env (linux)
`source .venv/bin/activate`

install python dependencies
`pip install -r requirements.txt`

### Step 3

Make sure to get Free Gemini API Key and create a *.env* file in the root that follows the *.env.example*   

Go to <ins>Google AI Studio</ins> and signin to your Google account, then click *'Get API Key'* and then *'Create API Key'* for you prokect.  
https://aistudio.google.com/api-keys

After downloading dependencies, program is ready to be executed.

## Run program

### Step 1

Run the training model 

**from root folder:**

```
cd model
python train_model.py
```

### Step 2

**from root folder:**

```
cd app
streamlit run app.py
```

## 📄 What You’ll See After Running the Program

When you run `main.py`, the program prints each stage of the process in the console. The output usually follows this order:

---
### **1. Dataset Preprocessing**
You will see messages confirming that the dataset was loaded correctly, along with:
- The extracted SOH values for cells U1–U21  
- Sorted versions of the values (depending on which sorting method is running)  
- Any notes or updates printed during preprocessing  
---
### **2. Model Training**
The console will show messages such as:
- When training begins  
- When the model finishes training  
- Any details or logs printed by scikit-learn during the process  
---
### **3. Evaluation Metrics**
After training, the program prints several metrics that show how the model performed:
- *R² Score*  
- *Mean Squared Error (MSE)*  
- *Mean Absolute Error (MAE)* 
- *Root Mean Squared Error (RMSE)*  
These numbers give an idea of how close the predictions are to the actual SOH values.
---
### **4. Final SOH Prediction**
Example output line: *Predicted SOH: 0.71*  
This represents the model’s estimate of the battery pack’s health.
---
### **5. Battery Health Classification**
Based on the required threshold rule (`0.6`): *Battery Status: Healthy* or *Battery Status Problem Detected*  

---  

## 🗂️ Repository Structure
algorithms-final-project/  
│── main.py            &emsp;    *# Main script: preprocessing → sorting → regression → evaluation*  
│── sorting.py         &emsp;    *# Sorting functions (may include selection & merge sort)*  
│── regression.py      &emsp;     *# Helper functions for regression*  
│── PulseBat Dataset.csv &emsp;   *# PulseBat dataset used by the program*  
│── README.md            &emsp;  *# Documentation file*  
│── .gitignore   

