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

## 🗂️ Repository Structure
algorithms-final-project/  
│── main.py            &emsp;    *# Main script: preprocessing → sorting → regression → evaluation*  
│── sorting.py         &emsp;    *# Sorting functions (may include selection & merge sort)*  
│── regression.py      &emsp;     *# Helper functions for regression*  
│── PulseBat Dataset.csv &emsp;   *# PulseBat dataset used by the program*  
│── README.md            &emsp;  *# Documentation file*  
│── .gitignore   

