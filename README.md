# Description

**Algorithms Final Project — PulseBat SOH Prediction** 

This repo predicts the *battery State of Health (SOH)* from PulseBat voltage features `U1...U21` using a simple linear regression model and row-wise sorting experiments with *Merge Sort* and *Selection Sort*. We will also design a simple chatbot model that will transmit the batter's health status (SOH) and answer some simple bettery related questions by linking our chatbot to a Chatbot API.

If SOH < 0.6 → Unhealthy Battery   

If SOH ≥ 0.6 → Healthy Battery



**Current Phase: Train & Evaluate Linear Model**
- Reads PulseBat data
- Trains Model
- Evaluates linear regression model based on standard metrics

# How to setup and run code


**in Git Bash**  
*git clone https://github.com/khalidb21/algorithms-final-project.git*  

*cd algorithms-final-project*

# create and activate *venv* in Windows or Git Bash
*python -m venv .venv*  

*source .venv/Scripts/activate*     

# install deps
*pip install pandas numpy scikit-learn*

# Run Code in Git Bash
*python main.py*
