# Predictive_Maintenance_System
I Use synthetic data to analyzing Anomaly, Predict Failure_Type, use pre processing to get better accuracy visualize with various plot to get the distribution and all visualize with Plotly dash
# Project Overview
This project focuses on building a machine learning system to predict:
Machine failures (binary classification)
Failure types (multiclass classification)
Anomaly detection for early warning
The goal is to enable predictive maintenance, reducing downtime and improving operational efficiency.
# Dataset Description
Contains sensor and operational data from industrial machines
Features include:
Temperature, pressure, vibration
Gas composition (CO, CO₂, O₂)
Cooling and system health indicators
Target variables:
FailureType → Type of failure (multiclass)
Anomaly → Normal vs abnormal condition
# Data Preprocessing
Handled missing values and inconsistencies
Feature scaling using StandardScaler
Converted categorical failure labels into numerical format
Train-test split for model evaluation
# Exploratory Data Analysis (EDA)
Distribution plots for all features
Correlation analysis with target variables
Identification of key factors influencing failure
Visualization of trends over time
# Feature Engineering
Created meaningful features from raw sensor data
Normalization and scaling applied
Selected important features based on correlation and model impact
# Model Development
1. Classification Models
Used models like:
Random Forest
XGBoost / Gradient Boosting
Tasks:
Binary classification (Failure / No Failure)
Multiclass classification (Failure Type)
# Anomaly Detection
Implemented Isolation Forest
Generated anomaly scores:
Anomaly_score = -model.decision_function(X)
Used for early fault detection
# Model Evaluation
Metrics used:
Accuracy
Precision, Recall
F1-score
ROC-AUC
PR-AUC
Visualization:
ROC Curve
Precision-Recall Curve
# Results & Insights
Identified key features affecting failures
Anomaly detection successfully detects early deviations
Model performs well on both binary and multiclass tasks

⭐ If you like this project, give it a star!
