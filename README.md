# Diabetes Prediction Model 

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/your-username/diabetes-prediction-model)](https://github.com/your-username/diabetes-prediction-model/issues)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/diabetes-prediction-model)](https://github.com/your-username/diabetes-prediction-model/stargazers)

A robust machine learning model for predicting diabetes using the PIMA Indians Diabetes Dataset. This project leverages advanced preprocessing, feature engineering, and ensemble learning to deliver accurate and reliable predictions.

---

## 📖 Project Overview

The **Diabetes Prediction Model** predicts the likelihood of diabetes based on clinical features like glucose levels, BMI, and insulin. By combining Random Forest, XGBoost, and LightGBM classifiers in a voting ensemble, the model ensures high performance and generalizability.

### Key Highlights
- **Data Preprocessing**: Handles missing values and outliers with RobustScaler.
- **Feature Engineering**: Creates interaction features and categorical encodings.
- **Model Optimization**: Uses GridSearchCV with stratified k-fold cross-validation.
- **Ensemble Learning**: Combines multiple models for improved accuracy.
- **Evaluation**: Comprehensive metrics including accuracy, ROC-AUC, and confusion matrix.
- **Visualization**: Feature importance plot to identify key predictors.

---
    
## 🌟 Features

- **Dataset**: PIMA Indians Diabetes Dataset (768 records, 9 features).
- **Preprocessing**:
  - Replaces invalid zeros with outcome-grouped medians.
  - Scales features using RobustScaler.
- **Feature Engineering**:
  - BMI and Age categorization.
  - Interaction features (e.g., Glucose × BMI, Age × BMI).
  - One-hot encoding for categorical variables.
- **Models**:
  - Random Forest Classifier
  - XGBoost Classifier
  - LightGBM Classifier
  - Soft Voting Ensemble
- **Metrics**:
  - Accuracy (~80-85%)
  - ROC-AUC (~0.85-0.90)
  - Precision, Recall, F1-Score
- **Visualization**: Feature importance plot.
- **Persistence**: Model and scaler saved using `joblib`.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Git

### Dependencies
Install required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm joblib
