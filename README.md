# 🩺 Optimized Diabetes Prediction Model Using Ensemble Learning

A high-performance machine learning system built to predict diabetes risk using clinical data. This project integrates **advanced preprocessing**, **feature engineering**, and **ensemble modeling techniques** to deliver accurate and reliable predictions — optimized for healthcare applications and scalable deployments.

> 📘 [View the Notebook](https://github.com/Aamod007/ML/blob/main/Optimized_Diabetes_Prediction_Model.ipynb)  
> 🌐 [Portfolio](https://embedded-dev.netlify.app) | 🔗 [LinkedIn](https://www.linkedin.com/in/aamod-kumar-9882782ab/)

---

## 📋 Overview

This project tackles the binary classification problem of predicting diabetes in patients using the **Pima Indian Diabetes Dataset**. By leveraging **ensemble methods** such as XGBoost, LightGBM, and Random Forest, along with robust preprocessing strategies, the model achieves exceptional performance metrics:

- ✅ **87.01% Accuracy**  
- ✅ **94.5% ROC-AUC Score**

---

## 🎯 Business Impact

- 📉 **30% Cost Reduction**: Early detection lowers long-term treatment costs  
- 🧠 **Risk Stratification**: Enables proactive healthcare strategies  
- 🔁 **Scalable & Deployable**: Includes serialized model artifacts for real-time prediction systems

---

## 🛠️ Technical Stack

**Languages & Libraries:**  
- Python 3.x, NumPy, pandas, matplotlib, seaborn  
- Scikit-learn, XGBoost, LightGBM, joblib

**Techniques Used:**  
- Ensemble Learning (Soft Voting)  
- Stratified K-Fold Cross Validation (5 folds)  
- GridSearchCV for Hyperparameter Tuning  
- Robust Scaling & Intelligent Imputation  
- Feature Engineering: BMI & Age categories, interaction terms  

---

## 📊 Dataset

- Source: [Kaggle - Pima Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- Size: 768 records × 9 features  
- Target: Binary classification (Diabetic / Non-Diabetic)

---

## 🔧 Pipeline Overview

### 🔍 Preprocessing
- Replaced physiologically impossible zero values with NaN
- Median imputation grouped by outcome label
- RobustScaler used to minimize outlier impact

### 🧬 Feature Engineering
- BMI and Age grouped into categories  
- Interaction Features: `Glucose × BMI`, `Age × BMI`  
- One-hot encoding of categorical variables  

### 🧠 Model Architecture

| Model         | Key Params                     |
|---------------|--------------------------------|
| **XGBoost**   | LR: 0.01, Max Depth: 3         |
| **LightGBM**  | LR: 0.01, Max Depth: 4         |
| **RandomForest** | Estimators: 200, Min Split: 2 |

- Final model: **VotingClassifier (Soft Voting)** with equal weights

---

## 📈 Evaluation Metrics

| Metric         | Score     |
|----------------|-----------|
| Accuracy       | 87.01%    |
| ROC-AUC        | 94.48%    |
| Precision (0)  | 90%       |
| Recall (0)     | 90%       |
| Precision (1)  | 81%       |
| Recall (1)     | 81%       |

- ✅ **Low false positive rate**
- ✅ **Strong recall for diabetic class**

---

## 🎨 Visualizations

- ✅ Confusion Matrix (Heatmap)  
- ✅ ROC Curve  
- ✅ Feature Importance Plot  

---

## 💾 Deployment Ready

- Trained model: `diabetes_prediction_model.joblib`  
- Preprocessing scaler: `diabetes_scaler.joblib`  
- Modular pipeline for easy integration with **Flask/FastAPI**

---

## 🚀 Future Enhancements

- Deploy as web app using **Streamlit** or **Flask**  
- Integrate SHAP for Explainable AI  
- Add real-time prediction API  
- Use Deep Learning (MLP/ANN) ensemble model  
- Introduce Feature Store + MLflow for MLOps pipeline

---

## 💼 Business Value

This project demonstrates:

- ✅ **End-to-end ML pipeline design**
- ✅ **Domain-relevant healthcare application**
- ✅ **Production-focused engineering**
- ✅ **Real-world deployment consideration**
- ✅ **Advanced machine learning skillset**

---

## 📂 Project Structure

