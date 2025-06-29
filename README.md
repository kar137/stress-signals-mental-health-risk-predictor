# 🧠 StressSignals: Predicting Mental Health Risk in Tech

[![Made with Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green?logo=scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)](https://shap.readthedocs.io/en/latest/)

> A machine learning-powered app to assess the likelihood of mental health treatment needs in the tech industry, built using data science and Streamlit.

---
## Live Demo

👉 **Try the App Here**:  
🔗 [https://stress-signals-mental-health-risk-predictor.streamlit.app/](https://stress-signals-mental-health-risk-predictor.streamlit.app/)

---
## Overview

**StressSignals** is a data-driven web application designed to raise awareness around mental health challenges faced by tech professionals. Trained on the [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey), this app predicts whether an individual is likely to seek mental health treatment based on various workplace and demographic factors.

---

## 👨‍💻 Tech Stack Used

<p align="center">
  <img src="https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Matplotlib-11557C?logo=plotly&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Seaborn-6B4C9A?logo=seaborn&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Random%20Forest-00C853?logo=scikit-learn&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-SHAP-FF7043?logo=shap&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Joblib-4B8BBE?logo=python&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=fff&style=for-the-badge" />
  <img src="https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=fff&style=for-the-badge" />
</p>

---
## Problem Statement

Mental health issues are prevalent in the tech industry, yet many professionals lack access to early indicators or support. This project aims to use data science to:

- Understand key mental health trends
- Predict treatment-seeking behavior
- Provide insights through explainable AI (SHAP)

---

## 📊 Exploratory Data Analysis (EDA)

Key insights from the data include:

- Age distribution of survey respondents
- Gender diversity and its correlation with mental health
- Influence of remote work, company size, and family history
- Correlation heatmaps and category distributions

EDA visualizations are available in the [`images/`](images/) folder and the [`notebooks/eda_analysis.ipynb`](notebooks/eda_analysis.ipynb) notebook.

---

## 🤖 Machine Learning Models

Three models were trained and evaluated:

| Model             | Accuracy | F1 Score |
|------------------|----------|----------|
| Logistic Regression | 71.4%      | 0.70    |
| XGBoost             | 65%      | 0.62     |
| 🏆 Random Forest     | **71.8%**  | **0.71** |

### 🏆 Best Model: Random Forest Classifier

We used SHAP to interpret feature importance and model predictions.

---

## Features Used

- Age, Gender, Family History
- Remote Work & Company Size
- Workplace support (benefits, leave policy, supervisor care)
- Perceptions of mental vs. physical health

---

## SHAP Interpretability

I used SHAP (SHapley Additive exPlanations) to understand how each feature impacts the prediction. See `images/shap_summary_rf.png` for the summary plot.

---

## 🌐 Streamlit App

The app allows users to:

- Enter key workplace and personal features
- Receive a prediction about mental health treatment likelihood
- Visualize how each input affects the outcome

---

## 💻 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/stress-signals-mental-health-risk-predictors.git
cd stress-signals-mental-health-risk-predictors

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
