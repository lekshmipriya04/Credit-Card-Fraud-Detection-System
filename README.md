# 🚀 Credit Card Fraud Detection System

A production-ready **end-to-end machine learning and deep learning system** for detecting fraudulent credit card transactions in real time. This project integrates advanced imbalance handling, multiple model architectures (ML + DL), modular pipelines, and a deployed Streamlit web application.

🔗 **Live App:**  
https://credit-card-fraud-detection-system-hyyu5xm7adgwjhhweyg8gt.streamlit.app/

---

## 📌 Problem Statement

Credit card fraud detection is challenging due to:

- Extremely **imbalanced datasets** (fraud < 1%)
- Continuously evolving fraud patterns
- Need for **real-time prediction systems**

This project aims to build a scalable and accurate system that detects fraudulent transactions while minimizing false positives.

---

## 🎯 Objectives

- Develop robust ML and DL models for fraud detection  
- Handle severe class imbalance using advanced techniques  
- Compare multiple models for performance optimization  
- Build a real-time prediction system  
- Deploy an interactive web interface  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Models:** Decision Tree, XGBoost, HGNN (Deep Learning)  
- **Imbalance Handling:** SMOTE, ADASYN  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Environment:** Jupyter Notebook  

---

## 📊 Dataset

- **Source:** Kaggle Credit Card Fraud Dataset  
- **Description:**
  - European cardholder transactions  
  - PCA-transformed features (`V1–V28`)  
  - Includes `Time`, `Amount`, and `Class` (0 = Legit, 1 = Fraud)  

- **Challenge:** Highly imbalanced dataset  

---

## 📁 Project Structure

```
fraud-detection/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── app.py                         # Streamlit web interface
├── predict_models_options.py      # CLI batch prediction script
│
├── src/
│   ├── config.py                  # Paths, hyperparameters, constants
│   ├── utils.py                   # Logging, helpers
│   ├── data_loader.py             # Load dataset
│   ├── feature_engineering.py     # PCA, feature selection
│   ├── preprocessing.py           # Cleaning & splitting
│   ├── models.py                  # ML & DL models
│   ├── training.py                # Training pipeline
│   ├── evaluation.py              # Metrics
│   ├── hgnn_utils.py              # Graph utilities
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── 02_train_decision_tree.ipynb
│   ├── 03_train_xgboost.ipynb
│   ├── 04_train_hgnn.ipynb
│   ├── 05_evaluation.ipynb
│
├── data/
├── models/
├── outputs/

```
