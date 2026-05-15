# 🚀 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-HGNN-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-brightgreen)

An end-to-end **Machine Learning and Deep Learning based Fraud Detection System** designed to identify fraudulent credit card transactions in real time. The project combines advanced preprocessing, imbalance handling techniques, multiple predictive models, and a deployed Streamlit application for interactive predictions.

---

# 📚 Course Details

| Field | Details |
|---|---|
| **Project Title** | Credit Card Fraud Detection System |
| **Course** | Machine Learning Laboratory |
| **Domain** | Financial Fraud Detection |
| **Technologies Used** | Python, Scikit-learn, XGBoost, Deep Learning, Streamlit |

---

# 👥 Team Members

- M.R Lekshmipriya  
- Ashna Jabin Nk 
- Hemalakshmi R

---

# 📌 Problem Statement

Credit card fraud has become a major challenge in the digital payment ecosystem. Financial institutions process millions of transactions daily, making it difficult to detect fraudulent activities manually.

The major challenges include:

- Extremely imbalanced datasets where fraudulent transactions are very rare
- Evolving fraud patterns and attack strategies
- Need for fast and accurate real-time detection
- Reducing false positives while maintaining high fraud recall

This project aims to develop a scalable and intelligent fraud detection system capable of accurately identifying fraudulent transactions using Machine Learning and Deep Learning approaches.

---

# 🎯 Project Objectives

- Build accurate fraud detection models using ML and DL techniques
- Handle severe class imbalance effectively
- Compare multiple models for performance evaluation
- Minimize false negatives and maximize fraud detection recall
- Deploy a real-time fraud prediction web application
- Create a modular and production-ready pipeline

---

# 📊 Dataset Description

## 🔹 Dataset Source

- **Dataset:** IEEE-CIS Fraud Detection Dataset  
- **Source:** Kaggle  

👉 Dataset Link:  
https://www.kaggle.com/competitions/ieee-fraud-detection/data

---

## 🔹 Dataset Overview

The dataset contains online transaction records collected from real-world e-commerce payment systems.

### Dataset Characteristics

| Attribute | Description |
|---|---|
| **Transactions** | Online payment transactions |
| **Target Variable** | `isFraud` |
| **Class Labels** | `0 = Legitimate`, `1 = Fraudulent` |
| **Feature Types** | Numerical, Categorical, Transactional |
| **Challenge** | Highly imbalanced fraud distribution |

---

## 🔹 Important Features

- `TransactionID`
- `TransactionAmt`
- `ProductCD`
- `card1 – card6`
- `addr1`, `addr2`
- `P_emaildomain`
- `R_emaildomain`
- `DeviceType`
- `DeviceInfo`
- Transaction timing features
- Identity-related attributes

---

## 🔹 Class Distribution

The dataset is highly imbalanced:

- Legitimate transactions form the majority
- Fraudulent transactions represent only a very small percentage

This imbalance makes fraud detection a challenging classification problem.

---

# 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| **Programming Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Deep Learning** | HGNN Architecture |
| **Imbalance Handling** | SMOTE, ADASYN |
| **Deployment** | Streamlit |
| **Development Environment** | Jupyter Notebook |

---

# 📁 Project Structure

```bash
Credit-Card-Fraud-Detection-System/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app.py
├── predict_models_options.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── utils.py
│   └── hgnn_utils.py
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── 02_train_decision_tree.ipynb
│   ├── 03_train_xgboost.ipynb
│   ├── 04_train_hgnn.ipynb
│   └── 05_evaluation.ipynb
│
├── data/
├── models/
├── outputs/
└── images/
```

---

# 🔄 Project Methodology

## 1️⃣ Data Collection

- Collected fraud transaction dataset from Kaggle
- Loaded and validated transactional records

---

## 2️⃣ Data Preprocessing

- Missing value handling
- Data cleaning and validation
- Feature scaling
- Encoding categorical variables
- Feature engineering
- Dimensionality reduction techniques

---

## 3️⃣ Handling Imbalanced Data

Advanced imbalance handling techniques were applied:

- **SMOTE (Synthetic Minority Oversampling Technique)**
- **ADASYN (Adaptive Synthetic Sampling)**

These methods improved fraud detection capability by balancing class distribution.

---

# 🧠 Model Development

## 🔹 Decision Tree

- Baseline interpretable model
- Used balanced class weighting
- Combined with SMOTE

---

## 🔹 XGBoost

- Gradient boosting-based ensemble model
- Optimized for imbalanced classification
- High precision and recall performance

---

## 🔹 HGNN-ATT-TD ⭐

Heterogeneous Graph Neural Network with:

- Attention mechanisms
- Temporal decay modeling
- Focal loss optimization
- Graph-based fraud relationship learning

---

# 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

# 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | High | Moderate | Moderate | Moderate |
| XGBoost | Very High | High | High | High |
| HGNN-ATT-TD | Best Performance | High | Very High | Very High |

---

## 🔹 Key Findings

- XGBoost significantly outperformed baseline models
- HGNN achieved superior fraud detection capability
- High recall reduced undetected fraud cases
- Resampling techniques improved minority class learning
- Deep learning models captured complex fraud relationships effectively

---

# 🌐 Streamlit Web Application

An interactive Streamlit web application was developed for real-time fraud prediction.

## ✨ Features

- User-friendly transaction input interface
- Instant fraud prediction
- Real-time inference
- Clean and responsive UI
- Easy deployment and accessibility

---

# 🔗 Live Deployment

👉 Live Streamlit App:  
https://credit-card-fraud-detection-system-hyyu5xm7adgwjhhweyg8gt.streamlit.app/

---

# 📸 Application Screenshots

## 🔹 Main Interface

![Application UI](images/UI.jpeg)

---

# 💻 Installation & Setup Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/lekshmipriya04/Credit-Card-Fraud-Detection-System.git

cd Credit-Card-Fraud-Detection-System
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Download Dataset

Download the dataset from Kaggle and place it inside the `data/` folder.

Dataset Link:  
https://www.kaggle.com/competitions/ieee-fraud-detection/data

---

## 5️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 📌 Future Enhancements

- Real-time streaming fraud detection
- Explainable AI integration
- Advanced ensemble learning
- Cloud deployment support
- Continuous fraud pattern learning
- API integration for banking systems

---

# ✅ Conclusion

This project demonstrates a complete fraud detection pipeline integrating Machine Learning and Deep Learning techniques for highly imbalanced financial datasets. The system successfully achieves high fraud detection performance while maintaining scalability and usability through a deployed Streamlit application.

---

# 📜 License

This project is developed for educational and academic purposes.

---

# ⭐ Acknowledgement

- Kaggle IEEE-CIS Fraud Detection Competition
- Open-source ML and DL communities
- Streamlit deployment platform
