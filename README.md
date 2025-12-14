# Credit Card Default Prediction – KNN Classifier

## Overview
This project implements a **basic machine learning workflow** to predict credit card default using a **K-Nearest Neighbors (KNN) classifier**.

The focus of the project is to demonstrate **core ML concepts**, including data preprocessing, normalization, model training, and evaluation, using a supervised learning approach.

---

## Dataset
- `ccdefault.csv`
- Contains customer credit and payment-related features
- Target variable: `DEFAULT` (whether a customer defaulted)

---

## Technologies Used
- Python
- pandas
- scikit-learn

---

## Project Workflow
1. Loaded and explored the dataset using pandas  
2. Separated features and target variable  
3. Removed non-informative columns (e.g., ID)  
4. Split data into training and testing sets  
5. Normalized features to support distance-based learning  
6. Trained a KNN classifier  
7. Evaluated model performance using accuracy  

---

## Model Details
- Algorithm: K-Nearest Neighbors (KNN)
- Number of neighbors: 5
- Distance-based classification
- Evaluation metric: Accuracy score

---

## Why Normalization Matters
KNN relies on distance calculations between data points.  
Feature normalization ensures that all variables contribute equally to distance computation and prevents features with larger scales from dominating the model.

---

## Strengths and Limitations
**Strengths**
- Simple and intuitive algorithm
- No explicit training phase
- Useful for learning distance-based classification

**Limitations**
- Computationally expensive for large datasets
- Sensitive to feature scaling
- Performance depends on choice of `k`

---

## Purpose
This project is intended to demonstrate:
- Understanding of supervised learning
- Practical data preprocessing steps
- Model evaluation basics
- When and why KNN is an appropriate choice

---

## Files
- `knn_classifier.py` – Python script implementing the ML workflow
- `ccdefault.csv` – Dataset used for training and evaluation
