Breast Cancer Classification Using Logistic Regression

This project uses Machine Learning to classify breast cancer tumors as malignant (cancerous) or benign (non-cancerous).  
It is based on the well-known Breast Cancer Wisconsin Dataset from Scikit-learn and demonstrates a complete classification workflow using Logistic Regression.

 Project Overview
Goal: Build a model that predicts whether a tumor is malignant or benign.  
Algorithm: Logistic Regression  
Dataset: Breast Cancer Wisconsin Diagnostic Dataset  
Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib  
Environment: Google Colab  

This project shows understanding of data preprocessing, scaling, model training, evaluation, and visualization.

Steps Performed

1. Import Libraries
Used standard ML libraries:
 numpy  
 pandas  
 scikit-learn (datasets, preprocessing, model, metrics)  
 matplotlib  

2. Load Dataset
Loaded the dataset directly from Scikit-learn.  
Features include medical measurements such as:
radius  
 texture  
 smoothness  
 symmetry  
 fractal dimension  

Target labels:
0 → Malignant
1 → Benign

3. Train–Test Split
Divided data into:
80% training data
20% testing data

This helps evaluate performance on unseen data.

4. Data Standardization
Used `StandardScaler()` to normalize all feature values.  
This improves Logistic Regression performance.

5. Model Training
Trained a Logistic Regression model with:
```python
model = LogisticRegression(max_iter=10000)
