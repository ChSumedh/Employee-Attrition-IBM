# Employee-Attrition-IBM
# Employee Attrition Prediction

## Problem Statement
Employee attrition refers to employees voluntarily leaving an organization.
This project aims to predict attrition using HR data to help companies
identify employees at risk and reduce hiring and training costs.

## Dataset
IBM HR Analytics Employee Attrition Dataset

## Approach
- Exploratory Data Analysis (EDA)
- Handling class imbalance
- Feature scaling and encoding
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest with threshold tuning
- Evaluation using accuracy, recall, and confusion matrix

## Key Insight
Accuracy alone is misleading due to class imbalance.
Recall for Attrition=Yes is prioritized to reduce false negatives.

## Best Model
Random Forest with custom threshold  
Accuracy ≈ 65%  
Recall ≈ 83%

## Tools
- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib, Plotly
