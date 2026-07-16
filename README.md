# Behavioral-feature-engineering-credit-risk
Comparative machine learning framework evaluating behavior-oriented feature engineering for credit default prediction using Logistic Regression, Random Forest, and XGBoost.

## Overview

Traditional credit scoring models primarily rely on financial indicators such as loan amount, annual income, debt-to-income ratio (DTI), and credit history. However, these variables may not fully capture borrower behavior, particularly in lending environments involving borrowers with non-traditional or irregular income patterns.

This project investigates whether **behavior-oriented feature engineering** can improve credit default prediction compared with traditional financial variables alone.

A two-tier experimental framework was developed using the Lending Club loan dataset:

- **Tier 1:** Traditional financial and credit-related variables
- **Tier 2:** Behavior-oriented representations engineered from selected financial variables

Three machine learning models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The objective was to determine whether behavioral feature engineering improves predictive performance while maintaining model interpretability.

---

# Research Question

> Can behavior-oriented feature engineering improve credit default prediction compared with traditional financial variables?

---

# Dataset

**Source:** Lending Club Loan Dataset

After preprocessing:

- 87,891 loan records
- Binary classification problem
- Target variable:
  - 0 → Non-default
  - 1 → Default

---

# Methodology

## Tier 1 — Traditional Financial Features

Baseline models were trained using conventional borrower information, including:

- Loan Amount
- Annual Income
- Interest Rate
- Debt-to-Income Ratio (DTI)
- Revolving Credit Utilization
- Credit History
- FICO Score
- Employment Information
- Loan Grade
- Loan Purpose

Models trained:

- Logistic Regression
- Random Forest
- XGBoost

---

## Tier 2 — Behavior-Oriented Feature Engineering

Selected financial variables were transformed into behavior-oriented representations to capture better borrower affordability, repayment burden, credit pressure, and financial stress.

Behavioral features included:

- Loan Burden Ratio
- Payment Burden Ratio
- Recent Credit Pressure
- Debt Exposure Score
- Credit Experience Category

The same preprocessing and modeling pipeline used in Tier 1 was repeated to ensure a fair comparison.

---

# Machine Learning Pipeline

```
Raw Lending Club Dataset
          │
          ▼
Data Cleaning & Preprocessing
          │
          ▼
Feature Engineering
          │
          ▼
Train-Test Split (80:20)
          │
          ▼
SMOTE (Training Set Only)
          │
          ▼
Model Training
 ├── Logistic Regression
 ├── Random Forest
 └── XGBoost
          │
          ▼
Model Evaluation
          │
          ▼
Feature Importance & SHAP
```

---

# Models Used

- Logistic Regression
- Random Forest
- XGBoost

---

# Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

Model interpretability was performed using:

- Feature Importance
- SHAP (SHapley Additive exPlanations)

---

# Results Summary

Behavior-oriented feature engineering produced measurable improvements across several models.

Key findings include:

- Logistic Regression achieved improved ROC-AUC and precision after introducing behavioral representations.
- Random Forest demonstrated the largest overall improvement across Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- XGBoost showed modest improvements in Accuracy, Precision, and ROC-AUC.

Overall, behavior-oriented feature engineering enhanced predictive performance without changing the underlying machine learning algorithms.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Statsmodels

---
# Limitations

- Behavioral variables were engineered from existing financial information rather than directly observed borrower behavior.
- The study used publicly available Lending Club data, which may not generalize to proprietary banking portfolios.
- Temporal borrower behavior and transaction-level information were unavailable.
- External validation on an independent dataset was not performed.


