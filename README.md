# 📊 CreditIQ — ML-Powered Loan Default Prediction Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

> Predict whether a loan applicant will default — with explainable AI showing exactly why.

---

## 🧠 Overview

CreditIQ is an end-to-end machine learning pipeline trained on **150,000 real loan applications** from Kaggle's Give Me Some Credit dataset. It predicts the probability of serious delinquency within 2 years and uses **SHAP explainability** to surface the exact features driving each individual prediction.

The project was built with production-quality practices — correct pipeline ordering to prevent data leakage, SMOTE for class imbalance, optimal threshold selection, 5-fold cross-validation, and XGBoost hyperparameter tuning via RandomizedSearchCV.

---

## ✨ Features

- **Live Risk Predictor** — Enter applicant details and get an instant default probability with a risk gauge
- **SHAP Explanations** — Per-prediction breakdown of which features increased or decreased risk
- **EDA Dashboard** — Interactive charts exploring the dataset, class imbalance, and feature distributions
- **Model Performance Page** — ROC curves, confusion matrix, and cross-validated AUC scores for all 3 models
- **Leakage-Free Pipeline** — Split-first architecture ensuring test data never influences preprocessing

---

## 🗂️ Project Structure

```
creditiq/
├── data/                        ← Download cs-training.csv here
├── models/                      ← Auto-generated after training
├── notebooks/
│   ├── 01_eda.ipynb             ← Exploratory Data Analysis
│   └── 02_modeling.ipynb        ← Training, evaluation & SHAP
├── src/
│   ├── preprocess.py            ← Leakage-free preprocessing pipeline
│   └── train.py                 ← Full training pipeline with CV & tuning
├── app.py                       ← Streamlit app (4 pages)
└── requirements.txt
```

---

## 🤖 ML Pipeline

```
Raw Data (150k loan applications)
        ↓
Train/Test Split (stratified, 80/20)
        ↓
Imputation → Outlier Capping → Feature Engineering
(all fit on train only — zero leakage)
        ↓
SMOTE Oversampling (14:1 → 3:1 ratio)
        ↓
Model Training with 5-Fold Cross-Validation
Logistic Regression → Random Forest → XGBoost
        ↓
RandomizedSearchCV Hyperparameter Tuning (XGBoost)
        ↓
Optimal Threshold via Precision-Recall Curve
        ↓
SHAP Explainability (TreeExplainer)
        ↓
Streamlit App — Live Risk Scoring
```

---

## 📊 Model Results

| Model | CV AUC (mean ± std) | Test AUC |
|-------|---------------------|----------|
| Dummy Baseline | ~0.500 | ~0.500 |
| Logistic Regression | — | — |
| Random Forest | — | — |
| **XGBoost (tuned)** | **—** | **—** |

> Run `python src/train.py` to populate your actual results.

---

## 🔑 Key Engineering Decisions

**Data Leakage Prevention**
The pipeline splits data first, then fits all preprocessing statistics (imputation medians, outlier caps, scaler) exclusively on the training set. The same saved statistics are applied to the test set — the test set never influences any computation.

**Class Imbalance**
The dataset has a 14:1 imbalance (93.3% non-default). A naive model that always predicts no-default gets 93% accuracy but catches zero actual defaults. SMOTE generates synthetic minority samples to give the model meaningful signal on the rare positive class.

**Threshold Selection**
Default threshold of 0.5 is suboptimal for imbalanced data. The optimal threshold is found by maximising F1 on the Precision-Recall curve and saved for use in the app.

**SHAP Explainability**
Feature importance tells you what matters globally. SHAP tells you why a specific prediction was made — which features pushed this applicant's risk up and by how much. Critical for finance applications where decisions must be justifiable.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
| Tracking | Python logging, JSON results |

---

## 📦 Dataset

**Give Me Some Credit** — Kaggle (2011)
150,000 loan applicants · 10 features · Binary target (serious delinquency within 2 years)

| Column | Description |
|--------|-------------|
| `SeriousDlqin2yrs` | **Target** — defaulted within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Credit card usage ratio |
| `age` | Borrower age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days late |
| `DebtRatio` | Monthly debt / monthly income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days late |
| `NumberOfDependents` | Dependents in family |

---

