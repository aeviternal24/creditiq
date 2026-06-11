# 📊 CreditIQ — ML-Powered Loan Default Prediction Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-189AB4?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://creditiq-gihxgcftdkbh5nptxebdhb.streamlit.app/)

---

## Overview

CreditIQ is an end-to-end machine learning project I built to predict whether a loan applicant is likely to default within two years. I trained the model on 150,000 real loan applications from Kaggle's Give Me Some Credit dataset and deployed it as an interactive Streamlit web app that scores new applicants in real time and explains every prediction using SHAP.

The goal was not just to build a model that works, but to build one the way it would need to work in production — with correct pipeline ordering, no data leakage, meaningful evaluation metrics, and decisions that can be justified to a non-technical stakeholder.

---

## The Problem

Consumer lending is an inherently imbalanced problem. In this dataset, only 6.7% of applicants defaulted — a 14:1 imbalance. A naive model that predicts "no default" for everyone gets 93% accuracy while catching zero actual defaults, which is completely useless for a bank. The real challenge was building a model that meaningfully identifies the minority class while remaining interpretable enough to explain individual decisions.

---

## Data & Preprocessing

The dataset contains 10 features covering credit utilization, payment history, debt ratios, income, and demographics. MonthlyIncome was missing for roughly 20% of rows and NumberOfDependents for about 2.5%.

I imputed missing income values with the training set median and dependents with the training set mode. For outliers, I used domain knowledge rather than blind percentile capping — DebtRatio was capped at 10 and RevolvingUtilization at 2.0. MonthlyIncome used the 99th percentile of the training set.

A critical design decision was the pipeline order. I split the data first, then fit all preprocessing statistics — imputation values, outlier caps, and the scaler — exclusively on the training set. This prevents data leakage through imputation and scaling, which is a subtle but serious bug that makes a model appear to perform better than it actually does.

---

## Feature Engineering

I created two new features. **TotalPastDue** combines all three delinquency severity columns into a single count capturing overall payment behaviour. **HasDependents** is a binary flag indicating whether the applicant has any financial dependents. I initially built an IncomePerDependent ratio but removed it — dividing by (dependents + 1) to avoid zero division distorts the feature for the majority of applicants who have no dependents. A clean binary flag captures the meaningful signal without mathematical artifacts.

---

## Handling Class Imbalance

I applied SMOTE — Synthetic Minority Oversampling Technique — to the training data. The dataset had 111,979 non-default cases and only 8,021 defaults before resampling. After SMOTE with a sampling strategy of 0.3, the default class grew to 33,593 synthetic and real samples, bringing the ratio down from 14:1 to approximately 3:1. I used 0.3 rather than full 1:1 balance because fully balancing the classes tends to hurt precision on credit data.

---

## Modeling

I trained three models in order of complexity — Logistic Regression as a linear baseline, Random Forest, and XGBoost — and included a DummyClassifier to confirm all three are genuinely learning from the data. I used ROC-AUC as the primary metric rather than accuracy, since accuracy is meaningless for imbalanced classification.

---

## Results

| Model | ROC-AUC | Avg Precision | Recall (Default) | Precision (Default) |
|-------|---------|---------------|------------------|---------------------|
| Logistic Regression | 0.8289 | 0.3213 | 0.7501 | 0.1823 |
| XGBoost | 0.8619 | 0.3905 | 0.2469 | 0.5386 |
| **Random Forest** ✅ | **0.8651** | **0.3985** | **0.3686** | **0.4749** |

**Best model: Random Forest** with ROC-AUC of 0.8651.

The results reveal an interesting tradeoff. Logistic Regression achieves the highest recall at 0.75 — it catches more defaults — but at the cost of very low precision at 0.18, meaning most of its positive predictions are false alarms. XGBoost flips this: high precision at 0.54 but low recall at 0.25, missing most actual defaults. Random Forest strikes the best overall balance and wins on ROC-AUC, the primary metric.

---

## Threshold Selection

The default 0.5 classification threshold is almost never optimal for imbalanced problems. I found the optimal threshold by computing F1 across every threshold on the Precision-Recall curve and selecting the point that maximises it. This threshold is saved and used by the Streamlit app for all predictions.

---

## Explainability

I used SHAP to generate per-prediction explanations, assigning each feature a contribution value that shows how much it pushed the probability toward or away from default for that specific individual. For a finance application this is not optional — it is what makes the model auditable. A bank needs to be able to tell an applicant which factors drove a rejection decision.

---

## Application

The deployed Streamlit app has four pages. The home page shows the dataset and pipeline overview. The EDA dashboard has interactive charts for class distribution, feature distributions by default status, default rate by age group, and the full correlation heatmap. The model performance page shows the comparison table, SHAP summary plot, and optimal threshold. The live predictor accepts applicant inputs and returns a default probability, a risk gauge, and a SHAP bar chart explaining which features drove the prediction.

> **Note:** The deployed version runs on a 2,000-row sample of the full dataset for EDA visualisations. The trained model was built on the full 150,000 rows.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
