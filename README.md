# 📊 CreditIQ — ML-Powered Loan Default Prediction Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## Overview

CreditIQ is an end-to-end machine learning project I built to predict whether a loan applicant is likely to default within two years. I trained the model on 150,000 real loan applications from Kaggle's Give Me Some Credit dataset and deployed it as an interactive Streamlit web app that scores new applicants in real time and explains every prediction using SHAP.

The goal was not just to build a model that works, but to build one the way it would need to work in production — with correct pipeline ordering, no data leakage, meaningful evaluation metrics, and decisions that can be justified to a non-technical stakeholder.

---

## The Problem

Consumer lending is an inherently imbalanced problem. In this dataset, only 6.7% of applicants defaulted — a 14:1 imbalance. A naive model that predicts "no default" for everyone gets 93% accuracy while catching zero actual defaults, which is completely useless for a bank. The real challenge was building a model that meaningfully identifies the minority class while remaining interpretable enough to explain individual decisions.

---

## Data & Preprocessing

The dataset contains 10 features covering credit utilization, payment history, debt ratios, income, and demographics. I found that roughly 20% of MonthlyIncome values and 2.5% of NumberOfDependents values were missing, and several columns had extreme outliers — DebtRatio had values in the hundreds of thousands, which are clearly data entry errors rather than real values.

I imputed missing income values with the training set median and dependents with the training set mode. For outliers, I used domain knowledge rather than blind percentile capping — DebtRatio was capped at 10 (a ratio above this is not a real financial situation) and RevolvingUtilization at 2.0. MonthlyIncome used the 99th percentile of the training set.

A critical design decision was the pipeline order. I split the data first, then fit all preprocessing statistics — imputation values, outlier caps, and the scaler — exclusively on the training set, saving them and applying the same values to the test set. This prevents data leakage through imputation and scaling, which is a subtle but serious bug that makes a model appear to perform better than it actually does.

---

## Feature Engineering

I created two new features beyond the raw columns. **TotalPastDue** combines all three delinquency severity columns — 30–59 days, 60–89 days, and 90+ days late — into a single count capturing overall payment behaviour. The individual columns were correlated with each other and collectively represented the same underlying signal.

I also added a **HasDependents** binary flag indicating whether the applicant has any financial dependents. I initially built an IncomePerDependent ratio but removed it — dividing by (dependents + 1) to avoid zero division distorts the feature for the majority of applicants who have no dependents, creating a systematic error. A clean binary flag captures the meaningful signal without mathematical artifacts.

---

## Handling Class Imbalance

I applied SMOTE — Synthetic Minority Oversampling Technique — to the training data. Rather than duplicating existing default samples, SMOTE creates synthetic ones by interpolating between real minority class examples in feature space. I used a sampling strategy of 0.3, bringing the minority class to 30% of the majority rather than full 1:1 balance. Fully balancing the classes tends to hurt precision on credit data because too many synthetic defaults push the decision boundary in unrealistic directions.

I applied SMOTE inside each cross-validation fold rather than before splitting, ensuring synthetic samples never leak into validation sets.

---

## Modeling

I trained three models in order of complexity — Logistic Regression as a linear baseline, Random Forest, and XGBoost. I included a DummyClassifier to confirm all three are genuinely learning from the data rather than exploiting class distribution.

For XGBoost I ran a RandomizedSearchCV over six hyperparameters — n_estimators, max_depth, learning_rate, subsample, colsample_bytree, and min_child_weight — with 5-fold stratified cross-validation over 20 random combinations. I report mean ± standard deviation AUC across folds rather than a single split score, which gives a more honest estimate of generalisation.

I used ROC-AUC as the primary metric rather than accuracy, since accuracy is meaningless for imbalanced classification. I also tracked Average Precision — the area under the Precision-Recall curve — which is more informative than ROC-AUC when the positive class is rare, because it does not reward correct identification of the majority class.

---

## Threshold Selection

The default 0.5 classification threshold is almost never optimal for imbalanced problems. I found the optimal threshold by computing F1 across every threshold on the Precision-Recall curve and selecting the point that maximises it. This threshold is saved and used by the Streamlit app for all predictions rather than defaulting to 0.5.

---

## Explainability

Feature importance tells you which features matter globally across the entire model. It cannot tell you why a specific applicant received a high risk score. I used SHAP — SHapley Additive exPlanations — to generate per-prediction explanations, assigning each feature a contribution value that shows how much it pushed the probability toward or away from default for that specific individual.

For a finance application this is not optional — it is what makes the model auditable. A bank needs to be able to tell an applicant which factors drove a rejection decision, and risk teams need to verify the model is not discriminating through proxy variables.

---

## Results

| Model | Test AUC |
|-------|----------|
| Dummy Baseline | ~0.500 |
| Logistic Regression | — |
| Random Forest | — |
| **XGBoost (tuned)** | **—** |

---

## Application

The Streamlit app has four pages. The home page shows the dataset and pipeline overview. The EDA dashboard has interactive charts for class distribution, feature distributions by default status, default rate by age group, and the full correlation heatmap. The model performance page shows ROC curves, confusion matrix, and the SHAP summary plot. The live predictor accepts applicant inputs and returns a default probability, a risk gauge, and a SHAP bar chart explaining exactly which features drove the prediction.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |

---
