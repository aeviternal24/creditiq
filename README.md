# 📊 CreditIQ — ML-Powered Loan Default Prediction Engine

> Predict whether a loan applicant will default — with explainable AI showing exactly why.

---

## 🧠 What This Project Does

CreditIQ is an end-to-end machine learning pipeline that:
- Trains on **150,000 real loan applications**
- Predicts the **probability of default** for new applicants
- Uses **SHAP values** to explain *why* the model made each decision
- Deploys as an interactive **Streamlit web app**

---

## 🗂️ Project Structure

```
creditiq/
├── data/                   ← Put your downloaded CSV here
├── models/                 ← Trained models saved here (auto-created)
├── notebooks/
│   ├── 01_eda.ipynb        ← Exploratory Data Analysis
│   └── 02_modeling.ipynb   ← ML Pipeline & Evaluation
├── src/
│   ├── preprocess.py       ← Data cleaning utilities
│   └── train.py            ← Model training pipeline
├── app.py                  ← Streamlit app (4 pages)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset — Give Me Some Credit (Kaggle)

1. Go to: https://www.kaggle.com/c/GiveMeSomeCredit/data
2. Download `cs-training.csv`
3. Place it in the `data/` folder

### Columns
| Column | Description |
|--------|-------------|
| `SeriousDlqin2yrs` | **TARGET** — 1 if defaulted within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Credit card usage ratio |
| `age` | Age of borrower |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30-59 days late |
| `DebtRatio` | Monthly debt / monthly income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60-89 days late |
| `NumberOfDependents` | Dependents in family |

---

## 🚀 Setup & Run

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/creditiq
cd creditiq
pip install -r requirements.txt
```

### 2. Run EDA Notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Train the Model
```bash
python src/train.py
```

### 4. Launch Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Model Results

| Model | ROC-AUC | Precision | Recall |
|-------|---------|-----------|--------|
| Logistic Regression | ~0.68 | — | — |
| Random Forest | ~0.73 | — | — |
| **XGBoost** | **~0.78** | — | — |

*(Fill in your actual results after training)*

---

## 🔍 Key Concepts Covered

- **Class Imbalance** — SMOTE oversampling (only ~6.7% of loans default)
- **Feature Engineering** — Creating meaningful ratios from raw data
- **Model Comparison** — Why XGBoost beats Logistic Regression here
- **SHAP Explainability** — Per-prediction explanation of risk drivers
- **Streamlit Deployment** — Interactive ML app without a backend

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `imbalanced-learn` `SHAP` `Streamlit` `Plotly` `Seaborn`

---

## 📝 Resume Bullets

- Built an end-to-end ML pipeline for credit default prediction on 150k+ real loan applications using XGBoost, achieving 0.78 ROC-AUC
- Engineered features and applied SMOTE oversampling to handle severe class imbalance (14:1 ratio)
- Implemented SHAP explainability to surface per-applicant risk drivers, making model decisions interpretable
- Deployed an interactive Streamlit application allowing real-time risk scoring with visual explanations
- Compared Logistic Regression, Random Forest, and XGBoost — documented trade-offs in structured model evaluation
