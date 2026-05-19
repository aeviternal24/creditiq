"""
preprocess.py — Data Cleaning & Preprocessing Utilities
=========================================================
All reusable functions for cleaning and preparing the
Give Me Some Credit dataset for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


# ── Column names for easy reference ──────────────────────────────────────────
TARGET_COL = "SeriousDlqin2yrs"

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV and drop the unnamed index column Kaggle adds.
    """
    df = pd.read_csv(filepath, index_col=0)
    print(f"✅ Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── 2. Missing Value Analysis ─────────────────────────────────────────────────
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing missing count and percentage per column.
    Useful for EDA — run this first to understand data quality.
    """
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %":     missing_pct.round(2)
    })
    return report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)


# ── 3. Handle Missing Values ──────────────────────────────────────────────────
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - MonthlyIncome  → fill with MEDIAN (skewed distribution, median is safer)
    - NumberOfDependents → fill with MODE (discrete count variable)
    
    Why not drop rows?
      We'd lose ~20% of data. Imputation is better here.
    Why not mean for income?
      Income is right-skewed (a few very high earners pull the mean up).
    """
    df = df.copy()

    # Median imputation for MonthlyIncome
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    # Mode imputation for NumberOfDependents
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].mode()[0]
    )

    print(f"✅ Missing values filled. Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ── 4. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that capture relationships not in raw data.
    
    Good features often matter MORE than model choice.
    """
    df = df.copy()

    # Total times EVER past due (combines all 3 delinquency columns)
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )

    # Income per dependent (financial stress indicator)
    # Add 1 to dependents to avoid division by zero
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # Age groups (binned) — risk varies a lot by age group
    df["AgeGroup"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 65, 120],
    labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2).astype(int)

    print(f"✅ Engineered features. New shape: {df.shape}")
    return df


# ── 5. Remove Outliers ────────────────────────────────────────────────────────
def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values at the 99th percentile instead of removing rows.
    
    Why cap instead of remove?
      Removing outliers shrinks the dataset. Capping keeps the row
      but limits the extreme value's influence on the model.
    """
    df = df.copy()
    cols_to_cap = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
    
    for col in cols_to_cap:
        cap_value = df[col].quantile(0.99)
        original_max = df[col].max()
        df[col] = df[col].clip(upper=cap_value)
        print(f"   {col}: capped {original_max:.2f} → {cap_value:.2f}")

    print("✅ Outliers capped")
    return df


# ── 6. Full Preprocessing Pipeline ───────────────────────────────────────────
def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_scaler: bool = True
):
    """
    Runs the full pipeline:
    load → clean → feature engineer → split → scale

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    # Step 1: Fill missing
    df = fill_missing(df)

    # Step 2: Cap outliers
    df = cap_outliers(df)

    # Step 3: Engineer features
    df = engineer_features(df)

    # Step 4: Separate features and target
    all_features = FEATURE_COLS + ["TotalPastDue", "IncomePerDependent", "AgeGroup"]
    X = df[all_features]
    y = df[TARGET_COL]

    print(f"\n📊 Class distribution:")
    print(y.value_counts(normalize=True).round(3))

    # Step 5: Train/test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # ← important for imbalanced data
    )

    # Step 6: Scale (fit ONLY on train, transform both)
    # Why? Fitting on test data would be "data leakage"
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(all_features, "models/feature_names.pkl")
        print("✅ Scaler and feature names saved to models/")

    print(f"\n✅ Preprocessing complete")
    print(f"   Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, all_features, scaler
