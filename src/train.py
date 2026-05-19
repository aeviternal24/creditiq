"""
train.py — Model Training Pipeline
=====================================
Trains 3 models, evaluates them, and saves the best one.
Run this script directly: python src/train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Import our preprocessing module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load & Preprocess
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prep(data_path="data/cs-training.csv"):
    print("=" * 60)
    print("STEP 1: Loading & Preprocessing Data")
    print("=" * 60)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Handle Class Imbalance with SMOTE
# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train, random_state=42):
    """
    SMOTE = Synthetic Minority Oversampling Technique
    
    The dataset has ~14:1 imbalance (non-default:default).
    Without fixing this, the model learns to just predict "no default"
    for everything and still gets 93% accuracy — which is USELESS.
    
    SMOTE creates synthetic new samples of the minority class (defaults)
    by interpolating between existing examples.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Applying SMOTE to Handle Class Imbalance")
    print("=" * 60)
    print(f"Before SMOTE: {np.bincount(y_train.astype(int))}")

    smote = SMOTE(random_state=random_state, sampling_strategy=0.3)
    # sampling_strategy=0.3 means minority class = 30% of majority
    # Full 1:1 balance often HURTS performance with real credit data
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE:  {np.bincount(y_resampled.astype(int))}")
    return X_resampled, y_resampled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Define Models
# ─────────────────────────────────────────────────────────────────────────────
def get_models():
    """
    Three models in order of complexity:
    
    1. Logistic Regression — simple linear baseline
       Pros: fast, interpretable, works well if features are linear
       Cons: can't capture non-linear patterns
    
    2. Random Forest — ensemble of decision trees
       Pros: handles non-linearity, robust to outliers
       Cons: slow to train on large data, memory-heavy
    
    3. XGBoost — gradient boosted trees (sequential improvement)
       Pros: usually best performance, handles imbalance well
       Cons: more hyperparameters, can overfit
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"  # another way to handle imbalance
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1  # use all CPU cores
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="auc",
            verbosity=0,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Train & Evaluate All Models
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(models, X_train, y_train, X_test, y_test, feature_names):
    """
    Train each model and collect evaluation metrics.
    
    Key metrics for imbalanced classification:
    - ROC-AUC: overall ability to rank positive vs negative (main metric)
    - Precision: of predicted defaults, how many were real?
    - Recall: of real defaults, how many did we catch? (very important for banks)
    - Average Precision: area under Precision-Recall curve
    """
    print("\n" + "=" * 60)
    print("STEP 3 & 4: Training Models & Evaluating")
    print("=" * 60)

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        model.fit(X_train, y_train)

        # Get probability scores (not just 0/1 predictions)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc      = roc_auc_score(y_test, y_pred_proba)
        avg_prec = average_precision_score(y_test, y_pred_proba)
        report   = classification_report(y_test, y_pred, output_dict=True)

        print(f"   ROC-AUC:           {auc:.4f}")
        print(f"   Avg Precision:     {avg_prec:.4f}")
        print(f"   Recall (default):  {report['1']['recall']:.4f}")
        print(f"   Precision (def):   {report['1']['precision']:.4f}")

        results[name] = {
            "auc":               round(auc, 4),
            "avg_precision":     round(avg_prec, 4),
            "recall_default":    round(report["1"]["recall"], 4),
            "precision_default": round(report["1"]["precision"], 4),
            "f1_default":        round(report["1"]["f1-score"], 4),
        }
        trained_models[name] = model

    return results, trained_models


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Save Best Model
# ─────────────────────────────────────────────────────────────────────────────
def save_best_model(results, trained_models):
    """Pick the model with highest ROC-AUC and save it."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving Best Model")
    print("=" * 60)

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = trained_models[best_name]

    print(f"🏆 Best model: {best_name} (AUC = {results[best_name]['auc']})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    # Save all results to JSON for the Streamlit app to read
    with open("models/results.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"✅ Model saved to models/best_model.pkl")
    print(f"✅ Results saved to models/results.json")

    return best_name, best_model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────
def generate_shap(model, X_test, feature_names, model_name, n_samples=500):
    """
    SHAP (SHapley Additive exPlanations) explains ML model predictions.
    
    For each prediction, SHAP tells you:
    "Feature X pushed the risk UP by 0.12, feature Y pushed it DOWN by 0.08..."
    
    This is what separates a real data scientist from someone who just
    calls model.fit() — you understand and can explain your model.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Generating SHAP Explanations")
    print("=" * 60)

    # Use a sample for speed (SHAP is slow on large datasets)
    X_sample = X_test[:n_samples]

    print(f"   Computing SHAP values on {n_samples} samples...")

    if model_name in ["XGBoost", "Random Forest"]:
        # TreeExplainer is fast and exact for tree-based models
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # LinearExplainer for Logistic Regression
        explainer  = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values might be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Save SHAP values for the app
    np.save("models/shap_values.npy", shap_values)
    np.save("models/X_sample.npy", X_sample)
    joblib.dump(feature_names, "models/feature_names.pkl")
    joblib.dump(explainer, "models/explainer.pkl")

    print("✅ SHAP values saved")

    # Plot and save summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("✅ SHAP summary plot saved to models/shap_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_and_prep()

    # 2. Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # 3. Get models
    models = get_models()

    # 4. Train & evaluate
    results, trained_models = train_and_evaluate(
        models, X_train_balanced, y_train_balanced, X_test, y_test, feature_names
    )

    # 5. Save best model
    best_name, best_model = save_best_model(results, trained_models)

    # 6. SHAP explainability
    generate_shap(best_model, X_test, feature_names, best_name)

    print("\n" + "=" * 60)
    print("🎉 Training complete! Now run: streamlit run app.py")
    print("=" * 60)
