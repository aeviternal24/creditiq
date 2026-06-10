"""
app.py — CreditIQ Streamlit Application
=========================================
4 pages:
  1. Home        — project overview & dataset stats
  2. EDA         — interactive charts (uses sample.csv on deployment)
  3. Model       — performance metrics & SHAP summary
  4. Predictor   — live risk scoring for a new applicant

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import shap

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .risk-low    { color: #27ae60; font-size: 2rem; font-weight: bold; }
    .risk-medium { color: #f39c12; font-size: 2rem; font-weight: bold; }
    .risk-high   { color: #e74c3c; font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load cached resources
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    # Use full dataset if available, fall back to sample for deployment
    if os.path.exists("data/cs-training.csv"):
        path = "data/cs-training.csv"
        is_sample = False
    elif os.path.exists("data/sample.csv"):
        path = "data/sample.csv"
        is_sample = True
    else:
        return None, False
    df = pd.read_csv(path, index_col=0)
    df["MonthlyIncome"]      = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].mode()[0])
    return df, is_sample


@st.cache_resource
def load_model_artifacts():
    required = ["models/best_model.pkl", "models/scaler.pkl",
                "models/feature_names.pkl", "models/results.json"]
    for path in required:
        if not os.path.exists(path):
            return None, None, None, None, None
    model         = joblib.load("models/best_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    with open("models/results.json") as f:
        results = json.load(f)
    # Load optimal threshold if available, fall back to 0.5
    if os.path.exists("models/optimal_threshold.pkl"):
        threshold = joblib.load("models/optimal_threshold.pkl")
    else:
        threshold = 0.5
    return model, scaler, feature_names, results, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 CreditIQ")
    st.markdown("*Loan Default Prediction Engine*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 EDA Dashboard", "📊 Model Performance", "🧪 Live Predictor"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("Python · XGBoost · SHAP  \nSMOTE · Scikit-learn · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("📊 CreditIQ")
    st.subheader("ML-Powered Loan Default Prediction Engine")
    st.markdown("""
    **CreditIQ** predicts whether a loan applicant is likely to default within 2 years —
    and explains *why* using SHAP explainability.
    """)
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples", "150,000+")
    col2.metric("Features", "13")
    col3.metric("Best Model", "XGBoost")
    col4.metric("Target Metric", "ROC-AUC")

    st.divider()
    st.markdown("### 🔄 Pipeline Overview")

    steps = [
        ("📥 Raw Data",            "150k loan applications from Kaggle"),
        ("🧹 Preprocessing",       "Fill missing values, cap outliers"),
        ("⚙️ Feature Engineering", "TotalPastDue, HasDependents"),
        ("⚖️ SMOTE",               "Fix 14:1 class imbalance"),
        ("🤖 Modeling",            "LogReg → Random Forest → XGBoost"),
        ("📊 Evaluation",          "ROC-AUC, Precision-Recall, Confusion Matrix"),
        ("🔍 SHAP",                "Explain every individual prediction"),
        ("🚀 Deployment",          "Streamlit app — real-time scoring"),
    ]

    cols = st.columns(4)
    for i, (title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='metric-card'>
                <b style='color:#f0f0f0;'>{title}</b><br>
                <small style='color:#aaaaaa;'>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📂 Dataset: Give Me Some Credit (Kaggle)")
    result = load_dataset()
    df, is_sample = result if result[0] is not None else (None, False)
    if df is not None:
        if is_sample:
            st.caption("⚠️ Showing a 2,000-row sample. Full dataset has 150,000 rows.")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Dataset not found. Download `cs-training.csv` from Kaggle and place it in `data/`.")
        st.markdown("[📥 Download Dataset](https://www.kaggle.com/c/GiveMeSomeCredit/data)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 EDA Dashboard":
    st.title("🔍 Exploratory Data Analysis")

    result = load_dataset()
    df, is_sample = result if result[0] is not None else (None, False)

    if df is None:
        st.warning("Dataset not found.")
        st.stop()

    if is_sample:
        st.info("📊 Showing analysis on a 2,000-row sample. Trends are representative of the full 150,000-row dataset.")

    # ── Class Imbalance ──────────────────────────────────────────────────────
    st.markdown("### Class Distribution (Target Variable)")
    col1, col2 = st.columns(2)

    target_counts = df["SeriousDlqin2yrs"].value_counts()
    with col1:
        fig = px.pie(
            values=target_counts.values,
            names=["No Default (0)", "Default (1)"],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            title="Loan Default Distribution"
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ratio = target_counts[0] / target_counts[1]
        st.markdown(f"""
        **Class Imbalance Summary**

        | Class | Count | % |
        |-------|-------|---|
        | No Default | {target_counts[0]:,} | {target_counts[0]/len(df)*100:.1f}% |
        | Default | {target_counts[1]:,} | {target_counts[1]/len(df)*100:.1f}% |

        **Imbalance ratio:** {ratio:.1f}:1

        This is why we use **SMOTE** — a naive model that always
        predicts "No Default" gets {target_counts[0]/len(df)*100:.1f}% accuracy
        but catches **zero actual defaults**.
        """)

    st.divider()

    # ── Feature Distribution ─────────────────────────────────────────────────
    st.markdown("### Feature Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "SeriousDlqin2yrs"]

    selected_feature = st.selectbox("Select a feature to explore:", numeric_cols)

    col1, col2 = st.columns(2)
    with col1:
        cap = df[selected_feature].quantile(0.99)
        fig = px.histogram(
            df[df[selected_feature] <= cap],
            x=selected_feature,
            color="SeriousDlqin2yrs",
            barmode="overlay",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={"SeriousDlqin2yrs": "Default"},
            title=f"{selected_feature} by Default Status"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df[df[selected_feature] <= cap],
            x="SeriousDlqin2yrs",
            y=selected_feature,
            color="SeriousDlqin2yrs",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={"SeriousDlqin2yrs": "Default (0=No, 1=Yes)"},
            title=f"{selected_feature} Box Plot"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Default Rate by Age Group ─────────────────────────────────────────────
    st.divider()
    st.markdown("### Default Rate by Age Group")

    df["AgeGroup"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 65, 120],
                             labels=["<25", "25-35", "35-50", "50-65", "65+"])
    default_by_age = df.groupby("AgeGroup", observed=True)["SeriousDlqin2yrs"].mean().reset_index()
    default_by_age.columns = ["Age Group", "Default Rate"]
    default_by_age["Default Rate %"] = (default_by_age["Default Rate"] * 100).round(2)

    fig = px.bar(default_by_age, x="Age Group", y="Default Rate %",
                 color="Default Rate %", color_continuous_scale="RdYlGn_r",
                 title="Default Rate (%) by Age Group",
                 text="Default Rate %")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Correlation Heatmap")

    corr = df.drop(columns=["AgeGroup"], errors="ignore").corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    model, scaler, feature_names, results_data, threshold = load_model_artifacts()
    if model is None:
        st.warning("Model not trained yet. Run `python src/train.py` first.")
        st.code("python src/train.py")
        st.stop()

    results   = results_data["results"]
    best_name = results_data["best_model"]

    # ── Model Comparison ─────────────────────────────────────────────────────
    st.markdown("### Model Comparison")
    col1, col2, col3 = st.columns(3)
    for i, (name, data) in enumerate(results.items()):
        col = [col1, col2, col3][i % 3]
        is_best = name == best_name
        col.metric(
            label=f"{'🏆 ' if is_best else ''}{name}",
            value=f"AUC: {data['auc']:.4f}",
        )

    st.dataframe(
        pd.DataFrame(results).T.rename(columns={
            "auc":               "ROC-AUC",
            "avg_precision":     "Avg Precision",
            "recall_default":    "Recall (Default)",
            "precision_default": "Precision (Default)",
            "f1_default":        "F1 (Default)"
        }),
        use_container_width=True
    )

    st.divider()
    st.markdown(f"**Optimal Classification Threshold:** `{threshold:.3f}`")
    st.caption("Found by maximising F1 on the Precision-Recall curve. Used in the Live Predictor.")

    # ── SHAP Summary ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### SHAP Feature Importance (Global)")
    st.markdown("SHAP values show which features influence predictions most, and in which direction.")

    if os.path.exists("models/shap_summary.png"):
        st.image("models/shap_summary.png", use_column_width=True)
    elif os.path.exists("models/shap_values.npy") and os.path.exists("models/X_sample.npy"):
        shap_values = np.load("models/shap_values.npy")
        X_sample    = np.load("models/X_sample.npy")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("SHAP plot not available. Retrain the model to generate it.")

    st.divider()
    st.markdown("""
    ### 📖 How to Read SHAP Values

    | Element | Meaning |
    |---------|---------|
    | **X-axis** | SHAP value — how much this feature pushes the prediction |
    | **Positive SHAP** | Pushes prediction toward DEFAULT ↑ |
    | **Negative SHAP** | Pushes prediction toward NO DEFAULT ↓ |
    | **Dot color (red)** | High feature value |
    | **Dot color (blue)** | Low feature value |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: LIVE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧪 Live Predictor":
    st.title("🧪 Live Default Risk Predictor")
    st.markdown("Enter an applicant's details to get their default risk score and explanation.")

    model, scaler, feature_names, results_data, threshold = load_model_artifacts()
    if model is None:
        st.warning("Model not trained yet. Run `python src/train.py` first.")
        st.code("python src/train.py")
        st.stop()

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👤 Personal Info")
        age         = st.slider("Age", 18, 90, 35)
        dependents  = st.number_input("Number of Dependents", 0, 20, 1)
        monthly_inc = st.number_input("Monthly Income ($)", 0, 100000, 5000, step=500)

    with col2:
        st.markdown("#### 💳 Credit Profile")
        revolving  = st.slider("Revolving Credit Utilization (0-1)", 0.0, 1.5, 0.3, step=0.01)
        debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, 0.35, step=0.01)
        open_loans = st.number_input("Open Credit Lines & Loans", 0, 50, 8)
        re_loans   = st.number_input("Real Estate Loans", 0, 20, 1)

    st.markdown("#### ⚠️ Delinquency History")
    col3, col4, col5 = st.columns(3)
    past_30 = col3.number_input("Times 30-59 Days Late", 0, 20, 0)
    past_60 = col4.number_input("Times 60-89 Days Late", 0, 20, 0)
    past_90 = col5.number_input("Times 90+ Days Late",   0, 20, 0)

    st.divider()

    if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):

        total_past_due  = past_30 + past_60 + past_90
        has_dependents  = 1 if dependents > 0 else 0

        raw_features = {
            "RevolvingUtilizationOfUnsecuredLines":  revolving,
            "age":                                   age,
            "NumberOfTime30-59DaysPastDueNotWorse":  past_30,
            "DebtRatio":                             debt_ratio,
            "MonthlyIncome":                         monthly_inc,
            "NumberOfOpenCreditLinesAndLoans":       open_loans,
            "NumberOfTimes90DaysLate":               past_90,
            "NumberRealEstateLoansOrLines":          re_loans,
            "NumberOfTime60-89DaysPastDueNotWorse":  past_60,
            "NumberOfDependents":                    dependents,
            "TotalPastDue":                          total_past_due,
            "HasDependents":                         has_dependents,
        }

        input_values = [raw_features.get(f, 0) for f in feature_names]
        X_input      = np.array(input_values).reshape(1, -1)
        X_scaled     = scaler.transform(X_input)

        prob = model.predict_proba(X_scaled)[0][1]

        # ── Risk Display ─────────────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prob < 0.25:
                risk_label = "LOW RISK"
                risk_class = "risk-low"
                risk_color = "#27ae60"
                emoji = "✅"
            elif prob < 0.50:
                risk_label = "MODERATE RISK"
                risk_class = "risk-medium"
                risk_color = "#f39c12"
                emoji = "⚠️"
            else:
                risk_label = "HIGH RISK"
                risk_class = "risk-high"
                risk_color = "#e74c3c"
                emoji = "🚨"

            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: #f8f9fa;
                        border-radius: 15px; border: 2px solid {risk_color};'>
                <p style='font-size: 1rem; color: #7f8c8d; margin-bottom: 0.5rem;'>
                    Default Probability
                </p>
                <p class='{risk_class}'>{prob:.1%}</p>
                <p style='font-size: 1.5rem;'>{emoji} {risk_label}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge Chart ──────────────────────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Default Risk (%)"},
            delta={"reference": 6.7, "suffix": "% vs avg"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range": [0,  25], "color": "#d5f5e3"},
                    {"range": [25, 50], "color": "#fef9e7"},
                    {"range": [50, 100], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 50},
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # ── SHAP Explanation ─────────────────────────────────────────────────
        st.markdown("### 🔍 Why this prediction?")

        try:
            if os.path.exists("models/explainer.pkl"):
                explainer = joblib.load("models/explainer.pkl")
            else:
                explainer = shap.TreeExplainer(model)

            shap_vals = explainer.shap_values(X_scaled)
            if isinstance(shap_vals, list):
                shap_vals = np.array(shap_vals[1])
            elif hasattr(shap_vals, 'values'):
                shap_vals = shap_vals.values
            else:
                shap_vals = np.array(shap_vals)
            shap_vals = shap_vals.flatten()

            shap_df = pd.DataFrame({
                "Feature":    feature_names,
                "Value":      [round(v, 3) for v in input_values],
                "SHAP Value": shap_vals,
                "Direction":  ["↑ Higher Risk" if v > 0 else "↓ Lower Risk" for v in shap_vals]
            })
            shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index)
            shap_df["SHAP Value"] = shap_df["SHAP Value"].round(4)

            colors_list = ["#e74c3c" if v > 0 else "#27ae60" for v in shap_df["SHAP Value"]]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors_list, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title("Feature Contributions\n(Red = increases risk, Green = decreases risk)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### Feature Breakdown")
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.info(f"SHAP explanation unavailable: {str(e)}")
