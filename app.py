"""
CreditIQ — Loan Default Prediction Engine
Streamlit Application
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.card {
    background: #1a1a2e;
    border-left: 4px solid #3498db;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.card b  { color: #e0e0e0; font-size: 0.95rem; }
.card small { color: #888; font-size: 0.82rem; }
.risk-box {
    text-align: center;
    padding: 2rem;
    border-radius: 12px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    for path in ["data/cs-training.csv", "data/sample.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df["MonthlyIncome"]      = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
            df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
                df["NumberOfDependents"].mode()[0])
            is_sample = "sample" in path
            return df, is_sample
    return None, False


@st.cache_resource
def get_model():
    needed = ["models/best_model.pkl", "models/scaler.pkl",
              "models/feature_names.pkl", "models/results.json"]
    if not all(os.path.exists(p) for p in needed):
        return None, None, None, None, 0.5

    model         = joblib.load("models/best_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = list(joblib.load("models/feature_names.pkl"))

    with open("models/results.json") as f:
        results = json.load(f)

    threshold = 0.5
    if os.path.exists("models/optimal_threshold.pkl"):
        threshold = float(joblib.load("models/optimal_threshold.pkl"))

    return model, scaler, feature_names, results, threshold


# ── SHAP helper ───────────────────────────────────────────────────────────────
def compute_shap(model, X_scaled, feature_names):
    """
    Returns (shap_array_1d, None) on success or (None, error_string) on failure.
    Works with XGBoost, RandomForest, and LogisticRegression.
    """
    try:
        import shap

        # TreeExplainer for tree models, LinearExplainer for linear
        model_type = type(model).__name__
        if model_type in ("XGBClassifier", "RandomForestClassifier",
                          "GradientBoostingClassifier", "DecisionTreeClassifier"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_scaled)

        raw = explainer.shap_values(X_scaled)

        # Unpack list form (binary classification gives [neg_class, pos_class])
        if isinstance(raw, list):
            raw = raw[1]

        raw = np.array(raw, dtype=float)

        # Flatten to 1-D
        raw = raw.flatten()

        # Trim or pad to match feature count
        n = len(feature_names)
        if len(raw) > n:
            raw = raw[:n]
        elif len(raw) < n:
            raw = np.concatenate([raw, np.zeros(n - len(raw))])

        return raw, None

    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 CreditIQ")
    st.markdown("*Loan Default Prediction Engine*")
    st.divider()
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔍 EDA Dashboard", "📊 Model Performance", "🧪 Live Predictor"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Stack**")
    st.markdown("Python · XGBoost · SHAP\nSMOTE · Scikit-learn · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📊 CreditIQ")
    st.subheader("ML-Powered Loan Default Prediction Engine")
    st.markdown(
        "**CreditIQ** predicts whether a loan applicant will default within 2 years "
        "and explains *why* using SHAP explainability."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", "150,000+")
    c2.metric("Features",         "12")
    c3.metric("Best Model",       "Random Forest")
    c4.metric("Primary Metric",   "ROC-AUC")

    st.divider()
    st.markdown("### Pipeline Overview")

    steps = [
        ("📥 Raw Data",            "150k loan applications — Kaggle"),
        ("🧹 Preprocessing",       "Imputation, outlier capping"),
        ("⚙️ Feature Engineering", "TotalPastDue, HasDependents"),
        ("⚖️ SMOTE",               "Fixes 14:1 class imbalance"),
        ("🤖 3 Models",            "LogReg → RF → XGBoost"),
        ("📊 Evaluation",          "ROC-AUC, PR curve, CV"),
        ("🔍 SHAP",                "Per-prediction explanations"),
        ("🚀 Deployed",            "Streamlit Cloud — live"),
    ]

    cols = st.columns(4)
    for i, (title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(
                f"<div class='card'><b>{title}</b><br><small>{desc}</small></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Dataset Preview")
    df, is_sample = get_data()
    if df is not None:
        if is_sample:
            st.caption("⚠️ Deployed version shows a 2,000-row sample. Full dataset: 150,000 rows.")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Dataset not found. Place `cs-training.csv` in the `data/` folder.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA Dashboard":
    st.title("🔍 Exploratory Data Analysis")

    df, is_sample = get_data()
    if df is None:
        st.error("Dataset not found. Add `cs-training.csv` or `sample.csv` to `data/`.")
        st.stop()

    if is_sample:
        st.info("Showing a 2,000-row sample. Patterns are representative of the full dataset.")

    # ── Target distribution
    st.markdown("### Target Variable — Class Imbalance")
    counts = df["SeriousDlqin2yrs"].value_counts().sort_index()
    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(
            values=counts.values,
            names=["No Default", "Default"],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            title="Class Distribution",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"""
**Why this matters**

| | Count | % |
|---|---|---|
| No Default | {counts[0]:,} | {counts[0]/len(df)*100:.1f}% |
| Default    | {counts[1]:,} | {counts[1]/len(df)*100:.1f}% |

**Ratio:** {counts[0]/counts[1]:.0f}:1

A model that always predicts "No Default" scores
**{counts[0]/len(df)*100:.1f}% accuracy** but catches
**zero defaults** — useless for a bank.
This is why we use SMOTE and ROC-AUC instead of accuracy.
        """)

    # ── Feature explorer
    st.divider()
    st.markdown("### Feature Explorer")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c != "SeriousDlqin2yrs"]
    feat = st.selectbox("Select feature:", num_cols)
    cap  = float(df[feat].quantile(0.99))
    dff  = df[df[feat] <= cap].copy()
    dff["Default"] = dff["SeriousDlqin2yrs"].map({0: "No Default", 1: "Default"})

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x=feat, color="Default", barmode="overlay",
                           color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"},
                           title=f"{feat} — Histogram")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(dff, x="Default", y=feat, color="Default",
                     color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"},
                     title=f"{feat} — Box Plot")
        st.plotly_chart(fig, use_container_width=True)

    # ── Default rate by age
    st.divider()
    st.markdown("### Default Rate by Age Group")
    tmp = df.copy()
    tmp["AgeGroup"] = pd.cut(
        tmp["age"],
        bins=[0, 25, 35, 50, 65, 120],
        labels=["<25", "25–35", "35–50", "50–65", "65+"],
    )
    age_def = (
        tmp.groupby("AgeGroup", observed=True)["SeriousDlqin2yrs"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
    )
    age_def.columns = ["Age Group", "Default Rate %"]
    fig = px.bar(
        age_def, x="Age Group", y="Default Rate %",
        color="Default Rate %", color_continuous_scale="RdYlGn_r",
        text="Default Rate %", title="Default Rate by Age Group",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap
    st.divider()
    st.markdown("### Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.4, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    model, scaler, feature_names, results_data, threshold = get_model()
    if model is None:
        st.error("Model not found. Make sure all files in `models/` are pushed to GitHub.")
        st.stop()

    results   = results_data["results"]
    best_name = results_data["best_model"]

    # ── Metrics table
    st.markdown("### Model Comparison")
    n_models = len(results)
    cols = st.columns(n_models)
    for i, (name, data) in enumerate(results.items()):
        cols[i].metric(
            label=f"{'🏆 ' if name == best_name else ''}{name}",
            value=f"AUC: {data['auc']:.4f}",
        )

    rename = {
        "auc":               "ROC-AUC",
        "avg_precision":     "Avg Precision",
        "recall_default":    "Recall (Default)",
        "precision_default": "Precision (Default)",
        "f1_default":        "F1 (Default)",
    }
    tbl = pd.DataFrame(results).T
    tbl = tbl[[k for k in rename if k in tbl.columns]].rename(columns=rename)
    st.dataframe(tbl.style.highlight_max(axis=0, color="#1a4a1a"), use_container_width=True)

    st.divider()
    st.markdown(f"**Optimal threshold:** `{threshold:.3f}` — maximises F1 on the Precision-Recall curve.")

    # ── SHAP summary
    st.divider()
    st.markdown("### SHAP Global Feature Importance")
    st.markdown("Which features drive default predictions most — and in which direction?")

    if os.path.exists("models/shap_summary.png"):
        st.image("models/shap_summary.png", use_column_width=True)
    elif os.path.exists("models/shap_values.npy") and os.path.exists("models/X_sample.npy"):
        try:
            import shap
            sv  = np.load("models/shap_values.npy")
            Xs  = np.load("models/X_sample.npy")
            fig, _ = plt.subplots(figsize=(10, 6))
            shap.summary_plot(sv, Xs, feature_names=feature_names, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render SHAP plot: {e}")
    else:
        st.info("SHAP plot not available. Re-run training to generate it.")

    st.divider()
    st.markdown("""
### How to Read SHAP Values
| | Meaning |
|---|---|
| **Positive SHAP (right)** | Feature pushes prediction toward DEFAULT |
| **Negative SHAP (left)**  | Feature pushes prediction toward NO DEFAULT |
| **Red dot** | Applicant has a high value for that feature |
| **Blue dot** | Applicant has a low value for that feature |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Live Predictor":
    st.title("🧪 Live Default Risk Predictor")
    st.markdown("Fill in the applicant details below and click **Predict**.")

    model, scaler, feature_names, results_data, threshold = get_model()
    if model is None:
        st.error("Model not found. Make sure all files in `models/` are pushed to GitHub.")
        st.stop()

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Personal Info")
        age         = st.slider("Age", 18, 90, 35)
        dependents  = st.number_input("Dependents", 0, 20, 0)
        monthly_inc = st.number_input("Monthly Income ($)", 0, 100000, 5000, step=500)

    with c2:
        st.markdown("#### Credit Profile")
        revolving  = st.slider("Revolving Utilization (0–1.5)", 0.0, 1.5, 0.3, step=0.01)
        debt_ratio = st.slider("Debt Ratio (0–5)", 0.0, 5.0, 0.35, step=0.01)
        open_loans = st.number_input("Open Credit Lines & Loans", 0, 50, 8)
        re_loans   = st.number_input("Real Estate Loans", 0, 20, 1)

    st.markdown("#### Delinquency History")
    c3, c4, c5 = st.columns(3)
    past_30 = c3.number_input("30–59 Days Late", 0, 20, 0)
    past_60 = c4.number_input("60–89 Days Late", 0, 20, 0)
    past_90 = c5.number_input("90+ Days Late",   0, 20, 0)

    st.divider()

    if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):

        # All possible features — covers old and new model versions
        all_possible = {
            "RevolvingUtilizationOfUnsecuredLines":  revolving,
            "age":                                   float(age),
            "NumberOfTime30-59DaysPastDueNotWorse":  float(past_30),
            "DebtRatio":                             debt_ratio,
            "MonthlyIncome":                         float(monthly_inc),
            "NumberOfOpenCreditLinesAndLoans":       float(open_loans),
            "NumberOfTimes90DaysLate":               float(past_90),
            "NumberRealEstateLoansOrLines":          float(re_loans),
            "NumberOfTime60-89DaysPastDueNotWorse":  float(past_60),
            "NumberOfDependents":                    float(dependents),
            "TotalPastDue":                          float(past_30 + past_60 + past_90),
            "HasDependents":                         float(dependents > 0),
            "IncomePerDependent":                    float(monthly_inc) / (float(dependents) + 1),
            "AgeGroup":                              float(
                pd.cut([age], bins=[0,25,35,50,65,120], labels=[0,1,2,3,4])[0]
            ),
        }

        # Use only features the model was trained on, in exact order
        input_values = [all_possible.get(f, 0.0) for f in feature_names]
        X = np.array(input_values, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prob = float(model.predict_proba(X_scaled)[0][1])

        # ── Risk label
        if prob < 0.25:
            label, color, emoji = "LOW RISK",      "#27ae60", "✅"
        elif prob < 0.50:
            label, color, emoji = "MODERATE RISK", "#f39c12", "⚠️"
        else:
            label, color, emoji = "HIGH RISK",     "#e74c3c", "🚨"

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(f"""
            <div class='risk-box' style='border: 2px solid {color};'>
                <p style='color:#aaa; margin:0 0 0.3rem 0;'>Default Probability</p>
                <p style='color:{color}; font-size:2.5rem; font-weight:bold; margin:0;'>{prob:.1%}</p>
                <p style='font-size:1.3rem; margin:0.5rem 0 0 0;'>{emoji} {label}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            title={"text": "Default Risk (%)"},
            delta={"reference": 6.7, "suffix": "% vs population avg"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": color},
                "steps": [
                    {"range": [0,  25],  "color": "#d5f5e3"},
                    {"range": [25, 50],  "color": "#fef9e7"},
                    {"range": [50, 100], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 50},
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ── SHAP
        st.markdown("### Why this prediction?")

        sv, err = compute_shap(model, X_scaled, feature_names)

        if sv is None:
            st.info(f"SHAP explanation could not be generated: {err}")
        else:
            n = len(feature_names)
            shap_df = pd.DataFrame({
                "Feature":    feature_names,
                "Your Value": [round(float(v), 3) for v in input_values[:n]],
                "Impact":     [round(float(v), 4) for v in sv[:n]],
                "Direction":  ["↑ Increases Risk" if v > 0 else "↓ Decreases Risk" for v in sv[:n]],
            })
            shap_df = shap_df.reindex(
                shap_df["Impact"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)

            # Bar chart
            bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["Impact"]]
            fig, ax = plt.subplots(figsize=(9, max(4, int(n * 0.45))))
            ax.barh(shap_df["Feature"], shap_df["Impact"], color=bar_colors, edgecolor="none")
            ax.axvline(0, color="#cccccc", linewidth=1)
            ax.set_xlabel("SHAP Value  (positive = increases default risk)")
            ax.set_title("What drove this prediction?", fontweight="bold", pad=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### Full Feature Breakdown")
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

            st.caption(
                "SHAP values show correlation-based attribution, not causation. "
                "A high SHAP value means this feature is associated with higher default risk "
                "for applicants with similar profiles — not that changing it would change the outcome."
            )
