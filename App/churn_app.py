import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc)

from data_utils import load_clean_encoded, clean, load_raw, BASE_DIR, REPO_ROOT, NUM_COLS, TARGET

st.set_page_config(page_title="Bank Customer Churn", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_FILENAME = "churn_model.pkl"
_MODEL_CANDIDATE_DIRS = [
    BASE_DIR, REPO_ROOT,
    os.path.join(REPO_ROOT, "Model"), os.path.join(BASE_DIR, "Model"),
    os.getcwd(), os.path.join(os.getcwd(), "Model"),
]


def _find_model_path():
    for d in _MODEL_CANDIDATE_DIRS:
        candidate = os.path.join(d, MODEL_FILENAME)
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource
def get_model_bundle():
    path = _find_model_path()
    if path is None:
        return None
    return joblib.load(path)


@st.cache_data
def get_data(_scaler, feature_columns):
    # _scaler prefixed with underscore so Streamlit doesn't try to hash the sklearn object
    return load_clean_encoded(_scaler, feature_columns)


bundle = get_model_bundle()

if bundle is None:
    st.error(
        f"No trained model found. Make sure `{MODEL_FILENAME}` sits in the "
        "`Model/` folder of the repo (or next to app.py)."
    )
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]

try:
    df_clean, df_encoded = get_data(scaler, feature_columns)
    data_error = None
except Exception as e:
    df_clean = df_encoded = None
    data_error = str(e)

# ---------------------------------------------------------------------------
st.title("🏦 Bank Customer Churn — EDA, Clustering & Prediction")
st.caption(
    "Kanak Baghel · IIT Guwahati DSBA · "
    "[GitHub repo](https://github.com/Kanakbaghel/Bank_Customer_Churn_Analysis)"
)

if data_error:
    st.error(
        "Couldn't load the churn dataset. This almost always means `Data.csv` "
        f"isn't where the app expects it.\n\n**Details:** {data_error}"
    )
    st.stop()

tab_overview, tab_eda, tab_predict, tab_cluster, tab_insights, tab_data = st.tabs(
    ["📋 Overview", "📊 EDA", "🔮 Predict Churn", "🧬 Clustering", "🧠 Model Insights", "🗂️ Raw Data"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers (after cleaning)", len(df_clean))
    c2.metric("Churn rate", f"{df_clean[TARGET].mean():.1%}")
    c3.metric("Avg. Credit Score", f"{df_clean['CreditScore'].mean():.0f}")
    c4.metric("Avg. Balance", f"${df_clean['Balance'].mean():,.0f}")

    st.markdown(
        """
        This app deploys the pipeline built in `Project_Notebook.ipynb`: raw customer
        records → cleaned (imputed + de-duplicated) → explored → clustered (PCA + K-Means)
        → used to train a **Random Forest** model that predicts whether a customer will
        **churn (exit)** or **stay**.
        """
    )

    churn_counts = df_clean[TARGET].value_counts().rename({0: "Stayed", 1: "Churned"}).reset_index()
    churn_counts.columns = ["status", "count"]
    fig = px.pie(churn_counts, names="status", values="count", hole=0.45,
                 color="status", color_discrete_map={"Stayed": "#2ecc71", "Churned": "#e74c3c"},
                 title="Customer churn mix")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_eda:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn rate by geography")
        rate = df_clean.groupby("Geography")[TARGET].mean().reset_index()
        fig = px.bar(rate, x="Geography", y=TARGET, color="Geography")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Churn rate by gender")
        rate = df_clean.groupby("Gender")[TARGET].mean().reset_index()
        fig = px.bar(rate, x="Gender", y=TARGET, color="Gender")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age distribution — churned vs stayed")
        fig = px.histogram(df_clean, x="Age", color=df_clean[TARGET].map({0: "Stayed", 1: "Churned"}),
                            barmode="overlay", opacity=0.6,
                            color_discrete_map={"Stayed": "#2ecc71", "Churned": "#e74c3c"})
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("Balance distribution — churned vs stayed")
        fig = px.histogram(df_clean, x="Balance", color=df_clean[TARGET].map({0: "Stayed", 1: "Churned"}),
                            barmode="overlay", opacity=0.6,
                            color_discrete_map={"Stayed": "#2ecc71", "Churned": "#e74c3c"})
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn rate by number of products")
    rate = df_clean.groupby("NumOfProducts")[TARGET].mean().reset_index()
    fig = px.bar(rate, x="NumOfProducts", y=TARGET)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature correlation heatmap")
    corr_cols = NUM_COLS + ["HasCrCard", "IsActiveMember", TARGET]
    corr = df_clean[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict whether a customer will churn")
    st.caption("Using the **Random Forest** pipeline from the notebook.")

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit score", 300, 900, 650)
        age = st.slider("Age", 18, 95, 38)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
    with col2:
        balance = st.number_input("Account balance", 0.0, 300000.0, 75000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=1000.0)
        num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
    with col3:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        has_cr_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Active member?", ["Yes", "No"], horizontal=True)

    if st.button("Predict", type="primary"):
        try:
            row = {
                "CreditScore": credit_score,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active == "Yes" else 0,
                "EstimatedSalary": estimated_salary,
                "Geography_Germany": 1 if geography == "Germany" else 0,
                "Geography_Spain": 1 if geography == "Spain" else 0,
                "Gender_Male": 1 if gender == "Male" else 0,
            }
            input_df = pd.DataFrame([row])
            # Scale ONLY the numeric columns, exactly like at training time —
            # HasCrCard / IsActiveMember / the one-hot columns stay as 0/1.
            input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])
            input_df = input_df[feature_columns]

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            if pred == 1:
                st.error(f"⚠️ Likely to **churn** — model confidence {proba:.0%}")
            else:
                st.success(f"✅ Likely to **stay** — model confidence {1 - proba:.0%}")

            fig = px.bar(x=["Stay", "Churn"], y=[1 - proba, proba],
                         color=["Stay", "Churn"],
                         color_discrete_map={"Stay": "#2ecc71", "Churn": "#e74c3c"})
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%",
                               yaxis_title="Predicted probability", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Model reaches ~86% accuracy / 0.55 F1 on held-out test data — treat this as illustrative, not financial advice.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
with tab_cluster:
    st.subheader("Unsupervised view: PCA + K-Means segments")
    st.caption(
        "Reproduces the notebook's unsupervised section — PCA compresses the "
        "customer features to 2D, then K-Means groups customers into 3 segments."
    )

    X_unsup = df_encoded.drop(columns=[TARGET])
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_unsup)

    k = st.slider("Number of clusters (K-Means)", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": clusters.astype(str),
        "Churned": df_clean[TARGET].map({0: "Stayed", 1: "Churned"}).values,
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", opacity=0.6,
                          title="Customer segments (K-Means on PCA space)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Churned", opacity=0.6,
                          color_discrete_map={"Stayed": "#2ecc71", "Churned": "#e74c3c"},
                          title="Same map, colored by actual churn")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"These 2 components explain {pca.explained_variance_ratio_[:2].sum():.1%} of total variance.")

    st.subheader("Churn rate within each segment")
    cross = (pd.DataFrame({"Cluster": clusters, "Churned": df_clean[TARGET].values})
             .groupby("Cluster")["Churned"].mean().reset_index())
    fig = px.bar(cross, x="Cluster", y="Churned")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("What drives the model's predictions")

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True)
        fig = px.bar(importance, x="importance", y="feature", orientation="h")
        fig.update_layout(yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Held-out performance")
    st.caption("Recomputed on an 80/20 stratified split (`random_state=42`), matching the notebook.")

    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.1%}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.1%}")
    m4.metric("F1 score", f"{f1_score(y_test, y_pred):.1%}")

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, x=["Predicted Stay", "Predicted Churn"],
                         y=["Actual Stay", "Actual Churn"], color_continuous_scale="Blues",
                         title="Confusion matrix")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig = px.line(x=fpr, y=tpr, title=f"ROC curve (AUC = {roc_auc:.2f})",
                       labels={"x": "False Positive Rate", "y": "True Positive Rate"})
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Recall on the churn class is lower than precision — the model is conservative "
        "about flagging churn, catching under half of actual churners. A recall-focused "
        "threshold or class-weighting would trade some precision for catching more at-risk customers."
    )

# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Cleaned dataset")
    st.dataframe(df_clean, use_container_width=True, hide_index=True)
    st.download_button(
        "Download cleaned data as CSV",
        df_clean.to_csv(index=False).encode("utf-8"),
        file_name="churn_cleaned.csv",
        mime="text/csv",
    )
