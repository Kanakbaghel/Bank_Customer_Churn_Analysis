<p align="center">
  <img src="https://img.shields.io/badge/Python-Pandas%20%7C%20Scikit--learn-3776AB?logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit Badge" />
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Completed Badge" />
  <img src="https://img.shields.io/badge/Model-Random%20Forest-informational" alt="Model Badge" />
</p>

<h1 align="center">🏦 Bank Customer Churn Analysis</h1>
<p align="center"><em>Graded Mini Project · Data Science & Business Analytics Program by IIT Guwahati (Emeritus)</em></p>

<p align="center">
  <a href="https://bankcustomerchurnanalysis-kb.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Launch_App-1f6feb?style=for-the-badge" alt="Live Demo" />
  </a>
</p>

---

## 📖 Introduction

In the highly competitive financial sector, customer retention remains a critical
determinant of sustained profitability. A prominent international bank is currently
facing elevated customer churn rates, with clients discontinuing accounts despite
access to a diverse portfolio of financial products. This concerning trend poses
significant risks to long-term growth and brand loyalty.

This project systematically addresses the objective end-to-end: **data cleaning &
preprocessing**, **exploratory data analysis (EDA)** to identify churn patterns,
**predictive modeling** for churn forecasting, and extraction of **actionable business
insights** — all deployed as an interactive Streamlit app.

The dataset encompasses customer demographics, account information, and behavioral
attributes, featuring a binary target variable `Exited` (1 = churned, 0 = retained).

---

## 📑 Table of Contents

1. [Project Objectives](#-project-objectives)
2. [Project Structure](#-project-structure)
3. [Libraries Utilized](#-libraries-utilized)
4. [Data Cleaning](#-data-cleaning)
5. [Key EDA Findings](#-key-eda-findings)
6. [Unsupervised Segmentation](#-unsupervised-segmentation)
7. [Model Performance](#-model-performance)
8. [Streamlit App](#️-streamlit-app)
9. [Getting Started](#-getting-started)

---

## 🎯 Project Objectives

As a data analyst, the primary goals are to:

- Examine customer data to identify pivotal churn drivers, such as demographic profiles and account usage patterns.
- Develop and rigorously evaluate predictive models to classify customers as likely to retain or depart.
- Generate strategic recommendations to support targeted retention initiatives, including personalized incentives and proactive interventions for high-risk segments.

---

## 📂 Project Structure

```
Bank_Customer_Churn_Analysis/
├── App/
│   ├── churn_app.py          # Streamlit app — EDA, prediction, clustering, model insights
│   └── data_utils.py          # Shared load/clean/encode logic (notebook + app)
├── Data/
│   └── Data.csv                # Raw customer records
├── Model/
│   └── churn_model.pkl          # Trained Random Forest + scaler + feature list
├── Project_Notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 🧰 Libraries Utilized

The analysis leverages the following key libraries to facilitate a robust end-to-end workflow, from data ingestion to insightful model evaluation:

- **pandas** — data manipulation, cleaning, and analysis
- **numpy** — numerical computations and array operations
- **matplotlib & seaborn** — visualizations and exploratory plots
- **scikit-learn** — preprocessing, model training, evaluation metrics (ROC curves, confusion matrices), PCA & K-Means
- **scipy** — advanced statistical functions, including hierarchical clustering for unsupervised customer segmentation
- **streamlit & plotly** — the deployed interactive app

---

## 🧹 Data Cleaning

- Dropped non-predictive identifiers: `CustomerId`, `Surname`
- Imputed missing values: `Geography`/`Gender` → mode, `Age` → median, `HasCrCard`/`IsActiveMember` → mode
- Removed 502 duplicate records (10,502 → 10,000 rows)
- One-hot encoded `Geography` and `Gender` (`drop_first=True`)
- Standard-scaled the 6 numeric features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`) — binary flags and one-hot columns were left unscaled, matching how the model was trained

---

## 🔍 Key EDA Findings

- **Zero-balance customers are common** — a large share of customers carry a €0 balance (checking-only accounts), and this segment shows a distinct churn profile worth targeting separately in retention campaigns.
- **Overall churn rate is ~20%** — a moderately imbalanced target, which shaped the choice of evaluation metrics (precision/recall/F1 over raw accuracy).
- Churn skews by **geography** and **number of products held** — customers with only 1 product churn noticeably more than those with 2.
- ⚠️ Note on the age distribution: imputing missing `Age` values with the median creates an artificial spike at that single value in histograms — this is an artifact of imputation, not a genuine behavioral pattern, and is worth calling out when presenting the EDA.

---

## 🧬 Unsupervised Segmentation

Beyond supervised prediction, the notebook explores customer segmentation using
**PCA** for dimensionality reduction and both **K-Means** and **hierarchical
clustering** to group customers by behavioral similarity — surfacing segments with
meaningfully different churn rates.

<p align="center">
  <img width="878" height="474" alt="Hierarchical clustering dendrogram" src="https://github.com/user-attachments/assets/5b628cb6-cba4-4184-a00a-e6383df2f20d" />
</p>
<p align="center"><strong>— Hierarchical Clustering —</strong></p>

---

## 🤖 Model Performance

A **Random Forest Classifier** was selected as the final model, benchmarked against Logistic Regression:

| Metric | Logistic Regression | Random Forest |
|--------|:---:|:---:|
| Accuracy | 81.2% | **85.7%** |
| Precision | 61.4% | **76.1%** |
| Recall | 21.1% | **43.0%** |
| F1 score | 31.4% | **54.9%** |

Recall on the churn class is the weakest metric for both models — a known challenge
with imbalanced churn data. The app's Model Insights tab surfaces the confusion
matrix and ROC curve live, so this trade-off is transparent rather than hidden.

---

## 🖥️ Streamlit App

**[🚀 Try the live app →](https://bankcustomerchurnanalysis-kb.streamlit.app/)**

| Tab | What it shows |
|-----|----------------|
| 📋 Overview | Headline KPIs + churn mix |
| 📊 EDA | Churn rate by geography/gender/products, age & balance distributions, correlation heatmap |
| 🔮 Predict Churn | Enter a customer's details and get a live churn prediction with probability |
| 🧬 Clustering | Interactive PCA + K-Means segmentation, adjustable number of clusters |
| 🧠 Model Insights | Feature importance, confusion matrix, ROC curve, held-out metrics |
| 🗂️ Raw Data | Browse and download the cleaned dataset |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Kanakbaghel/Bank_Customer_Churn_Analysis.git
cd Bank_Customer_Churn_Analysis
pip install -r requirements.txt
streamlit run App/churn_app.py
```

---

> *"Data becomes meaningful when it tells a story that leads to better decisions."*

<p align="center"><em>Crafted with ♥ by <strong>Kanak Baghel</strong> | <a href="https://www.linkedin.com/in/kanakbaghel">LinkedIn</a> | <a href="https://github.com/Kanakbaghel">GitHub</a></em></p>
