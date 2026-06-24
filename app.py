from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.churn_pipeline import MODEL_INPUT_FEATURES, assign_risk_band, build_preprocessor, get_feature_names, split_features_target


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DATA_PATH = ROOT / "data" / "churn_dataset.csv"
DEFAULT_DATA = Path(r"C:\Users\Admin\Downloads\churn dataset.csv")


@st.cache_data
def load_data():
    path = DATA_PATH if DATA_PATH.exists() else DEFAULT_DATA
    return pd.read_csv(path)


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    thresholds = np.r_[thresholds, 1.0]
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    feasible = np.where(precision >= 0.60)[0]
    if len(feasible):
        return float(thresholds[feasible[np.argmax(f1[feasible])]])
    return float(thresholds[np.argmax(f1)])


def train_cloud_model(df: pd.DataFrame) -> dict:
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    predictions = (probabilities >= threshold).astype(int)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": get_feature_names(preprocessor),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "model_name": "Gradient Boosting",
        "feature_importance": importance.to_dict(orient="records"),
        "metrics": {
            "model": "Gradient Boosting",
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "threshold": threshold,
        },
    }


@st.cache_resource
def load_model(_df: pd.DataFrame):
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH), "Loaded saved model artifact."
    except Exception as exc:
        return train_cloud_model(_df), f"Rebuilt model from CSV because the saved artifact could not be loaded: {exc.__class__.__name__}."
    return train_cloud_model(_df), "Rebuilt model from CSV because no saved artifact was found."


def predict_probability(bundle: dict, row: dict) -> float:
    frame = pd.DataFrame([row], columns=MODEL_INPUT_FEATURES)
    return float(bundle["pipeline"].predict_proba(frame)[:, 1][0])


st.set_page_config(page_title="Bank Churn Risk Scoring", layout="wide")
st.title("Bank Customer Churn Risk Scoring")

df = load_data()
bundle, model_status = load_model(df)
threshold = bundle["threshold"]
st.caption(model_status)

left, right = st.columns([0.34, 0.66])

with left:
    st.subheader("Customer Inputs")
    credit_score = st.slider("Credit score", 300, 900, 650, 1)
    geography = st.selectbox("Geography", sorted(df["Geography"].unique()))
    gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
    age = st.slider("Age", 18, 95, 40, 1)
    tenure = st.slider("Tenure", 0, 10, 3, 1)
    balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
    num_products = st.slider("Number of products", 1, 4, 1, 1)
    has_card = st.toggle("Has credit card", value=True)
    active = st.toggle("Active member", value=True)
    salary = st.number_input("Estimated salary", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)

customer = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": int(has_card),
    "IsActiveMember": int(active),
    "EstimatedSalary": salary,
}

probability = predict_probability(bundle, customer)
flag = int(probability >= threshold)
risk_band = assign_risk_band(probability)

with right:
    st.subheader("Risk Score")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn probability", f"{probability:.1%}")
    m2.metric("Risk band", risk_band)
    m3.metric("Churn flag", "Yes" if flag else "No")
    m4.metric("Model", bundle["model_name"])

    st.progress(min(probability, 1.0))

    st.subheader("What-If Scenario Simulator")
    scenario = customer.copy()
    c1, c2, c3 = st.columns(3)
    scenario["IsActiveMember"] = int(c1.toggle("Scenario active member", value=bool(active)))
    scenario["NumOfProducts"] = c2.slider("Scenario products", 1, 4, num_products, 1)
    scenario["Balance"] = c3.number_input("Scenario balance", min_value=0.0, max_value=300000.0, value=float(balance), step=1000.0)
    scenario_probability = predict_probability(bundle, scenario)
    delta = scenario_probability - probability
    st.metric("Scenario churn probability", f"{scenario_probability:.1%}", delta=f"{delta:+.1%}")

tab1, tab2, tab3 = st.tabs(["Probability Distribution", "Feature Importance", "Model Performance"])

with tab1:
    probabilities = bundle["pipeline"].predict_proba(df[MODEL_INPUT_FEATURES])[:, 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probabilities, bins=30, color="#3a7ca5", edgecolor="white")
    ax.axvline(threshold, color="#d1495b", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Customers")
    ax.legend()
    st.pyplot(fig)

with tab2:
    importance = pd.DataFrame(bundle["feature_importance"]).head(15)
    st.dataframe(importance, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = importance.sort_values("importance")
    ax.barh(ordered["feature"], ordered["importance"], color="#3a7ca5")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

with tab3:
    st.dataframe(pd.DataFrame([bundle["metrics"]]).drop(columns=["pipeline"], errors="ignore"), use_container_width=True)
    st.caption("Metrics are computed on a stratified 20% holdout set.")
