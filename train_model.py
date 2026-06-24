from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.churn_pipeline import build_preprocessor, get_feature_names, split_features_target

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r"C:\Users\Admin\Downloads\churn dataset.csv")
DATA_PATH = ROOT / "data" / "churn_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def load_data() -> pd.DataFrame:
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


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=1)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "threshold": threshold,
        "pipeline": pipeline,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def build_models() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
    return models


def plot_outputs(df: pd.DataFrame, best: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Exited", hue="Exited", palette=["#3a7ca5", "#d1495b"], legend=False)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Exited")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(best["probabilities"], bins=30, color="#3a7ca5")
    plt.axvline(best["threshold"], color="#d1495b", linestyle="--", label=f"Threshold {best['threshold']:.2f}")
    plt.title("Predicted Churn Probability Distribution")
    plt.xlabel("Churn probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "probability_distribution.png", dpi=180)
    plt.close()

    cm = confusion_matrix(y_test, best["predictions"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Retained", "Churn"], yticklabels=["Retained", "Churn"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    pipeline = best["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(names))

    importance = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    top = importance.head(12).sort_values("importance")
    plt.barh(top["feature"], top["importance"], color="#3a7ca5")
    plt.title("Top Churn Drivers")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close()

    try:
        sample = X_test_processed[:200]
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        (OUTPUT_DIR / "shap_summary_unavailable.txt").write_text(str(exc), encoding="utf-8")

    return importance


def write_reports(df: pd.DataFrame, metrics: pd.DataFrame, best: dict, importance: pd.DataFrame, class_report: str) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.select_dtypes(include=["float", "float64"]).columns:
            display[column] = display[column].map(lambda value: f"{value:.3f}")
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
        return "\n".join(lines)

    report = f"""# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Dataset Overview
- Observations: {len(df):,}
- Churn rate: {df['Exited'].mean():.2%}
- Countries covered: {', '.join(sorted(df['Geography'].unique()))}
- Target: `Exited` where 1 means churned and 0 means retained.

## Methodology
The modeling workflow removes non-informative identifiers, engineers relationship-strength features, one-hot encodes categorical variables, and uses a stratified train-test split to preserve the churn class balance. Logistic Regression is retained as an interpretable benchmark, while tree-based and boosting models capture non-linear churn patterns.

Engineered features:
- `BalanceToSalaryRatio`
- `ProductDensity`
- `EngagementProductInteraction`
- `AgeTenureInteraction`

## Model Results

{markdown_table(metrics.drop(columns=['threshold']))}

Selected model: **{best['model']}**

Decision threshold: **{best['threshold']:.3f}**. The threshold is selected from the precision-recall curve to reduce false positives while maintaining useful recall.

## Classification Report

```text
{class_report}
```

## Main Churn Drivers

{markdown_table(importance.head(10))}

## Business Recommendations
- Prioritize customers with high age, high balance exposure, low activity, and limited product engagement for retention actions.
- Use churn probabilities as campaign ranking scores, not only as yes/no decisions.
- Apply differentiated interventions by risk band: service outreach for medium risk, personalized retention offers for high risk.
- Monitor model precision before campaigns to control wasted incentives and customer contact fatigue.

## Regulatory and Governance Notes
- The model excludes direct identifiers such as `CustomerId` and `Surname`.
- Feature importance and SHAP outputs are generated for explainability.
- Thresholds should be reviewed periodically because campaign costs and churn patterns can change over time.
"""
    (ROOT / "research_paper.md").write_text(report, encoding="utf-8")

    executive = f"""# Executive Summary: Bank Customer Churn Risk Intelligence

This project converts historical churn data into a proactive risk-scoring system. The selected model, **{best['model']}**, achieved ROC-AUC of **{best['roc_auc']:.3f}** on the holdout test set and produces a probability score for each customer.

The strongest risk signals are relationship and engagement variables, especially product utilization, activity status, age, balance exposure, and engineered interaction features. This supports a policy direction where banks focus retention efforts on relationship quality rather than broad demographic segmentation.

Recommended actions for stakeholders:
- Use churn risk scores to rank customers for retention campaigns.
- Reserve costly incentives for high-probability churn cases.
- Track false positives through precision to avoid inefficient outreach.
- Require explainability reports before operational deployment.
- Re-train and validate the model quarterly or after major product or policy changes.
"""
    (ROOT / "executive_summary.md").write_text(executive, encoding="utf-8")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    results = []
    for name, model in build_models().items():
        print(f"Training {name}...")
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    best = max(results, key=lambda item: (item["roc_auc"], item["f1"]))
    metrics = pd.DataFrame(
        [
            {k: v for k, v in result.items() if k not in {"pipeline", "probabilities", "predictions"}}
            for result in results
        ]
    ).sort_values("roc_auc", ascending=False)

    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    class_report = classification_report(y_test, best["predictions"], target_names=["Retained", "Churn"], zero_division=0)
    importance = plot_outputs(df, best, X_test, y_test)

    bundle = {
        "pipeline": best["pipeline"],
        "threshold": best["threshold"],
        "model_name": best["model"],
        "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in best.items() if k not in {"pipeline", "probabilities", "predictions"}},
        "feature_importance": importance.to_dict(orient="records"),
    }
    joblib.dump(bundle, MODEL_DIR / "churn_model.joblib")
    (OUTPUT_DIR / "best_model.json").write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    write_reports(df, metrics, best, importance, class_report)
    print(metrics.to_string(index=False))
    print(f"\nBest model: {best['model']} at threshold {best['threshold']:.3f}")


if __name__ == "__main__":
    main()
