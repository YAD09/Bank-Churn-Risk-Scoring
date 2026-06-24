from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Exited"
DROP_COLUMNS = ["CustomerId", "Surname", "Year"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "BalanceToSalaryRatio",
    "ProductDensity",
    "EngagementProductInteraction",
    "AgeTenureInteraction",
]
MODEL_INPUT_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add churn-risk features used consistently during training and scoring."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        salary = X["EstimatedSalary"].replace(0, np.nan)
        X["BalanceToSalaryRatio"] = (X["Balance"] / salary).replace([np.inf, -np.inf], np.nan).fillna(0)
        X["ProductDensity"] = X["NumOfProducts"] / (X["Tenure"] + 1)
        X["EngagementProductInteraction"] = X["IsActiveMember"] * X["NumOfProducts"]
        X["AgeTenureInteraction"] = X["Age"] * X["Tenure"]
        return X


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    usable = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    X = usable.drop(columns=[TARGET])
    y = usable[TARGET]
    return X, y


def build_preprocessor(scale_numeric: bool = False) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformer = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", FeatureEngineer()), ("preprocess", transformer)])


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    return list(preprocessor.named_steps["preprocess"].get_feature_names_out())


def assign_risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"
