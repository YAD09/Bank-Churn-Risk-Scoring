# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Dataset Overview
- Observations: 10,000
- Churn rate: 20.37%
- Countries covered: France, Germany, Spain
- Target: `Exited` where 1 means churned and 0 means retained.

## Methodology
The modeling workflow removes non-informative identifiers, engineers relationship-strength features, one-hot encodes categorical variables, and uses a stratified train-test split to preserve the churn class balance. Logistic Regression is retained as an interpretable benchmark, while tree-based and boosting models capture non-linear churn patterns.

Engineered features:
- `BalanceToSalaryRatio`
- `ProductDensity`
- `EngagementProductInteraction`
- `AgeTenureInteraction`

## Model Results

| model | accuracy | precision | recall | f1 | roc_auc | cv_roc_auc_mean | cv_roc_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 0.845 | 0.603 | 0.690 | 0.644 | 0.867 | 0.862 | 0.007 |
| XGBoost | 0.857 | 0.664 | 0.602 | 0.631 | 0.866 | 0.861 | 0.006 |
| Random Forest | 0.847 | 0.614 | 0.663 | 0.638 | 0.864 | 0.851 | 0.008 |
| Decision Tree | 0.841 | 0.647 | 0.477 | 0.549 | 0.821 | 0.828 | 0.004 |
| Logistic Regression | 0.807 | 0.602 | 0.152 | 0.243 | 0.776 | 0.766 | 0.013 |

Selected model: **Gradient Boosting**

Decision threshold: **0.265**. The threshold is selected from the precision-recall curve to reduce false positives while maintaining useful recall.

## Classification Report

```text
              precision    recall  f1-score   support

    Retained       0.92      0.88      0.90      1593
       Churn       0.60      0.69      0.64       407

    accuracy                           0.84      2000
   macro avg       0.76      0.79      0.77      2000
weighted avg       0.85      0.84      0.85      2000

```

## Main Churn Drivers

| feature | importance |
| --- | --- |
| Age | 0.387 |
| NumOfProducts | 0.290 |
| EngagementProductInteraction | 0.078 |
| Balance | 0.062 |
| Geography_Germany | 0.054 |
| IsActiveMember | 0.043 |
| BalanceToSalaryRatio | 0.024 |
| CreditScore | 0.018 |
| EstimatedSalary | 0.015 |
| AgeTenureInteraction | 0.009 |

## Business Recommendations
- Prioritize customers with high age, high balance exposure, low activity, and limited product engagement for retention actions.
- Use churn probabilities as campaign ranking scores, not only as yes/no decisions.
- Apply differentiated interventions by risk band: service outreach for medium risk, personalized retention offers for high risk.
- Monitor model precision before campaigns to control wasted incentives and customer contact fatigue.

## Regulatory and Governance Notes
- The model excludes direct identifiers such as `CustomerId` and `Surname`.
- Feature importance and SHAP outputs are generated for explainability.
- Thresholds should be reviewed periodically because campaign costs and churn patterns can change over time.
