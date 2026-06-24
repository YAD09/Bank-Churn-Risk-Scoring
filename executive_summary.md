# Executive Summary: Bank Customer Churn Risk Intelligence

This project converts historical churn data into a proactive risk-scoring system. The selected model, **Gradient Boosting**, achieved ROC-AUC of **0.867** on the holdout test set and produces a probability score for each customer.

The strongest risk signals are relationship and engagement variables, especially product utilization, activity status, age, balance exposure, and engineered interaction features. This supports a policy direction where banks focus retention efforts on relationship quality rather than broad demographic segmentation.

Recommended actions for stakeholders:
- Use churn risk scores to rank customers for retention campaigns.
- Reserve costly incentives for high-probability churn cases.
- Track false positives through precision to avoid inefficient outreach.
- Require explainability reports before operational deployment.
- Re-train and validate the model quarterly or after major product or policy changes.
