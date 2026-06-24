# Predictive Modeling and Risk Scoring for Bank Customer Churn

This project builds a churn prediction and risk-scoring system for retail bank customers. It includes model training, explainability outputs, stakeholder summaries, and a Streamlit dashboard for individual risk scoring and what-if simulation.

## Project Structure

- `train_model.py` trains and evaluates Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost when available.
- `app.py` runs the Streamlit churn risk calculator and dashboard.
- `src/churn_pipeline.py` contains shared feature engineering and preprocessing logic.
- `outputs/` stores metrics and plots generated during training.
- `models/` stores the trained model artifact.
- `research_paper.md` and `executive_summary.md` are generated after training.

## Run

```powershell
python train_model.py
streamlit run app.py
```

The training script reads `data/churn_dataset.csv` if present. Otherwise, it uses `C:\Users\Admin\Downloads\churn dataset.csv`.

## Methodology

The workflow uses stratified train-test splitting, categorical one-hot encoding, numeric preprocessing, and engineered features:

- Balance-to-salary ratio
- Product density
- Engagement-product interaction
- Age-tenure interaction

The selected model is saved with a threshold chosen from the precision-recall curve to reduce false positives while preserving useful churn detection.
