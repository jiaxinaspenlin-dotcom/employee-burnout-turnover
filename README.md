# Employee Burnout Risk Prediction

Predicting employee burnout risk using a Hugging Face dataset (~850K rows). Includes EDA, feature selection, and model benchmarking with LightGBM tuning.

## Dataset
- Source: Hugging Face (`BrotherTony/employee-burnout-turnover-prediction-800k`)
- Size: ~850,000 rows, 31 columns
- Target: `burnout_risk` (regression)

## Approach
- Data cleaning + exploratory analysis (distributions, correlations)
- Feature selection (SelectKBest, Lasso, tree-based methods)
- Models: Linear Regression, tree-based baselines, LightGBM
- Hyperparameter tuning: GridSearchCV

## Key Results
- Baseline Linear Regression: R² ≈ 0.678
- Tuned LightGBM: R² ≈ 0.994, RMSE ≈ 0.025

## How to Run
Open the notebook and run top-to-bottom.  
If using Colab, click the badge/link in the notebook header. 
