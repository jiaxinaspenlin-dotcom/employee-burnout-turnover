# Employee Burnout Risk Prediction

Predicting employee burnout risk using a Hugging Face dataset (~850K rows). Includes EDA, feature selection, and model benchmarking with LightGBM tuning.

**Goal:** Identify key drivers of burnout risk and translate findings into actionable retention/workload recommendations.

## Tech Stack
Python, pandas, scikit-learn, LightGBM, matplotlib/seaborn, Hugging Face `datasets`

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

## Notes / Limitations
- The dataset includes synthetic/engineered variables (e.g., `turnover_probability_generated`), which may inflate predictive performance. Results are reported as-is and interpreted with caution.

## Repo Contents
- `Employee_Burnout_Turnover_Prediction.ipynb` — full analysis and modeling workflow
- `requirements.txt` — dependencies

## How to Run
Open the notebook and run top-to-bottom.  
If using Colab, click the badge/link in the notebook header.
