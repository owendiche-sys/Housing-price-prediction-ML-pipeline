# Housing Price Prediction ML Pipeline

Portfolio project for cleaning housing transaction data, exploring price drivers, and comparing baseline versus non-linear regression models for sale price prediction.

## Project Snapshot

RealAgents wants more accurate listing prices so agents can reduce time-to-sale and identify properties that need manual valuation review. This repository turns a raw housing dataset into:

- a cleaned analysis table with consistent dates, categories, numeric areas, and imputed missing values
- exploratory pricing summaries by bedroom count, city, property type, listing duration, and sale month
- a reproducible model comparison between Ridge Regression and Histogram Gradient Boosting
- an interactive Streamlit dashboard for market exploration, model review, and scenario valuation

## Repository Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard with market filters, model diagnostics, feature importance, and a valuation lab. |
| `Housing-price-prediction-ML-pipeline.ipynb` | Notebook version of the original data cleaning and modeling workflow. |
| `Housing.csv` | Source housing transactions used by the notebook and dashboard. |
| `scripts/validate_model.py` | Command-line validation script that retrains both models and prints holdout metrics. |
| `requirements.txt` | Python dependencies for the app and validation workflow. |

## Dataset

The project uses `Housing.csv`.

Key fields:

- `house_id`: property identifier
- `city`: market location
- `sale_price`: model target
- `sale_date`: transaction date
- `months_listed`: time on market
- `bedrooms`: room count
- `house_type`: detached, semi-detached, or terraced
- `area`: property area, stored as text and cleaned into square metres

## Modeling Approach

The pipeline applies deterministic cleaning rules before modeling:

- standardizes text fields and placeholder missing values
- fills missing city values as `Unknown`
- imputes missing dates with `2023-01-01`
- imputes numeric columns with mean values where required
- normalizes house type labels such as `Det.` and `Semi Detached`
- extracts numeric area values from strings like `107.8 sq.m.`
- drops rows without a valid target price

Two models are evaluated on a fixed 80/20 holdout split:

- **Baseline:** Ridge Regression with one-hot encoded categorical features and scaled numeric features
- **Comparison:** Histogram Gradient Boosting Regressor using the same engineered feature set

The dashboard reports RMSE, MAE, R², absolute error distributions, difficult validation cases, and permutation importance.

## Streamlit Dashboard

Run the dashboard locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard views:

- **Executive Overview:** headline market KPIs, pricing trend, city leaderboard, and model snapshot
- **Market Drivers:** bedroom, house type, area, listing-duration, and city benchmark analysis
- **Model Review:** Ridge versus HistGradientBoosting diagnostics and validation error review
- **Valuation Lab:** scenario-based price estimates for custom property profiles
- **Insights:** generated takeaways from the current filter selection

## Reproduce Model Metrics

Run the validation script:

```bash
python scripts/validate_model.py
```

Expected output includes row counts plus holdout metrics for both models. On the current dataset, Histogram Gradient Boosting should outperform the Ridge baseline on RMSE.

Example metrics from the current data and fixed split:

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Ridge Regression | about 21,500 | reported by script | reported by script |
| Histogram Gradient Boosting | about 15,950 | reported by script | reported by script |

## Portfolio Notes

This project demonstrates:

- practical tabular data cleaning for messy business data
- interpretable baseline modeling before using a stronger non-linear estimator
- model evaluation beyond a single score
- an applied analytics interface that turns a notebook workflow into a usable decision-support tool

## Next Improvements

- add cross-validation and hyperparameter tuning for the gradient boosting model
- persist trained model artifacts for faster deployment startup
- add SHAP or partial dependence plots for deeper interpretation
- package the cleaning and modeling code into reusable modules shared by the notebook, app, and validation script
