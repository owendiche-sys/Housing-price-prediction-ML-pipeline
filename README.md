# \# RealAgents Housing Price Prediction

# 

# \## Author: Owen Nda Diche

# 

# \## Project overview

# RealAgents operates in a metropolitan area and wants to reduce time-to-sale by setting more accurate listing prices. This project cleans historical sales data, explores price patterns, and trains two regression models to predict sale price.

# 

# \## Dataset

# \*\*File:\*\* `Housing.csv`  

# \*\*Target:\*\* `sale\_price`  

# \*\*Key features:\*\*

# \- `city` (categorical)

# \- `sale\_date` (date)

# \- `months\_listed` (numeric)

# \- `bedrooms` (numeric)

# \- `house\_type` (categorical/ordinal)

# \- `area` (numeric extracted from text like `"107.8 sq.m."`)

# 

# \## Tasks implemented (1–5)

# \### Task 1 — Missing value audit

# \- Computed the number of missing `city` values.

# \- In this dataset, missing cities are represented as `"--"`.

# 

# \### Task 2 — Data cleaning (rule-based)

# Created `clean\_data` by applying strict cleaning rules:

# \- Standardized text fields (trim whitespace, collapse spaces, normalize case)

# \- Treated common placeholder values (e.g., `"--"`, `"NA"`) as missing

# \- Applied rule-based imputation:

# &nbsp; - `city` → `"Unknown"`

# &nbsp; - `sale\_date` → `"2023-01-01"` if missing

# &nbsp; - `months\_listed` → mean (rounded to 1 dp)

# &nbsp; - `bedrooms` → mean (rounded to nearest int)

# &nbsp; - `house\_type` → mode (standardized to `Terraced`, `Semi-detached`, `Detached`)

# &nbsp; - `area` → extracted numeric value from strings (then mean imputation, 1 dp)

# \- Removed records with missing `sale\_price` and cast it to whole dollars (int)

# 

# \### Task 3 — Grouped aggregation

# Created `price\_by\_rooms` to analyze pricing by bedroom count:

# \- `avg\_price`: mean sale price per bedroom group (1 dp)

# \- `var\_price`: variance of sale price per bedroom group (1 dp)

# 

# \### Task 4 — Baseline model

# \- Trained a \*\*Ridge Regression\*\* model using one-hot encoding for categoricals.

# \- Produced `base\_result` with `house\_id` and predicted `price`.

# 

# \### Task 5 — Comparison model

# \- Trained a \*\*Histogram Gradient Boosting Regressor\*\* (non-linear model).

# \- Produced `compare\_result` with `house\_id` and predicted `price`.

# 

# \## Results

# Validation RMSE (lower is better):

# \- \*\*Baseline (Ridge):\*\* ~ \*\*21,464\*\*

# \- \*\*Comparison (HistGB):\*\* ~ \*\*16,598\*\*

# 

# The boosted model improved predictive accuracy substantially, suggesting non-linear relationships and feature interactions are important for pricing.

# 

# \## Key insights

# \- \*\*Bedrooms are a strong driver\*\* of average price, but high variance within groups indicates other factors matter.

# \- \*\*City, house type, and area\*\* provide additional predictive signal beyond bedrooms.

# \- \*\*Gradient boosting outperforms linear regression\*\*, supporting the use of non-linear models for real estate pricing.

# 

# \## How to run

# 1\. Place `Housing.csv` in the same folder as the notebook.

# 2\. Open the notebook and run cells top-to-bottom.

# 3\. Outputs generated:

# &nbsp;  - `missing\_city`

# &nbsp;  - `clean\_data`

# &nbsp;  - `price\_by\_rooms`

# &nbsp;  - `base\_result`

# &nbsp;  - `compare\_result`

# &nbsp;  - RMSE comparison table

# 

# \## Next improvements

# \- Add cross-validation and hyperparameter tuning for the boosted model.

# \- Engineer richer time features (e.g., quarter, seasonal indicators).

# \- Add interpretation (permutation importance or SHAP) for feature impact and pricing drivers.



