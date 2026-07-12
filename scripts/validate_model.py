from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Housing.csv"
MISSING_MARKERS = {"--", "-", "---", "na", "n/a", "null", "none", "nan", "missing", ""}


def normalize_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return text.mask(text.str.lower().isin(MISSING_MARKERS), pd.NA)


def clean_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["house_id"] = clean["house_id"].astype("string").str.strip()
    clean["city"] = normalize_text(clean["city"]).str.title().fillna("Unknown").astype("string")

    clean["sale_price"] = pd.to_numeric(normalize_text(clean["sale_price"]), errors="coerce")
    clean = clean.dropna(subset=["sale_price"])
    clean["sale_price"] = clean["sale_price"].round(0).astype(int)

    clean["sale_date"] = normalize_text(clean["sale_date"]).fillna("2023-01-01")
    sale_date = pd.to_datetime(clean["sale_date"], errors="coerce").fillna(pd.Timestamp("2023-01-01"))
    clean["sale_year"] = sale_date.dt.year.astype(int)
    clean["sale_month"] = sale_date.dt.month.astype(int)

    clean["months_listed"] = pd.to_numeric(normalize_text(clean["months_listed"]), errors="coerce")
    clean["months_listed"] = clean["months_listed"].fillna(clean["months_listed"].mean()).round(1)

    clean["bedrooms"] = pd.to_numeric(normalize_text(clean["bedrooms"]), errors="coerce")
    clean["bedrooms"] = clean["bedrooms"].fillna(clean["bedrooms"].mean()).round(0).astype(int)

    house_type = normalize_text(clean["house_type"]).astype("string").str.lower()
    house_type = house_type.str.replace(r"\.+$", "", regex=True)
    house_type = house_type.str.replace("semi detached", "semi-detached", regex=False)
    house_type = house_type.str.replace("semidetached", "semi-detached", regex=False)
    house_type_map = {
        "detached": "Detached",
        "det": "Detached",
        "semi-detached": "Semi-detached",
        "semi": "Semi-detached",
        "terraced": "Terraced",
        "terr": "Terraced",
    }
    clean["house_type"] = house_type.map(house_type_map)
    clean["house_type"] = clean["house_type"].fillna(clean["house_type"].mode().iloc[0]).astype("string")

    area_text = normalize_text(clean["area"]).astype("string")
    area_numeric = area_text.str.replace(",", "", regex=False).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    clean["area"] = pd.to_numeric(area_numeric, errors="coerce")
    clean["area"] = clean["area"].fillna(clean["area"].mean()).round(1)

    return clean[
        [
            "house_id",
            "city",
            "sale_price",
            "months_listed",
            "bedrooms",
            "house_type",
            "area",
            "sale_year",
            "sale_month",
        ]
    ].copy()


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(model) -> Pipeline:
    cat_cols = ["city", "house_type"]
    num_cols = ["months_listed", "bedrooms", "area", "sale_year", "sale_month"]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    clean = clean_housing_data(raw)

    feature_cols = ["city", "house_type", "months_listed", "bedrooms", "area", "sale_year", "sale_month"]
    X = clean[feature_cols]
    y = clean["sale_price"].astype(float)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=0),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_depth=6,
            max_iter=600,
            min_samples_leaf=20,
            random_state=0,
        ),
    }

    print(f"Loaded rows: {len(raw):,}")
    print(f"Clean rows used for modeling: {len(clean):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print()
    print(f"{'Model':<30} {'RMSE':>12} {'MAE':>12} {'R2':>8}")
    print("-" * 66)

    for name, model in models.items():
        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)
        pred = np.clip(pipeline.predict(X_valid), 0, None)
        print(
            f"{name:<30} "
            f"{rmse(y_valid, pred):>12,.0f} "
            f"{mean_absolute_error(y_valid, pred):>12,.0f} "
            f"{r2_score(y_valid, pred):>8.3f}"
        )


if __name__ == "__main__":
    main()
