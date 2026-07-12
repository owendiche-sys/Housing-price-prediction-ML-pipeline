from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="RealAgents Housing Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Theme
# =========================
BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "rgba(15,23,42,0.68)"
BORDER = "rgba(15,23,42,0.08)"
PRIMARY = "#2563EB"
PRIMARY_STRONG = "#1E40AF"
PRIMARY_SOFT = "rgba(37,99,235,0.12)"
SUCCESS = "#059669"
WARNING = "#D97706"
DANGER = "#DC2626"

pio.templates["realagents_light"] = go.layout.Template(
    layout=go.Layout(
        font=dict(color=TEXT),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        colorway=["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2"],
        xaxis=dict(
            color=TEXT,
            gridcolor="rgba(15,23,42,0.08)",
            zerolinecolor="rgba(15,23,42,0.12)",
        ),
        yaxis=dict(
            color=TEXT,
            gridcolor="rgba(15,23,42,0.08)",
            zerolinecolor="rgba(15,23,42,0.12)",
        ),
        legend=dict(font=dict(color=TEXT)),
        title=dict(font=dict(color=TEXT)),
    )
)
pio.templates.default = "realagents_light"
px.defaults.template = "realagents_light"

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: {BG};
        color: {TEXT};
    }}
    .block-container {{
        padding-top: 1.9rem;
        padding-bottom: 1.5rem;
        max-width: 1460px;
    }}
    #MainMenu, footer {{
        visibility: hidden;
    }}
    section[data-testid="stSidebar"] > div {{
        background: {CARD};
        border-right: 1px solid {BORDER};
    }}
    .stApp :where(p, li, span, label, div, h1, h2, h3, h4, h5, h6),
    :where([data-testid="stMarkdownContainer"]),
    :where([data-testid="stMarkdownContainer"] p),
    :where([data-testid="stWidgetLabel"]),
    :where([data-testid="stWidgetLabel"] p),
    :where([data-testid="stSidebar"] *),
    :where([data-testid="stDataFrame"] *) {{
        color: {TEXT};
    }}
    :where([data-baseweb="select"] *),
    :where([data-baseweb="popover"] *),
    :where([data-baseweb="menu"] *) {{
        color: {TEXT};
    }}
    .stApp span[data-baseweb="tag"],
    .stApp span[data-baseweb="tag"] span,
    .stApp span[data-baseweb="tag"] svg,
    .stApp span[data-baseweb="tag"] [role="presentation"] {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
    [data-testid="stSliderThumbValue"],
    [data-testid="stSliderThumbValue"] * {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
    .stApp div[style*="translate(-50%, -50%)"]:has(> [data-testid="stSliderThumbValue"]) {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
    input,
    textarea,
    [contenteditable="true"] {{
        color: {TEXT} !important;
        background: {CARD} !important;
    }}
    .stAlert,
    .stAlert * {{
        color: {TEXT};
    }}
    .hero {{
        background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(37,99,235,0.04));
        border: 1px solid rgba(37,99,235,0.14);
        border-radius: 24px;
        padding: 22px 22px 20px 22px;
        box-shadow: 0 14px 35px rgba(15,23,42,0.05);
    }}
    .badge {{
        display: inline-block;
        background: {PRIMARY_SOFT};
        color: {PRIMARY_STRONG};
        border: 1px solid rgba(37,99,235,0.14);
        border-radius: 999px;
        padding: 6px 12px;
        font-weight: 800;
        font-size: 12px;
        letter-spacing: 0.02em;
        margin-bottom: 10px;
    }}
    .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 16px 16px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.05);
    }}
    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.05);
        min-height: 122px;
    }}
    .metric-title {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 700;
    }}
    .metric-value {{
        color: {TEXT};
        font-size: 29px;
        font-weight: 800;
        margin-top: 6px;
        line-height: 1.1;
    }}
    .metric-sub {{
        color: {MUTED};
        font-size: 12px;
        margin-top: 8px;
    }}
    .section-title {{
        color: {TEXT};
        font-size: 21px;
        font-weight: 850;
    }}
    .section-sub {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 4px;
    }}
    .insight-item {{
        padding: 9px 0;
        border-bottom: 1px solid rgba(15,23,42,0.06);
    }}
    .insight-item:last-child {{
        border-bottom: none;
    }}
    .small {{
        color: {MUTED};
        font-size: 12px;
    }}
    hr {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 10px 0 14px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# UI helpers
# =========================
def kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card_start(title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">{title}</div>
            {sub}
            <hr>
        """,
        unsafe_allow_html=True,
    )


def section_card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def display_insights(items: list[str], limit: Optional[int] = None) -> None:
    shown = items if limit is None else items[:limit]
    if not shown:
        st.write("No insights available for the current selection.")
        return
    for item in shown:
        st.markdown(f'<div class="insight-item">{item}</div>', unsafe_allow_html=True)


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def pct_change(new_value: float, base_value: float) -> float:
    if base_value == 0 or pd.isna(base_value):
        return np.nan
    return float(((new_value - base_value) / base_value) * 100)


# =========================
# Data loading and cleaning
# =========================
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "Housing.csv"
MISSING_MARKERS = {"--", "-", "---", "—", "–", "na", "n/a", "null", "none", "nan", "missing", ""}


@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def normalize_text(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    s_lower = s.str.lower()
    return s.mask(s_lower.isin(MISSING_MARKERS), pd.NA)


def make_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_data = df.copy()

    clean_data["house_id"] = clean_data["house_id"].astype("string").str.strip()

    clean_data["city"] = normalize_text(clean_data["city"]).str.title().fillna("Unknown").astype("string")

    clean_data["sale_price"] = pd.to_numeric(normalize_text(clean_data["sale_price"]), errors="coerce")
    clean_data = clean_data.dropna(subset=["sale_price"])
    clean_data["sale_price"] = clean_data["sale_price"].round(0).astype(int)

    clean_data["sale_date"] = normalize_text(clean_data["sale_date"]).fillna("2023-01-01")
    clean_data["sale_date"] = pd.to_datetime(clean_data["sale_date"], errors="coerce").fillna(pd.Timestamp("2023-01-01"))
    clean_data["sale_date"] = clean_data["sale_date"].dt.strftime("%Y-%m-%d").astype("string")

    clean_data["months_listed"] = pd.to_numeric(normalize_text(clean_data["months_listed"]), errors="coerce")
    clean_data["months_listed"] = clean_data["months_listed"].fillna(clean_data["months_listed"].mean()).round(1)

    clean_data["bedrooms"] = pd.to_numeric(normalize_text(clean_data["bedrooms"]), errors="coerce")
    clean_data["bedrooms"] = clean_data["bedrooms"].fillna(clean_data["bedrooms"].mean()).round(0).astype(int)

    ht_raw = normalize_text(clean_data["house_type"]).astype("string").str.strip().str.lower()
    ht_raw = ht_raw.str.replace(r"\.+$", "", regex=True)
    ht_raw = ht_raw.str.replace("semi detached", "semi-detached", regex=False)
    ht_raw = ht_raw.str.replace("semidetached", "semi-detached", regex=False)

    ht_map = {
        "detached": "Detached",
        "det": "Detached",
        "semi-detached": "Semi-detached",
        "semi": "Semi-detached",
        "terraced": "Terraced",
        "terr": "Terraced",
    }
    clean_data["house_type"] = ht_raw.map(ht_map)
    mode_ht = clean_data["house_type"].mode(dropna=True)
    fill_ht = mode_ht.iloc[0] if len(mode_ht) else "Terraced"
    clean_data["house_type"] = clean_data["house_type"].fillna(fill_ht).astype("string")

    area_txt = normalize_text(clean_data["area"]).astype("string").str.strip()
    area_num = area_txt.str.replace(",", "", regex=False).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    clean_data["area"] = pd.to_numeric(area_num, errors="coerce")
    clean_data["area"] = clean_data["area"].fillna(clean_data["area"].mean()).round(1)

    clean_data = clean_data[
        ["house_id", "city", "sale_price", "sale_date", "months_listed", "bedrooms", "house_type", "area"]
    ].copy()

    return clean_data


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["sale_date"], errors="coerce")
    out["sale_year"] = dt.dt.year.astype(int)
    out["sale_month"] = dt.dt.month.astype(int)
    out["sale_year_month"] = dt.dt.to_period("M").astype(str)
    out["price_per_sqm"] = out["sale_price"] / out["area"].replace(0, np.nan)
    return out


# =========================
# Modeling helpers
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocess(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
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
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@st.cache_resource(show_spinner=False)
def train_models_cached(df_model: pd.DataFrame):
    target = "sale_price"
    id_col = "house_id"
    feature_cols = ["city", "house_type", "months_listed", "bedrooms", "area", "sale_year", "sale_month"]
    cat_cols = ["city", "house_type"]
    num_cols = ["months_listed", "bedrooms", "area", "sale_year", "sale_month"]

    md = df_model.copy()
    X = md[[id_col] + feature_cols].copy()
    y = md[target].astype(float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocess = make_preprocess(cat_cols, num_cols)

    ridge_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    ridge_model.fit(X_train[feature_cols], y_train)
    ridge_pred = np.clip(ridge_model.predict(X_valid[feature_cols]), 0, None)

    histgb_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.06,
                    max_depth=6,
                    max_iter=600,
                    min_samples_leaf=20,
                    random_state=0,
                ),
            ),
        ]
    )
    histgb_model.fit(X_train[feature_cols], y_train)
    histgb_pred = np.clip(histgb_model.predict(X_valid[feature_cols]), 0, None)

    results = pd.DataFrame(
        {
            "Model": ["Baseline (Ridge)", "Comparison (HistGradientBoosting)"],
            "RMSE": [rmse(y_valid, ridge_pred), rmse(y_valid, histgb_pred)],
            "MAE": [mean_absolute_error(y_valid, ridge_pred), mean_absolute_error(y_valid, histgb_pred)],
            "R²": [r2_score(y_valid, ridge_pred), r2_score(y_valid, histgb_pred)],
        }
    ).sort_values("RMSE").reset_index(drop=True)

    pred_frame = X_valid[["house_id", "city", "house_type", "bedrooms", "area", "months_listed", "sale_year", "sale_month"]].copy()
    pred_frame["actual_price"] = y_valid.values
    pred_frame["ridge_prediction"] = ridge_pred
    pred_frame["histgb_prediction"] = histgb_pred
    pred_frame["ridge_abs_error"] = np.abs(pred_frame["actual_price"] - pred_frame["ridge_prediction"])
    pred_frame["histgb_abs_error"] = np.abs(pred_frame["actual_price"] - pred_frame["histgb_prediction"])

    model_lookup = {
        "Baseline (Ridge)": {
            "pipeline": ridge_model,
            "pred": ridge_pred,
            "rmse": rmse(y_valid, ridge_pred),
            "mae": mean_absolute_error(y_valid, ridge_pred),
            "r2": r2_score(y_valid, ridge_pred),
            "error_col": "ridge_abs_error",
        },
        "Comparison (HistGradientBoosting)": {
            "pipeline": histgb_model,
            "pred": histgb_pred,
            "rmse": rmse(y_valid, histgb_pred),
            "mae": mean_absolute_error(y_valid, histgb_pred),
            "r2": r2_score(y_valid, histgb_pred),
            "error_col": "histgb_abs_error",
        },
    }

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "results": results,
        "pred_frame": pred_frame,
        "model_lookup": model_lookup,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }


@st.cache_data(show_spinner=False)
def permutation_importance_cached(df_model: pd.DataFrame, focus_model: str) -> pd.DataFrame:
    fitted = train_models_cached(df_model)
    X_valid = fitted["X_valid"]
    y_valid = fitted["y_valid"]
    feature_cols = fitted["feature_cols"]
    pipeline = fitted["model_lookup"][focus_model]["pipeline"]

    sample_n = min(len(X_valid), 400)
    X_sample = X_valid[feature_cols].sample(sample_n, random_state=0)
    y_sample = y_valid.loc[X_sample.index]

    perm = permutation_importance(
        pipeline,
        X_sample,
        y_sample,
        n_repeats=8,
        random_state=0,
        scoring="neg_root_mean_squared_error",
    )

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)

    return importance_df.reset_index(drop=True)


# =========================
# Analysis helpers
# =========================
def market_summary(df: pd.DataFrame) -> dict:
    out = {}
    out["records"] = len(df)
    out["avg_price"] = float(df["sale_price"].mean()) if len(df) else np.nan
    out["median_price"] = float(df["sale_price"].median()) if len(df) else np.nan
    out["avg_price_per_sqm"] = float(df["price_per_sqm"].mean()) if len(df) else np.nan
    out["avg_months_listed"] = float(df["months_listed"].mean()) if len(df) else np.nan
    return out


def city_market_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby("city", as_index=False)
        .agg(
            listings=("house_id", "count"),
            avg_price=("sale_price", "mean"),
            median_price=("sale_price", "median"),
            avg_price_per_sqm=("price_per_sqm", "mean"),
            avg_months_listed=("months_listed", "mean"),
        )
        .sort_values(["avg_price", "listings"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return table


def house_type_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby("house_type", as_index=False)
        .agg(
            listings=("house_id", "count"),
            avg_price=("sale_price", "mean"),
            median_price=("sale_price", "median"),
            avg_area=("area", "mean"),
            avg_months_listed=("months_listed", "mean"),
        )
        .sort_values("avg_price", ascending=False)
        .reset_index(drop=True)
    )
    return table


def bedroom_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby("bedrooms", as_index=False)
        .agg(
            listings=("house_id", "count"),
            avg_price=("sale_price", "mean"),
            price_variance=("sale_price", "var"),
            avg_area=("area", "mean"),
        )
        .sort_values("bedrooms")
        .reset_index(drop=True)
    )
    return table


def monthly_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby("sale_year_month", as_index=False)
        .agg(
            listings=("house_id", "count"),
            avg_price=("sale_price", "mean"),
            median_price=("sale_price", "median"),
        )
        .sort_values("sale_year_month")
        .reset_index(drop=True)
    )
    return table


def listing_speed_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(
        df["months_listed"],
        bins=[-0.001, 1, 3, 6, 12, np.inf],
        labels=["Up to 1 month", "1 to 3 months", "3 to 6 months", "6 to 12 months", "Over 12 months"],
    )
    tmp = df.copy()
    tmp["listing_band"] = bins
    table = (
        tmp.groupby("listing_band", as_index=False, observed=True)
        .agg(
            listings=("house_id", "count"),
            avg_price=("sale_price", "mean"),
            avg_price_per_sqm=("price_per_sqm", "mean"),
        )
        .sort_values("listing_band")
        .reset_index(drop=True)
    )
    return table


def generate_data_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    if df.empty:
        return insights

    summary = market_summary(df)
    city_table = city_market_table(df)
    type_table = house_type_table(df)
    bed_table = bedroom_table(df)
    trend_table = monthly_trend_table(df)
    speed_table = listing_speed_table(df)

    insights.append(
        f"The filtered market view contains {summary['records']:,} transactions with an average sale price of {format_money(summary['avg_price'])} "
        f"and a median of {format_money(summary['median_price'])}."
    )

    if not city_table.empty:
        top_city = city_table.iloc[0]
        insights.append(
            f"{top_city['city']} is the highest-priced city in the current view, averaging {format_money(top_city['avg_price'])} "
            f"across {int(top_city['listings']):,} listings."
        )

    if len(type_table) >= 2:
        premium = pct_change(type_table.iloc[0]["avg_price"], type_table.iloc[-1]["avg_price"])
        insights.append(
            f"{type_table.iloc[0]['house_type']} homes command the strongest pricing, with an average price {premium:.1f}% above "
            f"{type_table.iloc[-1]['house_type']} homes in the current selection."
        )

    if not bed_table.empty:
        best_bed = bed_table.loc[bed_table["avg_price"].idxmax()]
        insights.append(
            f"The highest average price by bedroom count is currently in the {int(best_bed['bedrooms'])}-bed segment at "
            f"{format_money(best_bed['avg_price'])}."
        )

    if not trend_table.empty and len(trend_table) > 1:
        recent = trend_table.iloc[-1]
        early = trend_table.iloc[0]
        change = pct_change(recent["avg_price"], early["avg_price"])
        insights.append(
            f"Average sale price moved {change:.1f}% between the earliest and latest months in view, ending at "
            f"{format_money(recent['avg_price'])} in {recent['sale_year_month']}."
        )

    if not speed_table.empty:
        top_speed = speed_table.sort_values("avg_price", ascending=False).iloc[0]
        insights.append(
            f"Listings in the '{top_speed['listing_band']}' time-on-market band have the highest average price at "
            f"{format_money(top_speed['avg_price'])}."
        )

    return insights


def generate_model_insights(df_model: pd.DataFrame, focus_model: str) -> list[str]:
    insights = []
    fitted = train_models_cached(df_model)
    results = fitted["results"]
    pred_frame = fitted["pred_frame"]

    best_row = results.iloc[0]
    selected_metrics = fitted["model_lookup"][focus_model]
    other_model = [m for m in results["Model"].tolist() if m != focus_model][0]
    other_metrics = fitted["model_lookup"][other_model]

    insights.append(
        f"{best_row['Model']} is the strongest holdout model in the current filtered market, with RMSE of {format_money(best_row['RMSE'])}."
    )

    improvement = pct_change(other_metrics["rmse"], selected_metrics["rmse"])
    if np.isfinite(improvement):
        direction = "lower" if selected_metrics["rmse"] < other_metrics["rmse"] else "higher"
        insights.append(
            f"The selected model produces {abs(improvement):.1f}% {direction} RMSE than {other_model}."
        )

    insights.append(
        f"Current validation quality for {focus_model}: RMSE {format_money(selected_metrics['rmse'])}, "
        f"MAE {format_money(selected_metrics['mae'])}, and R² {selected_metrics['r2']:.3f}."
    )

    abs_err = pred_frame[selected_metrics["error_col"]]
    q50, q80, q90 = np.quantile(abs_err, [0.5, 0.8, 0.9])
    insights.append(
        f"Absolute validation error bands for {focus_model} are {format_money(q50)} at the median, "
        f"{format_money(q80)} at the 80th percentile, and {format_money(q90)} at the 90th percentile."
    )

    try:
        importance_df = permutation_importance_cached(df_model, focus_model)
        if not importance_df.empty:
            top_features = ", ".join(importance_df.head(4)["feature"].tolist())
            insights.append(f"The strongest model drivers are {top_features}.")
    except Exception:
        pass

    return insights


# =========================
# Header
# =========================
st.markdown(
    """
    <div class="hero">
      <div class="badge">FEATURED PROJECT</div>
      <div style="font-size:32px; font-weight:900; color:#0F172A; line-height:1.15;">
        RealAgents Housing Analytics
      </div>
      <div style="margin-top:10px; color:rgba(15,23,42,0.72); font-size:15px; max-width:980px;">
        An insight-led housing analytics dashboard for understanding pricing patterns, comparing valuation models,
        and testing estimated sale prices across different property scenarios.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# =========================
# Sidebar
# =========================
st.sidebar.title("Configuration")
page = st.sidebar.radio(
    "View",
    ["Executive Overview", "Market Drivers", "Model Review", "Valuation Lab", "Insights"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Data source")
use_local_file = st.sidebar.checkbox("Load Housing.csv from app folder", value=True)

if use_local_file:
    data_path_value = st.sidebar.text_input("Data file path", value=str(DEFAULT_DATA_PATH.name))
    try:
        raw_df = load_raw(data_path_value)
        source_label = f"Loaded from {data_path_value}"
    except Exception as exc:
        st.error(f"Could not load '{data_path_value}'. {exc}")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file or enable local Housing.csv loading.")
        st.stop()
    raw_df = load_uploaded_csv(uploaded_file)
    source_label = "Loaded from uploaded CSV"

clean_data = make_clean_data(raw_df)
model_data = add_time_features(clean_data)

st.sidebar.divider()
st.sidebar.subheader("Market filters")
city_options = sorted(model_data["city"].dropna().unique().tolist())
type_options = sorted(model_data["house_type"].dropna().unique().tolist())
bed_min = int(model_data["bedrooms"].min())
bed_max = int(model_data["bedrooms"].max())
date_min = pd.to_datetime(model_data["sale_date"]).min()
date_max = pd.to_datetime(model_data["sale_date"]).max()

sel_city = st.sidebar.multiselect("City", options=city_options, default=city_options)
sel_type = st.sidebar.multiselect("House type", options=type_options, default=type_options)
sel_bed = st.sidebar.slider("Bedrooms", min_value=bed_min, max_value=bed_max, value=(bed_min, bed_max))
sel_date = st.sidebar.date_input(
    "Sale date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date(),
)

focus_model = st.sidebar.selectbox(
    "Primary valuation model",
    options=["Comparison (HistGradientBoosting)", "Baseline (Ridge)"],
    index=0,
)

st.sidebar.caption(source_label)

mask = (
    model_data["city"].isin(sel_city)
    & model_data["house_type"].isin(sel_type)
    & model_data["bedrooms"].between(sel_bed[0], sel_bed[1])
    & pd.to_datetime(model_data["sale_date"]).dt.date.between(sel_date[0], sel_date[1])
)
df_f = model_data.loc[mask].copy()

if df_f.empty:
    st.warning("No properties match the current filter selection. Expand the filters to continue.")
    st.stop()

summary = market_summary(df_f)
city_table = city_market_table(df_f)
type_table = house_type_table(df_f)
bed_table = bedroom_table(df_f)
trend_table = monthly_trend_table(df_f)
speed_table = listing_speed_table(df_f)
data_insights = generate_data_insights(df_f)

if len(df_f) < 80:
    st.warning("The current filtered view is too small for stable model evaluation. Expand the filters to unlock the model sections.")
    modeling_available = False
    fitted_models = None
    model_results = pd.DataFrame()
    model_insights = []
else:
    modeling_available = True
    with st.spinner("Refreshing model outputs..."):
        fitted_models = train_models_cached(df_f)
        model_results = fitted_models["results"]
        model_insights = generate_model_insights(df_f, focus_model)


# =========================
# Page: Executive Overview
# =========================
if page == "Executive Overview":
    best_city = city_table.iloc[0]["city"] if not city_table.empty else "Not available"
    best_model = model_results.iloc[0]["Model"] if modeling_available else "Not available"

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Average sale price", format_money(summary["avg_price"]), "Across the current market view")
    with c2:
        kpi_card("Median sale price", format_money(summary["median_price"]), "Middle transaction value in the filtered market")
    with c3:
        kpi_card("Highest-priced city", best_city, "City leading the current market by average sale price")
    with c4:
        kpi_card("Best valuation model", best_model, "Lowest holdout RMSE in the current market view")

    st.write("")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        section_card_start("Executive Summary", "The strongest pricing and model takeaways from the current market view")
        st.markdown("**Data-driven insights**")
        display_insights(data_insights, limit=5)
        if modeling_available:
            st.write("")
            st.markdown("**Model-driven insights**")
            display_insights(model_insights, limit=4)
        section_card_end()

    with right:
        section_card_start("Market Snapshot", "A compact view of valuation quality and market mix")
        snap1, snap2 = st.columns(2)
        with snap1:
            kpi_card("Listings in view", f"{summary['records']:,}", "Transactions remaining after filters")
        with snap2:
            kpi_card("Average price per sq.m.", format_money(summary["avg_price_per_sqm"]), "Pricing intensity across available area")

        st.write("")
        type_mix = df_f["house_type"].value_counts(dropna=False).reset_index()
        type_mix.columns = ["house_type", "listings"]
        fig_mix = px.pie(type_mix, names="house_type", values="listings", hole=0.55)
        fig_mix.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_mix, use_container_width=True)
        section_card_end()

    st.write("")

    a, b = st.columns([1.05, 0.95], gap="large")
    with a:
        section_card_start("Pricing trend", "How average sale price changes across the selected time window")
        if trend_table.empty:
            st.info("Not enough time coverage is available for a pricing trend.")
        else:
            fig_trend = px.line(
                trend_table,
                x="sale_year_month",
                y="avg_price",
                markers=True,
                labels={"sale_year_month": "Sale month", "avg_price": "Average sale price"},
            )
            fig_trend.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_trend, use_container_width=True)
        section_card_end()

    with b:
        section_card_start("City pricing leaderboard", "Cities with the strongest average sale prices in the current view")
        top_city_table = city_table.head(10).iloc[::-1]
        fig_city = px.bar(
            top_city_table,
            x="avg_price",
            y="city",
            orientation="h",
            labels={"avg_price": "Average sale price", "city": "City"},
        )
        fig_city.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_city, use_container_width=True)
        section_card_end()

    if modeling_available:
        st.write("")
        section_card_start("Model comparison snapshot", "The dashboard automatically compares the baseline and non-linear valuation models")
        st.dataframe(model_results, use_container_width=True, hide_index=True)
        section_card_end()


# =========================
# Page: Market Drivers
# =========================
elif page == "Market Drivers":
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Average price per sq.m.", format_money(summary["avg_price_per_sqm"]), "Pricing density across area")
    with c2:
        top_type = type_table.iloc[0]["house_type"] if not type_table.empty else "Not available"
        kpi_card("Highest-priced house type", top_type, "Segment with the highest average sale price")
    with c3:
        fastest_band = speed_table.sort_values("avg_price", ascending=False).iloc[0]["listing_band"] if not speed_table.empty else "Not available"
        kpi_card("Strongest listing band", str(fastest_band), "Time-on-market band with the highest average price")
    with c4:
        kpi_card("Average months listed", f"{summary['avg_months_listed']:.1f}", "Typical time on market in the current view")

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        section_card_start("Price by bedrooms", "Bedroom count remains one of the clearest structural pricing signals")
        fig_bed = px.bar(
            bed_table,
            x="bedrooms",
            y="avg_price",
            text="listings",
            labels={"bedrooms": "Bedrooms", "avg_price": "Average sale price", "listings": "Listings"},
        )
        fig_bed.update_traces(textposition="outside")
        fig_bed.update_layout(height=370, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig_bed, use_container_width=True)
        section_card_end()

    with right:
        section_card_start("Price by house type", "Property type changes both average price and typical property size")
        fig_type = go.Figure()
        fig_type.add_trace(go.Bar(x=type_table["house_type"], y=type_table["avg_price"], name="Average sale price"))
        fig_type.add_trace(go.Bar(x=type_table["house_type"], y=type_table["avg_area"], name="Average area"))
        fig_type.update_layout(
            barmode="group",
            height=370,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="House type",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_type, use_container_width=True)
        section_card_end()

    st.write("")

    a, b = st.columns([1.0, 1.0], gap="large")
    with a:
        section_card_start("Area and price relationship", "Larger homes generally support higher valuations, but the spread also reveals market dispersion")
        fig_scatter = px.scatter(
            df_f,
            x="area",
            y="sale_price",
            color="house_type",
            hover_data=["city", "bedrooms", "months_listed"],
            labels={"area": "Area (sq.m.)", "sale_price": "Sale price"},
            opacity=0.65,
        )
        fig_scatter.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
        section_card_end()

    with b:
        section_card_start("Time on market and pricing", "This view tests whether longer listing duration is associated with different pricing levels")
        fig_speed = px.bar(
            speed_table,
            x="listing_band",
            y="avg_price",
            text="listings",
            labels={"listing_band": "Months listed band", "avg_price": "Average sale price"},
        )
        fig_speed.update_traces(textposition="outside")
        fig_speed.update_layout(height=420, margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Listing duration")
        st.plotly_chart(fig_speed, use_container_width=True)
        section_card_end()

    st.write("")
    section_card_start("City benchmark table", "Useful for comparing pricing, price density, and listing speed across cities")
    st.dataframe(city_table, use_container_width=True, hide_index=True)
    section_card_end()


# =========================
# Page: Model Review
# =========================
elif page == "Model Review":
    if not modeling_available:
        st.info("Expand the filters to include more rows and unlock model review.")
        st.stop()

    pred_frame = fitted_models["pred_frame"].copy()
    selected = fitted_models["model_lookup"][focus_model]
    selected_pred = selected["pred"]
    y_valid = fitted_models["y_valid"]

    try:
        importance_df = permutation_importance_cached(df_f, focus_model)
    except Exception:
        importance_df = pd.DataFrame(columns=["feature", "importance"])

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Selected model RMSE", format_money(selected["rmse"]), "Lower values indicate tighter pricing accuracy")
    with c2:
        kpi_card("Selected model MAE", format_money(selected["mae"]), "Typical absolute valuation gap")
    with c3:
        kpi_card("Selected model R²", f"{selected['r2']:.3f}", "Share of price variance explained")
    with c4:
        improvement_vs_other = pct_change(
            fitted_models["model_lookup"]["Baseline (Ridge)"]["rmse"] if focus_model == "Comparison (HistGradientBoosting)" else fitted_models["model_lookup"]["Comparison (HistGradientBoosting)"]["rmse"],
            selected["rmse"],
        )
        kpi_card("Relative RMSE advantage", f"{improvement_vs_other:.1f}%" if np.isfinite(improvement_vs_other) else "—", "Difference versus the alternative model")

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        section_card_start("Model comparison", "The stronger model should justify itself on more than one metric")
        st.dataframe(model_results, use_container_width=True, hide_index=True)
        section_card_end()

    with right:
        section_card_start("Model-driven insights", "Interpretation of what the validation results mean for pricing decisions")
        display_insights(model_insights, limit=6)
        section_card_end()

    st.write("")

    a, b = st.columns([1.0, 1.0], gap="large")
    with a:
        section_card_start("Actual vs predicted", "A stronger model keeps predictions closer to the diagonal line")
        eval_df = pd.DataFrame(
            {
                "Actual price": y_valid.values,
                "Predicted price": selected_pred,
            }
        )
        fig_eval = px.scatter(eval_df, x="Actual price", y="Predicted price", opacity=0.65)
        min_val = min(eval_df["Actual price"].min(), eval_df["Predicted price"].min())
        max_val = max(eval_df["Actual price"].max(), eval_df["Predicted price"].max())
        fig_eval.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect fit",
            )
        )
        fig_eval.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_eval, use_container_width=True)
        section_card_end()

    with b:
        section_card_start("Absolute error distribution", "This shows how widely the validation pricing errors are spread")
        fig_err = px.histogram(
            pred_frame,
            x=selected["error_col"],
            nbins=30,
            labels={selected["error_col"]: "Absolute error"},
        )
        fig_err.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_err, use_container_width=True)
        section_card_end()

    st.write("")

    c, d = st.columns([1.0, 1.0], gap="large")
    with c:
        section_card_start("Main valuation drivers", "Permutation importance shows which original inputs matter most to the selected model")
        if importance_df.empty:
            st.info("Feature importance could not be computed in this environment.")
        else:
            fig_imp = px.bar(
                importance_df.head(10).iloc[::-1],
                x="importance",
                y="feature",
                orientation="h",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig_imp.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)
        section_card_end()

    with d:
        section_card_start("Most difficult validation cases", "These properties are the strongest candidates for manual review or future feature engineering")
        review_cols = [
            "house_id",
            "city",
            "house_type",
            "bedrooms",
            "area",
            "months_listed",
            "actual_price",
            "ridge_prediction",
            "histgb_prediction",
            selected["error_col"],
        ]
        st.dataframe(
            pred_frame.sort_values(selected["error_col"], ascending=False)[review_cols].head(15),
            use_container_width=True,
            hide_index=True,
        )
        section_card_end()


# =========================
# Page: Valuation Lab
# =========================
elif page == "Valuation Lab":
    if not modeling_available:
        st.info("Expand the filters to include more rows and unlock the valuation lab.")
        st.stop()

    feature_pool = df_f.copy() if len(df_f) >= 20 else model_data.copy()
    city_pool = sorted(feature_pool["city"].dropna().unique().tolist())
    type_pool = sorted(feature_pool["house_type"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        kpi_card("Primary valuation model", focus_model, "Current model used for the lead estimate")
    with c2:
        kpi_card("Filtered market median", format_money(summary["median_price"]), "Useful anchor for comparing scenario output")
    with c3:
        kpi_card("Filtered average price per sq.m.", format_money(summary["avg_price_per_sqm"]), "Useful for judging scenario intensity")

    st.write("")

    section_card_start("Scenario inputs", "Build a property profile and estimate its sale price using both valuation models")
    i1, i2, i3 = st.columns(3, gap="large")
    with i1:
        scenario_city = st.selectbox("City", options=city_pool, index=0)
        scenario_bedrooms = st.slider(
            "Bedrooms",
            min_value=int(feature_pool["bedrooms"].min()),
            max_value=int(feature_pool["bedrooms"].max()),
            value=int(feature_pool["bedrooms"].median()),
        )
    with i2:
        scenario_type = st.selectbox("House type", options=type_pool, index=0)
        scenario_area = st.slider(
            "Area (sq.m.)",
            min_value=float(feature_pool["area"].quantile(0.05)),
            max_value=float(feature_pool["area"].quantile(0.95)),
            value=float(round(feature_pool["area"].median(), 1)),
        )
    with i3:
        scenario_months_listed = st.slider(
            "Months listed",
            min_value=float(feature_pool["months_listed"].quantile(0.05)),
            max_value=float(feature_pool["months_listed"].quantile(0.95)),
            value=float(round(feature_pool["months_listed"].median(), 1)),
        )
        year_values = sorted(feature_pool["sale_year"].unique().tolist())
        month_values = sorted(feature_pool["sale_month"].unique().tolist())
        scenario_year = st.selectbox("Sale year", options=year_values, index=len(year_values) - 1)
        scenario_month = st.selectbox("Sale month", options=month_values, index=min(len(month_values) - 1, 0))
    section_card_end()

    scenario_df = pd.DataFrame(
        [
            {
                "city": scenario_city,
                "house_type": scenario_type,
                "months_listed": float(scenario_months_listed),
                "bedrooms": int(scenario_bedrooms),
                "area": float(scenario_area),
                "sale_year": int(scenario_year),
                "sale_month": int(scenario_month),
            }
        ]
    )

    ridge_price = float(fitted_models["model_lookup"]["Baseline (Ridge)"]["pipeline"].predict(scenario_df)[0])
    histgb_price = float(fitted_models["model_lookup"]["Comparison (HistGradientBoosting)"]["pipeline"].predict(scenario_df)[0])
    selected_price = histgb_price if focus_model == "Comparison (HistGradientBoosting)" else ridge_price
    low_estimate = float(min(ridge_price, histgb_price))
    high_estimate = float(max(ridge_price, histgb_price))
    scenario_price_per_sqm = selected_price / scenario_area if scenario_area else np.nan

    segment_mask = (
        (df_f["city"] == scenario_city)
        & (df_f["house_type"] == scenario_type)
        & (df_f["bedrooms"] == scenario_bedrooms)
    )
    segment_df = df_f.loc[segment_mask]
    segment_median = float(segment_df["sale_price"].median()) if not segment_df.empty else summary["median_price"]
    premium_vs_segment = pct_change(selected_price, segment_median)

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        section_card_start("Scenario valuation result", "The lead estimate uses the primary valuation model, with both model outputs shown for context")
        r1, r2 = st.columns(2)
        with r1:
            kpi_card("Primary estimate", format_money(selected_price), "Lead valuation from the selected model")
        with r2:
            kpi_card("Estimated range", f"{format_money(low_estimate)} to {format_money(high_estimate)}", "Spread across both models")
        st.write("")
        st.write(f"Baseline (Ridge): **{format_money(ridge_price)}**")
        st.write(f"Comparison (HistGradientBoosting): **{format_money(histgb_price)}**")
        st.write(f"Scenario price per sq.m.: **{format_money(scenario_price_per_sqm)}**")
        st.write(f"Difference vs comparable segment median: **{premium_vs_segment:.1f}%**")
        section_card_end()

    with right:
        section_card_start("Valuation gauge", "This compares the primary estimate against the current filtered market median")
        gauge_max = max(summary["median_price"] * 2, high_estimate * 1.15)
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=selected_price,
                number={"prefix": "$", "valueformat": ",.0f"},
                title={"text": "Primary estimated price"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": PRIMARY},
                    "steps": [
                        {"range": [0, summary["median_price"] * 0.9], "color": "rgba(5,150,105,0.18)"},
                        {"range": [summary["median_price"] * 0.9, summary["median_price"] * 1.1], "color": "rgba(217,119,6,0.18)"},
                        {"range": [summary["median_price"] * 1.1, gauge_max], "color": "rgba(220,38,38,0.18)"},
                    ],
                    "threshold": {
                        "line": {"color": DANGER, "width": 4},
                        "thickness": 0.75,
                        "value": summary["median_price"],
                    },
                },
            )
        )
        fig_gauge.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        section_card_end()

    st.write("")

    section_card_start("Scenario interpretation", "This explains how the property profile compares with the current market view")
    scenario_notes = [
        f"The selected model places this property at {format_money(selected_price)}, compared with a filtered market median of {format_money(summary['median_price'])}.",
        f"The estimate implies a price density of {format_money(scenario_price_per_sqm)} per sq.m., versus a filtered market average of {format_money(summary['avg_price_per_sqm'])}.",
        f"Within the matched city, property type, and bedroom segment, the estimate is {premium_vs_segment:.1f}% relative to the observed median benchmark.",
        "The model spread is narrow when both models agree closely and wider when the scenario sits in a less stable or less common part of the market.",
    ]
    display_insights(scenario_notes)
    section_card_end()


# =========================
# Page: Insights
# =========================
elif page == "Insights":
    section_card_start("Insights", "This section starts with data-driven findings and then moves to model-driven interpretation")
    st.markdown("**Data-driven insights**")
    display_insights(data_insights, limit=8)

    st.write("")
    st.markdown("**Model-driven insights**")
    if modeling_available:
        display_insights(model_insights, limit=8)
    else:
        st.info("Expand the filters to include more rows and unlock model-driven insights.")
    section_card_end()

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        section_card_start("Bedroom pricing table", "This preserves the notebook's bedroom aggregation but presents it in a cleaner business format")
        bed_out = bed_table.copy()
        bed_out["avg_price"] = bed_out["avg_price"].round(1)
        bed_out["price_variance"] = bed_out["price_variance"].round(1)
        st.dataframe(bed_out, use_container_width=True, hide_index=True)
        section_card_end()

    with right:
        section_card_start("Model comparison table", "This preserves the notebook's baseline versus comparison model story")
        if modeling_available:
            st.dataframe(model_results, use_container_width=True, hide_index=True)
        else:
            st.info("Expand the filters to unlock the model comparison table.")
        section_card_end()
