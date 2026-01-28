import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="RealAgents Housing Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Styling
# =========================
CARD_CSS = """
<style>
:root {
  --card-bg: #ffffff;
  --card-bd: rgba(0,0,0,0.08);
  --card-shadow: 0 6px 16px rgba(0,0,0,0.06);
  --muted: rgba(0,0,0,0.55);
}
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: var(--muted); font-size: 0.92rem; }

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px)  { .kpi-grid { grid-template-columns: 1fr; } }

.card {
  background: var(--card-bg);
  border: 1px solid var(--card-bd);
  border-radius: 16px;
  box-shadow: var(--card-shadow);
  padding: 14px 16px;
}
.kpi-label { font-size: 0.82rem; color: var(--muted); margin-bottom: 4px; }
.kpi-value { font-size: 1.35rem; font-weight: 800; line-height: 1.1; }
.kpi-sub { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }

.section-grid {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 12px;
}
@media (max-width: 1100px) { .section-grid { grid-template-columns: 1fr; } }

hr.soft {
  border: none;
  border-top: 1px solid rgba(0,0,0,0.08);
  margin: 12px 0;
}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)


# =========================
# Cleaning utilities 
# =========================
MISSING_MARKERS = {"--", "-", "---", "—", "–", "na", "n/a", "null", "none", "nan", "missing", ""}


def normalize_text(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    s_lower = s.str.lower()
    return s.mask(s_lower.isin(MISSING_MARKERS), pd.NA)


@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def make_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_data = df.copy()

    # house_id
    clean_data["house_id"] = clean_data["house_id"].astype("string").str.strip()

    # city
    clean_data["city"] = normalize_text(clean_data["city"]).str.title().fillna("Unknown").astype("string")

    # sale_price (drop missing)
    clean_data["sale_price"] = pd.to_numeric(normalize_text(clean_data["sale_price"]), errors="coerce")
    clean_data = clean_data.dropna(subset=["sale_price"])
    clean_data["sale_price"] = clean_data["sale_price"].round(0).astype(int)

    # sale_date (fill missing with 2023-01-01, ISO string)
    clean_data["sale_date"] = normalize_text(clean_data["sale_date"]).fillna("2023-01-01")
    clean_data["sale_date"] = pd.to_datetime(clean_data["sale_date"], errors="coerce").fillna(pd.Timestamp("2023-01-01"))
    clean_data["sale_date"] = clean_data["sale_date"].dt.strftime("%Y-%m-%d").astype("string")

    # months_listed
    clean_data["months_listed"] = pd.to_numeric(normalize_text(clean_data["months_listed"]), errors="coerce")
    ml_mean = clean_data["months_listed"].mean()
    clean_data["months_listed"] = clean_data["months_listed"].fillna(ml_mean).round(1)

    # bedrooms
    clean_data["bedrooms"] = pd.to_numeric(normalize_text(clean_data["bedrooms"]), errors="coerce")
    br_mean = clean_data["bedrooms"].mean()
    clean_data["bedrooms"] = clean_data["bedrooms"].fillna(br_mean).round(0).astype(int)

    # house_type
    ht_raw = normalize_text(clean_data["house_type"]).astype("string").str.strip().str.lower()
    ht_raw = ht_raw.str.replace(r"\.+$", "", regex=True)  # "det." -> "det"
    ht_raw = ht_raw.str.replace("semi detached", "semi-detached", regex=False)
    ht_raw = ht_raw.str.replace("semidetached", "semi-detached", regex=False)

    ht_map = {
        "detached": "Detached", "det": "Detached",
        "semi-detached": "Semi-detached", "semi": "Semi-detached",
        "terraced": "Terraced", "terr": "Terraced",
    }
    clean_data["house_type"] = ht_raw.map(ht_map)

    mode_ht = clean_data["house_type"].mode(dropna=True)
    fill_ht = mode_ht.iloc[0] if len(mode_ht) else "Terraced"
    clean_data["house_type"] = clean_data["house_type"].fillna(fill_ht).astype("string")

    # area (extract first number)
    area_txt = normalize_text(clean_data["area"]).astype("string").str.strip()
    area_num = (
        area_txt.str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    )
    clean_data["area"] = pd.to_numeric(area_num, errors="coerce")
    area_mean = clean_data["area"].mean()
    clean_data["area"] = clean_data["area"].fillna(area_mean).round(1)

    # final order
    clean_data = clean_data[
        ["house_id", "city", "sale_price", "sale_date", "months_listed", "bedrooms", "house_type", "area"]
    ]
    return clean_data


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["sale_date"], errors="coerce")
    out["sale_year"] = dt.dt.year.astype(int)
    out["sale_month"] = dt.dt.month.astype(int)
    out["sale_year_month"] = dt.dt.to_period("M").astype(str)
    return out


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def kpi_card_html(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {sub_html}
    </div>
    """

def fig_hist(series: pd.Series, title: str, bins: int = 30):
    fig = plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    return fig


def fig_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str):
    fig = plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig


def fig_bar(x, y, title: str, xlabel: str, ylabel: str):
    fig = plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    return fig


# =========================
# Sidebar: navigation then filters
# =========================
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    label="Go to",
    options=["Summary", "Data explorer", "Modeling", "Insights"],
    index=0,
)

st.sidebar.markdown("## Filters")
DATA_PATH = st.sidebar.text_input("Data file path", value="Housing.csv", help="Place Housing.csv in the same folder as app.py")

try:
    raw_df = load_raw(DATA_PATH)
except Exception as e:
    st.error(f"Could not load '{DATA_PATH}'. Make sure the file exists next to app.py. Error: {e}")
    st.stop()

clean_data = make_clean_data(raw_df)
model_data = add_time_features(clean_data)

# filter controls
cities = sorted(model_data["city"].unique().tolist())
types_ = sorted(model_data["house_type"].unique().tolist())
bed_min, bed_max = int(model_data["bedrooms"].min()), int(model_data["bedrooms"].max())
date_min = pd.to_datetime(model_data["sale_date"]).min()
date_max = pd.to_datetime(model_data["sale_date"]).max()

sel_city = st.sidebar.multiselect("City", options=cities, default=cities)
sel_type = st.sidebar.multiselect("House type", options=types_, default=types_)
sel_bed = st.sidebar.slider("Bedrooms", min_value=bed_min, max_value=bed_max, value=(bed_min, bed_max))
sel_date = st.sidebar.date_input(
    "Sale date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date(),
)

st.sidebar.markdown("## Data")
st.sidebar.markdown(f"<div class='small-muted'>Rows after cleaning: <b>{len(model_data):,}</b></div>", unsafe_allow_html=True)

# apply filters
df_f = model_data[
    (model_data["city"].isin(sel_city))
    & (model_data["house_type"].isin(sel_type))
    & (model_data["bedrooms"].between(sel_bed[0], sel_bed[1]))
    & (pd.to_datetime(model_data["sale_date"]).dt.date.between(sel_date[0], sel_date[1]))
].copy()

# =========================
# Main header
# =========================
st.title("RealAgents Housing Analytics")
st.markdown(
    "A cleaned housing dataset with price analysis, model comparison, and insights to support listing price decisions."
)


# =========================
# Page: Summary
# =========================
if page == "Summary":
    st.subheader("Summary")

    # KPIs (ONLY here)
    total = len(df_f)
    avg_price = df_f["sale_price"].mean()
    med_price = df_f["sale_price"].median()
    avg_area = df_f["area"].mean()
    avg_bed = df_f["bedrooms"].mean()
    unknown_city_share = (df_f["city"].eq("Unknown").mean() * 100.0) if total > 0 else 0.0


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(kpi_card_html("Rows in view", f"{total:,}", "After cleaning and filters"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card_html("Average sale price", f"{avg_price:,.0f}", "Whole dollars"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card_html("Median sale price", f"{med_price:,.0f}", "Whole dollars"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card_html("Average area (sq.m.)", f"{avg_area:,.1f}", f"Average bedrooms: {avg_bed:,.1f}"), unsafe_allow_html=True)



    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.pyplot(fig_hist(df_f["sale_price"].rename("sale_price"), "Sale price distribution", bins=30), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        by_bed = df_f.groupby("bedrooms")["sale_price"].mean().sort_index()
        st.pyplot(fig_bar(by_bed.index.astype(str), by_bed.values, "Average price by bedrooms", "Bedrooms", "Average sale price"), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-grid'>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.pyplot(
            fig_scatter(
                df_f["area"], df_f["sale_price"],
                "Sale price vs area",
                "Area (sq.m.)",
                "Sale price"
            ),
            clear_figure=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        by_city = df_f.groupby("city")["sale_price"].mean().sort_values(ascending=False).head(10)
        st.pyplot(fig_bar(by_city.index.tolist(), by_city.values, "Top cities by average price", "City", "Average sale price"), clear_figure=True)
        st.markdown(
            f"<div class='small-muted'>Unknown city share in view: <b>{unknown_city_share:.1f}%</b></div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Page: Data explorer
# =========================
elif page == "Data explorer":
    st.subheader("Data explorer")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Cleaned dataset (filtered view)")
    st.dataframe(
        df_f[["house_id", "city", "sale_price", "sale_date", "months_listed", "bedrooms", "house_type", "area"]],
        use_container_width=True,
        height=450
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Summary statistics")
    st.dataframe(df_f.describe(include="all").T, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Page: Modeling
# =========================
elif page == "Modeling":
    st.subheader("Modeling")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "Train two models on the cleaned dataset and compare performance on a held-out validation split. "
        "The comparison model captures non-linear relationships and interactions."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # prepare modeling set (use filtered view to keep consistent with user choices)
    md = df_f.copy()

    if len(md) < 100:
        st.warning("Not enough rows after filtering to train a stable model. Expand filters to include more data.")
        st.stop()

    target = "sale_price"
    id_col = "house_id"
    X = md.drop(columns=[target])
    y = md[target].astype(float)

    cat_cols = ["city", "house_type"]
    num_cols = ["months_listed", "bedrooms", "area", "sale_year", "sale_month"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline
    baseline_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    baseline_model.fit(X_train, y_train)
    base_pred = np.clip(baseline_model.predict(X_valid), 0, None)
    base_rmse = rmse(y_valid, base_pred)

    # Comparison
    compare_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.06,
                max_depth=6,
                max_iter=600,
                min_samples_leaf=20,
                random_state=0
            )),
        ]
    )
    compare_model.fit(X_train, y_train)
    comp_pred = np.clip(compare_model.predict(X_valid), 0, None)
    comp_rmse = rmse(y_valid, comp_pred)

    # Results table
    res = pd.DataFrame({
        "model": ["Baseline (Ridge)", "Comparison (HistGB)"],
        "rmse": [base_rmse, comp_rmse],
    }).sort_values("rmse")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Validation performance")
    st.dataframe(res, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Predictions output like the exam objects
    base_result = pd.DataFrame({"house_id": X_valid[id_col].astype("string"), "price": base_pred})
    compare_result = pd.DataFrame({"house_id": X_valid[id_col].astype("string"), "price": comp_pred})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Baseline predictions sample")
        st.dataframe(base_result.head(10), use_container_width=True)
        st.download_button(
            "Download baseline predictions (CSV)",
            data=base_result.to_csv(index=False).encode("utf-8"),
            file_name="base_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Comparison predictions sample")
        st.dataframe(compare_result.head(10), use_container_width=True)
        st.download_button(
            "Download comparison predictions (CSV)",
            data=compare_result.to_csv(index=False).encode("utf-8"),
            file_name="compare_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Error distribution
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Error distribution (comparison model)")
    errors = (y_valid.values - comp_pred)
    st.pyplot(fig_hist(pd.Series(errors, name="error"), "Validation errors (actual - predicted)", bins=30), clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Page: Insights (data-driven then model-driven)
# =========================
elif page == "Insights":
    st.subheader("Insights")

    # -------- Data-driven insights --------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Data-driven insights")

    if len(df_f) == 0:
        st.info("No rows available after filters. Expand filters to generate insights.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Insight 1: bedroom impact
    by_bed = df_f.groupby("bedrooms")["sale_price"].agg(["mean", "count"]).sort_index()
    strongest_bed = by_bed["mean"].idxmax()
    st.markdown(
        f"- Average sale price increases with bedrooms. The highest average in view is for **{strongest_bed} bedrooms** "
        f"at **{by_bed.loc[strongest_bed, 'mean']:,.0f}** (n={int(by_bed.loc[strongest_bed, 'count'])})."
    )

    # Insight 2: top city
    by_city = df_f.groupby("city")["sale_price"].mean().sort_values(ascending=False)
    top_city = by_city.index[0]
    st.markdown(f"- The highest average sale price by city in view is **{top_city}** at **{by_city.iloc[0]:,.0f}**.")

    # Insight 3: area relationship (correlation)
    corr = df_f[["sale_price", "area", "months_listed", "bedrooms"]].corr(numeric_only=True)["sale_price"].drop("sale_price")
    corr_sorted = corr.abs().sort_values(ascending=False)
    top_driver = corr_sorted.index[0]
    st.markdown(
        f"- Strongest linear association with price (absolute correlation) in view is **{top_driver}** "
        f"(|r|={corr[top_driver]:.2f})."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Model-driven insights --------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model-driven insights")

    # train comparison model on filtered data for insights
    md = df_f.copy()
    if len(md) < 150:
        st.markdown("- Not enough rows after filters to compute stable model-driven insights. Expand filters for more data.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    target = "sale_price"
    X = md.drop(columns=[target])
    y = md[target].astype(float)

    cat_cols = ["city", "house_type"]
    num_cols = ["months_listed", "bedrooms", "area", "sale_year", "sale_month"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.06,
                max_depth=6,
                max_iter=600,
                min_samples_leaf=20,
                random_state=0
            )),
        ]
    )
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_valid), 0, None)
    model_rmse = rmse(y_valid, pred)

    st.markdown(f"- Validation RMSE (comparison model): **{model_rmse:,.0f}**")

    # Permutation importance (model-driven interpretability)
    # Use a small cap to keep it fast
    sample_n = min(len(X_valid), 500)
    Xv = X_valid.sample(sample_n, random_state=0)
    yv = y_valid.loc[Xv.index]

    try:
        perm = permutation_importance(
            model, Xv, yv,
            n_repeats=8,
            random_state=0,
            scoring="neg_root_mean_squared_error"
        )
        imp = pd.DataFrame({
            "feature": X.columns,
            "importance": perm.importances_mean
        }).sort_values("importance", ascending=False).head(10)

        st.markdown("- Top drivers (permutation importance on validation sample):")
        st.dataframe(imp, use_container_width=True)
    except Exception as e:
        st.markdown(f"- Feature importance could not be computed in this environment. Error: {e}")

    # Error bands (regression analogue to probability bands)
    abs_err = np.abs(y_valid.values - pred)
    q50, q80, q90 = np.quantile(abs_err, [0.5, 0.8, 0.9])
    st.markdown(
        f"- Absolute error bands: median **{q50:,.0f}**, 80th percentile **{q80:,.0f}**, 90th percentile **{q90:,.0f}**."
    )

    st.markdown("</div>", unsafe_allow_html=True)
