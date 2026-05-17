import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Paths and data loading
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = DATA_DIR / "lgb_model.pkl"
MODEL_FALLBACK_PATH = DATA_DIR / "lgb_model_latest.pkl"
METRICS_PATH = DATA_DIR / "model_metrics.json"
INSIGHTS_MD_PATH = DATA_DIR / "business_insights.md"
INSIGHTS_JSON_PATH = DATA_DIR / "business_insights.json"
TRAIN_PROCESSED_PATH = DATA_DIR / "train_processed.csv"
TEST_PRED_PATH = DATA_DIR / "test_predictions.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.png"
FEATURE_IMPORTANCE_FALLBACK_PATH = DATA_DIR / "feature_importance_latest.png"
SHAP_IMPORTANCE_PLOT_PATH = DATA_DIR / "feature_importance_shap.png"
SHAP_IMPORTANCE_PLOT_FALLBACK_PATH = DATA_DIR / "feature_importance_shap_latest.png"

@st.cache_resource
def load_model():
    if MODEL_FALLBACK_PATH.exists():
        return joblib.load(MODEL_FALLBACK_PATH)
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError("No trained model found. Expected data/lgb_model.pkl")


@st.cache_data
def load_json_if_exists(path: Path, default_value):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_value


@st.cache_data
def load_text_if_exists(path: Path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


@st.cache_data
def load_csv_if_exists(path: Path, parse_dates=None):
    if path.exists():
        return pd.read_csv(path, parse_dates=parse_dates)
    return None


model = load_model()
metrics = load_json_if_exists(METRICS_PATH, {})
insights_json = load_json_if_exists(INSIGHTS_JSON_PATH, {})
insights_markdown = load_text_if_exists(INSIGHTS_MD_PATH)
train_hist = load_csv_if_exists(TRAIN_PROCESSED_PATH, parse_dates=["Date"])

group_std = None
global_std = None
if train_hist is not None:
    global_std = float(train_hist["Weekly_Sales"].std())
    group_std = (
        train_hist.groupby(["Store", "Dept"])["Weekly_Sales"]
        .std()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_std)
    )

pred_table = load_csv_if_exists(TEST_PRED_PATH, parse_dates=["Date"])


def decision_action(prediction, lower_pred, upper_pred, benchmark):
    ci_width_pct = None
    if lower_pred is not None and upper_pred is not None:
        ci_width_pct = float((upper_pred - lower_pred) / max(abs(prediction), 1e-6) * 100)

    demand_gap_pct = None
    if benchmark is not None and np.isfinite(benchmark):
        demand_gap_pct = float((prediction - benchmark) / max(abs(benchmark), 1e-6) * 100)

    if ci_width_pct is None:
        risk = "Medium"
    elif ci_width_pct >= 60:
        risk = "High"
    elif ci_width_pct >= 30:
        risk = "Medium"
    else:
        risk = "Low"

    if benchmark is None or not np.isfinite(benchmark):
        action = "No benchmark available: use planner judgment."
        return action, risk, ci_width_pct, demand_gap_pct

    ratio = prediction / max(benchmark, 1e-6)
    if ratio >= 1.10:
        action = "High-demand signal: increase inventory and staffing."
    elif ratio <= 0.90:
        action = "Low-demand signal: reduce replenishment and targeted promos."
    else:
        action = "Stable-demand signal: maintain standard operating plan."

    if risk == "High":
        action += " Risk is high, use buffer stock and tighter monitoring."
    elif risk == "Medium":
        action += " Risk is moderate, review weekly."
    else:
        action += " Risk is low."

    return action, risk, ci_width_pct, demand_gap_pct


def build_feature_row(
    date_value,
    is_holiday,
    temperature,
    fuel_price,
    markdown1,
    markdown2,
    markdown3,
    markdown4,
    markdown5,
    cpi,
    unemployment,
    store_size,
    store_type,
):
    return pd.DataFrame(
        {
            "IsHoliday": [1 if is_holiday == "Yes" else 0],
            "Temperature": [temperature],
            "Fuel_Price": [fuel_price],
            "MarkDown1": [markdown1],
            "MarkDown2": [markdown2],
            "MarkDown3": [markdown3],
            "MarkDown4": [markdown4],
            "MarkDown5": [markdown5],
            "CPI": [cpi],
            "Unemployment": [unemployment],
            "Size": [store_size],
            "Type_B": [1 if store_type == "B" else 0],
            "Type_C": [1 if store_type == "C" else 0],
            "Year": [date_value.year],
            "Month": [date_value.month],
            "Week": [date_value.isocalendar()[1]],
            "Day": [date_value.day],
        }
    )


def interval_from_group_std(prediction, store_id, dept_id, std_multiplier=1.5):
    if group_std is None or global_std is None:
        return None, None
    std_val = float(group_std.get((int(store_id), int(dept_id)), global_std))
    half_width = std_multiplier * std_val
    max_half_width = max(abs(prediction) * 0.75, 2500.0)
    half_width = min(half_width, max_half_width)
    lower = max(prediction - half_width, 0.0)
    upper = prediction + half_width
    return lower, upper


# -------------------------
# Page config and style
# -------------------------
st.set_page_config(
    page_title="Walmart Forecast Decision Tool",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --primary: #e98552;
    --primary-strong: #d9703f;
    --accent: #334155;
    --bg-a: #fcf6ec;
    --bg-b: #f3e4cf;
    --text-main: #374151;
    --text-muted: #6b7280;
    --card-bg: #fffdfa;
    --card-border: #e4ccb0;
    --tab-bg: #f5e8d7;
    --shadow: 0 14px 34px rgba(15, 23, 42, 0.12);
}
.stApp {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, var(--bg-a), var(--bg-b));
    color: var(--text-main);
}
[data-testid="stAppViewContainer"] * {
    color: var(--text-main);
}
[data-testid="stHeader"] {
    background: linear-gradient(90deg, #f7ecdc, #efdfc7) !important;
    border-bottom: 1px solid #d7bc9c !important;
}
[data-testid="stToolbar"] * {
    color: #5b6673 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0e2cf 0%, #e9d5bb 100%);
    border-right: 1px solid #d7bc9c;
}
[data-testid="stSidebar"] * {
    color: #17212b !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"] > div {
    background-color: #fffaf2 !important;
    border: 1px solid #cdb18d !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}
[data-testid="stSidebar"] div[data-baseweb="input"] input {
    color: #102030 !important;
    -webkit-text-fill-color: #102030 !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder {
    color: #5f6b78 !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #fffaf2 !important;
    border: 1px solid #cdb18d !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #102030 !important;
}
[data-testid="stSidebar"] .stDateInput input {
    color: #102030 !important;
    -webkit-text-fill-color: #102030 !important;
}
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] button[title="Increment"],
[data-testid="stSidebar"] button[title="Decrement"] {
    color: #102030 !important;
    background: #f6e8d2 !important;
    border-color: #c9b08d !important;
}
[data-testid="stSidebar"] .stNumberInput button,
[data-testid="stSidebar"] .stNumberInput button * {
    color: #102030 !important;
    fill: #102030 !important;
    stroke: #102030 !important;
}
[data-testid="stSidebar"] .stNumberInput [data-baseweb="button-group"] {
    background: #f6e8d2 !important;
    border-left: 1px solid #c9b08d !important;
}
[data-testid="stSidebar"] .stNumberInput [data-baseweb="button-group"] button {
    background: #f6e8d2 !important;
    color: #102030 !important;
    border: none !important;
}
[data-testid="stSidebar"] .stNumberInput [data-baseweb="button-group"] button:hover,
[data-testid="stSidebar"] .stNumberInput [data-baseweb="button-group"] button:focus,
[data-testid="stSidebar"] .stNumberInput [data-baseweb="button-group"] button:active {
    background: #efdcbf !important;
    color: #102030 !important;
}
[data-testid="stSidebar"] .stNumberInput [data-baseweb="button-group"] svg,
[data-testid="stSidebar"] .stNumberInput [data-baseweb="button-group"] svg path {
    fill: #102030 !important;
    stroke: #102030 !important;
}
[data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] {
    background-color: #fffaf2 !important;
    border: 1px solid #c9b08d !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    color: #102030 !important;
    -webkit-text-fill-color: #102030 !important;
}
h1, h2, h3 {
    color: var(--accent) !important;
    letter-spacing: -0.02em;
}
p, label, span, div {
    color: var(--text-main);
}
[data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}
.hero-card {
    background:
      radial-gradient(circle at 0% 0%, rgba(255, 107, 53, 0.14), transparent 35%),
      linear-gradient(140deg, #fffdf9 0%, #fff6ea 100%);
    border-radius: 22px;
    padding: 24px 26px;
    border: 1px solid #e8d3b8;
    box-shadow: var(--shadow);
    margin-bottom: 14px;
}
.decision-card {
    background: linear-gradient(180deg, #fffefb, #fff7ea);
    border-radius: 18px;
    padding: 18px 22px;
    border: 1px solid var(--card-border);
    box-shadow: var(--shadow);
}
.kpi-value {
    color: #475569;
    font-weight: 700;
    font-size: 1.65rem;
    letter-spacing: -0.02em;
}
[data-testid="stMetric"] {
    background: linear-gradient(165deg, #fffdfa 0%, #fff6e9 100%);
    border: 1px solid #e6cfb3;
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 8px 20px rgba(12, 20, 33, 0.09);
}
[data-testid="stMetric"] > div {
    gap: 2px;
}
[data-baseweb="tab-list"] {
    background: var(--tab-bg);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid #dfc4a4;
}
[data-baseweb="tab"] {
    color: #263645 !important;
    font-weight: 700;
    border-radius: 10px;
    margin-right: 4px;
    padding: 8px 14px;
}
[aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, var(--primary), var(--primary-strong)) !important;
    border-bottom-color: transparent !important;
}
[data-testid="stMetricLabel"] {
    color: #2f3d4a !important;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #475569 !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}
[data-testid="stAlert"] {
    color: #122030 !important;
}
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #e2cf91;
    background: linear-gradient(180deg, #fff4c7 0%, #f7e9b2 100%);
}
[data-testid="stHorizontalBlock"] {
    gap: 1rem 1rem;
}
div.stPlotlyChart, div.stPyplot {
    background: linear-gradient(170deg, #fffdf9 0%, #fff5e8 100%);
    border: 1px solid #e3cab0;
    border-radius: 14px;
    padding: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class='hero-card'>
    <div style="display:inline-block; margin-bottom:8px; padding:5px 10px; border-radius:999px; background:#f2dfc3; color:#6b7280; font-size:0.75rem; font-weight:600;">
        Decision Intelligence Dashboard
    </div>
    <h1 style="margin:0 0 6px 0; font-size: 3rem;">Walmart Demand Forecast Decision Tool</h1>
    <p style="margin:0; color: var(--text-muted); font-size: 1.04rem;">
        Forecast + baseline comparison + business insights to support weekly inventory and staffing decisions.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if not metrics:
    st.warning(
        "Metrics artifact not found. Run `python src/train_model.py` to generate "
        "`model_metrics.json`, baseline comparison, and insights."
    )
else:
    interval_coverage = metrics.get("prediction_interval", {}).get("validation_coverage_pct")
    if interval_coverage is not None and interval_coverage < 70:
        st.warning(
            f"Prediction interval confidence is currently low (coverage: {interval_coverage:.2f}%). "
            "Treat interval-based risk levels as directional only."
        )


# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Scenario Input")
store = st.sidebar.number_input("Store ID", min_value=1, max_value=45, value=1)
dept = st.sidebar.number_input("Dept ID", min_value=1, max_value=100, value=1)
date = st.sidebar.date_input("Week Start Date")
is_holiday = st.sidebar.selectbox("Is Holiday?", ["No", "Yes"])
store_type = st.sidebar.selectbox("Store Type", ["A", "B", "C"])

st.sidebar.subheader("External Signals")
temperature = st.sidebar.slider("Temperature (F)", min_value=-20.0, max_value=120.0, value=70.0)
fuel_price = st.sidebar.slider("Fuel Price", min_value=1.0, max_value=6.0, value=3.0)
cpi = st.sidebar.slider("CPI", min_value=150.0, max_value=260.0, value=210.0)
unemployment = st.sidebar.slider("Unemployment", min_value=2.0, max_value=15.0, value=7.0)
store_size = st.sidebar.number_input("Store Size", min_value=10000, max_value=300000, value=150000)

st.sidebar.subheader("Promotion Plan")
markdown1 = st.sidebar.number_input("MarkDown1", value=0.0)
markdown2 = st.sidebar.number_input("MarkDown2", value=0.0)
markdown3 = st.sidebar.number_input("MarkDown3", value=0.0)
markdown4 = st.sidebar.number_input("MarkDown4", value=0.0)
markdown5 = st.sidebar.number_input("MarkDown5", value=0.0)


input_df = build_feature_row(
    date_value=date,
    is_holiday=is_holiday,
    temperature=temperature,
    fuel_price=fuel_price,
    markdown1=markdown1,
    markdown2=markdown2,
    markdown3=markdown3,
    markdown4=markdown4,
    markdown5=markdown5,
    cpi=cpi,
    unemployment=unemployment,
    store_size=store_size,
    store_type=store_type,
)

feature_columns = list(getattr(model, "feature_name_", input_df.columns))
input_df = input_df.reindex(columns=feature_columns, fill_value=0.0)
prediction = float(model.predict(input_df)[0])
lower_prediction, upper_prediction = interval_from_group_std(prediction, store, dept, std_multiplier=1.5)

benchmark = None
if train_hist is not None:
    segment = train_hist[(train_hist["Store"] == store) & (train_hist["Dept"] == dept)]
    if not segment.empty:
        benchmark = float(segment["Weekly_Sales"].median())

action_text, risk_level, ci_width_pct, demand_gap_pct = decision_action(
    prediction, lower_prediction, upper_prediction, benchmark
)


# -------------------------
# Dashboard tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive KPIs", "Decision Simulator", "Portfolio Planner", "Business Insights"]
)

with tab1:
    col1, col2, col3 = st.columns(3)

    lgb_rmse = metrics.get("lightgbm", {}).get("rmse")
    lgb_wmape = metrics.get("lightgbm", {}).get("wmape")
    base_rmse = metrics.get("baseline_moving_average_4w", {}).get("rmse")
    base_wmape = metrics.get("baseline_moving_average_4w", {}).get("wmape")
    imp_rmse = metrics.get("improvement_vs_baseline_pct", {}).get("rmse")
    imp_wmape = metrics.get("improvement_vs_baseline_pct", {}).get("wmape")
    ci_cov = metrics.get("prediction_interval", {}).get("validation_coverage_pct")
    avg_interval_width_pct = metrics.get("prediction_interval", {}).get("avg_interval_width_pct")

    col1.metric("LGBM RMSE", f"{lgb_rmse:.2f}" if lgb_rmse is not None else "n/a")
    col1.metric("LGBM WMAPE", f"{lgb_wmape:.2f}%" if lgb_wmape is not None else "n/a")

    col2.metric("Baseline RMSE (MA-4)", f"{base_rmse:.2f}" if base_rmse is not None else "n/a")
    col2.metric("Baseline WMAPE (MA-4)", f"{base_wmape:.2f}%" if base_wmape is not None else "n/a")

    col3.metric("RMSE Improvement", f"{imp_rmse:.2f}%" if imp_rmse is not None else "n/a")
    col3.metric("WMAPE Improvement", f"{imp_wmape:.2f}%" if imp_wmape is not None else "n/a")
    col3.metric("CI Coverage (10-90)", f"{ci_cov:.2f}%" if ci_cov is not None else "n/a")
    col3.metric("Avg Interval Width", f"{avg_interval_width_pct:.2f}%" if avg_interval_width_pct is not None else "n/a")

    st.markdown("<div class='decision-card'>", unsafe_allow_html=True)
    st.subheader("Current Forecast Decision")
    st.write(f"Store **{store}**, Dept **{dept}**, Week **{date}**")
    st.markdown(f"<p class='kpi-value'>Predicted Weekly Sales: ${prediction:,.2f}</p>", unsafe_allow_html=True)
    if lower_prediction is not None and upper_prediction is not None:
        st.write(
            f"Prediction Interval (P10-P90): **USD {lower_prediction:,.2f} to USD {upper_prediction:,.2f}**"
        )
    st.write(f"Risk Level: **{risk_level}**")
    if ci_width_pct is not None:
        st.write(f"Uncertainty (interval width / prediction): **{ci_width_pct:.2f}%**")
    if benchmark is not None:
        st.write(f"Historical Median Benchmark: **${benchmark:,.2f}**")
    if demand_gap_pct is not None:
        st.write(f"Demand gap vs benchmark: **{demand_gap_pct:+.2f}%**")
    st.info(action_text)
    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.subheader("What-If Scenario Simulator")
    st.caption("Compare your base plan with an adjusted promotion/environment scenario.")

    c1, c2 = st.columns(2)
    promo_uplift = c1.slider("Promotion multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
    temp_delta = c2.slider("Temperature delta", min_value=-20.0, max_value=20.0, value=0.0, step=1.0)

    scenario_df = input_df.copy()
    for col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        scenario_df[col] = scenario_df[col] * promo_uplift
    scenario_df["Temperature"] = scenario_df["Temperature"] + temp_delta
    scenario_df = scenario_df.reindex(columns=feature_columns, fill_value=0.0)

    scenario_pred = float(model.predict(scenario_df)[0])
    scenario_lower, scenario_upper = interval_from_group_std(scenario_pred, store, dept, std_multiplier=1.5)
    delta_abs = scenario_pred - prediction
    delta_pct = (delta_abs / max(abs(prediction), 1e-6)) * 100

    c3, c4, c5 = st.columns(3)
    c3.metric("Base forecast", f"${prediction:,.2f}")
    c4.metric("Scenario forecast", f"${scenario_pred:,.2f}")
    c5.metric("Scenario impact", f"{delta_pct:+.2f}%")
    if scenario_lower is not None and scenario_upper is not None:
        st.write(
            f"Scenario interval (P10-P90): **USD {scenario_lower:,.2f} to USD {scenario_upper:,.2f}**"
        )

    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = ["Base", "Scenario"]
    values = [prediction, scenario_pred]
    ax.bar(labels, values, color=["#173f5f", "#e8682f"])
    ax.set_ylabel("Predicted Weekly Sales")
    ax.set_title("Forecast Comparison")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Suggested action: "
        + ("prepare upside capacity." if delta_pct > 8 else "keep baseline operating plan." if delta_pct >= -8 else "manage downside risk and avoid overstock.")
    )


with tab3:
    st.subheader("Portfolio Planner")
    st.caption("Prioritize top Store-Dept combinations from the forecast horizon.")

    if pred_table is None or pred_table.empty:
        st.warning("No `test_predictions.csv` found. Run training first to populate planner table.")
    else:
        horizon = pred_table.copy()
        horizon["Gap_vs_Baseline"] = (
            horizon["Weekly_Sales_Predicted"] - horizon.get("Baseline_MA4_Predicted", 0)
        )
        top_n = st.slider("Show top N opportunities", min_value=5, max_value=50, value=15, step=5)
        sort_col = st.selectbox(
            "Rank by",
            ["Weekly_Sales_Predicted", "Gap_vs_Baseline"],
        )
        ranked = horizon.sort_values(sort_col, ascending=False).head(top_n)
        st.dataframe(
            ranked[
                [
                    "Store",
                    "Dept",
                    "Date",
                    "Weekly_Sales_Predicted",
                    "Baseline_MA4_Predicted",
                    "Gap_vs_Baseline",
                ]
            ],
            use_container_width=True,
        )

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.hist(horizon["Weekly_Sales_Predicted"], bins=30, color="#173f5f", alpha=0.85)
        ax2.set_title("Distribution of Predicted Weekly Sales")
        ax2.set_xlabel("Predicted Weekly Sales")
        ax2.set_ylabel("Count")
        ax2.grid(axis="y", alpha=0.22, linestyle="--")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)
        plt.close(fig2)


with tab4:
    st.subheader("Automated Business Insights")
    if insights_markdown:
        st.markdown(insights_markdown)
    else:
        st.info("No `business_insights.md` found. Run training to generate automated insights.")

    seasonality = insights_json.get("seasonality", {})
    trend = insights_json.get("trend", {})
    if seasonality or trend:
        st.markdown("### Structured Summary")
        if seasonality:
            st.write(
                f"Seasonality: best month **{seasonality.get('best_month', 'n/a')}**, "
                f"worst month **{seasonality.get('worst_month', 'n/a')}**, "
                f"peak week **{seasonality.get('peak_week', 'n/a')}**, "
                f"swing **{seasonality.get('seasonality_swing_pct', 'n/a')}%**."
            )
        if trend:
            st.write(
                f"Trend: **{trend.get('direction', 'n/a')}**, slope **{trend.get('slope_per_period', 'n/a')}**."
            )

    if insights_json.get("top_features_shap"):
        feat_df = pd.DataFrame(insights_json["top_features_shap"])
        st.write("Top feature drivers:")
        st.dataframe(feat_df, use_container_width=True)

    if SHAP_IMPORTANCE_PLOT_PATH.exists():
        st.image(str(SHAP_IMPORTANCE_PLOT_PATH), caption="SHAP feature importance")
    elif SHAP_IMPORTANCE_PLOT_FALLBACK_PATH.exists():
        st.image(str(SHAP_IMPORTANCE_PLOT_FALLBACK_PATH), caption="SHAP feature importance")
    elif FEATURE_IMPORTANCE_PATH.exists():
        st.image(str(FEATURE_IMPORTANCE_PATH), caption="Model feature importance")
    elif FEATURE_IMPORTANCE_FALLBACK_PATH.exists():
        st.image(str(FEATURE_IMPORTANCE_FALLBACK_PATH), caption="Model feature importance")
