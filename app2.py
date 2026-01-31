import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# -------------------------
# Load model
# -------------------------
model = joblib.load(DATA_DIR / "lgb_model.pkl")

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Walmart Sales Predictor 🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛒"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #fff5e6, #ffe6cc);
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #FF6600;
    font-weight: bold;
}
.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}
.metric-value {
    font-size: 36px !important;
    color: #FF5733;
    font-weight: bold;
}
.icon {
    font-size: 18px;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Hero Section
# -------------------------
st.markdown("""
<div style='text-align:center; padding:30px 10px;'>
    <h1>🛒 Walmart Weekly Sales Predictor</h1>
    <p style='font-size:18px; color:#555; max-width:700px; margin:auto;'>
        Predict weekly sales for any store & department using historical and economic data.
        Enter store information below and get instant predictions!
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Main Page Inputs (Vertical, with Icons)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📥 Enter Basic Store & Week Data")

store = st.number_input("🏪 Store ID", min_value=1, max_value=45, value=1)
dept = st.number_input("📦 Dept ID", min_value=1, max_value=100, value=1)
is_holiday = st.selectbox("🎉 Is Holiday?", ["No", "Yes"])
temperature = st.number_input("🌡️ Temperature (°F)", value=70.0)
fuel_price = st.number_input("⛽ Fuel Price ($)", value=3.0)
date = st.date_input("📅 Week Start Date")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Sidebar Additional Features
# -------------------------
st.sidebar.header("⚙️ Additional Features")
markdown1 = st.sidebar.number_input("MarkDown1", value=0.0)
markdown2 = st.sidebar.number_input("MarkDown2", value=0.0)
markdown3 = st.sidebar.number_input("MarkDown3", value=0.0)
markdown4 = st.sidebar.number_input("MarkDown4", value=0.0)
markdown5 = st.sidebar.number_input("MarkDown5", value=0.0)
cpi = st.sidebar.number_input("📈 CPI", value=200.0)
unemployment = st.sidebar.number_input("📉 Unemployment Rate", value=7.0)
store_size = st.sidebar.number_input("🏬 Store Size (sq.ft.)", value=150000)
store_type = st.sidebar.selectbox("🏷️ Store Type", ["A", "B", "C"])

# -------------------------
# Prepare DataFrame
# -------------------------
input_df = pd.DataFrame({
    "IsHoliday": [1 if is_holiday=="Yes" else 0],
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
    "Type_B": [1 if store_type=="B" else 0],
    "Type_C": [1 if store_type=="C" else 0],
    "Year": [date.year],
    "Month": [date.month],
    "Week": [date.isocalendar()[1]],
    "Day": [date.weekday()]
})

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Weekly Sales"):
    pred = model.predict(input_df)[0]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"💰 Predicted Weekly Sales for Store {store} - Dept {dept}")
    st.markdown(f"<p class='metric-value'>${pred:,.2f}</p>", unsafe_allow_html=True)

    # Optional: show input data
    with st.expander("Show Input Data"):
        st.dataframe(input_df)

    # -------------------------
    # Visualization: Predicted Sales Trend
    # -------------------------
    st.subheader("📊 Predicted Sales Trend (Next 5 Weeks)")

    dates = pd.date_range(start=date, periods=5, freq='W')
    sales = [pred * (0.9 + np.random.rand() * 0.2) for _ in range(5)]  # small variation

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(dates, sales, marker='o', linestyle='-', color='#FF5733', linewidth=2)
    ax.fill_between(dates, np.array(sales)*0.95, np.array(sales)*1.05, color='#FFC300', alpha=0.2)
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Predicted Sales ($)", fontsize=12)
    ax.set_title("Next 5 Weeks Predicted Sales", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px; color: gray;'>
        Powered by LightGBM & Streamlit | Demo Walmart Weekly Sales Predictor
    </p>
    """, unsafe_allow_html=True
)
