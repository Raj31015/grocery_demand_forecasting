from pathlib import Path
import json
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import SKU_PROFILE_PATH
from src.predict import predict_row


def load_profiles() -> list[dict]:
    if not SKU_PROFILE_PATH.exists():
        return []
    return json.loads(SKU_PROFILE_PATH.read_text())


def lookup_profile(profiles: list[dict], store_nbr: int, item_nbr: int) -> dict | None:
    for row in profiles:
        if int(row["store_nbr"]) == int(store_nbr) and int(row["item_nbr"]) == int(item_nbr):
            return row
    return None


st.set_page_config(page_title="Grocery Forecasting", layout="wide")
st.title("Grocery Demand Forecasting")
st.write("Forecast next-day unit demand and a stock-pressure proxy using the Favorita grocery dataset.")

profiles = load_profiles()
if not profiles:
    st.warning("No trained profiles found yet. Run `python -m src.train` first.")
    st.stop()

store_options = sorted({int(row["store_nbr"]) for row in profiles})
default_store = store_options[0]
default_item = int(next(row["item_nbr"] for row in profiles if int(row["store_nbr"]) == default_store))

with st.form("forecast_form"):
    col1, col2 = st.columns(2)
    with col1:
        store_nbr = st.selectbox("Store", store_options)
        item_nbr = st.number_input("Item Number", min_value=1, value=default_item, step=1)
        forecast_date = st.date_input("Forecast Date")
        onpromotion = st.selectbox("Promotion", [False, True], index=0)
        current_inventory = st.number_input("Current Inventory (optional but useful)", min_value=0.0, value=0.0)
    with col2:
        transactions = st.number_input("Expected Store Transactions (optional)", min_value=0.0, value=0.0)
        dcoilwtico = st.number_input("Oil Price Override (optional)", min_value=0.0, value=0.0)
        holiday_flag = st.selectbox("Holiday/Event Flag", [0, 1], index=0)
        transferred_holiday = st.selectbox("Transferred Holiday", [0, 1], index=0)
        holiday_event_count = st.number_input("Holiday/Event Count", min_value=0, value=0, step=1)

    submitted = st.form_submit_button("Predict")

profile = lookup_profile(profiles, int(store_nbr), int(item_nbr))
if profile:
    st.caption(
        f"Profile preview: family={profile['family']}, avg_daily_units={profile['avg_daily_units']:.2f}, "
        f"rolling_mean_7={profile['rolling_mean_7']:.2f}"
    )

if submitted:
    payload = {
        "store_nbr": int(store_nbr),
        "item_nbr": int(item_nbr),
        "date": str(forecast_date),
        "onpromotion": bool(onpromotion),
        "current_inventory": float(current_inventory),
        "transactions": None if transactions == 0 else float(transactions),
        "dcoilwtico": None if dcoilwtico == 0 else float(dcoilwtico),
        "holiday_flag": int(holiday_flag),
        "transferred_holiday": int(transferred_holiday),
        "holiday_event_count": int(holiday_event_count),
    }

    try:
        result = predict_row(payload)
    except ValueError as exc:
        st.error(str(exc))
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Units Sold", f"{result['predicted_unit_sales']:.1f}")
        col2.metric("Stock-Pressure Probability", f"{result['stockout_probability'] * 100:.1f}%")
        col3.metric("Risk Tier", result["stockout_risk_tier"].title())

        st.subheader("Replenishment Guidance")
        st.write(f"Recommended order quantity: **{result['recommended_order_qty']:.1f} units**")
        st.write(f"Reorder point: **{result['reorder_point']:.1f} units**")
        st.write(f"Safety stock: **{result['recommended_safety_stock']:.1f} units**")
        st.write(f"Inventory after forecasted demand: **{result['projected_inventory_gap']:.1f} units**")

        st.subheader("Historical Context")
        st.write(f"Average daily units for this store-item pair: **{result['avg_daily_units']:.2f}**")
        st.write(f"Demand standard deviation: **{result['demand_std']:.2f}**")
