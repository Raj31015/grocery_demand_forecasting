from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import TREE_DEMAND_MODEL_PATH, TREE_SKU_PROFILE_PATH, TREE_STOCKOUT_MODEL_PATH


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@st.cache_resource(show_spinner=False)
def load_model() -> dict:
    """Load trained artifacts and build fast lookup structures for the UI."""
    if not TREE_DEMAND_MODEL_PATH.exists() or not TREE_STOCKOUT_MODEL_PATH.exists() or not TREE_SKU_PROFILE_PATH.exists():
        raise FileNotFoundError("Missing trained model artifacts. Run training before opening the app.")

    demand_model = joblib.load(TREE_DEMAND_MODEL_PATH)
    stockout_model = joblib.load(TREE_STOCKOUT_MODEL_PATH)
    profiles = json.loads(TREE_SKU_PROFILE_PATH.read_text())
    profiles_by_key = {
        (int(profile["store_nbr"]), int(profile["item_nbr"])): profile
        for profile in profiles
    }

    # Favorita does not include human-readable item names, so we create a clean mapping for UI display.
    item_mapping: dict[int, str] = {}
    item_family: dict[int, str] = {}
    for profile in profiles:
        item_nbr = int(profile["item_nbr"])
        family = str(profile.get("family", "Unknown"))
        item_family[item_nbr] = family
        item_mapping[item_nbr] = f"{family.title()} SKU ({item_nbr})"

    return {
        "demand_model": demand_model,
        "stockout_model": stockout_model,
        "profiles": profiles,
        "profiles_by_key": profiles_by_key,
        "item_mapping": item_mapping,
        "item_family": item_family,
    }


def lookup_profile(artifacts: dict, store_nbr: int, item_nbr: int) -> dict | None:
    """Return the saved profile for the selected store-item pair."""
    return artifacts["profiles_by_key"].get((int(store_nbr), int(item_nbr)))


def preprocess_input(
    store_nbr: int,
    item_nbr: int,
    forecast_date,
    onpromotion: bool,
    transactions: float,
    oil_price: float,
    holiday_flag: int,
    transferred_holiday: int,
    holiday_event_count: int,
    inventory: float,
) -> dict:
    """Convert UI inputs into a model-ready request payload."""
    return {
        "store_nbr": int(store_nbr),
        "item_nbr": int(item_nbr),
        "date": str(forecast_date),
        "onpromotion": bool(onpromotion),
        "transactions": float(transactions),
        "dcoilwtico": float(oil_price),
        "holiday_flag": int(holiday_flag),
        "transferred_holiday": int(transferred_holiday),
        "holiday_event_count": int(holiday_event_count),
        "current_inventory": float(inventory),
    }


def build_feature_row(payload: dict, profile: dict) -> tuple[dict, float]:
    """Assemble the exact feature row required by the trained model."""
    forecast_date = pd.to_datetime(payload["date"])
    current_inventory = float(payload["current_inventory"])

    row = {
        "store_nbr": int(payload["store_nbr"]),
        "item_nbr": int(payload["item_nbr"]),
        "family": profile["family"],
        "class": int(profile["class"]),
        "perishable": int(profile["perishable"]),
        "city": profile["city"],
        "state": profile["state"],
        "store_type": profile["store_type"],
        "cluster": int(profile["cluster"]),
        "dcoilwtico": float(payload["dcoilwtico"]),
        "transactions": float(payload["transactions"]),
        "holiday_flag": int(payload["holiday_flag"]),
        "transferred_holiday": int(payload["transferred_holiday"]),
        "holiday_event_count": int(payload["holiday_event_count"]),
        "local_holiday_count": 0,
        "regional_holiday_count": 0,
        "national_holiday_count": 0,
        "promo_flag": int(bool(payload["onpromotion"])),
        "day_of_week": int(forecast_date.dayofweek),
        "week_of_year": int(forecast_date.isocalendar().week),
        "month": int(forecast_date.month),
        "day_of_month": int(forecast_date.day),
        "is_month_start": int(forecast_date.is_month_start),
        "is_month_end": int(forecast_date.is_month_end),
        "is_weekend": int(forecast_date.dayofweek >= 5),
        "is_payday": int(forecast_date.day == 15 or forecast_date.is_month_end),
        "lag_1": float(profile.get("lag_1", profile.get("avg_daily_units", 0.0))),
        "lag_7": float(profile.get("lag_7", profile.get("avg_daily_units", 0.0))),
        "lag_14": float(profile.get("lag_14", profile.get("avg_daily_units", 0.0))),
        "lag_28": float(profile.get("lag_28", profile.get("avg_daily_units", 0.0))),
        "diff_lag_1_7": float(profile.get("diff_lag_1_7", 0.0)),
        "rolling_mean_7": float(profile.get("rolling_mean_7", profile.get("avg_daily_units", 0.0))),
        "rolling_mean_14": float(profile.get("rolling_mean_14", profile.get("avg_daily_units", 0.0))),
        "rolling_mean_28": float(profile.get("rolling_mean_28", profile.get("avg_daily_units", 0.0))),
        "rolling_std_7": float(profile.get("rolling_std_7", profile.get("demand_std", 0.0))),
        "weekday_avg_sales": float(profile.get("weekday_avg_sales", profile.get("avg_daily_units", 0.0))),
        "relative_momentum": float(profile.get("relative_momentum", 1.0)),
    }
    return row, current_inventory


def predict(payload: dict, artifacts: dict) -> dict:
    """Run prediction using the already-loaded model bundles."""
    profile = lookup_profile(artifacts, payload["store_nbr"], payload["item_nbr"])
    if profile is None:
        raise ValueError("No trained profile found for the selected store/item combination.")

    feature_row, current_inventory = build_feature_row(payload, profile)
    X = pd.DataFrame([feature_row])
    demand_value = float(np.expm1(artifacts["demand_model"].predict(X)[0]))
    demand_int = max(0, int(round(demand_value)))
    stockout_probability = float(artifacts["stockout_model"].predict_proba(X)[0, 1])

    avg_daily_units = float(profile.get("avg_daily_units", feature_row["rolling_mean_7"]))
    demand_std = max(1.0, float(profile.get("demand_std", feature_row["rolling_std_7"])))
    safety_stock = max(2.0, 1.65 * demand_std)
    reorder_point = round((avg_daily_units * 2) + safety_stock, 2)
    recommended_order_qty = max(0.0, round(reorder_point + demand_int - current_inventory, 2))
    projected_ending_inventory = round(current_inventory - demand_int, 2)

    return {
        "predicted_demand": demand_int,
        "demand_range_low": max(0, int(round(demand_int * 0.8))),
        "demand_range_high": int(round(demand_int * 1.2)),
        "stockout_probability": stockout_probability,
        "reorder_point": reorder_point,
        "recommended_order_qty": recommended_order_qty,
        "projected_ending_inventory": projected_ending_inventory,
        "recommended_safety_stock": round(safety_stock, 2),
        "avg_daily_units": round(avg_daily_units, 2),
        "demand_std": round(demand_std, 2),
        "feature_row": feature_row,
        "profile": profile,
    }


def compute_stock_pressure(predicted_demand: int, inventory: float) -> tuple[float, str, str]:
    """Translate predicted demand into a business-friendly stock pressure score."""
    safe_inventory = max(float(inventory), 1.0)
    pressure = predicted_demand / safe_inventory
    if pressure > 0.7:
        return pressure, "Restock needed", "#d93025"
    if pressure >= 0.3:
        return pressure, "Watch", "#f9ab00"
    return pressure, "Safe", "#188038"


def simulate_explanations(payload: dict, prediction: dict) -> list[tuple[str, str]]:
    """Approximate top drivers when SHAP is not available."""
    feature_row = prediction["feature_row"]
    profile = prediction["profile"]
    explanations: list[tuple[str, float, str]] = []

    if payload["onpromotion"]:
        explanations.append(("Promotion", 35.0, "Promotion is active, which typically lifts next-day demand."))
    else:
        explanations.append(("Promotion", 0.0, "No promotion is applied today."))

    avg_transactions = max(float(profile.get("avg_transactions", payload["transactions"])), 1.0)
    tx_ratio = payload["transactions"] / avg_transactions
    explanations.append(
        (
            "Store traffic",
            round((tx_ratio - 1.0) * 25, 1),
            "Expected transactions are higher than usual." if tx_ratio > 1.05 else "Expected transactions are near or below normal.",
        )
    )

    weekday_avg = max(float(feature_row["weekday_avg_sales"]), 1.0)
    avg_daily_units = max(float(profile.get("avg_daily_units", weekday_avg)), 1.0)
    weekday_ratio = weekday_avg / avg_daily_units
    explanations.append(
        (
            "Day-of-week pattern",
            round((weekday_ratio - 1.0) * 20, 1),
            "This weekday is historically stronger for the selected SKU." if weekday_ratio > 1.05 else "This weekday is historically typical for the selected SKU.",
        )
    )

    holiday_effect = 12.0 if payload["holiday_flag"] else 0.0
    explanations.append(
        (
            "Holiday effect",
            holiday_effect,
            "Holiday/event demand uplift applied." if payload["holiday_flag"] else "No holiday effect detected.",
        )
    )

    momentum = float(feature_row["relative_momentum"])
    explanations.append(
        (
            "Recent momentum",
            round((momentum - 1.0) * 15, 1),
            "Recent sales are running above rolling average." if momentum > 1.02 else "Recent sales are close to the rolling average.",
        )
    )

    explanations.sort(key=lambda item: abs(item[1]), reverse=True)
    return [(name, f"{effect:+.1f}%  {note}") for name, effect, note in explanations[:3]]


def recommendation_text(stock_pressure: float) -> tuple[str, str]:
    """Map stock pressure into an operational recommendation."""
    if stock_pressure > 0.7:
        return "Urgent restock required", "warning"
    if stock_pressure >= 0.3:
        return "Monitor inventory", "warning"
    return "Stock sufficient", "success"


st.set_page_config(page_title="Grocery Demand Forecasting", layout="wide")
st.title("Grocery Demand Forecasting")
st.write("Interview-ready demand forecasting dashboard for the Favorita retail dataset.")

try:
    artifacts = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

profiles = artifacts["profiles"]
store_options = sorted({int(profile["store_nbr"]) for profile in profiles})
default_store = store_options[0]
store_items = sorted(
    {int(profile["item_nbr"]) for profile in profiles if int(profile["store_nbr"]) == default_store}
)
default_item = store_items[0]
default_profile = lookup_profile(artifacts, default_store, default_item)
default_transactions = clamp(float(default_profile.get("avg_transactions", 1000.0)), 800.0, 1200.0)
default_oil = clamp(float(default_profile.get("dcoilwtico", 70.0)), 60.0, 80.0)
default_inventory = clamp(float(default_profile.get("avg_daily_units", 100.0) * 2.0), 50.0, 150.0)

with st.form("forecast_form"):
    st.subheader("Product Info")
    product_col1, product_col2 = st.columns(2)
    with product_col1:
        store_nbr = st.selectbox("Store", store_options)
        available_item_numbers = sorted(
            {int(profile["item_nbr"]) for profile in profiles if int(profile["store_nbr"]) == int(store_nbr)}
        )
        display_options = {
            item_nbr: artifacts["item_mapping"].get(item_nbr, f"Unknown SKU ({item_nbr})")
            for item_nbr in available_item_numbers
        }
        selected_item_label = st.selectbox(
            "Item",
            [display_options[item_nbr] for item_nbr in available_item_numbers],
        )
        label_to_item = {label: item_nbr for item_nbr, label in display_options.items()}
        item_nbr = label_to_item[selected_item_label]
    with product_col2:
        forecast_date = st.date_input("Forecast Date")
        onpromotion = st.toggle("Promotion active", value=False)
        selected_profile = lookup_profile(artifacts, int(store_nbr), int(item_nbr))
        family_label = selected_profile["family"] if selected_profile else "Unknown"
        st.markdown(f"**Item Family / Category:** `{family_label}`")
        st.caption(f"Average daily units: {selected_profile.get('avg_daily_units', 0.0):.2f}")

    st.subheader("Store Conditions")
    store_col1, store_col2 = st.columns(2)
    with store_col1:
        default_inventory_value = clamp(
            float(selected_profile.get("avg_daily_units", 100.0) * 2.0),
            50.0,
            150.0,
        )
        current_inventory = st.number_input(
            "Current Inventory",
            min_value=0.0,
            value=float(default_inventory_value),
            step=5.0,
        )
        transactions_default = clamp(
            float(selected_profile.get("avg_transactions", default_transactions)),
            800.0,
            1200.0,
        )
        transactions = st.number_input(
            "Expected Transactions",
            min_value=0.0,
            value=float(transactions_default),
            step=25.0,
        )
    with store_col2:
        holiday_flag = st.selectbox("Holiday / Event", [0, 1], index=0)
        transferred_holiday = st.selectbox("Transferred Holiday", [0, 1], index=0)
        holiday_event_count = st.number_input("Holiday Event Count", min_value=0, value=0, step=1)

    st.subheader("External Factors")
    external_col1, external_col2 = st.columns(2)
    with external_col1:
        oil_default = clamp(float(selected_profile.get("dcoilwtico", default_oil)), 60.0, 80.0)
        dcoilwtico = st.number_input(
            "Oil Price",
            min_value=0.0,
            value=float(oil_default),
            step=1.0,
        )
    with external_col2:
        st.info(
            "Backend time features used: `day_of_week`, `month`, `lag_7`, and `rolling_mean_7` "
            "along with longer lags and calendar signals."
        )

    submitted = st.form_submit_button("Predict Demand")

if submitted:
    payload = preprocess_input(
        store_nbr=store_nbr,
        item_nbr=item_nbr,
        forecast_date=forecast_date,
        onpromotion=onpromotion,
        transactions=transactions,
        oil_price=dcoilwtico,
        holiday_flag=holiday_flag,
        transferred_holiday=transferred_holiday,
        holiday_event_count=holiday_event_count,
        inventory=current_inventory,
    )

    try:
        result = predict(payload, artifacts)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    stock_pressure, pressure_label, pressure_color = compute_stock_pressure(
        result["predicted_demand"],
        current_inventory,
    )

    st.subheader("Model Output")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted Demand", f"{result['predicted_demand']}")
    metric_col2.metric(
        "Demand Range",
        f"{result['demand_range_low']} - {result['demand_range_high']}",
    )
    metric_col3.metric("Stock Pressure", f"{stock_pressure:.2f}")

    st.markdown(
        f"""
        <div style="padding:0.8rem 1rem;border-radius:0.6rem;background:{pressure_color};color:white;font-weight:600;">
            Risk Status: {pressure_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("Reorder Point", f"{result['reorder_point']:.1f}")
    detail_col2.metric("Recommended Order Qty", f"{result['recommended_order_qty']:.1f}")
    detail_col3.metric("Projected Ending Inventory", f"{result['projected_ending_inventory']:.1f}")

    st.subheader("Recommendation")
    recommendation, message_type = recommendation_text(stock_pressure)
    if message_type == "success":
        st.success(recommendation)
    else:
        st.warning(recommendation)

    st.subheader("Business Insights")
    st.write(
        f"Historical avg daily units: **{result['avg_daily_units']:.2f}** | "
        f"Demand std: **{result['demand_std']:.2f}** | "
        f"Safety stock: **{result['recommended_safety_stock']:.2f}**"
    )

    st.subheader("Top Drivers")
    for factor_name, factor_detail in simulate_explanations(payload, result):
        st.write(f"- **{factor_name}:** {factor_detail}")

st.markdown("---")
st.caption("Built for retail demand forecasting using Favorita dataset")
