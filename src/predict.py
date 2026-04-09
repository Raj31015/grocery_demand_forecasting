from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .config import DEMAND_MODEL_PATH, HASH_FEATURE_DIMENSIONS, SKU_PROFILE_PATH, STOCKOUT_MODEL_PATH
from .features import make_hashed_features
from .train import build_hasher


def _load_profiles() -> list[dict]:
    if not SKU_PROFILE_PATH.exists():
        return []
    return json.loads(SKU_PROFILE_PATH.read_text())


def _lookup_profile(store_nbr: int, item_nbr: int) -> dict | None:
    for row in _load_profiles():
        if int(row["store_nbr"]) == int(store_nbr) and int(row["item_nbr"]) == int(item_nbr):
            return row
    return None


def _risk_tier(stockout_probability: float) -> str:
    if stockout_probability >= 0.7:
        return "high"
    if stockout_probability >= 0.4:
        return "medium"
    return "low"


def _coalesce(value: object, fallback: object) -> object:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    return value


def _build_feature_row(features: dict, profile: dict) -> tuple[dict, float]:
    forecast_date = pd.to_datetime(features.get("date", pd.Timestamp.today().normalize()))
    current_inventory = float(features.get("current_inventory", 0.0))
    transactions = float(_coalesce(features.get("transactions"), profile.get("avg_transactions", 0.0)))
    oil_price = float(_coalesce(features.get("dcoilwtico"), profile.get("dcoilwtico", 0.0)))

    row = {
        "store_nbr": int(features["store_nbr"]),
        "item_nbr": int(features["item_nbr"]),
        "family": profile["family"],
        "class": int(profile["class"]),
        "perishable": int(profile["perishable"]),
        "city": profile["city"],
        "state": profile["state"],
        "store_type": profile["store_type"],
        "cluster": int(profile["cluster"]),
        "dcoilwtico": oil_price,
        "transactions": transactions,
        "holiday_flag": int(features.get("holiday_flag", 0)),
        "transferred_holiday": int(features.get("transferred_holiday", 0)),
        "holiday_event_count": int(features.get("holiday_event_count", 0)),
        "local_holiday_count": int(features.get("local_holiday_count", 0)),
        "regional_holiday_count": int(features.get("regional_holiday_count", 0)),
        "national_holiday_count": int(features.get("national_holiday_count", 0)),
        "promo_flag": int(bool(features.get("onpromotion", False))),
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
        "diff_lag_1_7": float(
            profile.get("diff_lag_1_7", profile.get("lag_1", 0.0) - profile.get("lag_7", 0.0))
        ),
        "rolling_mean_7": float(profile.get("rolling_mean_7", profile.get("avg_daily_units", 0.0))),
        "rolling_mean_14": float(profile.get("rolling_mean_14", profile.get("avg_daily_units", 0.0))),
        "rolling_mean_28": float(profile.get("rolling_mean_28", profile.get("avg_daily_units", 0.0))),
        "rolling_std_7": float(profile.get("rolling_std_7", profile.get("demand_std", 0.0))),
        "weekday_avg_sales": float(profile.get("weekday_avg_sales", profile.get("avg_daily_units", 0.0))),
        "relative_momentum": float(profile.get("relative_momentum", 1.0)),
    }
    return row, current_inventory


def predict_row(features: dict) -> dict:
    demand_bundle = joblib.load(DEMAND_MODEL_PATH)
    stockout_bundle = joblib.load(STOCKOUT_MODEL_PATH)
    hasher = build_hasher()
    demand_model = demand_bundle["model"]
    stockout_model = stockout_bundle["model"]
    scaler = demand_bundle["scaler"]

    profile = _lookup_profile(features["store_nbr"], features["item_nbr"])
    if profile is None:
        raise ValueError("No trained profile found for the selected store/item combination.")

    row, current_inventory = _build_feature_row(features, profile)
    X = scaler.transform(hasher.transform([make_hashed_features(row)]))

    demand = float(np.expm1(demand_model.predict(X)[0]))
    stockout_prob = float(stockout_model.predict_proba(X)[0, 1])
    risk_tier = _risk_tier(stockout_prob)

    avg_daily_units = float(profile.get("avg_daily_units", row["rolling_mean_7"]))
    demand_std = max(1.0, float(profile.get("demand_std", row["rolling_std_7"])))
    lead_time_days = 2
    safety_stock = max(2.0, 1.65 * demand_std)
    reorder_point = (avg_daily_units * lead_time_days) + safety_stock
    recommended_order_qty = max(0.0, round(reorder_point + demand - current_inventory, 2))
    projected_inventory_gap = round(current_inventory - demand, 2)

    return {
        "predicted_unit_sales": round(max(demand, 0.0), 2),
        "stockout_probability": round(stockout_prob, 4),
        "stockout_risk_tier": risk_tier,
        "recommended_order_qty": recommended_order_qty,
        "recommended_safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "projected_inventory_gap": projected_inventory_gap,
        "avg_daily_units": round(avg_daily_units, 2),
        "demand_std": round(demand_std, 2),
    }
