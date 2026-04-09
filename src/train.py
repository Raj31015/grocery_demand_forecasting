from __future__ import annotations

import json
import sys
import time
from collections import deque
from math import sqrt

import joblib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BACKTEST_PATH,
    DEMAND_MODEL_PATH,
    HASH_FEATURE_DIMENSIONS,
    HOLDOUT_RATIO,
    METRICS_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    SKU_PROFILE_PATH,
    STOCKOUT_MODEL_PATH,
    STREAM_MAX_ROWS,
    TRAIN_CHUNK_SIZE,
    TRAIN_DATA_PATH,
)
from data_utils import iter_train_chunks, load_reference_maps
from features import make_hashed_features


def build_hasher() -> FeatureHasher:
    return FeatureHasher(n_features=HASH_FEATURE_DIMENSIONS, input_type="dict", alternate_sign=False)


def build_models() -> tuple[SGDRegressor, SGDClassifier]:
    demand_model = SGDRegressor(
        loss="huber",
        penalty="l2",
        alpha=5e-5,
        learning_rate="adaptive",
        eta0=0.01,
        epsilon=0.1,
        average=True,
        random_state=RANDOM_STATE,
    )
    stockout_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        average=True,
        random_state=RANDOM_STATE,
    )
    return demand_model, stockout_model


def _compute_balanced_sample_weights(labels: list[int]) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    if y.size == 0:
        return np.array([], dtype=float)

    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        return np.ones(y.shape[0], dtype=float)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y,
    )
    weight_map = dict(zip(unique_classes.tolist(), class_weights.tolist()))
    return np.array([weight_map[int(label)] for label in y], dtype=float)


def _date_key(timestamp) -> str:
    return timestamp.date().isoformat()


def _safe_lag(values: list[float], periods: int) -> float:
    if not values:
        return 0.0
    if len(values) >= periods:
        return float(values[-periods])
    return float(values[-1])


def _window_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))


def _window_std(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    return float(np.std(values[-window:], ddof=0))


def _make_feature_row(row, state: dict, refs: dict[str, dict]) -> dict:
    date_key = _date_key(row.date)
    item = refs["items"].get(int(row.item_nbr), {})
    store = refs["stores"].get(int(row.store_nbr), {})
    holiday = refs["holidays"].get(
        date_key,
        {
            "holiday_flag": 0,
            "transferred_holiday": 0,
            "holiday_event_count": 0,
            "local_holiday_count": 0,
            "regional_holiday_count": 0,
            "national_holiday_count": 0,
        },
    )
    history_values = list(state["history"])
    day_of_week = int(row.date.dayofweek)
    lag_1 = _safe_lag(history_values, 1)
    lag_7 = _safe_lag(history_values, 7)
    lag_14 = _safe_lag(history_values, 14)
    lag_28 = _safe_lag(history_values, 28)
    rolling_mean_7 = _window_mean(history_values, 7)
    rolling_mean_14 = _window_mean(history_values, 14)
    rolling_mean_28 = _window_mean(history_values, 28)
    rolling_std_7 = _window_std(history_values, 7)
    weekday_count = state["weekday_count"][day_of_week]
    weekday_avg_sales = (
        state["weekday_sum"][day_of_week] / weekday_count
        if weekday_count > 0
        else rolling_mean_7
    )
    return {
        "store_nbr": int(row.store_nbr),
        "item_nbr": int(row.item_nbr),
        "family": item.get("family", "unknown"),
        "class": int(item.get("class", -1)),
        "perishable": int(item.get("perishable", 0)),
        "city": store.get("city", "unknown"),
        "state": store.get("state", "unknown"),
        "store_type": store.get("store_type", "unknown"),
        "cluster": int(store.get("cluster", -1)),
        "dcoilwtico": float(refs["oil"].get(date_key, 0.0)),
        "transactions": float(refs["transactions"].get((date_key, int(row.store_nbr)), 0.0)),
        
        "holiday_flag": int(holiday["holiday_flag"]),
        "transferred_holiday": int(holiday["transferred_holiday"]),
        "holiday_event_count": int(holiday["holiday_event_count"]),
        "local_holiday_count": int(holiday["local_holiday_count"]),
        "regional_holiday_count": int(holiday["regional_holiday_count"]),
        "national_holiday_count": int(holiday["national_holiday_count"]),
        "promo_flag": int(bool(row.onpromotion)),
        
        "day_of_week": day_of_week,
        "week_of_year": int(row.date.isocalendar().week),
        "month": int(row.date.month),
        "day_of_month": int(row.date.day),
        "is_month_start": int(row.date.is_month_start),
        "is_month_end": int(row.date.is_month_end),
        "is_weekend": int(day_of_week >= 5),
        "is_payday": int(row.date.day == 15 or row.date.is_month_end),

        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "diff_lag_1_7": lag_1 - lag_7,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_14": rolling_mean_14,
        "rolling_mean_28": rolling_mean_28,
        "rolling_std_7": rolling_std_7,
        "weekday_avg_sales": float(weekday_avg_sales),
        "relative_momentum": lag_1 / (rolling_mean_7 + 1e-5),
    }


def _stock_pressure_label(unit_sales: float, rolling_mean_7: float, rolling_std_7: float) -> int:
    threshold = max(rolling_mean_7 + rolling_std_7, (rolling_mean_7 * 1.2) + 1.0)
    return int(unit_sales > threshold)


def _update_state(state: dict, row, feature_row: dict) -> None:
    unit_sales = float(max(row.unit_sales, 0.0))
    day_of_week = int(row.date.dayofweek)
    state["history"].append(unit_sales)
    state["count"] += 1
    state["sum_units"] += unit_sales
    state["sum_units_sq"] += unit_sales * unit_sales
    state["promo_count"] += int(bool(row.onpromotion))
    state["transactions_sum"] += float(feature_row["transactions"])
    state["transactions_count"] += 1
    state["weekday_sum"][day_of_week] += unit_sales
    state["weekday_count"][day_of_week] += 1
    state["latest"] = {
        "family": feature_row["family"],
        "class": feature_row["class"],
        "perishable": feature_row["perishable"],
        "city": feature_row["city"],
        "state": feature_row["state"],
        "store_type": feature_row["store_type"],
        "cluster": feature_row["cluster"],
        "lag_1": feature_row["lag_1"],
        "lag_7": feature_row["lag_7"],
        "lag_14": feature_row["lag_14"],
        "lag_28": feature_row["lag_28"],
        "diff_lag_1_7": feature_row["diff_lag_1_7"],
        "rolling_mean_7": feature_row["rolling_mean_7"],
        "rolling_mean_14": feature_row["rolling_mean_14"],
        "rolling_mean_28": feature_row["rolling_mean_28"],
        "rolling_std_7": feature_row["rolling_std_7"],
        "weekday_avg_sales": feature_row["weekday_avg_sales"],
        "relative_momentum": feature_row["relative_momentum"],
        "dcoilwtico": feature_row["dcoilwtico"],
    }


def _count_rows() -> int:
    if STREAM_MAX_ROWS and STREAM_MAX_ROWS > 0:
        return STREAM_MAX_ROWS
    with open(TRAIN_DATA_PATH, "rb") as handle:
        return sum(buffer.count(b"\n") for buffer in iter(lambda: handle.read(1024 * 1024), b"")) - 1


def _profiles_from_state(states: dict[tuple[int, int], dict]) -> list[dict]:
    profiles = []
    for (store_nbr, item_nbr), state in states.items():
        latest = state.get("latest", {})
        if not latest:
            continue
        avg_daily_units = state["sum_units"] / max(1, state["count"])
        variance = max(0.0, (state["sum_units_sq"] / max(1, state["count"])) - (avg_daily_units**2))
        profiles.append(
            {
                "store_nbr": store_nbr,
                "item_nbr": item_nbr,
                "family": latest["family"],
                "class": latest["class"],
                "perishable": latest["perishable"],
                "city": latest["city"],
                "state": latest["state"],
                "store_type": latest["store_type"],
                "cluster": latest["cluster"],
                "lag_1": latest["lag_1"],
                "lag_7": latest["lag_7"],
                "lag_14": latest["lag_14"],
                "lag_28": latest["lag_28"],
                "diff_lag_1_7": latest["diff_lag_1_7"],
                "rolling_mean_7": latest["rolling_mean_7"],
                "rolling_mean_14": latest["rolling_mean_14"],
                "rolling_mean_28": latest["rolling_mean_28"],
                "rolling_std_7": latest["rolling_std_7"],
                "weekday_avg_sales": latest["weekday_avg_sales"],
                "relative_momentum": latest["relative_momentum"],
                "dcoilwtico": latest["dcoilwtico"],
                "avg_daily_units": avg_daily_units,
                "demand_std": sqrt(variance),
                "promo_rate": state["promo_count"] / max(1, state["count"]),
                "avg_transactions": state["transactions_sum"] / max(1, state["transactions_count"]),
            }
        )
    return profiles


def _render_progress(processed_rows: int, total_rows: int, start_time: float) -> None:
    if total_rows <= 0:
        return
    ratio = min(1.0, processed_rows / total_rows)
    bar_width = 30
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.time() - start_time
    rows_per_sec = processed_rows / elapsed if elapsed > 0 else 0.0
    message = (
        f"\rTraining [{bar}] {ratio * 100:6.2f}% "
        f"({processed_rows:,}/{total_rows:,} rows) "
        f"| {rows_per_sec:,.0f} rows/s | {elapsed:,.1f}s"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    refs = load_reference_maps()
    total_rows = _count_rows()
    holdout_start = int(total_rows * (1.0 - HOLDOUT_RATIO))
    start_time = time.time()

    hasher = build_hasher()
    demand_model, stockout_model = build_models()
    scaler = MaxAbsScaler()
    states: dict[tuple[int, int], dict] = {}

    processed_rows = 0
    processed_preview = []
    demand_truth: list[float] = []
    demand_preds: list[float] = []
    cls_truth: list[int] = []
    cls_preds: list[int] = []
    naive_preds: list[float] = []
    classifier_initialized = False

    for chunk in iter_train_chunks(path=TRAIN_DATA_PATH, chunk_size=TRAIN_CHUNK_SIZE, max_rows=STREAM_MAX_ROWS or None):
        train_features = []
        train_y_reg = []
        train_y_cls = []
        holdout_features = []
        holdout_y_reg = []
        holdout_y_cls = []
        holdout_naive = []

        for row in chunk.itertuples(index=False):
            key = (int(row.store_nbr), int(row.item_nbr))
            if key not in states:
                states[key] = {
                    "history": deque(maxlen=28),
                    "count": 0,
                    "sum_units": 0.0,
                    "sum_units_sq": 0.0,
                    "promo_count": 0,
                    "transactions_sum": 0.0,
                    "transactions_count": 0,
                    "weekday_sum": [0.0] * 7,
                    "weekday_count": [0] * 7,
                    "latest": {},
                }
            state = states[key]
            feature_row = _make_feature_row(row, state, refs)
            hashed = make_hashed_features(feature_row)
            stock_label = _stock_pressure_label(float(row.unit_sales), feature_row["rolling_mean_7"], feature_row["rolling_std_7"])

            if processed_rows < 2000:
                processed_preview.append({**feature_row, "unit_sales": float(row.unit_sales), "stockout_risk": stock_label})

            if processed_rows < holdout_start:
                train_features.append(hashed)
                train_y_reg.append(np.log1p(float(row.unit_sales)))
                train_y_cls.append(stock_label)
            else:
                holdout_features.append(hashed)
                holdout_y_reg.append(float(row.unit_sales))
                holdout_y_cls.append(stock_label)
                holdout_naive.append(feature_row["lag_1"])

            _update_state(state, row, feature_row)
            processed_rows += 1

        if train_features:
            X_train = hasher.transform(train_features)
            scaler.partial_fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            demand_model.partial_fit(X_train_scaled, np.array(train_y_reg))
            cls_sample_weight = _compute_balanced_sample_weights(train_y_cls)
            if not classifier_initialized:
                stockout_model.partial_fit(
                    X_train_scaled,
                    np.array(train_y_cls),
                    classes=np.array([0, 1]),
                    sample_weight=cls_sample_weight,
                )
                classifier_initialized = True
            else:
                stockout_model.partial_fit(
                    X_train_scaled,
                    np.array(train_y_cls),
                    sample_weight=cls_sample_weight,
                )

        if holdout_features and classifier_initialized:
            X_holdout = hasher.transform(holdout_features)
            X_holdout_scaled = scaler.transform(X_holdout)
            chunk_demand_preds = np.expm1(demand_model.predict(X_holdout_scaled)).clip(min=0)
            chunk_cls_preds = stockout_model.predict(X_holdout_scaled)
            demand_truth.extend(holdout_y_reg)
            demand_preds.extend(chunk_demand_preds.tolist())
            cls_truth.extend(holdout_y_cls)
            cls_preds.extend(chunk_cls_preds.tolist())
            naive_preds.extend(holdout_naive)

        _render_progress(processed_rows, total_rows, start_time)

    sys.stdout.write("\n")
    sys.stdout.flush()

    if processed_preview:
        import pandas as pd

        pd.DataFrame(processed_preview).to_csv(PROCESSED_DATA_PATH, index=False)

    sku_profiles = _profiles_from_state(states)
    metrics = {
        "demand_mae": float(mean_absolute_error(demand_truth, demand_preds)) if demand_truth else None,
        "naive_baseline_mae": float(mean_absolute_error(demand_truth, naive_preds)) if demand_truth else None,
        "stock_pressure_accuracy": float(accuracy_score(cls_truth, cls_preds)) if cls_truth else None,
        "stock_pressure_f1": float(f1_score(cls_truth, cls_preds, zero_division=0)) if cls_truth else None,
        "average_backtest_mae": None,
        "rows_used": int(total_rows),
        "holdout_rows": int(len(demand_truth)),
        "training_mode": "streaming_full_dataset",
        "chunk_size": int(TRAIN_CHUNK_SIZE),
    }

    backtest_results = [
        {
            "mode": "streaming_holdout",
            "rows_used": int(total_rows),
            "holdout_rows": int(len(demand_truth)),
            "mae_model": metrics["demand_mae"],
            "mae_naive_baseline": metrics["naive_baseline_mae"],
            "pressure_f1": metrics["stock_pressure_f1"],
        }
    ]

    joblib.dump({"model": demand_model, "scaler": scaler}, DEMAND_MODEL_PATH)
    joblib.dump({"model": stockout_model, "scaler": scaler}, STOCKOUT_MODEL_PATH)
    SKU_PROFILE_PATH.write_text(json.dumps(sku_profiles, indent=2))
    BACKTEST_PATH.write_text(json.dumps(backtest_results, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
