from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TREE_BACKTEST_PATH,
    TREE_DEMAND_MODEL_PATH,
    TREE_MAX_ROWS,
    TREE_METRICS_PATH,
    TREE_SKU_PROFILE_PATH,
    TREE_STOCKOUT_MODEL_PATH,
)
from src.data_utils import load_dataset


TREE_FEATURE_COLS = [
    "store_nbr",
    "item_nbr",
    "family",
    "class",
    "perishable",
    "city",
    "state",
    "store_type",
    "cluster",
    "dcoilwtico",
    "transactions",
    "holiday_flag",
    "transferred_holiday",
    "holiday_event_count",
    "local_holiday_count",
    "regional_holiday_count",
    "national_holiday_count",
    "promo_flag",
    "day_of_week",
    "week_of_year",
    "month",
    "day_of_month",
    "is_month_start",
    "is_month_end",
    "is_weekend",
    "is_payday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "diff_lag_1_7",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "weekday_avg_sales",
    "relative_momentum",
]

CATEGORICAL_FEATURES = ["family", "city", "state", "store_type"]
NUMERIC_FEATURES = [col for col in TREE_FEATURE_COLS if col not in CATEGORICAL_FEATURES]


def _add_tree_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
    data["promo_flag"] = data["onpromotion"].fillna(False).astype(int)
    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["month"] = data["date"].dt.month
    data["day_of_month"] = data["date"].dt.day
    data["is_month_start"] = data["date"].dt.is_month_start.astype(int)
    data["is_month_end"] = data["date"].dt.is_month_end.astype(int)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_payday"] = ((data["day_of_month"] == 15) | data["date"].dt.is_month_end).astype(int)

    group_cols = ["store_nbr", "item_nbr"]
    grouped_sales = data.groupby(group_cols, sort=False)["unit_sales"]
    for lag in [1, 7, 14, 28]:
        data[f"lag_{lag}"] = grouped_sales.shift(lag)

    data["diff_lag_1_7"] = data["lag_1"] - data["lag_7"]
    data["rolling_mean_7"] = grouped_sales.transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    data["rolling_mean_14"] = grouped_sales.transform(lambda s: s.shift(1).rolling(14, min_periods=2).mean())
    data["rolling_mean_28"] = grouped_sales.transform(lambda s: s.shift(1).rolling(28, min_periods=2).mean())
    data["rolling_std_7"] = grouped_sales.transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
    data["weekday_avg_sales"] = (
        data.groupby(["store_nbr", "item_nbr", "day_of_week"], sort=False)["unit_sales"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    global_median = float(data["unit_sales"].median())
    fill_with_global = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]
    for col in fill_with_global:
        data[col] = data[col].fillna(global_median)

    data["rolling_std_7"] = data["rolling_std_7"].fillna(0.0)
    data["weekday_avg_sales"] = data["weekday_avg_sales"].fillna(data["rolling_mean_7"])
    data["diff_lag_1_7"] = data["diff_lag_1_7"].fillna(0.0)
    data["relative_momentum"] = data["lag_1"] / (data["rolling_mean_7"] + 1e-5)
    data["transactions"] = data["transactions"].fillna(data["transactions"].median())
    data["dcoilwtico"] = data["dcoilwtico"].ffill().bfill()

    threshold = np.maximum(
        data["rolling_mean_7"] + data["rolling_std_7"],
        (data["rolling_mean_7"] * 1.2) + 1.0,
    )
    data["stockout_risk"] = (data["unit_sales"] > threshold).astype(int)
    return data


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _build_models() -> tuple[Pipeline, Pipeline]:
    demand_model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                GradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=6,
                    n_estimators=220,
                    min_samples_leaf=40,
                    subsample=0.9,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cls_model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                GradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=5,
                    n_estimators=180,
                    min_samples_leaf=40,
                    subsample=0.9,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return demand_model, cls_model


def _build_profiles(df: pd.DataFrame) -> list[dict]:
    latest_rows = df.sort_values("date").groupby(["store_nbr", "item_nbr"], as_index=False).tail(1).copy()
    profiles = (
        latest_rows.merge(
            df.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
                avg_daily_units=("unit_sales", "mean"),
                demand_std=("unit_sales", "std"),
                promo_rate=("promo_flag", "mean"),
                avg_transactions=("transactions", "mean"),
            ),
            on=["store_nbr", "item_nbr"],
            how="left",
        )
        .fillna({"demand_std": 0.0, "avg_transactions": 0.0, "promo_rate": 0.0})
    )
    return profiles[
        [
            "store_nbr",
            "item_nbr",
            "family",
            "class",
            "perishable",
            "city",
            "state",
            "store_type",
            "cluster",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "diff_lag_1_7",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28",
            "rolling_std_7",
            "weekday_avg_sales",
            "relative_momentum",
            "dcoilwtico",
            "avg_daily_units",
            "demand_std",
            "promo_rate",
            "avg_transactions",
        ]
    ].to_dict(orient="records")


def train_tree() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(max_rows=TREE_MAX_ROWS)
    df = _add_tree_features(raw_df)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    demand_model, cls_model = _build_models()
    X_train = train_df[TREE_FEATURE_COLS]
    X_test = test_df[TREE_FEATURE_COLS]

    demand_model.fit(X_train, np.log1p(train_df["unit_sales"]))
    cls_model.fit(X_train, train_df["stockout_risk"])

    demand_preds = np.expm1(demand_model.predict(X_test)).clip(min=0)
    cls_preds = cls_model.predict(X_test)
    naive_lag1 = test_df["lag_1"]
    naive_lag7 = test_df["lag_7"]
    naive_roll7 = test_df["rolling_mean_7"]

    metrics = {
        "tree_subset_rows": int(len(df)),
        "holdout_rows": int(len(test_df)),
        "demand_mae": float(mean_absolute_error(test_df["unit_sales"], demand_preds)),
        "naive_lag1_mae": float(mean_absolute_error(test_df["unit_sales"], naive_lag1)),
        "naive_lag7_mae": float(mean_absolute_error(test_df["unit_sales"], naive_lag7)),
        "naive_roll7_mae": float(mean_absolute_error(test_df["unit_sales"], naive_roll7)),
        "demand_rmsle": float(np.sqrt(mean_squared_log_error(test_df["unit_sales"], demand_preds))),
        "stock_pressure_accuracy": float(accuracy_score(test_df["stockout_risk"], cls_preds)),
        "stock_pressure_f1": float(f1_score(test_df["stockout_risk"], cls_preds, zero_division=0)),
        "training_mode": "tree_subset",
    }

    backtest = [
        {
            "mode": "subset_holdout",
            "rows_used": int(len(df)),
            "holdout_rows": int(len(test_df)),
            "mae_model": metrics["demand_mae"],
            "mae_naive_lag1": metrics["naive_lag1_mae"],
            "mae_naive_lag7": metrics["naive_lag7_mae"],
            "mae_naive_roll7": metrics["naive_roll7_mae"],
            "pressure_f1": metrics["stock_pressure_f1"],
        }
    ]

    joblib.dump(demand_model, TREE_DEMAND_MODEL_PATH)
    joblib.dump(cls_model, TREE_STOCKOUT_MODEL_PATH)
    TREE_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    TREE_BACKTEST_PATH.write_text(json.dumps(backtest, indent=2))
    TREE_SKU_PROFILE_PATH.write_text(json.dumps(_build_profiles(df), indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_tree()
