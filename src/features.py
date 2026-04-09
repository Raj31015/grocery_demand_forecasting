from __future__ import annotations

import numpy as np
import pandas as pd


def _group_shift_feature(
    data: pd.DataFrame,
    group_cols: list[str],
    source_col: str,
    periods: int,
) -> pd.Series:
    return data.groupby(group_cols, sort=False)[source_col].shift(periods)


def _group_rolling_feature(
    data: pd.DataFrame,
    group_cols: list[str],
    source_col: str,
    window: int,
    stat: str,
) -> pd.Series:
    if stat == "mean":
        return data.groupby(group_cols, sort=False)[source_col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).mean()
        )
    if stat == "std":
        return data.groupby(group_cols, sort=False)[source_col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )
    raise ValueError(f"Unsupported rolling stat: {stat}")


def make_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    data = df.copy().sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
    data["promo_flag"] = data["onpromotion"].fillna(False).astype(int)
    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["month"] = data["date"].dt.month
    data["day_of_month"] = data["date"].dt.day
    data["is_month_start"] = data["date"].dt.is_month_start.astype(int)
    data["is_month_end"] = data["date"].dt.is_month_end.astype(int)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    group_cols = ["store_nbr", "item_nbr"]
    data["lag_1"] = _group_shift_feature(data, group_cols, "unit_sales", 1)
    data["lag_7"] = _group_shift_feature(data, group_cols, "unit_sales", 7)
    data["rolling_mean_7"] = _group_rolling_feature(data, group_cols, "unit_sales", 7, "mean")
    data["rolling_std_7"] = _group_rolling_feature(data, group_cols, "unit_sales", 7, "std")

    global_median = float(data["unit_sales"].median()) if "unit_sales" in data else 0.0
    data["lag_1"] = data["lag_1"].fillna(global_median)
    data["lag_7"] = data["lag_7"].fillna(global_median)
    data["rolling_mean_7"] = data["rolling_mean_7"].fillna(data["lag_1"])
    data["rolling_std_7"] = data["rolling_std_7"].fillna(0.0)
    data["transactions"] = data["transactions"].fillna(data["transactions"].median())
    data["dcoilwtico"] = data["dcoilwtico"].ffill().bfill()

    if training and "unit_sales" in data:
        threshold = np.maximum(
            data["rolling_mean_7"] + data["rolling_std_7"],
            (data["rolling_mean_7"] * 1.2) + 1.0,
        )
        data["stockout_risk"] = (data["unit_sales"] > threshold).astype(int)

    return data


def make_hashed_features(feature_row: dict) -> dict[str, float]:
    hashed: dict[str, float] = {}
    categorical_cols = [
        "store_nbr",
        "item_nbr",
        "family",
        "class",
        "perishable",
        "city",
        "state",
        "store_type",
        "cluster",
    ]
    for col in categorical_cols:
        hashed[f"{col}={feature_row[col]}"] = 1.0

    interaction_tokens = [
        f"item_dayofweek={feature_row['item_nbr']}_{feature_row['day_of_week']}",
        f"store_item={feature_row['store_nbr']}_{feature_row['item_nbr']}",
        f"store_item_day={feature_row['store_nbr']}_{feature_row['item_nbr']}_{feature_row['day_of_week']}",
        f"family_dayofweek={feature_row['family']}_{feature_row['day_of_week']}",
        f"promo_item={feature_row['item_nbr']}_{feature_row['promo_flag']}",
        f"item_weekofyear={feature_row['item_nbr']}_{feature_row['week_of_year']}",
    ]
    for token in interaction_tokens:
        hashed[token] = 1.0

    numeric_cols = [
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
    for col in numeric_cols:
        hashed[col] = float(feature_row[col])
    return hashed
