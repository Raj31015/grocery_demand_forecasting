from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    HASH_FEATURE_DIMENSIONS,
    DEFAULT_MAX_TRAIN_ROWS,
    HOLIDAYS_DATA_PATH,
    ITEMS_DATA_PATH,
    OIL_DATA_PATH,
    STORES_DATA_PATH,
    STREAM_MAX_ROWS,
    TRAIN_CHUNK_SIZE,
    TRAIN_DATA_PATH,
    TRANSACTIONS_DATA_PATH,
)


TRAIN_DTYPES = {
    "id": "int64",
    "store_nbr": "int16",
    "item_nbr": "int32",
    "unit_sales": "float32",
    "onpromotion": "boolean",
}


def _load_train(path: Path = TRAIN_DATA_PATH, max_rows: int | None = DEFAULT_MAX_TRAIN_ROWS) -> pd.DataFrame:
    train = pd.read_csv(
        path,
        nrows=max_rows,
        parse_dates=["date"],
        dtype=TRAIN_DTYPES,
        usecols=["id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
    )
    train["onpromotion"] = train["onpromotion"].fillna(False).astype(bool)
    train["unit_sales"] = train["unit_sales"].clip(lower=0)
    return train


def _load_items() -> pd.DataFrame:
    return pd.read_csv(
        ITEMS_DATA_PATH,
        dtype={
            "item_nbr": "int32",
            "family": "string",
            "class": "int32",
            "perishable": "int8",
        },
    )


def _load_stores() -> pd.DataFrame:
    stores = pd.read_csv(
        STORES_DATA_PATH,
        dtype={
            "store_nbr": "int16",
            "city": "string",
            "state": "string",
            "type": "string",
            "cluster": "int8",
        },
    )
    return stores.rename(columns={"type": "store_type"})


def _load_oil() -> pd.DataFrame:
    oil = pd.read_csv(OIL_DATA_PATH, parse_dates=["date"])
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    return oil


def _load_transactions() -> pd.DataFrame:
    return pd.read_csv(
        TRANSACTIONS_DATA_PATH,
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "transactions": "float32"},
    )


def _load_holidays() -> pd.DataFrame:
    holidays = pd.read_csv(HOLIDAYS_DATA_PATH, parse_dates=["date"])
    holidays["transferred"] = holidays["transferred"].fillna(False).astype(bool)
    holidays["is_holiday_event"] = holidays["type"].isin(["Holiday", "Additional", "Bridge", "Event"])
    holiday_features = (
        holidays.groupby("date").agg(
            holiday_flag=("is_holiday_event", "max"),
            transferred_holiday=("transferred", "max"),
            holiday_event_count=("description", "count"),
            local_holiday_count=("locale", lambda s: int((s == "Local").sum())),
            regional_holiday_count=("locale", lambda s: int((s == "Regional").sum())),
            national_holiday_count=("locale", lambda s: int((s == "National").sum())),
        )
    ).reset_index()
    bool_cols = ["holiday_flag", "transferred_holiday"]
    for col in bool_cols:
        holiday_features[col] = holiday_features[col].astype(int)
    return holiday_features


def load_dataset(max_rows: int | None = DEFAULT_MAX_TRAIN_ROWS) -> pd.DataFrame:
    train = _load_train(max_rows=max_rows)
    items = _load_items()
    stores = _load_stores()
    oil = _load_oil()
    transactions = _load_transactions()
    holidays = _load_holidays()

    merged = train.merge(items, on="item_nbr", how="left")
    merged = merged.merge(stores, on="store_nbr", how="left")
    merged = merged.merge(oil, on="date", how="left")
    merged = merged.merge(transactions, on=["date", "store_nbr"], how="left")
    merged = merged.merge(holidays, on="date", how="left")

    fill_zero_cols = [
        "transactions",
        "holiday_flag",
        "transferred_holiday",
        "holiday_event_count",
        "local_holiday_count",
        "regional_holiday_count",
        "national_holiday_count",
    ]
    for col in fill_zero_cols:
        merged[col] = merged[col].fillna(0)

    text_cols = ["family", "city", "state", "store_type"]
    for col in text_cols:
        merged[col] = merged[col].fillna("unknown")

    merged["dcoilwtico"] = merged["dcoilwtico"].ffill().bfill()
    return merged.sort_values(["date", "store_nbr", "item_nbr"]).reset_index(drop=True)


def iter_train_chunks(
    path: Path = TRAIN_DATA_PATH,
    chunk_size: int = TRAIN_CHUNK_SIZE,
    max_rows: int | None = STREAM_MAX_ROWS or None,
):
    rows_left = max_rows
    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        parse_dates=["date"],
        dtype=TRAIN_DTYPES,
        usecols=["id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
    )
    for chunk in reader:
        if rows_left is not None:
            if rows_left <= 0:
                break
            if len(chunk) > rows_left:
                chunk = chunk.iloc[:rows_left].copy()
            rows_left -= len(chunk)
        chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype(bool)
        chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)
        yield chunk


def load_reference_maps() -> dict[str, dict]:
    items = _load_items()
    stores = _load_stores()
    oil = _load_oil()
    transactions = _load_transactions()
    holidays = _load_holidays()

    items_map = items.set_index("item_nbr").to_dict(orient="index")
    stores_map = stores.set_index("store_nbr").to_dict(orient="index")
    oil_map = {
        row.date.date().isoformat(): float(row.dcoilwtico)
        for row in oil.itertuples(index=False)
    }
    transactions_map = {
        (row.date.date().isoformat(), int(row.store_nbr)): float(row.transactions)
        for row in transactions.itertuples(index=False)
    }
    holidays_map = {
        row.date.date().isoformat(): {
            "holiday_flag": int(row.holiday_flag),
            "transferred_holiday": int(row.transferred_holiday),
            "holiday_event_count": int(row.holiday_event_count),
            "local_holiday_count": int(row.local_holiday_count),
            "regional_holiday_count": int(row.regional_holiday_count),
            "national_holiday_count": int(row.national_holiday_count),
        }
        for row in holidays.itertuples(index=False)
    }

    return {
        "items": items_map,
        "stores": stores_map,
        "oil": oil_map,
        "transactions": transactions_map,
        "holidays": holidays_map,
    }
