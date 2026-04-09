from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "favorita_merged_sample.csv"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"
ITEMS_DATA_PATH = RAW_DATA_DIR / "items.csv"
STORES_DATA_PATH = RAW_DATA_DIR / "stores.csv"
OIL_DATA_PATH = RAW_DATA_DIR / "oil.csv"
TRANSACTIONS_DATA_PATH = RAW_DATA_DIR / "transactions.csv"
HOLIDAYS_DATA_PATH = RAW_DATA_DIR / "holidays_events.csv"

MODELS_DIR = BASE_DIR / "models"
DEMAND_MODEL_PATH = MODELS_DIR / "demand_model.joblib"
STOCKOUT_MODEL_PATH = MODELS_DIR / "stock_pressure_model.joblib"
SKU_PROFILE_PATH = MODELS_DIR / "sku_profiles.json"
BACKTEST_PATH = MODELS_DIR / "backtest.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
TREE_DEMAND_MODEL_PATH = MODELS_DIR / "tree_demand_model.joblib"
TREE_STOCKOUT_MODEL_PATH = MODELS_DIR / "tree_stock_pressure_model.joblib"
TREE_METRICS_PATH = MODELS_DIR / "tree_metrics.json"
TREE_BACKTEST_PATH = MODELS_DIR / "tree_backtest.json"
TREE_SKU_PROFILE_PATH = MODELS_DIR / "tree_sku_profiles.json"

DEFAULT_MAX_TRAIN_ROWS = int(os.getenv("GROCERY_MAX_TRAIN_ROWS", "1000000"))
RANDOM_STATE = 42
TRAIN_CHUNK_SIZE = int(os.getenv("GROCERY_TRAIN_CHUNK_SIZE", "100000"))
HASH_FEATURE_DIMENSIONS = int(os.getenv("GROCERY_HASH_FEATURES", str(2**20)))
HOLDOUT_RATIO = float(os.getenv("GROCERY_HOLDOUT_RATIO", "0.1"))
STREAM_MAX_ROWS = int(os.getenv("GROCERY_STREAM_MAX_ROWS", "2500000"))
TREE_MAX_ROWS = int(os.getenv("GROCERY_TREE_MAX_ROWS", "300000"))
