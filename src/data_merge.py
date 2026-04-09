from __future__ import annotations

from .config import DEFAULT_MAX_TRAIN_ROWS, PROCESSED_DATA_PATH, PROCESSED_DIR
from .data_utils import load_dataset


def build_merged_dataset(max_rows: int = DEFAULT_MAX_TRAIN_ROWS) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(max_rows=max_rows)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    build_merged_dataset()
