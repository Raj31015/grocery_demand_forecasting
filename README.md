# Grocery Demand Forecasting

## Problem

This project now uses the real Favorita grocery dataset already present in `data/raw/`:

- `train.csv`
- `test.csv`
- `items.csv`
- `stores.csv`
- `oil.csv`
- `transactions.csv`
- `holidays_events.csv`

The pipeline merges those files, engineers time-series and calendar features, predicts next-day unit demand, and trains a secondary stock-pressure classifier as a proxy risk signal. The classifier is a proxy because the real dataset does not include true inventory or stockout labels.

## Features Used

The training code now works with columns that actually exist in the raw data:

- sales history: `unit_sales`, `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_std_7`
- promotions: `onpromotion`
- item metadata: `family`, `class`, `perishable`
- store metadata: `city`, `state`, `type`, `cluster`
- external signals: `transactions`, `dcoilwtico`
- calendar and holiday features derived from `date` and `holidays_events.csv`

Fields from the previous synthetic version such as `inventory_on_hand`, `unit_price`, `category`, and `grocery_sales.csv` are no longer required.

## Run

```powershell
python -m src.train
streamlit run app\streamlit_app.py
```

By default, training now streams the first `500000` rows from `train.csv`, which is a practical balance for a resume-ready project on a local machine. You can change that with:

```powershell
$env:GROCERY_STREAM_MAX_ROWS="300000"
python -m src.train
```

## Notes

- The saved merged sample is written to `data/processed/favorita_merged_sample.csv`.
- The app expects trained models and SKU profiles to exist before it starts.
- Replenishment outputs use historical demand plus optional user-entered current inventory, since inventory is not part of the raw dataset.
- You can rerun training anytime with `python -m src.train`; it will overwrite the saved models and metrics.
