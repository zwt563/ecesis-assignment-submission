# Assignment 2 Outputs and Cache

This folder contains both final deliverables and cached artifacts used by the notebook.

## Final Forecasts

- `assignment2_next_day_forecast_2025.parquet`
- `assignment2_next_month_forecast_2025.parquet`

These are the primary full-year bus-level forecasts. Each has about 88 million rows, so Parquet is used instead of full CSV.

## Review Files

- `assignment2_model_comparison.csv`
- `assignment2_metrics.csv`
- `assignment2_method1_direct_bus_metrics.csv`
- `assignment2_weather_metrics.csv`
- `assignment2_next_day_forecast_sample.csv`
- `assignment2_next_month_forecast_sample.csv`
- `assignment2_weather_next_day_forecast_sample.csv`
- `assignment2_weather_next_month_forecast_sample.csv`
- `assignment2_method2_zone_accuracy_by_zone.csv`
- `assignment2_weather_zone_accuracy_by_zone.csv`

## Cached Artifacts

Keep these files/folders if you want the notebook to run in reuse/validation mode without rebuilding expensive stages:

- `assignment2_source_share_cache/`
- `weather_data/`
- `assignment2_next_day_zone_predictions.parquet`
- `assignment2_next_month_zone_predictions.parquet`
- `assignment2_weather_next_day_zone_predictions.parquet`
- `assignment2_weather_next_month_zone_predictions.parquet`
- `assignment2_method1_direct_bus_imputer.pkl`
- Method 1 `.pt` model checkpoints
- Method 1 training/profile parquet files
