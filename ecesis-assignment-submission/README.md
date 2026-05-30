# Ecesis Assignment Submission

This repository mirror contains the completed Ecesis Investments 2026 Summer internship assignments.

## Layout

- `data/` contains the raw input data needed by the notebooks.
- `assignment1_constraint_mapping/` contains the constraint-mapping notebook, matched output CSV, and report.
- `assignment2_load_forecasting/` contains the bus-level load-forecasting notebook, code modules, final forecasts, metrics, reports, AI log, and cached artifacts. The core Python modules are kept under `code/` for review and mirrored under `outputs/` so the notebook can import the runtime copy beside the cached outputs.
- `assignment3_bus_mapping/` contains the optional bus-mapping notebook, output CSV, and report.

## Assignment 2 Runtime Note

Assignment 2 is intentionally packaged with cached artifacts because the full bus-share allocation writes roughly 88 million bus-hour rows per horizon and is slow to rebuild. The notebook defaults are set to reuse existing outputs and validation caches.

Important cached folders/files:

- `assignment2_load_forecasting/outputs/assignment2_source_share_cache/`
- `assignment2_load_forecasting/outputs/weather_data/`
- `assignment2_load_forecasting/outputs/assignment2_next_day_zone_predictions.parquet`
- `assignment2_load_forecasting/outputs/assignment2_next_month_zone_predictions.parquet`
- Method 1 `.pt`, `.pkl`, and training-cache parquet files in `assignment2_load_forecasting/outputs/`

The full Assignment 2 forecast files are stored as Parquet rather than CSV because each full-year bus forecast has about 88 million rows. Sample CSVs and metrics CSVs are included for quick review.

Primary full forecast files:

- `assignment2_load_forecasting/outputs/assignment2_next_day_forecast_2025.parquet`
- `assignment2_load_forecasting/outputs/assignment2_next_month_forecast_2025.parquet`

Template-friendly aliases are also provided:

- `assignment2_load_forecasting/outputs/assignment2_next_day_forecast.parquet`
- `assignment2_load_forecasting/outputs/assignment2_next_month_forecast.parquet`

## How to Review

1. Create an environment from `requirements.txt`.
2. Open each notebook from its `notebook/` folder.
3. For Assignment 2, use the existing default flags. They are set to validation/reuse mode, not retraining mode.
4. Review reports under each assignment's `report/` folder and outputs under each assignment's `outputs/` folder.

## Main Deliverables

Assignment 1:

- `assignment1_constraint_mapping/outputs/assignment1_matched_results.csv`
- `assignment1_constraint_mapping/report/assignment1_summary_report.md`

Assignment 2:

- `assignment2_load_forecasting/outputs/assignment2_next_day_forecast_2025.parquet`
- `assignment2_load_forecasting/outputs/assignment2_next_month_forecast_2025.parquet`
- `assignment2_load_forecasting/outputs/assignment2_model_comparison.csv`
- `assignment2_load_forecasting/report/assignment2_summary_report.md`
- `assignment2_load_forecasting/ai_usage_logs/assignment2_ai_usage_log.md`

Assignment 3:

- `assignment3_bus_mapping/outputs/assignment3_matched_results.csv`
- `assignment3_bus_mapping/report/assignment3_summary_report.md`
