# Data Directory

This folder keeps the raw input files used by the notebooks.

## Included Data

- `Assignment 1 - constraint_mapping/`
  - `Market PJMISO constraint list.csv`
  - `Dayzer PJMISO constraint list.csv`
  - `Pano PJMISO constraint list.csv`

- `Assignment 2 - load_forecasting/`
  - `bus_load_2022.parquet`
  - `bus_load_2023.parquet`
  - `bus_load_2024.parquet`
  - `bus_load_2025.parquet`
  - `zone_load_2022.parquet`
  - `zone_load_2023.parquet`
  - `zone_load_2024.parquet`
  - `zone_load_2025.parquet`

- `Assignment 3 - bus_mapping/`
  - `DAYZER_BUS_GEO.csv`
  - `DAYZER_BRANCH.csv`
  - `PANO_BUS_GEO.csv`
  - `PANO_BRANCH.csv`

The Assignment 2 notebook also uses cached intermediate artifacts stored under `assignment2_load_forecasting/outputs/` so reviewers do not need to rerun the multi-hour bus-share allocation stage.
