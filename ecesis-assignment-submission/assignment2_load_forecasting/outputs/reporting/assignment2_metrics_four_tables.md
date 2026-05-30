# Assignment 2 Metrics Summary Tables

Source files:

- assignment2_method1_direct_bus_metrics.csv
- assignment2_metrics.csv
- assignment2_weather_metrics.csv

## Table 1. Next-day bus-level accuracy

| Model                           |   MAE |   RMSE |   WMAPE |
|:--------------------------------|------:|-------:|--------:|
| direct_bus_next_day             | 1.387 | 11.308 |   0.249 |
| baseline_hour_weekday_avg       | 1.931 |  9.283 |   0.347 |
| baseline_previous_week          | 0.905 |  5.204 |   0.163 |
| zone_share_transformer_next_day | 0.892 |  4.561 |   0.16  |
| zone_share_weather_transformer_next_day         | 0.946 |  4.731 |   0.17  |

## Table 2. Next-day zone-level aggregated accuracy

| Model                           |      MAE |     RMSE |   WMAPE |
|:--------------------------------|---------:|---------:|--------:|
| direct_bus_next_day             |  760.635 | 1194.13  |   0.109 |
| baseline_hour_weekday_avg       | 1197.83  | 1768.27  |   0.172 |
| baseline_previous_week          |  787.322 | 1449.35  |   0.113 |
| zone_share_transformer_next_day |  619.951 | 1004.41  |   0.089 |
| zone_share_weather_transformer_next_day         |  623.694 |  863.149 |   0.09  |

## Table 3. Next-month bus-level accuracy

| Model                             |   MAE |   RMSE |   WMAPE |
|:----------------------------------|------:|-------:|--------:|
| direct_bus_next_month             | 2.382 | 12.988 |   0.428 |
| baseline_hour_weekday_avg         | 1.931 |  9.283 |   0.347 |
| baseline_previous_year            | 1.833 |  9.332 |   0.33  |
| zone_share_transformer_next_month | 1.836 |  9.155 |   0.33  |
| zone_share_weather_transformer_next_month         | 1.743 |  9.037 |   0.313 |

## Table 4. Next-month zone-level aggregated accuracy

| Model                             |      MAE |    RMSE |   WMAPE |
|:----------------------------------|---------:|--------:|--------:|
| direct_bus_next_month             | 1597.52  | 2401.65 |   0.229 |
| baseline_hour_weekday_avg         | 1197.83  | 1768.27 |   0.172 |
| baseline_previous_year            |  798.181 | 1330.65 |   0.115 |
| zone_share_transformer_next_month | 1027.47  | 1672.51 |   0.147 |
| zone_share_weather_transformer_next_month         |  783.115 | 1273.68 |   0.112 |
