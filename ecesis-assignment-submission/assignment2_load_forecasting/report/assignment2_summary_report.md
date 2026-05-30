
# Assignment 2 Summary Report

## Objective

The goal of this assignment is to build a reproducible bus-level load forecasting pipeline for 2025. The required forecast tasks are:

* next-day load forecast
* next-month load forecast

The final forecast output is at the bus-hour level, but I also evaluate the results after aggregating bus forecasts back to the zone level. This is important because the model has two related goals: it should predict total zone load well, and it should also distribute that load reasonably across buses.

The metrics used in this report are:

`MAE = mean(abs(actual - forecast))`

`RMSE = sqrt(mean((actual - forecast)^2))`

`WMAPE = sum(abs(actual - forecast)) / sum(actual)`

I evaluate separately for:

* next-day bus-level accuracy
* next-day zone-level aggregated accuracy
* next-month bus-level accuracy
* next-month zone-level aggregated accuracy

## Model Used

My main model is a zone-level Transformer forecast plus bus-share allocation pipeline.

I chose this structure because the bus-level dataset is very large, sparse, and noisy. Many buses have zero load, missing `pd`, unstable history, or appear only in some years. If I train one model directly on all bus-hour rows, the model can be dominated by zero or missing rows. Zone-level load is more stable because it aggregates many buses, so it is a better target for learning the main load pattern.

The pipeline has two stages.

First, I forecast zone-hour load using a Transformer regression model. Each input feature is treated as a token, so the model can learn interactions among calendar variables, lagged load, historical averages, and optional weather variables. The model is trained separately by zone because different zones have different load patterns.

Second, I allocate the predicted zone-hour load back to individual buses using historical bus shares. For each target `date`, `he`, and `zone_name`, I calculate how much each bus historically contributed to that zone-hour load. Then I multiply the predicted zone load by that bus share.

The bus-share logic is:

`bus_share = historical_bus_load / historical_total_zone_load`

`bus_forecast = zone_forecast * bus_share`

I also tested a direct bus-level Transformer model. This model predicts bus-hour load directly using bus-level features, zone embeddings, hashed bus-id embeddings, and historical baseline features. However, the direct bus model performed worse than the zone-share approach, especially for the next-month task. Because of this, I treat the zone forecast plus bus-share allocation method as the main approach.

I also built a weather-aware version of Method 2. The reason is that next-month forecasting is more sensitive to weather and temperature-driven demand. The no-weather model can learn average seasonal patterns from historical load, but it cannot fully know if a target month is unusually hot or cold. The weather-aware model adds zone-level weather features based on representative geographic points for each zone.

## Baselines Compared Against

I compared the models against three main baselines.

### Previous-Week Baseline

For the next-day task, I use the same bus and hour from the previous week. This is a strong baseline because load has a clear weekly pattern.

### Previous-Year Baseline

For the next-month task, I use the same bus, calendar date, and hour from the previous year. This baseline captures seasonal load patterns and is especially strong for month-ahead forecasting.

### Historical Hour-Weekday Average Baseline

I also use a historical average baseline based on bus, hour-ending, and weekday using 2022-2024 data. This is a simple but useful baseline because it smooths over individual noisy days.

## Features Created

## Calendar Features

I created basic time features:

* hour-ending (`he`)
* day of week
* month
* day of year
* weekend flag
* U.S. federal holiday flag

These features are useful because load has strong hourly, weekly, and seasonal patterns.

## Next-Day Zone Features

For next-day forecasting, I used lagged zone load features:

* D-7 load
* D-14 load
* D-28 load
* D-365 load
* historical zone-hour averages

I did not use D-1 actual load because the forecast is created at D-1 00:01. At that time, the full D-1 actual load is not available. Using it would create future data leakage.

## Next-Month Zone Features

For next-month forecasting, I used longer-horizon seasonal features:

* previous-year load
* month-lag features, such as 3-month, 6-month, and 12-month lags
* historical month/day/hour averages
* fallback historical month/hour averages

These features are only built from data available before the forecast creation time.

## Weather Features

For the weather-aware model, I used zone-level hourly weather features:

* temperature
* dew point
* humidity
* wind speed
* precipitation
* cooling degree hours
* heating degree hours
* extreme heat flag
* extreme cold flag

I added these features because next-month forecasting was difficult using only load history and calendar variables. Load is strongly affected by weather, especially during hot or cold periods. Weather features help the model adjust forecasts when the target period is different from normal seasonal conditions.

## Bus-Share Features

After the zone forecast is produced, I allocate it to buses using historical bus shares.

For the next-day task, I use recent weekly share information:

* D-7
* D-14
* D-28

For the next-month task, I use a previous-year same-season window:

* previous-year same calendar date
* previous-year date minus 14 days
* previous-year date minus 7 days
* previous-year date plus 7 days
* previous-year date plus 14 days

If the task-specific share is missing, the pipeline falls back to a 2024 bus-hour average share. If a zone-hour group still has no usable historical share, the share remains zero instead of forcing an equal split. I chose this because some buses are all-missing, inactive, or have no historical nonzero load. Giving them artificial load could make the bus-level forecast misleading.

## Missing Data Treatment

The bus-level data contains many missing or zero `pd` values. I audited bus missingness by year and separated buses into different cases.

For partial-missing buses with some nonzero historical load, I used historical imputation. The fallback order is:

1. same bus + same hour + nearby historical days
2. same bus + same hour + same day-of-week average
3. same bus + same hour average
4. same zone + same hour + same day-of-week average multiplied by historical bus share
5. zero fallback

All-missing buses are not used to estimate bus shares because they do not provide reliable historical evidence of load.

## How I Avoided Using Future Data

I used strict forecast cutoff rules.

For the next-day task, target day D is forecast at D-1 00:01. Therefore, I do not use the full D-1 actual load. The model only uses historical information such as D-7, D-14, D-28, and earlier data.

For the next-month task, the target month is forecast at the first day of the previous month. For example, a February forecast is created at the beginning of January. The model only uses data available before that forecast creation date. Month-lag features and historical averages are also built from the same cutoff.

For weather features, I avoid using future observed weather in the default leakage-safe setting. Historical weather can be used for training rows, but future target rows should use weather normals or forecast-like weather inputs. Actual 2025 target weather is only an optional experiment and should not be treated as the default final production setting.

For bus-level actuals, the 2025 `pd` values are used only for evaluation. They are not used as model features.

## Metrics Summary

WMAPE is shown as a percentage in the tables below.

## Next-Day Bus-Level Accuracy

| Model                                    |   MAE |   RMSE | WMAPE |
| ---------------------------------------- | ----: | -----: | ----: |
| Direct bus Transformer                   | 1.387 | 11.308 | 24.9% |
| Historical hour-weekday average baseline | 1.931 |  9.283 | 34.7% |
| Previous-week baseline                   | 0.905 |  5.204 | 16.3% |
| Zone-share Transformer                   | 0.892 |  4.561 | 16.0% |
| Weather-aware zone-share Transformer     | 0.946 |  4.731 | 17.0% |

For next-day bus-level forecasting, the no-weather zone-share Transformer performs best. It has the lowest MAE, RMSE, and WMAPE. The previous-week baseline is already very strong, but the zone-share Transformer slightly improves on it.

The weather-aware model is slightly worse than the no-weather model for this task. My interpretation is that next-day forecasting already has very strong recent load signals. D-7, D-14, and D-28 load patterns directly describe recent demand behavior. Since the forecast horizon is short, these recent load features are more useful than weather normals. The added weather features may introduce noise or duplicate seasonal information instead of adding new useful information.

## Next-Day Zone-Level Aggregated Accuracy

| Model                                    |      MAE |     RMSE | WMAPE |
| ---------------------------------------- | -------: | -------: | ----: |
| Direct bus Transformer                   |  760.635 | 1194.130 | 10.9% |
| Historical hour-weekday average baseline | 1197.830 | 1768.270 | 17.2% |
| Previous-week baseline                   |  787.322 | 1449.350 | 11.3% |
| Zone-share Transformer                   |  619.951 | 1004.410 |  8.9% |
| Weather-aware zone-share Transformer     |  623.694 |  863.149 |  9.0% |

At the zone-aggregated level, the zone-share Transformer clearly improves over the baselines. The no-weather zone-share Transformer has the best MAE and WMAPE. The weather-aware model has the best RMSE, which means it reduces some large zone-level errors, but its average absolute error and WMAPE are slightly worse than the no-weather model.

Overall, for next-day forecasting, I would choose the no-weather zone-share Transformer.

## Next-Month Bus-Level Accuracy

| Model                                    |   MAE |   RMSE | WMAPE |
| ---------------------------------------- | ----: | -----: | ----: |
| Direct bus Transformer                   | 2.382 | 12.988 | 42.8% |
| Historical hour-weekday average baseline | 1.931 |  9.283 | 34.7% |
| Previous-year baseline                   | 1.833 |  9.332 | 33.0% |
| Zone-share Transformer                   | 1.836 |  9.155 | 33.0% |
| Weather-aware zone-share Transformer     | 1.743 |  9.037 | 31.3% |

For next-month bus-level forecasting, the weather-aware zone-share Transformer performs best. It improves over the previous-year baseline and the no-weather zone-share Transformer.

This result makes sense because next-month forecasting is harder than next-day forecasting. Recent weekly load patterns are less directly useful for a month-ahead target. The no-weather model mostly depends on previous-year load, month-lag features, and historical seasonal averages. These features capture normal seasonality, but they cannot fully capture whether the target month is hotter or colder than usual. Weather features help the model capture temperature-driven demand, so they are more useful for the next-month task.

The direct bus Transformer performs poorly for next-month forecasting. Its WMAPE is much higher than the zone-share models and baselines. This suggests that direct bus-level forecasting is not stable enough for this sparse bus-level dataset.

## Next-Month Zone-Level Aggregated Accuracy

| Model                                    |      MAE |     RMSE | WMAPE |
| ---------------------------------------- | -------: | -------: | ----: |
| Direct bus Transformer                   | 1597.520 | 2401.650 | 22.9% |
| Historical hour-weekday average baseline | 1197.830 | 1768.270 | 17.2% |
| Previous-year baseline                   |  798.181 | 1330.650 | 11.5% |
| Zone-share Transformer                   | 1027.470 | 1672.510 | 14.7% |
| Weather-aware zone-share Transformer     |  783.115 | 1273.680 | 11.2% |

At the next-month zone level, the weather-aware model is also the best. It slightly improves over the previous-year baseline and clearly improves over the no-weather zone-share Transformer.

This is one of the main findings of the assignment. Weather does not help much for next-day forecasting, but it helps for next-month forecasting because month-ahead load depends more on weather-driven demand and less on recent weekly history.

## Main Findings

The best next-day model is the no-weather zone-share Transformer. For a one-day horizon, recent historical load is very informative. Weather features do not improve the result because the D-7, D-14, and D-28 features already capture recent demand conditions.

The best next-month model is the weather-aware zone-share Transformer. For a month-ahead horizon, historical load and calendar features are not enough. Weather features help explain heating and cooling demand, so the weather-aware model performs better.

The direct bus Transformer is not the best model. Bus-level data is sparse and unstable, so directly predicting every bus-hour is difficult. Zone-level forecasting is more stable, and bus-share allocation is a practical way to produce the required bus-level output.

The previous-week and previous-year baselines are strong. This is expected because load has strong weekly and yearly seasonality. The model needs to beat these simple baselines to be useful.

## Where the Model Works Well

The model works well when zone-level load patterns are stable. Zone load is easier to predict than individual bus load because it aggregates many buses and reduces local noise.

The next-day model works well because weekly lag features are strong. Buses and zones often have similar load behavior from week to week.

The bus-share allocation works well for buses with stable historical load contribution. If a bus has a consistent share of its zone-hour load, the allocation method can produce reasonable bus-level forecasts.

The weather-aware model works well for next-month forecasting. Weather features help capture temperature-related load changes that are not fully explained by previous-year or historical-average features.

## Where the Model Performs Poorly

The model performs poorly for buses with missing, sparse, or unstable history. If a bus is new, retired, all-missing, or has no historical nonzero load, the bus-share method has little information to estimate its future load.

The model also has difficulty with newly added buses. The bus roster changes from year to year. If a bus appears in 2025 but does not have enough historical load in 2022-2024, the bus-share model cannot estimate a reliable share. In the current implementation, these buses generally receive zero allocation rather than an artificial equal share. This avoids creating fake load, but it can under-forecast new buses that actually carry load.

The model can also perform poorly when there are topology changes or load transfers between buses. Even if the zone forecast is accurate, the bus-level allocation can be wrong if historical bus shares no longer represent the current network structure.

The no-weather next-month model is weaker because it cannot observe unusual weather conditions. It only learns average seasonal patterns from historical load.

The direct bus model performs poorly because the bus-level data is very sparse. I did not cluster buses by similar profiles before training. Without clustering, one global direct bus model has to learn many different types of buses at the same time, including stable load buses, sparse buses, zero-heavy buses, and missing buses.

The data is also noisy. I used missing-value audit and historical imputation, but I did not apply strong smoothing or outlier treatment before modeling. Abnormal spikes, short missing intervals, or sudden profile changes can still affect both training and bus-share estimation.

## Limitations

There are several limitations in my current solution.

First, I did not run full rolling-origin hyperparameter tuning for every zone model. A better approach would train and validate each zone over multiple rolling forecast windows, then choose hyperparameters separately for each horizon and possibly each zone. Because of training time and hardware limits, I used a main set of hyperparameters and reused cached outputs after validation.

Second, the bus-share model does not fully solve the new-bus problem. If a bus appears in 2025 without enough historical nonzero load, the model cannot infer a reliable share from its own history. The current fallback is conservative and may assign zero rather than fake load. This is safer, but it can miss real load from new buses.

Third, the direct bus model is not fully optimized. I did not first cluster buses by historical load-profile similarity. A better direct bus model would group similar bus profiles and train separate models by cluster, or add cluster embeddings as model features.

Fourth, I did not do enough smoothing of the raw bus-level data. The current pipeline handles missingness and imputation, but it does not fully smooth noisy time series, abnormal spikes, or sudden discontinuities.

Fifth, the weather-aware model uses weather information in a simplified way. A production model should use actual weather forecasts available at forecast creation time, not only historical weather normals. Better weather forecast inputs could improve next-month forecasting.

Sixth, the bus-share method assumes historical bus shares are stable. This assumption can fail when topology changes, buses are added or removed, or load is transferred between buses.

## Improvements With More Time

With more time, I would make several improvements.

First, I would run rolling-origin validation for all zone models. I would train and validate across multiple historical forecast windows, then tune learning rate, dropout, number of epochs, early stopping, lag features, and model size. This would make the validation closer to real forecasting.

Second, I would use zone-specific hyperparameters. Different zones may have different load patterns, so one common parameter setting may not be best for every zone.

Third, I would improve new-bus handling. Instead of assigning zero share when there is no history, I would estimate the share using similar buses, bus profile clusters, zone averages, or grid metadata if available.

Fourth, I would improve the direct bus model by clustering buses before training. Buses with similar historical load profiles could be modeled together. This would reduce noise and make direct bus forecasting more meaningful.

Fifth, I would add smoothing and outlier treatment before modeling. This could make historical profiles, baselines, and bus shares more robust.

Sixth, I would use real weather forecast data for the weather-aware model. Since weather improved next-month results, better forecasted weather inputs would likely help further.

Seventh, I would add explicit topology-change detection. If a bus appears, disappears, or changes behavior suddenly, the model should detect that and switch to a different fallback strategy.

## Conclusion

My final approach is a zone-level Transformer forecast with bus-share allocation. This approach is more robust than direct bus-level forecasting because zone load is less sparse and easier to model. The bus-share step then converts the zone forecast into the required bus-level output.

For next-day forecasting, the no-weather zone-share Transformer performs best. Recent weekly load features are strong enough, and weather features do not improve the short-horizon forecast.

For next-month forecasting, the weather-aware zone-share Transformer performs best. Weather information helps because month-ahead load depends more on heating and cooling demand.

Overall, the results show that zone forecasting plus bus-share allocation is a practical approach for this dataset. The main weakness is that bus-level allocation still depends on historical bus shares, so new buses, sparse buses, missing buses, and topology changes remain difficult.
