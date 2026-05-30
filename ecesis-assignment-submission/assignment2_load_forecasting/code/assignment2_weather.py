from __future__ import annotations

import os
import shutil
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from assignment2_method2_zone_share_transformer import (
    FORECAST_COLUMNS,
    FORECAST_COMPRESSION,
    FORECAST_SCHEMA,
    FORECAST_USE_DICTIONARY,
    MetricAccumulator,
    TRANSFORMER_NEXT_DAY_MODEL_NAME,
    TRANSFORMER_NEXT_MONTH_MODEL_NAME,
    add_zone_residual_target,
    add_calendar_features,
    add_day_lag_features,
    add_month_lag_features,
    add_year_ago_feature,
    all_missing_bus_set,
    build_bus_pd_imputer,
    compute_2024_bus_hour_share,
    compute_bus_hour_weekday_average_baseline,
    dynamic_share_for_dates,
    expected_2025_bus_rows,
    find_assignment2_data_dir,
    find_project_root,
    fit_predict_zone_transformer_models,
    historical_active_bus_set,
    infer_cached_zone_model_name,
    load_zone_data,
    normalize_shares,
    normalize_month_lookbacks,
    process_bus_forecasts,
    read_bus_day,
    update_metric,
    write_forecast_rows,
    write_sample_csv,
    restore_zone_residual_prediction,
    zone_prediction_lookup,
)


WEATHER_NEXT_DAY_MODEL_NAME = "zone_share_weather_transformer_next_day"
WEATHER_NEXT_MONTH_MODEL_NAME = "zone_share_weather_transformer_next_month"
WEATHER_RESIDUAL_NEXT_DAY_MODEL_NAME = "zone_share_weather_transformer_next_day"
WEATHER_RESIDUAL_NEXT_MONTH_MODEL_NAME = "zone_share_weather_transformer_next_month"
WEATHER_FEATURE_COLUMNS = [
    "weather_temp_f",
    "weather_dewpoint_f",
    "weather_humidity_pct",
    "weather_wind_speed_kph",
    "weather_precip_mm",
    "weather_cdd65",
    "weather_hdd65",
    "weather_extreme_heat",
    "weather_extreme_cold",
]


ZONE_WEATHER_POINTS = {
    "COAS": [
        ("Houston", 29.7604, -95.3698),
        ("Galveston", 29.3013, -94.7977),
        ("Corpus_Christi", 27.8006, -97.3964),
    ],
    "EAST": [
        ("Tyler", 32.3513, -95.3011),
        ("Longview", 32.5007, -94.7405),
        ("Lufkin", 31.3382, -94.7291),
    ],
    "FWES": [
        ("Midland", 31.9973, -102.0779),
        ("Odessa", 31.8457, -102.3676),
        ("Fort_Stockton", 30.8940, -102.8793),
    ],
    "NCEN": [
        ("Dallas", 32.7767, -96.7970),
        ("Fort_Worth", 32.7555, -97.3308),
        ("Waco", 31.5493, -97.1467),
    ],
    "NOTH": [
        ("Wichita_Falls", 33.9137, -98.4934),
        ("Abilene", 32.4487, -99.7331),
        ("Sherman", 33.6357, -96.6089),
    ],
    "SCEN": [
        ("Austin", 30.2672, -97.7431),
        ("San_Antonio", 29.4241, -98.4936),
        ("Killeen", 31.1171, -97.7278),
    ],
    "SOUT": [
        ("Laredo", 27.5306, -99.4803),
        ("McAllen", 26.2034, -98.2300),
        ("Brownsville", 25.9017, -97.4975),
    ],
    "WEST": [
        ("San_Angelo", 31.4638, -100.4370),
        ("Lubbock", 33.5779, -101.8552),
        ("Amarillo", 35.2220, -101.8313),
    ],
    "ISOLATED": [
        ("Texas_State_Center", 31.0000, -99.0000),
    ],
}


def weather_paths(output_dir: Path) -> dict[str, Path]:
    """Return weather cache paths used by the notebook and weather-aware model."""
    weather_dir = output_dir / "weather_data"
    return {
        "weather_dir": weather_dir,
        "zone_hourly_weather": weather_dir / "assignment2_zone_hourly_weather_2022_2025.parquet",
        "station_manifest": weather_dir / "assignment2_weather_zone_points.csv",
    }


def weather_model_paths(output_dir: Path) -> dict[str, Path]:
    """Return independent weather-aware forecast paths so the no-weather model is not overwritten."""
    return {
        "next_day": output_dir / "assignment2_weather_next_day_forecast_2025.parquet",
        "next_month": output_dir / "assignment2_weather_next_month_forecast_2025.parquet",
        "next_day_sample": output_dir / "assignment2_weather_next_day_forecast_sample.csv",
        "next_month_sample": output_dir / "assignment2_weather_next_month_forecast_sample.csv",
        "metrics": output_dir / "assignment2_weather_metrics.csv",
        "training_log": output_dir / "assignment2_weather_training_log.csv",
        "next_day_zone": output_dir / "assignment2_weather_next_day_zone_predictions.parquet",
        "next_month_zone": output_dir / "assignment2_weather_next_month_zone_predictions.parquet",
        "missingness": output_dir / "assignment2_weather_bus_pd_missingness_by_year.csv",
        "missing_bus_audit": output_dir / "assignment2_weather_missing_bus_forecast_audit.csv",
        "zone_accuracy_by_zone": output_dir / "assignment2_weather_zone_accuracy_by_zone.csv",
        "report": output_dir / "assignment2_weather_summary_report.md",
        "ai_log": output_dir / "assignment2_weather_ai_usage_log.md",
    }


def weather_residual_model_paths(output_dir: Path) -> dict[str, Path]:
    """Return weather-aware residual Method 2B zone cache paths."""
    return {
        "training_log": output_dir / "assignment2_weather_method2B_residual_training_log.csv",
        "next_day_zone": output_dir / "assignment2_weather_method2B_residual_next_day_zone_predictions.parquet",
        "next_month_zone": output_dir / "assignment2_weather_method2B_residual_next_month_zone_predictions.parquet",
    }


def _standardize_meteostat_hourly(raw: pd.DataFrame, zone_name: str, point_name: str) -> pd.DataFrame:
    """Convert one Meteostat hourly response into normalized weather columns."""
    frame = raw.reset_index().rename(
        columns={
            "time": "timestamp",
            "temp": "temp_c",
            "dwpt": "dewpoint_c",
            "rhum": "humidity_pct",
            "wspd": "wind_speed_kph",
            "prcp": "precip_mm",
            "pres": "pressure_hpa",
        }
    )
    if "timestamp" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    for column in ["temp_c", "dewpoint_c", "humidity_pct", "wind_speed_kph", "precip_mm", "pressure_hpa"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None)
    frame["target_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["he"] = (frame["timestamp"].dt.hour + 1).astype("int16")
    frame["zone_name"] = zone_name
    frame["weather_point"] = point_name
    return frame[
        [
            "zone_name",
            "weather_point",
            "timestamp",
            "target_date",
            "he",
            "temp_c",
            "dewpoint_c",
            "humidity_pct",
            "wind_speed_kph",
            "precip_mm",
            "pressure_hpa",
        ]
    ]


def _add_derived_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Fahrenheit, degree-hour, and extreme-weather features."""
    out = frame.copy()
    out["weather_temp_f"] = (out["temp_c"] * 9.0 / 5.0 + 32.0).astype("float32")
    out["weather_dewpoint_f"] = (out["dewpoint_c"] * 9.0 / 5.0 + 32.0).astype("float32")
    out["weather_humidity_pct"] = out["humidity_pct"].astype("float32")
    out["weather_wind_speed_kph"] = out["wind_speed_kph"].fillna(0.0).astype("float32")
    out["weather_precip_mm"] = out["precip_mm"].fillna(0.0).astype("float32")
    out["weather_cdd65"] = np.maximum(out["weather_temp_f"] - 65.0, 0.0).astype("float32")
    out["weather_hdd65"] = np.maximum(65.0 - out["weather_temp_f"], 0.0).astype("float32")
    out["weather_extreme_heat"] = out["weather_temp_f"].ge(95.0).astype("float32")
    out["weather_extreme_cold"] = out["weather_temp_f"].le(32.0).astype("float32")
    return out


def _download_open_meteo_hourly(zone_name: str, point_name: str, latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    """Download one representative point from Open-Meteo archive by latitude/longitude."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,dew_point_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "America/Chicago",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    with urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temp_c": hourly.get("temperature_2m"),
            "dewpoint_c": hourly.get("dew_point_2m"),
            "humidity_pct": hourly.get("relative_humidity_2m"),
            "wind_speed_kph": hourly.get("wind_speed_10m"),
            "precip_mm": hourly.get("precipitation"),
            "pressure_hpa": np.nan,
        }
    )
    frame["target_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["he"] = (frame["timestamp"].dt.hour + 1).astype("int16")
    frame["zone_name"] = zone_name
    frame["weather_point"] = point_name
    return frame[
        [
            "zone_name",
            "weather_point",
            "timestamp",
            "target_date",
            "he",
            "temp_c",
            "dewpoint_c",
            "humidity_pct",
            "wind_speed_kph",
            "precip_mm",
            "pressure_hpa",
        ]
    ]


def _download_meteostat_hourly(zone_name: str, point_name: str, latitude: float, longitude: float, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Download one representative point from Meteostat; return an empty frame if that point has no rows."""
    from meteostat import Hourly, Point

    raw = Hourly(Point(latitude, longitude), start_dt, end_dt).fetch()
    if raw.empty:
        return pd.DataFrame()
    return _standardize_meteostat_hourly(raw, zone_name, point_name)


def download_zone_weather(
    force_download: bool = False,
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    provider: str = "open_meteo",
) -> dict[str, Path]:
    """Download hourly zone weather and save a zone-level parquet cache.

    The default provider is Open-Meteo archive because it downloads by latitude/longitude
    and does not depend on individual airport station files. Meteostat remains available
    as an optional provider for station-style experiments.
    """
    output_dir = Path(__file__).resolve().parent
    paths = weather_paths(output_dir)
    paths["weather_dir"].mkdir(parents=True, exist_ok=True)
    if paths["zone_hourly_weather"].exists() and not force_download:
        return paths

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end).replace(hour=23)
    weather_frames = []
    manifest_rows = []
    for zone_name, points in ZONE_WEATHER_POINTS.items():
        for point_name, latitude, longitude in points:
            if provider == "meteostat":
                frame = _download_meteostat_hourly(zone_name, point_name, latitude, longitude, start_dt, end_dt)
            elif provider == "open_meteo":
                frame = _download_open_meteo_hourly(zone_name, point_name, latitude, longitude, start, end)
            else:
                raise ValueError("provider must be either 'open_meteo' or 'meteostat'.")
            manifest_rows.append(
                {
                    "zone_name": zone_name,
                    "weather_point": point_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "provider": provider,
                    "rows_downloaded": int(len(frame)),
                }
            )
            if frame.empty:
                continue
            weather_frames.append(frame)

    if not weather_frames:
        raise RuntimeError(f"No hourly weather rows were downloaded from {provider}.")

    raw_weather = pd.concat(weather_frames, ignore_index=True)
    aggregate_columns = ["temp_c", "dewpoint_c", "humidity_pct", "wind_speed_kph", "precip_mm", "pressure_hpa"]
    zone_weather = (
        raw_weather.groupby(["zone_name", "target_date", "he"], observed=True, as_index=False)[aggregate_columns]
        .mean(numeric_only=True)
        .sort_values(["zone_name", "target_date", "he"])
    )
    zone_weather = _add_derived_weather_features(zone_weather)
    keep_columns = ["zone_name", "target_date", "he"] + WEATHER_FEATURE_COLUMNS
    zone_weather[keep_columns].to_parquet(paths["zone_hourly_weather"], index=False)
    pd.DataFrame(manifest_rows).to_csv(paths["station_manifest"], index=False)
    return paths


def load_zone_weather(output_dir: Path | None = None) -> pd.DataFrame:
    """Load cached zone-level hourly weather features."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    path = weather_paths(output_dir)["zone_hourly_weather"]
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}. Run download_zone_weather() first.")
    weather = pd.read_parquet(path)
    weather["target_date"] = pd.to_datetime(weather["target_date"])
    weather["he"] = weather["he"].astype("int16")
    return weather[["zone_name", "target_date", "he"] + WEATHER_FEATURE_COLUMNS]


def _weather_normals_for_target(target: pd.DataFrame, available_weather: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe weather normals for target dates using only available historical weather."""
    target_keys = add_calendar_features(target[["zone_name", "target_date", "he"]].copy())
    available = add_calendar_features(available_weather.copy())
    exact = (
        available.groupby(["zone_name", "month", "day", "he"], observed=True)[WEATHER_FEATURE_COLUMNS]
        .mean(numeric_only=True)
        .reset_index()
    )
    exact = exact.rename(columns={column: f"{column}_normal_exact" for column in WEATHER_FEATURE_COLUMNS})
    fallback = (
        available.groupby(["zone_name", "month", "he"], observed=True)[WEATHER_FEATURE_COLUMNS]
        .mean(numeric_only=True)
        .reset_index()
    )
    fallback = fallback.rename(columns={column: f"{column}_normal_fallback" for column in WEATHER_FEATURE_COLUMNS})
    out = target_keys.merge(exact, on=["zone_name", "month", "day", "he"], how="left")
    out = out.merge(fallback, on=["zone_name", "month", "he"], how="left")
    for column in WEATHER_FEATURE_COLUMNS:
        out[column] = out[f"{column}_normal_exact"].fillna(out[f"{column}_normal_fallback"])
    return out[["zone_name", "target_date", "he"] + WEATHER_FEATURE_COLUMNS]


def _merge_weather(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
    available_weather: pd.DataFrame,
    use_observed_target_weather: bool,
) -> pd.DataFrame:
    """Merge either observed weather or leakage-safe weather normals into a zone frame."""
    if use_observed_target_weather:
        weather_features = weather[["zone_name", "target_date", "he"] + WEATHER_FEATURE_COLUMNS].copy()
    else:
        weather_features = _weather_normals_for_target(frame, available_weather)
    out = frame.merge(weather_features, on=["zone_name", "target_date", "he"], how="left")
    medians = available_weather[WEATHER_FEATURE_COLUMNS].median(numeric_only=True)
    for column in WEATHER_FEATURE_COLUMNS:
        out[column] = out[column].fillna(medians.get(column, 0.0)).fillna(0.0).astype("float32")
    return out


def build_next_day_weather_zone_predictions(
    zone: pd.DataFrame,
    weather: pd.DataFrame,
    use_observed_target_weather: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.0015,
    dropout: float = 0.10,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    training_log_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Train next-day zone Transformer with weather features and produce 2025 zone predictions."""
    print("weather model: building next-day zone/weather features", flush=True)
    frame = add_calendar_features(zone)
    frame = add_day_lag_features(frame, [7, 14, 28, 365])
    train_mask = frame["target_date"].between("2022-01-01", "2024-12-31")
    train = frame.loc[train_mask].copy()
    hist_dow = train.groupby(["zone_name", "he", "dayofweek"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_dow")
    hist_month = train.groupby(["zone_name", "he", "month"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_month")
    frame = frame.merge(hist_dow.reset_index(), on=["zone_name", "he", "dayofweek"], how="left")
    frame = frame.merge(hist_month.reset_index(), on=["zone_name", "he", "month"], how="left")
    train = frame.loc[train_mask].copy()
    predict = frame.loc[frame["target_date"].between("2025-01-01", "2025-12-31")].copy()
    train_weather = weather[weather["target_date"].between("2022-01-01", "2024-12-31")].copy()
    train = _merge_weather(train, weather, train_weather, use_observed_target_weather=True)
    predict = _merge_weather(predict, weather, train_weather, use_observed_target_weather=use_observed_target_weather)
    features = [
        "he",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "is_holiday",
        "lag_7d",
        "lag_14d",
        "lag_28d",
        "lag_365d",
        "hist_avg_zone_he_dow",
        "hist_avg_zone_he_month",
    ] + WEATHER_FEATURE_COLUMNS
    model_name = WEATHER_NEXT_DAY_MODEL_NAME
    print(f"weather model: training next-day zone Transformer with {epochs} epochs", flush=True)
    result = fit_predict_zone_transformer_models(
        train,
        predict,
        features,
        model_name,
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        training_log_rows=training_log_rows,
        training_context={
            "method": "method2_zone_share_transformer_with_weather",
            "task": "next_day",
            "target_month": "2025_full_year",
            "use_observed_target_weather": use_observed_target_weather,
        },
    )
    result["baseline_zone_pd"] = result["lag_7d"].fillna(result["hist_avg_zone_he_dow"]).fillna(result["hist_avg_zone_he_month"]).fillna(0.0)
    result["target_date_str"] = result["target_date"].dt.strftime("%Y-%m-%d")
    result["forecast_created_at"] = (result["target_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d 00:01:00")
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]


def build_monthly_weather_features(
    frame: pd.DataFrame,
    actuals: pd.DataFrame,
    available: pd.DataFrame,
    weather: pd.DataFrame,
    available_weather: pd.DataFrame,
    use_observed_target_weather: bool,
    lookback_months: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Build next-month features with either observed weather or leakage-safe weather normals."""
    out = add_calendar_features(frame)
    out = add_year_ago_feature(out, actuals)
    out = add_month_lag_features(out, available, lookback_months)
    available_calendar = add_calendar_features(available)
    hist = available_calendar.groupby(["zone_name", "month", "day", "he"], observed=True)["actual_pd"].mean().rename("hist_avg_month_day_he")
    out = out.merge(hist.reset_index(), on=["zone_name", "month", "day", "he"], how="left")
    fallback = available_calendar.groupby(["zone_name", "month", "he"], observed=True)["actual_pd"].mean().rename("hist_avg_month_he")
    out = out.merge(fallback.reset_index(), on=["zone_name", "month", "he"], how="left")
    out["hist_avg_pd"] = out["hist_avg_month_day_he"].fillna(out["hist_avg_month_he"])
    return _merge_weather(out, weather, available_weather, use_observed_target_weather=use_observed_target_weather)


def build_next_month_weather_zone_predictions(
    zone: pd.DataFrame,
    weather: pd.DataFrame,
    use_observed_target_weather: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.0015,
    dropout: float = 0.10,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
    training_log_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Train rolling next-month zone Transformer with weather features and produce 2025 zone predictions."""
    all_predictions = []
    month_lookbacks = normalize_month_lookbacks(next_month_lookback_months)
    features = [
        "he",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "is_holiday",
        "year_ago_pd",
        "hist_avg_pd",
    ] + [f"lag_{months}m" for months in month_lookbacks] + WEATHER_FEATURE_COLUMNS
    model_name = WEATHER_NEXT_MONTH_MODEL_NAME
    for month_start in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        print(f"weather model: training next-month zone Transformer for {month_start.strftime('%Y-%m')} with {epochs} epochs", flush=True)
        created_date = month_start - pd.DateOffset(months=1)
        available = zone[zone["target_date"].lt(created_date)].copy()
        target = zone[zone["target_date"].ge(month_start) & zone["target_date"].lt(month_start + pd.DateOffset(months=1))].copy()
        available_weather = weather[weather["target_date"].lt(created_date)].copy()
        train = build_monthly_weather_features(available, zone, available, weather, available_weather, use_observed_target_weather=True, lookback_months=month_lookbacks)
        predict = build_monthly_weather_features(
            target,
            zone,
            available,
            weather,
            available_weather,
            use_observed_target_weather=use_observed_target_weather,
            lookback_months=month_lookbacks,
        )
        month_prediction = fit_predict_zone_transformer_models(
            train,
            predict,
            features,
            model_name,
            epochs=epochs,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            learning_rate=learning_rate,
            dropout=dropout,
            use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_lr=lr_scheduler_min_lr,
            training_log_rows=training_log_rows,
            training_context={
                "method": "method2_zone_share_transformer_with_weather",
                "task": "next_month",
                "target_month": month_start.strftime("%Y-%m"),
                "forecast_created_at": created_date.strftime("%Y-%m-%d 00:01:00"),
                "use_observed_target_weather": use_observed_target_weather,
                "next_month_lookback_months": ",".join(str(value) for value in month_lookbacks),
            },
        )
        month_prediction["baseline_zone_pd"] = month_prediction["year_ago_pd"].fillna(month_prediction["hist_avg_pd"]).fillna(0.0)
        month_prediction["target_date_str"] = month_prediction["target_date"].dt.strftime("%Y-%m-%d")
        month_prediction["forecast_created_at"] = created_date.strftime("%Y-%m-%d 00:01:00")
        all_predictions.append(month_prediction)
    result = pd.concat(all_predictions, ignore_index=True)
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]


def build_next_day_weather_zone_predictions_residual(
    zone: pd.DataFrame,
    weather: pd.DataFrame,
    use_observed_target_weather: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.0015,
    dropout: float = 0.10,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    training_log_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Train residual next-day weather zone Transformer anchored on lag_7d/history baseline."""
    print("weather residual model: building next-day zone/weather features", flush=True)
    frame = add_calendar_features(zone)
    frame = add_day_lag_features(frame, [7, 14, 28, 365])
    train_mask = frame["target_date"].between("2022-01-01", "2024-12-31")
    train_base = frame.loc[train_mask].copy()
    hist_dow = train_base.groupby(["zone_name", "he", "dayofweek"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_dow")
    hist_month = train_base.groupby(["zone_name", "he", "month"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_month")
    frame = frame.merge(hist_dow.reset_index(), on=["zone_name", "he", "dayofweek"], how="left")
    frame = frame.merge(hist_month.reset_index(), on=["zone_name", "he", "month"], how="left")
    frame["baseline_zone_pd"] = frame["lag_7d"].fillna(frame["hist_avg_zone_he_dow"]).fillna(frame["hist_avg_zone_he_month"]).fillna(0.0)
    train = frame.loc[train_mask].copy()
    predict = frame.loc[frame["target_date"].between("2025-01-01", "2025-12-31")].copy()
    train_weather = weather[weather["target_date"].between("2022-01-01", "2024-12-31")].copy()
    train = _merge_weather(train, weather, train_weather, use_observed_target_weather=True)
    predict = _merge_weather(predict, weather, train_weather, use_observed_target_weather=use_observed_target_weather)
    train = add_zone_residual_target(train)
    features = [
        "he",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "is_holiday",
        "lag_7d",
        "lag_14d",
        "lag_28d",
        "lag_365d",
        "hist_avg_zone_he_dow",
        "hist_avg_zone_he_month",
        "baseline_zone_pd",
    ] + WEATHER_FEATURE_COLUMNS
    model_name = WEATHER_RESIDUAL_NEXT_DAY_MODEL_NAME
    result = fit_predict_zone_transformer_models(
        train,
        predict,
        features,
        model_name,
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        training_log_rows=training_log_rows,
        training_context={
            "method": "method2B_zone_residual_transformer_with_weather",
            "task": "next_day",
            "target_month": "2025_full_year",
            "use_observed_target_weather": use_observed_target_weather,
        },
        target_column="target_residual_log",
        prediction_column="predict_residual_log",
        clip_prediction=False,
        loss_units="standardized_log1p_zone_residual_mse",
    )
    result = restore_zone_residual_prediction(result)
    result["target_date_str"] = result["target_date"].dt.strftime("%Y-%m-%d")
    result["forecast_created_at"] = (result["target_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d 00:01:00")
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]


def build_next_month_weather_zone_predictions_residual(
    zone: pd.DataFrame,
    weather: pd.DataFrame,
    use_observed_target_weather: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.0015,
    dropout: float = 0.10,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
    training_log_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Train residual rolling next-month weather zone Transformer anchored on previous-year/history baseline."""
    all_predictions = []
    month_lookbacks = normalize_month_lookbacks(next_month_lookback_months)
    features = [
        "he",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "is_holiday",
        "year_ago_pd",
        "hist_avg_pd",
        "baseline_zone_pd",
    ] + [f"lag_{months}m" for months in month_lookbacks] + WEATHER_FEATURE_COLUMNS
    model_name = WEATHER_RESIDUAL_NEXT_MONTH_MODEL_NAME
    for month_start in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        print(f"weather residual model: training next-month zone Transformer for {month_start.strftime('%Y-%m')}", flush=True)
        created_date = month_start - pd.DateOffset(months=1)
        available = zone[zone["target_date"].lt(created_date)].copy()
        target = zone[zone["target_date"].ge(month_start) & zone["target_date"].lt(month_start + pd.DateOffset(months=1))].copy()
        available_weather = weather[weather["target_date"].lt(created_date)].copy()
        train = build_monthly_weather_features(available, zone, available, weather, available_weather, use_observed_target_weather=True, lookback_months=month_lookbacks)
        predict = build_monthly_weather_features(
            target,
            zone,
            available,
            weather,
            available_weather,
            use_observed_target_weather=use_observed_target_weather,
            lookback_months=month_lookbacks,
        )
        train["baseline_zone_pd"] = train["year_ago_pd"].fillna(train["hist_avg_pd"]).fillna(0.0)
        predict["baseline_zone_pd"] = predict["year_ago_pd"].fillna(predict["hist_avg_pd"]).fillna(0.0)
        train = add_zone_residual_target(train)
        month_prediction = fit_predict_zone_transformer_models(
            train,
            predict,
            features,
            model_name,
            epochs=epochs,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            learning_rate=learning_rate,
            dropout=dropout,
            use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_lr=lr_scheduler_min_lr,
            training_log_rows=training_log_rows,
            training_context={
                "method": "method2B_zone_residual_transformer_with_weather",
                "task": "next_month",
                "target_month": month_start.strftime("%Y-%m"),
                "forecast_created_at": created_date.strftime("%Y-%m-%d 00:01:00"),
                "use_observed_target_weather": use_observed_target_weather,
                "next_month_lookback_months": ",".join(str(value) for value in month_lookbacks),
            },
            target_column="target_residual_log",
            prediction_column="predict_residual_log",
            clip_prediction=False,
            loss_units="standardized_log1p_zone_residual_mse",
        )
        month_prediction = restore_zone_residual_prediction(month_prediction)
        month_prediction["target_date_str"] = month_prediction["target_date"].dt.strftime("%Y-%m-%d")
        month_prediction["forecast_created_at"] = created_date.strftime("%Y-%m-%d 00:01:00")
        all_predictions.append(month_prediction)
    result = pd.concat(all_predictions, ignore_index=True)
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]


def _merge_metric_accumulators(
    target: dict[tuple[str, str, str], MetricAccumulator],
    source: dict[tuple[str, str, str], MetricAccumulator],
) -> None:
    """Merge one worker's streaming metric totals into the full-run totals."""
    for key, accumulator in source.items():
        if key not in target:
            target[key] = MetricAccumulator()
        target[key].n += accumulator.n
        target[key].abs_error_sum += accumulator.abs_error_sum
        target[key].sq_error_sum += accumulator.sq_error_sum
        target[key].actual_sum += accumulator.actual_sum


def _metric_accumulators_to_frame(metrics: dict[tuple[str, str, str], MetricAccumulator]) -> pd.DataFrame:
    """Convert streaming metric totals into the assignment metrics table."""
    rows = []
    for (task, level, model_name), accumulator in sorted(metrics.items()):
        row = {"task": task, "level": level, "model_name": model_name}
        row.update(accumulator.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _update_weather_zone_metric_by_zone(
    zone_metrics: dict[tuple[str, str, str], MetricAccumulator],
    task: str,
    model_name: str,
    zone_eval: pd.DataFrame,
    predicted_column: str,
) -> None:
    """Accumulate weather-aware zone-level metrics separately for each zone_id."""
    for zone_id, group in zone_eval.groupby("zone_name", observed=True):
        key = (task, str(zone_id), model_name)
        update_metric(zone_metrics, key, group["actual_pd"].to_numpy(), group[predicted_column].to_numpy())


def _zone_metric_accumulators_to_frame(zone_metrics: dict[tuple[str, str, str], MetricAccumulator]) -> pd.DataFrame:
    """Convert per-zone metric totals into task, zone_id, model_name, n, mae, rmse, wmape, sum_actual rows."""
    rows = []
    for (task, zone_id, model_name), accumulator in sorted(zone_metrics.items()):
        row = {"task": task, "zone_id": zone_id, "model_name": model_name}
        row.update(accumulator.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_month_weather_bus_parts(
    month_start: pd.Timestamp,
    data_dir: Path,
    part_dir: Path,
    next_day_zone: pd.DataFrame,
    next_month_zone: pd.DataFrame,
    fallback: pd.DataFrame,
    historical_average_lookup: pd.DataFrame,
    imputer: object,
    all_missing_2025: set[str],
    active_history: set[str],
    next_day_model_name: str,
    next_month_model_name: str,
) -> dict[str, object]:
    """Generate one month of weather-aware bus forecasts into independent parquet part files."""
    month_start = pd.Timestamp(month_start)
    month_label = month_start.strftime("%Y_%m")
    month_end = month_start + pd.offsets.MonthEnd(0)
    next_day_part = part_dir / "next_day" / f"part_{month_label}.parquet"
    next_month_part = part_dir / "next_month" / f"part_{month_label}.parquet"
    next_day_part.parent.mkdir(parents=True, exist_ok=True)
    next_month_part.parent.mkdir(parents=True, exist_ok=True)
    for part_path in [next_day_part, next_month_part]:
        if part_path.exists():
            part_path.unlink()

    metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
    zone_metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
    audit_counts = {
        ("next_day", "all_missing_2025_bus_rows"): 0,
        ("next_month", "all_missing_2025_bus_rows"): 0,
        ("next_day", "no_historical_nonzero_bus_rows"): 0,
        ("next_month", "no_historical_nonzero_bus_rows"): 0,
        ("next_day", "zero_share_group_rows"): 0,
        ("next_month", "zero_share_group_rows"): 0,
    }

    next_day_writer = pq.ParquetWriter(next_day_part, FORECAST_SCHEMA, compression=FORECAST_COMPRESSION, use_dictionary=FORECAST_USE_DICTIONARY)
    next_month_writer = pq.ParquetWriter(next_month_part, FORECAST_SCHEMA, compression=FORECAST_COMPRESSION, use_dictionary=FORECAST_USE_DICTIONARY)
    try:
        for target_date in pd.date_range(month_start, month_end, freq="D"):
            date_str = target_date.strftime("%Y-%m-%d")
            target = read_bus_day(data_dir, target_date)
            target["actual_pd"] = target["pd"].fillna(0.0).astype("float32")
            target["_dayofweek"] = np.int16(target_date.dayofweek)
            target = target.merge(historical_average_lookup, on=["bus_unique_id", "zone_name", "he", "_dayofweek"], how="left")
            target["baseline_bus_hour_weekday_avg"] = target["baseline_bus_hour_weekday_avg"].fillna(0.0).astype("float32")
            target = target.drop(columns=["_dayofweek"])

            target_bus_ids = target["bus_unique_id"].astype(str)
            all_missing_count = int(target_bus_ids.isin(all_missing_2025).sum())
            no_history_count = int((~target_bus_ids.isin(active_history)).sum())
            audit_counts[("next_day", "all_missing_2025_bus_rows")] += all_missing_count
            audit_counts[("next_month", "all_missing_2025_bus_rows")] += all_missing_count
            audit_counts[("next_day", "no_historical_nonzero_bus_rows")] += no_history_count
            audit_counts[("next_month", "no_historical_nonzero_bus_rows")] += no_history_count

            lag_date = target_date - pd.Timedelta(days=7)
            next_day_share_sources = [
                (target_date - pd.Timedelta(days=7), 0.60),
                (target_date - pd.Timedelta(days=14), 0.30),
                (target_date - pd.Timedelta(days=28), 0.10),
            ]
            lag_source = dynamic_share_for_dates(data_dir, next_day_share_sources, "next_day_raw_share", "baseline_next_day_pd", lag_date, imputer)
            day_frame = target.merge(lag_source, on=["bus_unique_id", "zone_name", "he"], how="left")
            day_frame["baseline_next_day_pd"] = day_frame["baseline_next_day_pd"].fillna(0.0).astype("float32")
            day_frame = normalize_shares(day_frame, "next_day_raw_share", fallback, "next_day_share")
            audit_counts[("next_day", "zero_share_group_rows")] += int(day_frame["next_day_share_zero_share_group"].sum())
            day_zone = zone_prediction_lookup(next_day_zone, date_str).rename(
                columns={
                    "predict_zone_pd": "next_day_zone_pred",
                    "baseline_zone_pd": "next_day_zone_baseline",
                    "forecast_created_at": "next_day_created_at",
                }
            )
            day_frame = day_frame.merge(day_zone, on=["zone_name", "he"], how="left")
            day_frame["next_day_predict_pd"] = (day_frame["next_day_zone_pred"].fillna(0.0) * day_frame["next_day_share"]).astype("float32")
            write_forecast_rows(next_day_writer, day_frame, next_day_model_name, "next_day_predict_pd", "next_day_created_at")

            previous_year_date = target_date - pd.DateOffset(years=1)
            next_month_share_sources = [
                (previous_year_date, 0.40),
                (previous_year_date - pd.Timedelta(days=7), 0.20),
                (previous_year_date + pd.Timedelta(days=7), 0.20),
                (previous_year_date - pd.Timedelta(days=14), 0.10),
                (previous_year_date + pd.Timedelta(days=14), 0.10),
            ]
            year_source = dynamic_share_for_dates(
                data_dir,
                next_month_share_sources,
                "next_month_raw_share",
                "baseline_next_month_pd",
                previous_year_date,
                imputer,
            )
            month_frame = target.merge(year_source, on=["bus_unique_id", "zone_name", "he"], how="left")
            month_frame["baseline_next_month_pd"] = month_frame["baseline_next_month_pd"].fillna(0.0).astype("float32")
            month_frame = normalize_shares(month_frame, "next_month_raw_share", fallback, "next_month_share")
            audit_counts[("next_month", "zero_share_group_rows")] += int(month_frame["next_month_share_zero_share_group"].sum())
            month_zone = zone_prediction_lookup(next_month_zone, date_str).rename(
                columns={
                    "predict_zone_pd": "next_month_zone_pred",
                    "baseline_zone_pd": "next_month_zone_baseline",
                    "forecast_created_at": "next_month_created_at",
                }
            )
            month_frame = month_frame.merge(month_zone, on=["zone_name", "he"], how="left")
            month_frame["next_month_predict_pd"] = (month_frame["next_month_zone_pred"].fillna(0.0) * month_frame["next_month_share"]).astype("float32")
            write_forecast_rows(next_month_writer, month_frame, next_month_model_name, "next_month_predict_pd", "next_month_created_at")

            update_metric(metrics, ("next_day", "bus", next_day_model_name), day_frame["actual_pd"].to_numpy(), day_frame["next_day_predict_pd"].to_numpy())
            update_metric(metrics, ("next_day", "bus", "baseline_previous_week"), day_frame["actual_pd"].to_numpy(), day_frame["baseline_next_day_pd"].to_numpy())
            update_metric(metrics, ("next_day", "bus", "baseline_bus_hour_weekday_avg"), day_frame["actual_pd"].to_numpy(), day_frame["baseline_bus_hour_weekday_avg"].to_numpy())
            update_metric(metrics, ("next_month", "bus", next_month_model_name), month_frame["actual_pd"].to_numpy(), month_frame["next_month_predict_pd"].to_numpy())
            update_metric(metrics, ("next_month", "bus", "baseline_previous_year"), month_frame["actual_pd"].to_numpy(), month_frame["baseline_next_month_pd"].to_numpy())
            update_metric(metrics, ("next_month", "bus", "baseline_bus_hour_weekday_avg"), month_frame["actual_pd"].to_numpy(), month_frame["baseline_bus_hour_weekday_avg"].to_numpy())

            day_zone_eval = day_frame.groupby(["date", "he", "zone_name"], observed=True).agg(
                actual_pd=("actual_pd", "sum"),
                model_pd=("next_day_predict_pd", "sum"),
                baseline_pd=("baseline_next_day_pd", "sum"),
                historical_avg_pd=("baseline_bus_hour_weekday_avg", "sum"),
            )
            update_metric(metrics, ("next_day", "zone", next_day_model_name), day_zone_eval["actual_pd"].to_numpy(), day_zone_eval["model_pd"].to_numpy())
            update_metric(metrics, ("next_day", "zone", "baseline_previous_week"), day_zone_eval["actual_pd"].to_numpy(), day_zone_eval["baseline_pd"].to_numpy())
            update_metric(metrics, ("next_day", "zone", "baseline_bus_hour_weekday_avg"), day_zone_eval["actual_pd"].to_numpy(), day_zone_eval["historical_avg_pd"].to_numpy())
            _update_weather_zone_metric_by_zone(zone_metrics, "next_day", next_day_model_name, day_zone_eval, "model_pd")
            _update_weather_zone_metric_by_zone(zone_metrics, "next_day", "baseline_previous_week", day_zone_eval, "baseline_pd")
            _update_weather_zone_metric_by_zone(zone_metrics, "next_day", "baseline_bus_hour_weekday_avg", day_zone_eval, "historical_avg_pd")

            month_zone_eval = month_frame.groupby(["date", "he", "zone_name"], observed=True).agg(
                actual_pd=("actual_pd", "sum"),
                model_pd=("next_month_predict_pd", "sum"),
                baseline_pd=("baseline_next_month_pd", "sum"),
                historical_avg_pd=("baseline_bus_hour_weekday_avg", "sum"),
            )
            update_metric(metrics, ("next_month", "zone", next_month_model_name), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["model_pd"].to_numpy())
            update_metric(metrics, ("next_month", "zone", "baseline_previous_year"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["baseline_pd"].to_numpy())
            update_metric(metrics, ("next_month", "zone", "baseline_bus_hour_weekday_avg"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["historical_avg_pd"].to_numpy())
            _update_weather_zone_metric_by_zone(zone_metrics, "next_month", next_month_model_name, month_zone_eval, "model_pd")
            _update_weather_zone_metric_by_zone(zone_metrics, "next_month", "baseline_previous_year", month_zone_eval, "baseline_pd")
            _update_weather_zone_metric_by_zone(zone_metrics, "next_month", "baseline_bus_hour_weekday_avg", month_zone_eval, "historical_avg_pd")
    finally:
        next_day_writer.close()
        next_month_writer.close()

    return {
        "month": month_label,
        "next_day_part": next_day_part,
        "next_month_part": next_month_part,
        "metrics": metrics,
        "zone_metrics": zone_metrics,
        "audit_counts": audit_counts,
    }


def _append_parquet_parts(part_paths: list[Path], final_path: Path) -> int:
    """Combine month-level parquet parts into one file that keeps the required assignment schema."""
    if final_path.exists() and final_path.is_file():
        final_path.unlink()
    elif final_path.exists() and final_path.is_dir():
        shutil.rmtree(final_path)
    total_rows = 0
    writer = pq.ParquetWriter(final_path, FORECAST_SCHEMA, compression=FORECAST_COMPRESSION, use_dictionary=FORECAST_USE_DICTIONARY)
    try:
        for part_path in part_paths:
            part_file = pq.ParquetFile(part_path)
            for row_group in range(part_file.num_row_groups):
                table = part_file.read_row_group(row_group, columns=FORECAST_COLUMNS)
                total_rows += table.num_rows
                writer.write_table(table)
    finally:
        writer.close()
    pf = pq.ParquetFile(final_path)
    if pf.num_row_groups:
        pf.read_row_group(0, columns=["model_name", "predict_pd"])
    return total_rows


def process_weather_bus_forecasts_parallel(
    data_dir: Path,
    output_dir: Path,
    paths: dict[str, Path],
    next_day_zone: pd.DataFrame,
    next_month_zone: pd.DataFrame,
    fallback: pd.DataFrame,
    historical_average_baseline: pd.DataFrame,
    imputer: object,
    next_day_model_name: str,
    next_month_model_name: str,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Generate weather-aware bus forecasts with optional month-level parallelism."""
    max_workers = max(1, int(max_workers))
    if max_workers <= 1:
        return process_bus_forecasts(
            data_dir,
            output_dir,
            paths,
            next_day_zone,
            next_month_zone,
            fallback,
            historical_average_baseline,
            imputer,
            next_day_model_name,
            next_month_model_name,
        )

    worker_limit = min(max_workers, 12)
    part_dir = output_dir / "assignment2_weather_parallel_parts"
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
    audit_counts = {
        ("next_day", "all_missing_2025_bus_rows"): 0,
        ("next_month", "all_missing_2025_bus_rows"): 0,
        ("next_day", "no_historical_nonzero_bus_rows"): 0,
        ("next_month", "no_historical_nonzero_bus_rows"): 0,
        ("next_day", "zero_share_group_rows"): 0,
        ("next_month", "zero_share_group_rows"): 0,
    }
    historical_average_lookup = historical_average_baseline.rename(columns={"dayofweek": "_dayofweek"})
    all_missing_2025 = all_missing_bus_set(imputer, 2025)
    active_history = historical_active_bus_set(imputer, (2022, 2023, 2024))
    month_starts = list(pd.date_range("2025-01-01", "2025-12-01", freq="MS"))
    results: list[dict[str, object]] = []

    print(f"weather model: parallel bus forecast workers={worker_limit}", flush=True)
    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = [
            executor.submit(
                _write_month_weather_bus_parts,
                month_start,
                data_dir,
                part_dir,
                next_day_zone,
                next_month_zone,
                fallback,
                historical_average_lookup,
                imputer,
                all_missing_2025,
                active_history,
                next_day_model_name,
                next_month_model_name,
            )
            for month_start in month_starts
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            _merge_metric_accumulators(metrics, result["metrics"])
            _merge_metric_accumulators(zone_metrics, result["zone_metrics"])
            for key, count in result["audit_counts"].items():
                audit_counts[key] += int(count)
            print(f"weather model: finished forecast part {index}/12 ({result['month']})", flush=True)

    results = sorted(results, key=lambda item: item["month"])
    next_day_parts = [result["next_day_part"] for result in results]
    next_month_parts = [result["next_month_part"] for result in results]
    print("weather model: merging parallel next-day parquet parts", flush=True)
    _append_parquet_parts(next_day_parts, paths["next_day"])
    print("weather model: merging parallel next-month parquet parts", flush=True)
    _append_parquet_parts(next_month_parts, paths["next_month"])

    metrics_df = _metric_accumulators_to_frame(metrics)
    metrics_df.to_csv(paths["metrics"], index=False)
    _zone_metric_accumulators_to_frame(zone_metrics).to_csv(paths["zone_accuracy_by_zone"], index=False)
    audit_df = pd.DataFrame(
        [{"task": task, "audit_item": item, "row_count": count} for (task, item), count in sorted(audit_counts.items())]
    )
    audit_df.to_csv(paths["missing_bus_audit"], index=False)
    shutil.rmtree(part_dir, ignore_errors=True)
    return metrics_df


def weather_outputs_complete(paths: dict[str, Path], data_dir: Path) -> bool:
    """Check whether weather-aware forecast outputs and metrics already exist."""
    required = [paths["next_day"], paths["next_month"], paths["metrics"], paths["zone_accuracy_by_zone"]]
    if any(not path.exists() for path in required):
        return False
    expected_rows = expected_2025_bus_rows(data_dir)
    try:
        for key in ["next_day", "next_month"]:
            pf = pq.ParquetFile(paths[key])
            if pf.schema_arrow.names != FORECAST_COLUMNS:
                return False
            if pf.metadata.num_rows != expected_rows:
                return False
        metrics = pd.read_csv(paths["metrics"])
    except Exception:
        return False
    model_names = set(metrics["model_name"].astype(str))
    expected_next_day = infer_cached_zone_model_name(paths, "next_day", WEATHER_NEXT_DAY_MODEL_NAME)
    expected_next_month = infer_cached_zone_model_name(paths, "next_month", WEATHER_NEXT_MONTH_MODEL_NAME)
    return expected_next_day in model_names and expected_next_month in model_names


def remove_weather_outputs(paths: dict[str, Path]) -> None:
    """Remove weather-aware forecast outputs before a full rebuild."""
    for path in paths.values():
        if path.exists() and path.is_file():
            path.unlink()
        elif path.exists() and path.is_dir() and path.name != "weather_data":
            shutil.rmtree(path)


def remove_weather_bus_outputs(paths: dict[str, Path]) -> None:
    """Remove only bus-share forecast outputs while keeping trained zone predictions."""
    for key in ["next_day", "next_month", "next_day_sample", "next_month_sample", "metrics", "zone_accuracy_by_zone", "missingness", "missing_bus_audit", "report", "ai_log"]:
        path = paths[key]
        if path.exists() and path.is_file():
            path.unlink()
        elif path.exists() and path.is_dir():
            shutil.rmtree(path)


def weather_zone_outputs_complete(paths: dict[str, Path]) -> bool:
    """Check whether cached zone-level weather predictions and training log are available."""
    required = [paths["next_day_zone"], paths["next_month_zone"], paths["training_log"]]
    if any(not path.exists() for path in required):
        return False
    try:
        next_day = pd.read_parquet(paths["next_day_zone"], columns=["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"])
        next_month = pd.read_parquet(paths["next_month_zone"], columns=["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"])
    except Exception:
        return False
    return not next_day.empty and not next_month.empty


def run_weather_zone_training_stage(
    force_rebuild: bool = False,
    force_download: bool = False,
    use_observed_target_weather: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.0015,
    dropout: float = 0.10,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
    use_residual: bool = True,
) -> dict[str, object]:
    """Train only the weather-aware zone-level Transformer stage; residual is the main formulation."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = weather_model_paths(output_dir)
    if force_rebuild:
        for key in ["next_day_zone", "next_month_zone", "training_log"]:
            paths[key].unlink(missing_ok=True)
    expected_next_day_model = WEATHER_RESIDUAL_NEXT_DAY_MODEL_NAME if use_residual else WEATHER_NEXT_DAY_MODEL_NAME
    expected_next_month_model = WEATHER_RESIDUAL_NEXT_MONTH_MODEL_NAME if use_residual else WEATHER_NEXT_MONTH_MODEL_NAME
    cache_matches_formulation = (
        weather_zone_outputs_complete(paths)
        and infer_cached_zone_model_name(paths, "next_day", "") == expected_next_day_model
        and infer_cached_zone_model_name(paths, "next_month", "") == expected_next_month_model
    )
    if cache_matches_formulation and not force_rebuild:
        training_log = pd.read_csv(paths["training_log"])
        return {
            "method_name": "method2_zone_share_transformer_with_weather",
            "paths": paths,
            "next_day_zone": pd.read_parquet(paths["next_day_zone"]),
            "next_month_zone": pd.read_parquet(paths["next_month_zone"]),
            "training_log": training_log,
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
            "epochs": epochs,
            "validation_fraction": validation_fraction,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "use_reduce_lr_on_plateau": use_reduce_lr_on_plateau,
            "lr_scheduler_factor": lr_scheduler_factor,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_min_lr": lr_scheduler_min_lr,
            "next_month_lookback_months": list(normalize_month_lookbacks(next_month_lookback_months)),
            "use_observed_target_weather": use_observed_target_weather,
            "use_residual": use_residual,
        }

    print("weather zone stage: preparing weather cache", flush=True)
    download_zone_weather(force_download=force_download)
    print("weather zone stage: loading cached weather features", flush=True)
    weather = load_zone_weather(output_dir)
    print("weather zone stage: loading zone load data", flush=True)
    zone = load_zone_data(data_dir)
    training_log_rows: list[dict[str, object]] = []
    next_day_builder = build_next_day_weather_zone_predictions_residual if use_residual else build_next_day_weather_zone_predictions
    next_month_builder = build_next_month_weather_zone_predictions_residual if use_residual else build_next_month_weather_zone_predictions
    formulation = "residual" if use_residual else "direct"
    print(f"weather zone stage: training {formulation} next-day zone Transformer", flush=True)
    next_day_zone = next_day_builder(
        zone,
        weather,
        use_observed_target_weather=use_observed_target_weather,
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        training_log_rows=training_log_rows,
    )
    print(f"weather zone stage: training {formulation} rolling next-month zone Transformer", flush=True)
    next_month_zone = next_month_builder(
        zone,
        weather,
        use_observed_target_weather=use_observed_target_weather,
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        next_month_lookback_months=next_month_lookback_months,
        training_log_rows=training_log_rows,
    )
    training_log = pd.DataFrame(training_log_rows)
    next_day_zone.to_parquet(paths["next_day_zone"], index=False)
    next_month_zone.to_parquet(paths["next_month_zone"], index=False)
    training_log.to_csv(paths["training_log"], index=False)
    return {
        "method_name": "method2_zone_share_transformer_with_weather",
        "paths": paths,
        "next_day_zone": next_day_zone,
        "next_month_zone": next_month_zone,
        "training_log": training_log,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": time.time() - start_time,
        "epochs": epochs,
        "validation_fraction": validation_fraction,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "use_reduce_lr_on_plateau": use_reduce_lr_on_plateau,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "lr_scheduler_min_lr": lr_scheduler_min_lr,
        "next_month_lookback_months": list(normalize_month_lookbacks(next_month_lookback_months)),
        "use_observed_target_weather": use_observed_target_weather,
        "use_residual": use_residual,
    }


def run_weather_residual_zone_training_stage(
    force_rebuild: bool = False,
    force_download: bool = False,
    use_observed_target_weather: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.0015,
    dropout: float = 0.10,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Train/cache weather-aware residual Method 2B zone predictions only."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = weather_residual_model_paths(output_dir)
    if force_rebuild:
        for path in paths.values():
            path.unlink(missing_ok=True)
    if weather_zone_outputs_complete(paths) and not force_rebuild:
        training_log = pd.read_csv(paths["training_log"])
        return {
            "method_name": "method2B_zone_residual_transformer_with_weather",
            "paths": paths,
            "next_day_zone": pd.read_parquet(paths["next_day_zone"]),
            "next_month_zone": pd.read_parquet(paths["next_month_zone"]),
            "training_log": training_log,
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
        }

    print("weather residual zone stage: preparing weather cache", flush=True)
    download_zone_weather(force_download=force_download)
    print("weather residual zone stage: loading cached weather features", flush=True)
    weather = load_zone_weather(output_dir)
    print("weather residual zone stage: loading zone load data", flush=True)
    zone = load_zone_data(data_dir)
    training_log_rows: list[dict[str, object]] = []
    next_day_zone = build_next_day_weather_zone_predictions_residual(
        zone,
        weather,
        use_observed_target_weather=use_observed_target_weather,
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        training_log_rows=training_log_rows,
    )
    next_month_zone = build_next_month_weather_zone_predictions_residual(
        zone,
        weather,
        use_observed_target_weather=use_observed_target_weather,
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        next_month_lookback_months=next_month_lookback_months,
        training_log_rows=training_log_rows,
    )
    training_log = pd.DataFrame(training_log_rows)
    next_day_zone.to_parquet(paths["next_day_zone"], index=False)
    next_month_zone.to_parquet(paths["next_month_zone"], index=False)
    training_log.to_csv(paths["training_log"], index=False)
    return {
        "method_name": "method2B_zone_residual_transformer_with_weather",
        "paths": paths,
        "next_day_zone": next_day_zone,
        "next_month_zone": next_month_zone,
        "training_log": training_log,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": time.time() - start_time,
    }


def run_weather_bus_share_stage(
    force_rebuild: bool = False,
    use_observed_target_weather: bool = False,
    forecast_workers: int = 1,
) -> dict[str, object]:
    """Run only the bus-share allocation stage from cached weather-aware zone predictions."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = weather_model_paths(output_dir)
    forecast_workers = max(1, min(int(forecast_workers), os.cpu_count() or 1, 12))
    if force_rebuild:
        remove_weather_bus_outputs(paths)
    if weather_outputs_complete(paths, data_dir) and not force_rebuild:
        metrics = pd.read_csv(paths["metrics"])
        return {
            "method_name": "method2_zone_share_transformer_with_weather",
            "paths": paths,
            "metrics": metrics,
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
            "forecast_workers": forecast_workers,
            "use_observed_target_weather": use_observed_target_weather,
        }
    if not weather_zone_outputs_complete(paths):
        raise FileNotFoundError(
            "Weather zone predictions are missing. Run run_weather_zone_training_stage(...) before run_weather_bus_share_stage(...)."
        )

    next_day_zone = pd.read_parquet(paths["next_day_zone"])
    next_month_zone = pd.read_parquet(paths["next_month_zone"])
    print("weather bus-share stage: building bus-level imputer for share allocation", flush=True)
    imputer = build_bus_pd_imputer(data_dir, paths["missingness"])
    print("weather bus-share stage: computing 2024 fallback bus shares", flush=True)
    fallback = compute_2024_bus_hour_share(data_dir, imputer)
    print("weather bus-share stage: computing bus-hour-weekday historical average baseline", flush=True)
    historical_average_baseline = compute_bus_hour_weekday_average_baseline(data_dir, imputer)
    default_next_day_model_name = WEATHER_NEXT_DAY_MODEL_NAME
    default_next_month_model_name = WEATHER_NEXT_MONTH_MODEL_NAME
    next_day_model_name = infer_cached_zone_model_name(paths, "next_day", default_next_day_model_name)
    next_month_model_name = infer_cached_zone_model_name(paths, "next_month", default_next_month_model_name)
    print("weather bus-share stage: streaming weather-aware 2025 bus forecasts", flush=True)
    metrics = process_weather_bus_forecasts_parallel(
        data_dir,
        output_dir,
        paths,
        next_day_zone,
        next_month_zone,
        fallback,
        historical_average_baseline,
        imputer,
        next_day_model_name,
        next_month_model_name,
        max_workers=forecast_workers,
    )
    write_sample_csv(paths["next_day"], paths["next_day_sample"])
    write_sample_csv(paths["next_month"], paths["next_month_sample"])
    return {
        "method_name": "method2_zone_share_transformer_with_weather",
        "paths": paths,
        "metrics": metrics,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": time.time() - start_time,
        "forecast_workers": forecast_workers,
        "use_observed_target_weather": use_observed_target_weather,
    }


def run_weather_validation_stage() -> dict[str, object]:
    """Validate already-written weather-aware Method 2 outputs without retraining or rewriting forecasts."""
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = weather_model_paths(output_dir)
    expected_rows = expected_2025_bus_rows(data_dir)
    checks: list[dict[str, object]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for key in ["next_day", "next_month"]:
        path = paths[key]
        if not path.exists():
            add_check(f"{key}_forecast_exists", False, f"Missing {path.name}")
            continue
        try:
            pf = pq.ParquetFile(path)
            add_check(f"{key}_schema", pf.schema_arrow.names == FORECAST_COLUMNS, ",".join(pf.schema_arrow.names))
            add_check(f"{key}_row_count", pf.metadata.num_rows == expected_rows, f"{pf.metadata.num_rows:,} rows; expected {expected_rows:,}")
            sample = pf.read_row_group(0, columns=["model_name", "predict_pd"]).to_pandas()
            default_model = WEATHER_NEXT_DAY_MODEL_NAME if key == "next_day" else WEATHER_NEXT_MONTH_MODEL_NAME
            expected_model = infer_cached_zone_model_name(paths, key, default_model)
            add_check(f"{key}_model_name", set(sample["model_name"].astype(str).unique()) == {expected_model}, ",".join(sorted(sample["model_name"].astype(str).unique())))
            add_check(f"{key}_nonnegative_sample", bool((sample["predict_pd"] >= 0).all()), "First row group checked")
        except Exception as exc:
            add_check(f"{key}_readable", False, repr(exc))

    if paths["metrics"].exists():
        metrics = pd.read_csv(paths["metrics"])
        required_metric_keys = {
            ("next_day", "bus", infer_cached_zone_model_name(paths, "next_day", WEATHER_NEXT_DAY_MODEL_NAME)),
            ("next_day", "zone", infer_cached_zone_model_name(paths, "next_day", WEATHER_NEXT_DAY_MODEL_NAME)),
            ("next_month", "bus", infer_cached_zone_model_name(paths, "next_month", WEATHER_NEXT_MONTH_MODEL_NAME)),
            ("next_month", "zone", infer_cached_zone_model_name(paths, "next_month", WEATHER_NEXT_MONTH_MODEL_NAME)),
            ("next_day", "bus", "baseline_previous_week"),
            ("next_day", "zone", "baseline_previous_week"),
            ("next_month", "bus", "baseline_previous_year"),
            ("next_month", "zone", "baseline_previous_year"),
            ("next_day", "bus", "baseline_bus_hour_weekday_avg"),
            ("next_day", "zone", "baseline_bus_hour_weekday_avg"),
            ("next_month", "bus", "baseline_bus_hour_weekday_avg"),
            ("next_month", "zone", "baseline_bus_hour_weekday_avg"),
        }
        actual_metric_keys = set(zip(metrics["task"], metrics["level"], metrics["model_name"]))
        missing = sorted(required_metric_keys - actual_metric_keys)
        add_check("metrics_required_rows", not missing, "missing=" + repr(missing[:5]))
    else:
        metrics = pd.DataFrame()
        add_check("metrics_exists", False, f"Missing {paths['metrics'].name}")

    if paths["zone_accuracy_by_zone"].exists():
        try:
            zone_accuracy = pd.read_csv(paths["zone_accuracy_by_zone"])
            required_columns = {"task", "zone_id", "model_name", "n", "mae", "rmse", "wmape", "sum_actual"}
            add_check("zone_accuracy_by_zone_schema", required_columns.issubset(zone_accuracy.columns), ",".join(zone_accuracy.columns))
            add_check("zone_accuracy_by_zone_nonempty", not zone_accuracy.empty, f"{len(zone_accuracy)} rows")
        except Exception as exc:
            add_check("zone_accuracy_by_zone_readable", False, repr(exc))
    else:
        add_check("zone_accuracy_by_zone_exists", False, f"Missing {paths['zone_accuracy_by_zone'].name}")

    add_check("zone_training_cache_exists", weather_zone_outputs_complete(paths), "Zone forecast cache and training log available")
    validation = pd.DataFrame(checks)
    return {
        "method_name": "method2_zone_share_transformer_with_weather",
        "paths": paths,
        "metrics": metrics,
        "validation": validation,
        "available": bool(validation["passed"].all()),
        "rebuilt": False,
    }


def run_zone_share_transformer_with_weather(
    force_rebuild: bool = False,
    force_download: bool = False,
    use_observed_target_weather: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.0015,
    dropout: float = 0.10,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
    forecast_workers: int = 1,
) -> dict[str, object]:
    """Run Method 2 with optional zone-level weather features as a separate model."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = weather_model_paths(output_dir)
    forecast_workers = max(1, min(int(forecast_workers), os.cpu_count() or 1, 12))
    if force_rebuild:
        remove_weather_outputs(paths)
    if weather_outputs_complete(paths, data_dir) and not force_rebuild:
        metrics = pd.read_csv(paths["metrics"])
        return {
            "method_name": "method2_zone_share_transformer_with_weather",
            "paths": paths,
            "metrics": metrics,
            "available": True,
            "rebuilt": False,
            "epochs": epochs,
            "validation_fraction": validation_fraction,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "use_reduce_lr_on_plateau": use_reduce_lr_on_plateau,
            "lr_scheduler_factor": lr_scheduler_factor,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_min_lr": lr_scheduler_min_lr,
            "next_month_lookback_months": list(normalize_month_lookbacks(next_month_lookback_months)),
            "forecast_workers": forecast_workers,
            "use_observed_target_weather": use_observed_target_weather,
        }

    zone_artifacts = run_weather_zone_training_stage(
        force_rebuild=False,
        force_download=force_download,
        use_observed_target_weather=use_observed_target_weather,
        epochs=epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
        next_month_lookback_months=next_month_lookback_months,
    )
    bus_artifacts = run_weather_bus_share_stage(
        force_rebuild=False,
        use_observed_target_weather=use_observed_target_weather,
        forecast_workers=forecast_workers,
    )
    metrics = bus_artifacts["metrics"]
    return {
        "method_name": "method2_zone_share_transformer_with_weather",
        "paths": paths,
        "metrics": metrics,
        "training_log": zone_artifacts["training_log"],
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": time.time() - start_time,
        "epochs": epochs,
        "validation_fraction": validation_fraction,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "use_reduce_lr_on_plateau": use_reduce_lr_on_plateau,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "lr_scheduler_min_lr": lr_scheduler_min_lr,
        "next_month_lookback_months": list(normalize_month_lookbacks(next_month_lookback_months)),
        "forecast_workers": forecast_workers,
        "use_observed_target_weather": use_observed_target_weather,
    }
