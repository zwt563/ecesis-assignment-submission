from __future__ import annotations  

import math  
import os  
import shutil  
import time  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from dataclasses import dataclass  
from pathlib import Path  

import numpy as np  
import pandas as pd  
import pyarrow as pa  
import pyarrow.parquet as pq  
import torch  
from pandas.tseries.holiday import USFederalHolidayCalendar  


TRANSFORMER_NEXT_DAY_MODEL_NAME = "zone_share_transformer_next_day"  
TRANSFORMER_NEXT_MONTH_MODEL_NAME = "zone_share_transformer_next_month"  
TRANSFORMER_RESIDUAL_NEXT_DAY_MODEL_NAME = "zone_share_transformer_next_day"
TRANSFORMER_RESIDUAL_NEXT_MONTH_MODEL_NAME = "zone_share_transformer_next_month"
FORECAST_COMPRESSION = "snappy"
FORECAST_USE_DICTIONARY = False
SOURCE_SHARE_CACHE_NAME = "assignment2_source_share_cache"


FORECAST_COLUMNS = [  
    "model_name",  
    "forecast_created_at",  
    "target_date",  
    "he",  
    "bus_id",  
    "zone_id",  
    "predict_pd",  
]  

FORECAST_SCHEMA = pa.schema(  
    [  
        pa.field("model_name", pa.string()),  
        pa.field("forecast_created_at", pa.string()),  
        pa.field("target_date", pa.string()),  
        pa.field("he", pa.int16()),  
        pa.field("bus_id", pa.string()),  
        pa.field("zone_id", pa.string()),  
        pa.field("predict_pd", pa.float32()),  
    ]  
)  


@dataclass  
class MetricAccumulator:  
    """Lightweight accumulator for streaming error metrics.

The bus forecast files are too large to load all predictions and actuals into memory at once.
This class stores only counts, absolute-error totals, squared-error totals, and actual-load totals;
MAE, RMSE, and WMAPE are computed from those totals at the end."""
    n: int = 0  
    abs_error_sum: float = 0.0  
    sq_error_sum: float = 0.0  
    actual_sum: float = 0.0  

    def update(self, actual: np.ndarray, predicted: np.ndarray) -> None:  
        """Add one batch of actual and predicted values to the metric accumulator."""
        actual_arr = np.asarray(actual, dtype=np.float64)  
        predicted_arr = np.asarray(predicted, dtype=np.float64)  
        error = predicted_arr - actual_arr  
        self.n += int(actual_arr.size)  
        self.abs_error_sum += float(np.abs(error).sum())  
        self.sq_error_sum += float(np.square(error).sum())  
        self.actual_sum += float(np.abs(actual_arr).sum())  

    def as_dict(self) -> dict[str, float]:  
        """Convert accumulated totals into MAE, RMSE, WMAPE, and total actual load."""
        mae = self.abs_error_sum / self.n if self.n else np.nan  
        rmse = math.sqrt(self.sq_error_sum / self.n) if self.n else np.nan  
        wmape = self.abs_error_sum / self.actual_sum if self.actual_sum else np.nan  
        return {  
            "n": self.n,  
            "mae": mae,  
            "rmse": rmse,  
            "wmape": wmape,  
            "sum_actual": self.actual_sum,  
        }


@dataclass  
class BusPdImputer:  
    """Container for the five-level time-aware historical bus-load imputation tables.

partial_missing_by_year stores buses that are partially missing but have at least one nonzero historical load.
lookup_by_source_year stores the historical averages and share lookup tables available for each source year.
missingness_by_year stores the bus-level missingness audit used for review and downstream filtering."""
    partial_missing_by_year: dict[int, set[str]]  
    lookup_by_source_year: dict[int, dict[str, pd.DataFrame]]  
    missingness_by_year: pd.DataFrame  

class FeatureTokenTransformer(torch.nn.Module):  
    """Main Assignment 2 neural network for zone-hour load forecasting.

Each numeric feature is treated as a token, projected into the Transformer hidden dimension,
and processed by a TransformerEncoder so the model can learn interactions among hour, calendar, lag-load,
and historical-average signals. The output head returns one zone-load prediction per row."""
    def __init__(  
        self,  
        n_features: int,  
        d_model: int = 16,  
        nhead: int = 4,  
        num_layers: int = 2,  
        dim_feedforward: int = 64,  
        dropout: float = 0.05,  
    ) -> None:  
        """Initialize the Transformer layers used for zone-load regression."""
        super().__init__()  
        self.value_projection = torch.nn.Linear(1, d_model)  
        self.feature_embedding = torch.nn.Parameter(torch.randn(n_features, d_model) * 0.02)  
        encoder_layer = torch.nn.TransformerEncoderLayer(  
            d_model=d_model,  
            nhead=nhead,  
            dim_feedforward=dim_feedforward,  
            dropout=dropout,  
            batch_first=True,  
            activation="gelu",  
            norm_first=True,  
        )  
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)  
        self.head = torch.nn.Sequential(  
            torch.nn.LayerNorm(d_model),  
            torch.nn.Linear(d_model, 32),  
            torch.nn.GELU(),  
            torch.nn.Linear(32, 1),  
        )  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """Run the forward pass from numeric features to one standardized zone-load prediction per row."""
        tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)  
        encoded = self.encoder(tokens)  
        pooled = encoded.mean(dim=1)  
        return self.head(pooled)  


def find_project_root() -> Path:  
    """Find the project root by walking upward from this script location.

This supports both the original Summer2026-main data layout and the final submission repository layout."""
    current = Path(__file__).resolve()  
    for candidate in [current.parent, *current.parents]:  
        has_source_data = (candidate / "data").exists() or (candidate / "assignment2_load_forecasting" / "input_data").exists()  
        if (candidate / "README.md").exists() and has_source_data:  
            return candidate  
    raise FileNotFoundError("Could not find Summer2026-main project root.")  


def find_assignment2_data_dir(project_root: Path) -> Path:  
    """Locate the Assignment 2 parquet data directory.

The function first checks the submission-friendly input_data layout and then falls back to the original downloaded data locations."""
    workspace_root = project_root.parent  
    candidates = [  
        Path(__file__).resolve().parent / "input_data",  
        project_root / "assignment2_load_forecasting" / "input_data",  
        workspace_root / "2026 Summer Intern-20260524T050101Z-3-001" / "2026 Summer Intern",  
        project_root / "data" / "Assignment 2 - load_forecasting",  
        project_root / "data" / "Assignment 2",  
    ]  
    for candidate in candidates:  
        if (candidate / "bus_load_2025.parquet").exists() and (candidate / "zone_load_2025.parquet").exists():  
            return candidate  
    raise FileNotFoundError("Could not find Assignment 2 bus/zone parquet files.")  


def output_paths(output_dir: Path) -> dict[str, Path]:  
    """Define all Assignment 2 output paths in one place to avoid hard-coded filenames throughout the pipeline."""
    return {  
        "next_day": output_dir / "assignment2_next_day_forecast_2025.parquet",  
        "next_month": output_dir / "assignment2_next_month_forecast_2025.parquet",  
        "next_day_sample": output_dir / "assignment2_next_day_forecast_sample.csv",  
        "next_month_sample": output_dir / "assignment2_next_month_forecast_sample.csv",  
        "metrics": output_dir / "assignment2_metrics.csv",  
        "training_log": output_dir / "assignment2_training_log.csv",  
        "missingness": output_dir / "assignment2_bus_pd_missingness_by_year.csv",  
        "missing_bus_audit": output_dir / "assignment2_missing_bus_forecast_audit.csv",  
        "zone_accuracy_by_zone": output_dir / "assignment2_method2_zone_accuracy_by_zone.csv",  # Per-zone zone-level accuracy table for diagnosing weak zones.
        "report": output_dir / "assignment2_summary_report.md",  
        "ai_log": output_dir / "assignment2_ai_usage_log.md",  
    }  


def bus_path(data_dir: Path, year: int | str) -> Path:  
    """Return the full path for bus_load_YYYY.parquet under the selected data directory."""
    return data_dir / f"bus_load_{year}.parquet"  


def zone_path(data_dir: Path, year: int | str) -> Path:  
    """Return the full path for zone_load_YYYY.parquet under the selected data directory."""
    return data_dir / f"zone_load_{year}.parquet"  


def load_zone_data(data_dir: Path) -> pd.DataFrame:  
    """Load 2022-2025 zone-level data and standardize the date, hour-ending, and actual-load fields."""
    frames = []  
    for year in [2022, 2023, 2024, 2025]:  
        frame = pd.read_parquet(zone_path(data_dir, year))  
        frame["source_year"] = year  
        frames.append(frame)  
    zone = pd.concat(frames, ignore_index=True)  
    zone["target_date"] = pd.to_datetime(zone["date"])  
    zone["actual_pd"] = zone["pd"].fillna(0.0).astype("float64")  
    zone["he"] = zone["he"].astype("int16")  
    return zone  


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:  
    """Create calendar features from target_date.

Load is strongly shaped by hour, weekday, month, weekends, and holidays, so these fields are used as model inputs."""
    out = frame.copy()  
    holidays = USFederalHolidayCalendar().holidays(  
        start=out["target_date"].min(),  
        end=out["target_date"].max(),  
    )  
    holiday_dates = set(pd.to_datetime(holidays).date)  
    out["dayofweek"] = out["target_date"].dt.dayofweek.astype("int16")  
    out["month"] = out["target_date"].dt.month.astype("int16")  
    out["day"] = out["target_date"].dt.day.astype("int16")  
    out["dayofyear"] = out["target_date"].dt.dayofyear.astype("int16")  
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype("int8")  
    out["is_holiday"] = out["target_date"].dt.date.isin(holiday_dates).astype("int8")  
    return out  


def add_day_lag_features(frame: pd.DataFrame, lags: list[int]) -> pd.DataFrame:  
    """Add historical daily lag features for the next-day model.

lag_7d, lag_14d, lag_28d, and lag_365d capture weekly, monthly, and annual seasonality for the same zone and hour."""
    out = frame.copy()  
    base = out[["zone_name", "target_date", "he", "actual_pd"]].copy()  
    for lag in lags:  
        lagged = base.copy()  
        lagged["target_date"] = lagged["target_date"] + pd.Timedelta(days=lag)  
        lagged = lagged.rename(columns={"actual_pd": f"lag_{lag}d"})  
        out = out.merge(lagged, on=["zone_name", "target_date", "he"], how="left")  
    return out  


def add_year_ago_feature(frame: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:  
    """Add the previous-year same-date same-hour zone load as a long-seasonality feature for next-month forecasting."""
    year_ago = actuals[["zone_name", "target_date", "he", "actual_pd"]].copy()  
    year_ago["target_date"] = year_ago["target_date"] + pd.DateOffset(years=1)  
    year_ago = year_ago.rename(columns={"actual_pd": "year_ago_pd"})  
    year_ago = year_ago.groupby(["zone_name", "target_date", "he"], observed=True, as_index=False)["year_ago_pd"].mean()  
    return frame.merge(year_ago, on=["zone_name", "target_date", "he"], how="left")  


def normalize_month_lookbacks(lookback_months: list[int] | tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Return clean positive month lookbacks for next-month lag features."""
    raw_values = [3, 6, 12] if lookback_months is None else list(lookback_months)
    cleaned = sorted({int(value) for value in raw_values if int(value) > 0})
    return tuple(cleaned)


def add_month_lag_features(
    frame: pd.DataFrame,
    actuals: pd.DataFrame,
    lookback_months: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Add leakage-safe zone load lags such as lag_3m, lag_6m, and lag_12m."""
    out = frame.copy()
    base = actuals[["zone_name", "target_date", "he", "actual_pd"]].copy()
    for months in normalize_month_lookbacks(lookback_months):
        column = f"lag_{months}m"
        lagged = base.copy()
        lagged["target_date"] = lagged["target_date"] + pd.DateOffset(months=months)
        lagged = lagged.rename(columns={"actual_pd": column})
        lagged = lagged.groupby(["zone_name", "target_date", "he"], observed=True, as_index=False)[column].mean()
        out = out.merge(lagged, on=["zone_name", "target_date", "he"], how="left")
    return out


def fit_predict_zone_transformer_models(  
    train: pd.DataFrame,  
    predict: pd.DataFrame,  
    features: list[str],  
    model_name: str,  
    epochs: int = 140,  
    validation_fraction: float = 0.15,  
    early_stopping_patience: int | None = 20,  
    early_stopping_min_delta: float = 1e-4,  
    learning_rate: float = 0.003,  # AdamW initial learning rate; weather-aware calls can lower this without changing the no-weather model.
    dropout: float = 0.05,  # Transformer dropout rate; weather-aware calls can raise this for stronger regularization.
    use_reduce_lr_on_plateau: bool = False,  # Whether to reduce learning rate when monitored loss plateaus.
    lr_scheduler_factor: float = 0.5,  # Multiplicative learning-rate decay used by ReduceLROnPlateau.
    lr_scheduler_patience: int = 6,  # Number of stale epochs before reducing learning rate.
    lr_scheduler_min_lr: float = 1e-4,  # Lower bound for ReduceLROnPlateau learning rate.
    training_log_rows: list[dict[str, object]] | None = None,  # Optional list that receives one row per epoch for later CSV export.
    training_context: dict[str, object] | None = None,  # Optional metadata such as task, month, or weather/no-weather method label.
    target_column: str = "actual_pd",
    prediction_column: str = "predict_zone_pd",
    clip_prediction: bool = True,
    loss_units: str | None = None,
) -> pd.DataFrame:  
    """Fit one Transformer per zone, use validation MSE for early stopping, and return zone-load predictions from the best checkpoint."""
    if not torch.cuda.is_available():  
        raise RuntimeError(  
            "GPU training is required for the Transformer addon, but torch.cuda.is_available() is False. "  
            "Run this script in a CUDA-enabled PyTorch environment, for example conda env `wentao_2025_10_13`."  
        )  
    torch.manual_seed(123)  
    torch.cuda.manual_seed_all(123)  
    device = torch.device("cuda")  
    predictions = []  
    log_context = dict(training_context or {})  # Copy caller-provided metadata so every training-log row is self-describing.
    for zone_name, zone_train in train.groupby("zone_name", sort=True):  
        zone_predict = predict[predict["zone_name"].eq(zone_name)].copy()  
        if zone_predict.empty:  
            continue  
        if zone_train[target_column].abs().sum() == 0:  
            zone_predict[prediction_column] = 0.0  
        else:  
            zone_train_sorted = zone_train.sort_values(["target_date", "he"]).copy()  
            use_validation = bool(0.0 < validation_fraction < 0.5 and len(zone_train_sorted) >= 240)  
            if use_validation:  
                split_index = int(len(zone_train_sorted) * (1.0 - validation_fraction))  
                split_index = min(max(split_index, 1), len(zone_train_sorted) - 1)  
                fit_frame = zone_train_sorted.iloc[:split_index].copy()  
                validation_frame = zone_train_sorted.iloc[split_index:].copy()  
            else:  
                fit_frame = zone_train_sorted  
                validation_frame = None  
            medians = fit_frame[features].median(numeric_only=True)  
            x_train_np = fit_frame[features].fillna(medians).fillna(0.0).astype("float32").to_numpy()  
            y_train_np = fit_frame[target_column].astype("float32").to_numpy()  
            x_predict_np = zone_predict[features].fillna(medians).fillna(0.0).astype("float32").to_numpy()  
            x_mean = x_train_np.mean(axis=0, keepdims=True)  
            x_std = x_train_np.std(axis=0, keepdims=True)  
            x_std = np.where(x_std < 1e-6, 1.0, x_std)  
            y_mean = float(y_train_np.mean())  
            y_std = float(y_train_np.std())  
            if y_std < 1e-6:  
                y_std = 1.0  
            x_train_tensor = torch.tensor((x_train_np - x_mean) / x_std, dtype=torch.float32, device=device)  
            y_train_tensor = torch.tensor((y_train_np - y_mean) / y_std, dtype=torch.float32, device=device).view(-1, 1)  
            if validation_frame is not None:  
                x_val_np = validation_frame[features].fillna(medians).fillna(0.0).astype("float32").to_numpy()  
                y_val_np = validation_frame[target_column].astype("float32").to_numpy()  
                x_val_tensor = torch.tensor((x_val_np - x_mean) / x_std, dtype=torch.float32, device=device)  
                y_val_tensor = torch.tensor((y_val_np - y_mean) / y_std, dtype=torch.float32, device=device).view(-1, 1)  
            else:  # ?? validation ??
                x_val_tensor = None  
                y_val_tensor = None  
            x_predict_tensor = torch.tensor((x_predict_np - x_mean) / x_std, dtype=torch.float32, device=device)  
            model = FeatureTokenTransformer(n_features=len(features), dropout=dropout).to(device)  
            weight_decay = 1e-4  # AdamW weight decay used as light regularization.
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  
            scheduler = (  # ReduceLROnPlateau lets long weather runs lower LR automatically after validation loss stalls.
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=lr_scheduler_factor,
                    patience=lr_scheduler_patience,
                    min_lr=lr_scheduler_min_lr,
                )
                if use_reduce_lr_on_plateau
                else None
            )
            loss_fn = torch.nn.MSELoss()  
            best_loss = float("inf")  
            best_epoch = 0  
            best_train_loss = np.nan  # Record the training loss from the checkpoint selected as best.
            best_val_loss = np.nan  # Record the validation loss from the checkpoint selected as best.
            best_learning_rate = learning_rate  # Record the learning rate used at the selected checkpoint.
            best_state = None  
            stale_epochs = 0  
            for epoch_index in range(epochs):  
                model.train()  
                optimizer.zero_grad(set_to_none=True)  
                prediction = model(x_train_tensor)  
                loss = loss_fn(prediction, y_train_tensor)  
                loss.backward()  
                optimizer.step()  
                train_loss = float(loss.detach().cpu())  # Record standardized training MSE for this epoch.
                if x_val_tensor is not None:  
                    model.eval()  
                    with torch.no_grad():  
                        val_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).detach().cpu())  # Record standardized validation MSE.
                    stop_loss = val_loss  # Early stopping monitors validation loss when validation rows exist.
                else:  # ?? validation ??
                    val_loss = np.nan  # No validation set is available for this small group.
                    stop_loss = train_loss  # Fall back to training loss only when no validation set exists.
                if scheduler is not None:  # Let the scheduler observe the same loss that early stopping monitors.
                    scheduler.step(stop_loss)
                current_learning_rate = float(optimizer.param_groups[0]["lr"])  # Record the actual LR after any scheduler update.
                if stop_loss < best_loss - early_stopping_min_delta:  
                    improved = True  # Mark this epoch as a new best checkpoint in the training log.
                    best_loss = stop_loss  
                    best_epoch = epoch_index + 1  
                    best_train_loss = train_loss  # Keep the train loss for the same checkpoint as best_loss.
                    best_val_loss = val_loss  # Keep the validation loss for the same checkpoint as best_loss.
                    best_learning_rate = current_learning_rate  # Keep the LR for the same checkpoint as best_loss.
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}  
                    stale_epochs = 0  
                else:  
                    improved = False  # This epoch did not beat the previous best monitored loss.
                    stale_epochs += 1  
                early_stopped = bool(early_stopping_patience and stale_epochs >= early_stopping_patience)  # Record whether this epoch triggered stopping.
                if training_log_rows is not None:  # Caller requested a persistent epoch-by-epoch training log.
                    training_log_rows.append(
                        {
                            **log_context,
                            "model_name": model_name,
                            "zone_name": zone_name,
                            "epoch": epoch_index + 1,
                            "max_epochs": epochs,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "monitored_loss": stop_loss,
                            "loss_units": loss_units or f"standardized_{target_column}_mse",
                            "learning_rate": current_learning_rate,
                            "initial_learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "dropout": dropout,
                            "lr_scheduler": "ReduceLROnPlateau" if scheduler is not None else "none",
                            "lr_scheduler_factor": lr_scheduler_factor if scheduler is not None else np.nan,
                            "lr_scheduler_patience": lr_scheduler_patience if scheduler is not None else np.nan,
                            "lr_scheduler_min_lr": lr_scheduler_min_lr if scheduler is not None else np.nan,
                            "n_train": int(len(fit_frame)),
                            "n_validation": int(0 if validation_frame is None else len(validation_frame)),
                            "validation_used": bool(validation_frame is not None),
                            "improved": improved,
                            "stale_epochs": stale_epochs,
                            "best_epoch_so_far": best_epoch,
                            "best_loss_so_far": best_loss,
                            "best_train_loss_so_far": best_train_loss,
                            "best_val_loss_so_far": best_val_loss,
                            "best_learning_rate_so_far": best_learning_rate,
                            "early_stopping_patience": early_stopping_patience,
                            "early_stopping_min_delta": early_stopping_min_delta,
                            "early_stopped": early_stopped,
                        }
                    )
                if early_stopped:  
                    break  
            if best_state is not None:  
                model.load_state_dict(best_state)  
            print(
                f"{model_name}: zone={zone_name} "
                f"best_epoch={best_epoch}/{epochs} "
                f"best_loss={best_loss:.6f} "
                f"best_train_loss={best_train_loss:.6f} "
                f"best_val_loss={best_val_loss:.6f} "
                f"best_lr={best_learning_rate:.6g}",
                flush=True,
            )  # Print the selected checkpoint loss breakdown and LR for training diagnostics.
            model.eval()  
            with torch.no_grad():  
                pred_scaled = model(x_predict_tensor).detach().cpu().numpy().reshape(-1)  
            raw_prediction = pred_scaled * y_std + y_mean
            zone_predict[prediction_column] = np.maximum(raw_prediction, 0.0) if clip_prediction else raw_prediction
            del model, optimizer, x_train_tensor, y_train_tensor, x_predict_tensor  
            if x_val_tensor is not None:  
                del x_val_tensor, y_val_tensor  
            torch.cuda.empty_cache()  
        zone_predict["model_name"] = model_name  
        predictions.append(zone_predict)  
    if not predictions:  
        raise ValueError("No Transformer zone predictions were produced.")  
    return pd.concat(predictions, ignore_index=True)  


def add_zone_residual_target(frame: pd.DataFrame, baseline_column: str = "baseline_zone_pd") -> pd.DataFrame:
    """Add log residual target relative to a nonnegative zone baseline."""
    out = frame.copy()
    baseline = np.maximum(out[baseline_column].fillna(0.0).to_numpy(dtype="float32"), 0.0)
    actual = np.maximum(out["actual_pd"].fillna(0.0).to_numpy(dtype="float32"), 0.0)
    out["target_residual_log"] = (np.log1p(actual) - np.log1p(baseline)).astype("float32")
    return out


def restore_zone_residual_prediction(frame: pd.DataFrame, baseline_column: str = "baseline_zone_pd") -> pd.DataFrame:
    """Convert predicted log residuals back to nonnegative zone load predictions."""
    out = frame.copy()
    baseline_log = np.log1p(np.maximum(out[baseline_column].fillna(0.0).to_numpy(dtype="float32"), 0.0))
    residual = out["predict_residual_log"].fillna(0.0).to_numpy(dtype="float32")
    out["predict_zone_pd"] = np.maximum(np.expm1(baseline_log + residual), 0.0).astype("float32")
    return out


def build_monthly_features(frame: pd.DataFrame, actuals: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:  
    """Build next-month features using only data available before forecast_created_at.

Historical averages are computed from the available-history window so target-month future information is not leaked."""
    out = add_calendar_features(frame)  
    out = add_year_ago_feature(out, actuals)  
    available_calendar = add_calendar_features(available)  
    hist = available_calendar.groupby(["zone_name", "month", "day", "he"], observed=True)["actual_pd"].mean().rename("hist_avg_month_day_he")  
    out = out.merge(hist.reset_index(), on=["zone_name", "month", "day", "he"], how="left")  
    fallback = available_calendar.groupby(["zone_name", "month", "he"], observed=True)["actual_pd"].mean().rename("hist_avg_month_he")  
    out = out.merge(fallback.reset_index(), on=["zone_name", "month", "he"], how="left")  
    out["hist_avg_pd"] = out["hist_avg_month_day_he"].fillna(out["hist_avg_month_he"])  
    return out  


def build_monthly_features(
    frame: pd.DataFrame,
    actuals: pd.DataFrame,
    available: pd.DataFrame,
    lookback_months: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Build next-month features using only history available before forecast_created_at."""
    out = add_calendar_features(frame)
    out = add_year_ago_feature(out, actuals)
    out = add_month_lag_features(out, available, lookback_months)
    available_calendar = add_calendar_features(available)
    hist = available_calendar.groupby(["zone_name", "month", "day", "he"], observed=True)["actual_pd"].mean().rename("hist_avg_month_day_he")
    out = out.merge(hist.reset_index(), on=["zone_name", "month", "day", "he"], how="left")
    fallback = available_calendar.groupby(["zone_name", "month", "he"], observed=True)["actual_pd"].mean().rename("hist_avg_month_he")
    out = out.merge(fallback.reset_index(), on=["zone_name", "month", "he"], how="left")
    out["hist_avg_pd"] = out["hist_avg_month_day_he"].fillna(out["hist_avg_month_he"])
    return out


def build_next_day_transformer_zone_predictions(
    zone: pd.DataFrame,
    training_log_rows: list[dict[str, object]] | None = None,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
) -> pd.DataFrame:  
    """Build the next-day zone training and 2025 prediction tables, train the Transformer, and return zone-hour forecasts with baselines."""
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
    features = [  
        "he",  
        "dayofweek",  
        "month",  
        "dayofyear",  
        "is_weekend",  
        "is_holiday",  
        "lag_7d",
        "lag_14d",
        "lag_7d",  
        "lag_14d",  
        "lag_28d",  
        "lag_365d",  
        "hist_avg_zone_he_dow",  
        "hist_avg_zone_he_month",  
    ]  
    result = fit_predict_zone_transformer_models(
        train,
        predict,
        features,
        TRANSFORMER_NEXT_DAY_MODEL_NAME,
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
        training_context={"method": "method2_zone_forecast_plus_bus_share_transformer", "task": "next_day", "target_month": "2025_full_year"},
    )  
    result["baseline_zone_pd"] = result["lag_7d"].fillna(result["hist_avg_zone_he_dow"]).fillna(result["hist_avg_zone_he_month"]).fillna(0.0)  
    result["target_date_str"] = result["target_date"].dt.strftime("%Y-%m-%d")  
    result["forecast_created_at"] = (result["target_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d 00:01:00")  
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]  


def build_next_day_transformer_zone_predictions(
    zone: pd.DataFrame,
    training_log_rows: list[dict[str, object]] | None = None,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
) -> pd.DataFrame:
    """Build leakage-safe next-day zone forecasts under a strict D-1 00:01 cutoff."""
    frame = add_calendar_features(zone)
    frame = add_day_lag_features(frame, [7, 14, 28, 365])
    train_mask = frame["target_date"].between("2022-01-01", "2024-12-31")
    train_base = frame.loc[train_mask].copy()
    hist_dow = train_base.groupby(["zone_name", "he", "dayofweek"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_dow")
    hist_month = train_base.groupby(["zone_name", "he", "month"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_month")
    frame = frame.merge(hist_dow.reset_index(), on=["zone_name", "he", "dayofweek"], how="left")
    frame = frame.merge(hist_month.reset_index(), on=["zone_name", "he", "month"], how="left")
    train = frame.loc[train_mask].copy()
    predict = frame.loc[frame["target_date"].between("2025-01-01", "2025-12-31")].copy()
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
    ]
    result = fit_predict_zone_transformer_models(
        train,
        predict,
        features,
        TRANSFORMER_NEXT_DAY_MODEL_NAME,
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
        training_context={"method": "method2_zone_forecast_plus_bus_share_transformer", "task": "next_day", "target_month": "2025_full_year"},
    )
    result["baseline_zone_pd"] = result["lag_7d"].fillna(result["hist_avg_zone_he_dow"]).fillna(result["hist_avg_zone_he_month"]).fillna(0.0)
    result["target_date_str"] = result["target_date"].dt.strftime("%Y-%m-%d")
    result["forecast_created_at"] = (result["target_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d 00:01:00")
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]


def build_next_month_transformer_zone_predictions(
    zone: pd.DataFrame,
    training_log_rows: list[dict[str, object]] | None = None,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:  
    """Build rolling next-month forecasts.

For each target month, the model is trained only on rows before that month's forecast creation date to satisfy the no-leakage requirement."""
    all_predictions = []  
    features = [  
        "he",  
        "dayofweek",  
        "month",  
        "dayofyear",  
        "is_weekend",  
        "is_holiday",  
        "year_ago_pd",  
        "hist_avg_pd",  
    ]  
    for month_start in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):  
        created_date = month_start - pd.DateOffset(months=1)  
        available = zone[zone["target_date"].lt(created_date)].copy()  
        target = zone[  
            zone["target_date"].ge(month_start)  
            & zone["target_date"].lt(month_start + pd.DateOffset(months=1))  
        ].copy()  
        train = build_monthly_features(available, zone, available)  
        predict = build_monthly_features(target, zone, available)  
        month_prediction = fit_predict_zone_transformer_models(
            train,
            predict,
            features,
            TRANSFORMER_NEXT_MONTH_MODEL_NAME,
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
                "method": "method2_zone_forecast_plus_bus_share_transformer",
                "task": "next_month",
                "target_month": month_start.strftime("%Y-%m"),
                "forecast_created_at": created_date.strftime("%Y-%m-%d 00:01:00"),
            },
        )  
        month_prediction["baseline_zone_pd"] = month_prediction["year_ago_pd"].fillna(month_prediction["hist_avg_pd"]).fillna(0.0)  
        month_prediction["target_date_str"] = month_prediction["target_date"].dt.strftime("%Y-%m-%d")  
        month_prediction["forecast_created_at"] = created_date.strftime("%Y-%m-%d 00:01:00")  
        all_predictions.append(month_prediction)  
    result = pd.concat(all_predictions, ignore_index=True)  
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]  


def build_next_month_transformer_zone_predictions(
    zone: pd.DataFrame,
    training_log_rows: list[dict[str, object]] | None = None,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Build rolling next-month zone forecasts with configurable month lookback features."""
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
    ] + [f"lag_{months}m" for months in month_lookbacks]
    all_predictions = []
    for month_start in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        created_date = month_start - pd.DateOffset(months=1)
        available = zone[zone["target_date"].lt(created_date)].copy()
        target = zone[
            zone["target_date"].ge(month_start)
            & zone["target_date"].lt(month_start + pd.DateOffset(months=1))
        ].copy()
        train = build_monthly_features(available, zone, available, month_lookbacks)
        predict = build_monthly_features(target, zone, available, month_lookbacks)
        month_prediction = fit_predict_zone_transformer_models(
            train,
            predict,
            features,
            TRANSFORMER_NEXT_MONTH_MODEL_NAME,
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
                "method": "method2_zone_forecast_plus_bus_share_transformer",
                "task": "next_month",
                "target_month": month_start.strftime("%Y-%m"),
                "forecast_created_at": created_date.strftime("%Y-%m-%d 00:01:00"),
                "next_month_lookback_months": ",".join(str(value) for value in month_lookbacks),
            },
        )
        month_prediction["baseline_zone_pd"] = month_prediction["year_ago_pd"].fillna(month_prediction["hist_avg_pd"]).fillna(0.0)
        month_prediction["target_date_str"] = month_prediction["target_date"].dt.strftime("%Y-%m-%d")
        month_prediction["forecast_created_at"] = created_date.strftime("%Y-%m-%d 00:01:00")
        all_predictions.append(month_prediction)
    result = pd.concat(all_predictions, ignore_index=True)
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]


def build_next_day_transformer_zone_predictions_residual(
    zone: pd.DataFrame,
    training_log_rows: list[dict[str, object]] | None = None,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
) -> pd.DataFrame:
    """Build residual-corrected next-day zone forecasts anchored on lag_7d/history baseline."""
    frame = add_calendar_features(zone)
    frame = add_day_lag_features(frame, [7, 14, 28, 365])
    train_mask = frame["target_date"].between("2022-01-01", "2024-12-31")
    train_base = frame.loc[train_mask].copy()
    hist_dow = train_base.groupby(["zone_name", "he", "dayofweek"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_dow")
    hist_month = train_base.groupby(["zone_name", "he", "month"], observed=True)["actual_pd"].mean().rename("hist_avg_zone_he_month")
    frame = frame.merge(hist_dow.reset_index(), on=["zone_name", "he", "dayofweek"], how="left")
    frame = frame.merge(hist_month.reset_index(), on=["zone_name", "he", "month"], how="left")
    frame["baseline_zone_pd"] = frame["lag_7d"].fillna(frame["hist_avg_zone_he_dow"]).fillna(frame["hist_avg_zone_he_month"]).fillna(0.0)
    train = add_zone_residual_target(frame.loc[train_mask].copy())
    predict = frame.loc[frame["target_date"].between("2025-01-01", "2025-12-31")].copy()
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
    ]
    result = fit_predict_zone_transformer_models(
        train,
        predict,
        features,
        TRANSFORMER_RESIDUAL_NEXT_DAY_MODEL_NAME,
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
        training_context={"method": "method2B_zone_residual_transformer", "task": "next_day", "target_month": "2025_full_year"},
        target_column="target_residual_log",
        prediction_column="predict_residual_log",
        clip_prediction=False,
        loss_units="standardized_log1p_zone_residual_mse",
    )
    result = restore_zone_residual_prediction(result)
    result["target_date_str"] = result["target_date"].dt.strftime("%Y-%m-%d")
    result["forecast_created_at"] = (result["target_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d 00:01:00")
    return result[["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]]


def build_next_month_transformer_zone_predictions_residual(
    zone: pd.DataFrame,
    training_log_rows: list[dict[str, object]] | None = None,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Build residual-corrected rolling next-month zone forecasts anchored on previous-year/history baseline."""
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
    ] + [f"lag_{months}m" for months in month_lookbacks]
    all_predictions = []
    for month_start in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        created_date = month_start - pd.DateOffset(months=1)
        available = zone[zone["target_date"].lt(created_date)].copy()
        target = zone[
            zone["target_date"].ge(month_start)
            & zone["target_date"].lt(month_start + pd.DateOffset(months=1))
        ].copy()
        train = build_monthly_features(available, zone, available, month_lookbacks)
        predict = build_monthly_features(target, zone, available, month_lookbacks)
        train["baseline_zone_pd"] = train["year_ago_pd"].fillna(train["hist_avg_pd"]).fillna(0.0)
        predict["baseline_zone_pd"] = predict["year_ago_pd"].fillna(predict["hist_avg_pd"]).fillna(0.0)
        train = add_zone_residual_target(train)
        month_prediction = fit_predict_zone_transformer_models(
            train,
            predict,
            features,
            TRANSFORMER_RESIDUAL_NEXT_MONTH_MODEL_NAME,
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
                "method": "method2B_zone_residual_transformer",
                "task": "next_month",
                "target_month": month_start.strftime("%Y-%m"),
                "forecast_created_at": created_date.strftime("%Y-%m-%d 00:01:00"),
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


def parquet_row_count(path: Path) -> int:  
    """Read the total row count from parquet metadata without loading the full table into memory."""
    return pq.ParquetFile(path).metadata.num_rows  


def expected_2025_bus_rows(data_dir: Path) -> int:  
    """Use the 2025 bus table row count as the required forecast coverage for each horizon."""
    return parquet_row_count(bus_path(data_dir, 2025))  


def input_coverage_note(data_dir: Path) -> str:  
    """Check whether the provided 2025 zone table covers every date, hour-ending, and zone combination, then return a report note."""
    zone_2025 = pd.read_parquet(zone_path(data_dir, 2025), columns=["date", "he", "zone_name"])  
    zones = sorted(zone_2025["zone_name"].unique())  
    expected = pd.MultiIndex.from_product(  
        [pd.date_range("2025-01-01", "2025-12-31").strftime("%Y-%m-%d"), range(1, 25), zones],  
        names=["date", "he", "zone_name"],  
    )  
    actual = pd.MultiIndex.from_frame(zone_2025[["date", "he", "zone_name"]].drop_duplicates())  
    missing = expected.difference(actual)  
    if len(missing) == 0:  
        return "The provided 2025 zone target data covers every calendar date and hour."  
    missing_frame = pd.DataFrame(list(missing), columns=["date", "he", "zone_name"])  
    missing_dates = sorted(missing_frame["date"].unique())  
    date_preview = ", ".join(missing_dates[:10])  
    if len(missing_dates) > 10:  
        date_preview += ", ..."  
    return (  
        f"The provided 2025 zone target data is missing {len(missing):,} date/hour/zone combinations "  
        f"across {len(missing_dates)} dates ({date_preview}). Forecast files therefore cover every "  
        "provided 2025 bus target row; missing source target rows cannot be evaluated without inventing a bus roster and actual load."  
    )  


def forecast_files_are_complete(paths: dict[str, Path], data_dir: Path) -> bool:  
    """Validate only the next-day and next-month forecast parquet files for row count, schema, and model_name."""
    if not paths["next_day"].exists() or not paths["next_month"].exists():  
        return False  
    expected_rows = expected_2025_bus_rows(data_dir)  
    stage_paths = method2_zone_stage_paths(paths["next_day"].parent)
    try:  
        for key in ["next_day", "next_month"]:  
            pf = pq.ParquetFile(paths[key])  
            if pf.metadata.num_rows != expected_rows:  
                return False  
            if pf.schema_arrow.names != FORECAST_COLUMNS:  
                return False  
            default_model = TRANSFORMER_NEXT_DAY_MODEL_NAME if key == "next_day" else TRANSFORMER_NEXT_MONTH_MODEL_NAME
            expected_model = infer_cached_zone_model_name(stage_paths, key, default_model)
            model_names = set(pf.read_row_group(0, columns=["model_name"]).to_pandas()["model_name"].unique())  
            if model_names != {expected_model}:  
                return False  
    except Exception:  
        return False  
    return True  


def outputs_are_complete(paths: dict[str, Path], data_dir: Path) -> bool:  
    """Check whether existing deliverables are complete so a rerun can validate outputs without retraining or rewriting large forecast files."""
    required = [paths["metrics"], paths["zone_accuracy_by_zone"], paths["missingness"], paths["missing_bus_audit"], paths["report"], paths["ai_log"]]  
    if any(not path.exists() for path in required):  
        return False  
    if not forecast_files_are_complete(paths, data_dir):  
        return False  
    try:  
        metrics = pd.read_csv(paths["metrics"])  
        stage_paths = method2_zone_stage_paths(paths["next_day"].parent)
        expected_next_day_model = infer_cached_zone_model_name(stage_paths, "next_day", TRANSFORMER_NEXT_DAY_MODEL_NAME)
        expected_next_month_model = infer_cached_zone_model_name(stage_paths, "next_month", TRANSFORMER_NEXT_MONTH_MODEL_NAME)
        required_metric_keys = {  
            ("next_day", "bus", expected_next_day_model),  
            ("next_day", "zone", expected_next_day_model),  
            ("next_month", "bus", expected_next_month_model),  
            ("next_month", "zone", expected_next_month_model),  
            ("next_day", "bus", "baseline_previous_week"),  
            ("next_day", "zone", "baseline_previous_week"),  
            ("next_month", "bus", "baseline_previous_year"),  
            ("next_month", "zone", "baseline_previous_year"),  
            ("next_day", "bus", "baseline_bus_hour_weekday_avg"),  
            ("next_day", "zone", "baseline_bus_hour_weekday_avg"),  
            ("next_month", "bus", "baseline_bus_hour_weekday_avg"),  
            ("next_month", "zone", "baseline_bus_hour_weekday_avg"),  
        }  
        allowed_models = {  
            expected_next_day_model,  
            expected_next_month_model,  
            "baseline_previous_week",  
            "baseline_previous_year",  
            "baseline_bus_hour_weekday_avg",  
        }  
        actual_metric_keys = set(zip(metrics["task"], metrics["level"], metrics["model_name"]))  
        if not required_metric_keys.issubset(actual_metric_keys):  
            return False  
        if not set(metrics["model_name"]).issubset(allowed_models):  
            return False  
    except Exception:  
        return False  
    return True  


def remove_old_outputs(paths: dict[str, Path]) -> None:  
    """Remove old deliverables when force_rebuild=True so new outputs do not mix with stale files."""
    for path in paths.values():  
        if path.exists():  
            if path.is_dir():  
                shutil.rmtree(path)  
            else:  
                path.unlink()  


def add_grouped_series(left: pd.Series | None, right: pd.Series) -> pd.Series:  
    """Add two grouped Series objects while allowing the first accumulator to be None."""
    if left is None:  
        return right.astype("float64")  
    return left.add(right.astype("float64"), fill_value=0.0)  


def make_mean_lookup(sum_series: pd.Series | None, count_series: pd.Series | None, value_name: str) -> pd.DataFrame:  
    """Convert accumulated historical load sums and counts into a named mean lookup table for imputation."""
    if sum_series is None or count_series is None:  
        return pd.DataFrame(columns=[value_name])  
    mean_series = sum_series.div(count_series.where(count_series.ne(0.0))).dropna()  
    lookup = mean_series.rename(value_name).reset_index()  
    lookup[value_name] = lookup[value_name].astype("float32")  
    return lookup  


def make_bus_share_lookup(bus_sum: pd.Series | None, zone_sum: pd.Series | None) -> pd.DataFrame:  
    """Compute bus shares from historical bus and zone load totals for the Level 4 zone fallback."""
    if bus_sum is None or zone_sum is None:  
        return pd.DataFrame(columns=["bus_unique_id", "zone_name", "he", "bus_share"])  
    bus = bus_sum.rename("bus_pd_sum").reset_index()  
    zone = zone_sum.rename("zone_pd_sum").reset_index()  
    out = bus.merge(zone, on=["zone_name", "he"], how="left")  
    out["bus_share"] = (out["bus_pd_sum"] / out["zone_pd_sum"].where(out["zone_pd_sum"].ne(0.0))).fillna(0.0).astype("float32")  
    return out[["bus_unique_id", "zone_name", "he", "bus_share"]]  


def build_source_year_imputation_lookup(accumulators: dict[int, dict[str, pd.Series | None]], source_year: int) -> dict[str, pd.DataFrame]:  
    """Convert raw historical accumulators into the five-level time-aware imputation lookup tables."""
    acc = accumulators[source_year]  
    return {  
        "level1_bus_he_doy": make_mean_lookup(acc["bus_he_doy_sum"], acc["bus_he_doy_count"], "impute_l1_pd"),  # Level 1: same bus + same HE + nearby historical days。
        "level2_bus_he_dow": make_mean_lookup(acc["bus_he_dow_sum"], acc["bus_he_dow_count"], "impute_l2_pd"),  # Level 2: same bus + same HE + same dayofweek historical mean。
        "level3_bus_he": make_mean_lookup(acc["bus_he_sum"], acc["bus_he_count"], "impute_l3_pd"),  # Level 3: same bus + same HE historical mean。
        "level4_zone_he_dow": make_mean_lookup(acc["zone_he_dow_sum"], acc["zone_he_dow_count"], "impute_l4_zone_pd"),  # Level 4: same zone + same HE + same dayofweek average。
        "bus_share": make_bus_share_lookup(acc["bus_share_sum"], acc["zone_share_sum"]),  
    }  

def build_bus_pd_imputer(data_dir: Path, output_path: Path | None = None) -> BusPdImputer:  
    """Scan 2022-2025 bus parquet files to build the missingness audit and historical imputation lookup tables."""
    total_by_year: dict[int, pd.Series | None] = {year: None for year in [2022, 2023, 2024, 2025]}  
    null_by_year: dict[int, pd.Series | None] = {year: None for year in [2022, 2023, 2024, 2025]}  
    zero_by_year: dict[int, pd.Series | None] = {year: None for year in [2022, 2023, 2024, 2025]}  
    nonzero_by_year: dict[int, pd.Series | None] = {year: None for year in [2022, 2023, 2024, 2025]}  
    accumulators: dict[int, dict[str, pd.Series | None]] = {  
        source_year: {  
            "bus_he_doy_sum": None,  
            "bus_he_doy_count": None,  
            "bus_he_dow_sum": None,  
            "bus_he_dow_count": None,  
            "bus_he_sum": None,  
            "bus_he_count": None,  
            "zone_he_dow_sum": None,  
            "zone_he_dow_count": None,  
            "bus_share_sum": None,  
            "zone_share_sum": None,  
        }
        for source_year in [2024, 2025]
    }

    for year in [2022, 2023, 2024, 2025]:  
        pf = pq.ParquetFile(bus_path(data_dir, year))  
        for row_group in range(pf.num_row_groups):  
            table = pf.read_row_group(row_group, columns=["bus_unique_id", "zone_name", "he", "pd", "date"])  
            chunk = table.to_pandas()  
            pd_values = chunk["pd"]  
            is_null = pd_values.isna()  
            is_zero = pd_values.fillna(0.0).eq(0.0) & ~is_null  
            is_nonzero = pd_values.fillna(0.0).ne(0.0) & ~is_null  
            total_by_year[year] = add_grouped_series(total_by_year[year], chunk.groupby("bus_unique_id", observed=True).size())  
            null_by_year[year] = add_grouped_series(null_by_year[year], is_null.groupby(chunk["bus_unique_id"], observed=True).sum())  
            zero_by_year[year] = add_grouped_series(zero_by_year[year], is_zero.groupby(chunk["bus_unique_id"], observed=True).sum())  
            nonzero_by_year[year] = add_grouped_series(nonzero_by_year[year], is_nonzero.groupby(chunk["bus_unique_id"], observed=True).sum())  
            if year not in [2022, 2023, 2024]:  
                continue  
            non_null = chunk.loc[~is_null, ["bus_unique_id", "zone_name", "he", "pd", "date"]].copy()  
            if non_null.empty:  
                continue  
            non_null["target_date"] = pd.to_datetime(non_null["date"])  
            non_null["dayofweek"] = non_null["target_date"].dt.dayofweek.astype("int16")  
            non_null["dayofyear"] = non_null["target_date"].dt.dayofyear.astype("int16")  
            target_source_years = [2025] if year == 2024 else [2024, 2025]  
            for source_year in target_source_years:  
                acc = accumulators[source_year]  
                grouped_sum = non_null.groupby(["bus_unique_id", "he", "dayofyear"], observed=True)["pd"].sum()  
                grouped_count = non_null.groupby(["bus_unique_id", "he", "dayofyear"], observed=True)["pd"].count()  
                acc["bus_he_doy_sum"] = add_grouped_series(acc["bus_he_doy_sum"], grouped_sum)  
                acc["bus_he_doy_count"] = add_grouped_series(acc["bus_he_doy_count"], grouped_count)  
                grouped_sum = non_null.groupby(["bus_unique_id", "he", "dayofweek"], observed=True)["pd"].sum()  
                grouped_count = non_null.groupby(["bus_unique_id", "he", "dayofweek"], observed=True)["pd"].count()  
                acc["bus_he_dow_sum"] = add_grouped_series(acc["bus_he_dow_sum"], grouped_sum)  
                acc["bus_he_dow_count"] = add_grouped_series(acc["bus_he_dow_count"], grouped_count)  
                grouped_sum = non_null.groupby(["bus_unique_id", "he"], observed=True)["pd"].sum()  
                grouped_count = non_null.groupby(["bus_unique_id", "he"], observed=True)["pd"].count()  
                acc["bus_he_sum"] = add_grouped_series(acc["bus_he_sum"], grouped_sum)  
                acc["bus_he_count"] = add_grouped_series(acc["bus_he_count"], grouped_count)  
                grouped_sum = non_null.groupby(["zone_name", "he", "dayofweek"], observed=True)["pd"].sum()  
                grouped_count = non_null.groupby(["zone_name", "he", "dayofweek"], observed=True)["pd"].count()  
                acc["zone_he_dow_sum"] = add_grouped_series(acc["zone_he_dow_sum"], grouped_sum)  
                acc["zone_he_dow_count"] = add_grouped_series(acc["zone_he_dow_count"], grouped_count)  
                bus_share_sum = non_null.groupby(["bus_unique_id", "zone_name", "he"], observed=True)["pd"].sum()  
                zone_share_sum = non_null.groupby(["zone_name", "he"], observed=True)["pd"].sum()  
                acc["bus_share_sum"] = add_grouped_series(acc["bus_share_sum"], bus_share_sum)  
                acc["zone_share_sum"] = add_grouped_series(acc["zone_share_sum"], zone_share_sum)  

    partial_missing_by_year: dict[int, set[str]] = {}  
    missingness_frames = []  
    for year in [2022, 2023, 2024, 2025]:  
        stats = pd.DataFrame({"total": total_by_year[year], "null": null_by_year[year], "zero": zero_by_year[year], "nonzero": nonzero_by_year[year]}).fillna(0).astype("int64")  
        stats["year"] = year  
        stats["missing_type"] = "no_missing"  
        stats.loc[stats["null"].eq(stats["total"]), "missing_type"] = "all_missing"  
        stats.loc[stats["null"].gt(0) & stats["null"].lt(stats["total"]), "missing_type"] = "partial_missing"  
        stats["activity_type"] = "has_nonzero_load"  
        stats.loc[stats["nonzero"].eq(0) & stats["null"].eq(stats["total"]), "activity_type"] = "all_missing"  
        stats.loc[stats["nonzero"].eq(0) & stats["zero"].eq(stats["total"]), "activity_type"] = "all_recorded_zero"  
        stats.loc[stats["nonzero"].eq(0) & (stats["null"] + stats["zero"]).eq(stats["total"]) & stats["null"].gt(0) & stats["zero"].gt(0), "activity_type"] = "only_zero_or_missing"  
        stats["imputation_eligible"] = stats["missing_type"].eq("partial_missing") & stats["activity_type"].eq("has_nonzero_load")  
        partial_missing_by_year[year] = set(stats.index[stats["imputation_eligible"]].astype(str))  
        missingness_frames.append(stats.reset_index(names="bus_unique_id"))  

    missingness_by_year = pd.concat(missingness_frames, ignore_index=True)  
    if output_path is not None:  
        missingness_by_year.to_csv(output_path, index=False)  

    lookup_by_source_year = {source_year: build_source_year_imputation_lookup(accumulators, source_year) for source_year in [2024, 2025]}  
    return BusPdImputer(partial_missing_by_year=partial_missing_by_year, lookup_by_source_year=lookup_by_source_year, missingness_by_year=missingness_by_year)  


def bus_ids_by_missingness(
    imputer: BusPdImputer,
    years: list[int] | tuple[int, ...],
    missing_type: str | None = None,
    activity_type: str | None = None,
) -> set[str]:  
    """Extract bus IDs from the missingness audit for the requested years and missingness/activity filters."""
    audit = imputer.missingness_by_year  
    mask = audit["year"].isin(list(years))  
    if missing_type is not None:  
        mask &= audit["missing_type"].eq(missing_type)  
    if activity_type is not None:  
        mask &= audit["activity_type"].eq(activity_type)  
    return set(audit.loc[mask, "bus_unique_id"].astype(str))  


def historical_active_bus_set(imputer: BusPdImputer, years: list[int] | tuple[int, ...] = (2022, 2023, 2024)) -> set[str]:  
    """Return buses with at least one nonzero historical load in the selected years."""
    audit = imputer.missingness_by_year  
    mask = audit["year"].isin(list(years)) & audit["nonzero"].gt(0)  
    return set(audit.loc[mask, "bus_unique_id"].astype(str))  


def all_missing_bus_set(imputer: BusPdImputer, year: int) -> set[str]:  
    """Return buses whose pd values are all missing for the selected year."""
    return bus_ids_by_missingness(imputer, [year], missing_type="all_missing")  


def apply_partial_missing_bus_imputation(frame: pd.DataFrame, source_year: int, imputer: BusPdImputer) -> pd.DataFrame:  
    """Apply five-level time-aware historical imputation only to eligible partial-missing active buses."""
    out = frame.copy()  
    out["history_pd"] = out["pd"].astype("float32")  
    partial_buses = imputer.partial_missing_by_year.get(source_year, set())  
    lookups = imputer.lookup_by_source_year.get(source_year)  
    if not partial_buses or not lookups:  
        return out  
    fill_mask = out["history_pd"].isna() & out["bus_unique_id"].astype(str).isin(partial_buses)  
    if not bool(fill_mask.any()):  
        return out  
    helper_columns: list[str] = []  

    if "date" in out.columns and not lookups["level1_bus_he_doy"].empty:  
        out["_target_dayofyear"] = pd.to_datetime(out["date"]).dt.dayofyear.astype("int16")  
        helper_columns.append("_target_dayofyear")  
        level1_frames = []  
        for doy in sorted(out.loc[fill_mask, "_target_dayofyear"].dropna().unique()):  
            lookup = lookups["level1_bus_he_doy"]  
            distance = (lookup["dayofyear"].astype("int16") - int(doy)).abs()  
            circular_distance = pd.Series(np.minimum(distance.to_numpy(), (366 - distance).to_numpy()), index=lookup.index)  
            nearby = lookup.loc[circular_distance.le(14), ["bus_unique_id", "he", "impute_l1_pd"]]  
            if nearby.empty:  
                continue  
            nearby = nearby.groupby(["bus_unique_id", "he"], observed=True, as_index=False)["impute_l1_pd"].mean()  
            nearby["_target_dayofyear"] = int(doy)  
            level1_frames.append(nearby)  
        if level1_frames:  
            level1_lookup = pd.concat(level1_frames, ignore_index=True)  
            out = out.merge(level1_lookup, on=["bus_unique_id", "he", "_target_dayofyear"], how="left")  
            helper_columns.append("impute_l1_pd")  
            level_mask = fill_mask & out["history_pd"].isna() & out["impute_l1_pd"].notna()  
            out.loc[level_mask, "history_pd"] = out.loc[level_mask, "impute_l1_pd"].astype("float32")  

    if "date" in out.columns and not lookups["level2_bus_he_dow"].empty:  
        out["_dayofweek"] = pd.to_datetime(out["date"]).dt.dayofweek.astype("int16")  
        helper_columns.append("_dayofweek")  
        level2_lookup = lookups["level2_bus_he_dow"].rename(columns={"dayofweek": "_dayofweek"})  
        out = out.merge(level2_lookup, on=["bus_unique_id", "he", "_dayofweek"], how="left")  
        helper_columns.append("impute_l2_pd")  
        level_mask = fill_mask & out["history_pd"].isna() & out["impute_l2_pd"].notna()  
        out.loc[level_mask, "history_pd"] = out.loc[level_mask, "impute_l2_pd"].astype("float32")  

    if not lookups["level3_bus_he"].empty:  
        out = out.merge(lookups["level3_bus_he"], on=["bus_unique_id", "he"], how="left")  
        helper_columns.append("impute_l3_pd")  
        level_mask = fill_mask & out["history_pd"].isna() & out["impute_l3_pd"].notna()  
        out.loc[level_mask, "history_pd"] = out.loc[level_mask, "impute_l3_pd"].astype("float32")  

    if "date" in out.columns and "zone_name" in out.columns and not lookups["level4_zone_he_dow"].empty and not lookups["bus_share"].empty:  
        if "_dayofweek" not in out.columns:  
            out["_dayofweek"] = pd.to_datetime(out["date"]).dt.dayofweek.astype("int16")  
            helper_columns.append("_dayofweek")  
        level4_lookup = lookups["level4_zone_he_dow"].rename(columns={"dayofweek": "_dayofweek"})  
        out = out.merge(level4_lookup, on=["zone_name", "he", "_dayofweek"], how="left")  
        out = out.merge(lookups["bus_share"], on=["bus_unique_id", "zone_name", "he"], how="left")  
        helper_columns.extend(["impute_l4_zone_pd", "bus_share"])  
        out["impute_l4_pd"] = (out["impute_l4_zone_pd"].fillna(0.0) * out["bus_share"].fillna(0.0)).astype("float32")  
        helper_columns.append("impute_l4_pd")  
        level_mask = fill_mask & out["history_pd"].isna() & out["impute_l4_pd"].gt(0.0)  
        out.loc[level_mask, "history_pd"] = out.loc[level_mask, "impute_l4_pd"]  

    remaining_mask = fill_mask & out["history_pd"].isna()  
    out.loc[remaining_mask, "history_pd"] = 0.0  
    return out.drop(columns=[column for column in dict.fromkeys(helper_columns) if column in out.columns])  

def compute_2024_bus_hour_share(data_dir: Path, imputer: BusPdImputer) -> pd.DataFrame:  
    """Compute 2024 bus-hour fallback shares used when previous-week or previous-year shares are unavailable."""
    pf = pq.ParquetFile(bus_path(data_dir, 2024))  
    bus_sum = None  
    zone_sum = None  
    excluded_all_missing = all_missing_bus_set(imputer, 2024)  
    for row_group in range(pf.num_row_groups):  
        table = pf.read_row_group(row_group, columns=["bus_unique_id", "zone_name", "he", "pd"])  
        chunk = table.to_pandas()  
        chunk = apply_partial_missing_bus_imputation(chunk, 2024, imputer)  
        if excluded_all_missing:  
            chunk = chunk.loc[~chunk["bus_unique_id"].astype(str).isin(excluded_all_missing)].copy()  
        chunk["actual_pd"] = chunk["history_pd"].fillna(0.0).astype("float64")  
        grouped_bus = chunk.groupby(["bus_unique_id", "zone_name", "he"], observed=True)["actual_pd"].sum()  
        grouped_zone = chunk.groupby(["zone_name", "he"], observed=True)["actual_pd"].sum()  
        bus_sum = grouped_bus if bus_sum is None else bus_sum.add(grouped_bus, fill_value=0.0)  
        zone_sum = grouped_zone if zone_sum is None else zone_sum.add(grouped_zone, fill_value=0.0)  
    fallback = bus_sum.rename("fallback_pd_sum").reset_index()  
    zone_total = zone_sum.rename("fallback_zone_sum").reset_index()  
    fallback = fallback.merge(zone_total, on=["zone_name", "he"], how="left")  
    fallback["fallback_share"] = np.where(  
        fallback["fallback_zone_sum"].gt(0.0),  
        fallback["fallback_pd_sum"] / fallback["fallback_zone_sum"],  
        0.0,  
    ).astype("float32")  
    return fallback[["bus_unique_id", "zone_name", "he", "fallback_share"]]  


def compute_bus_hour_weekday_average_baseline(data_dir: Path, imputer: BusPdImputer) -> pd.DataFrame:  
    """Compute the bus + hour-ending + weekday historical-average baseline required by the README hints."""
    value_sum = None  
    value_count = None  
    for year in [2022, 2023, 2024]:  
        pf = pq.ParquetFile(bus_path(data_dir, year))  
        excluded_all_missing = all_missing_bus_set(imputer, year)  
        for row_group in range(pf.num_row_groups):  
            table = pf.read_row_group(row_group, columns=["bus_unique_id", "zone_name", "he", "pd", "date"])  
            chunk = table.to_pandas()  
            chunk = apply_partial_missing_bus_imputation(chunk, year, imputer)  
            if excluded_all_missing:  
                chunk = chunk.loc[~chunk["bus_unique_id"].astype(str).isin(excluded_all_missing)].copy()  
            chunk["dayofweek"] = pd.to_datetime(chunk["date"]).dt.dayofweek.astype("int16")  
            chunk["baseline_bus_hour_weekday_avg"] = chunk["history_pd"].fillna(0.0).astype("float64")  
            group_keys = ["bus_unique_id", "zone_name", "he", "dayofweek"]  
            grouped_sum = chunk.groupby(group_keys, observed=True)["baseline_bus_hour_weekday_avg"].sum()  
            grouped_count = chunk.groupby(group_keys, observed=True)["baseline_bus_hour_weekday_avg"].count()  
            value_sum = grouped_sum if value_sum is None else value_sum.add(grouped_sum, fill_value=0.0)  
            value_count = grouped_count if value_count is None else value_count.add(grouped_count, fill_value=0.0)  
    if value_sum is None or value_count is None:  
        return pd.DataFrame(columns=["bus_unique_id", "zone_name", "he", "dayofweek", "baseline_bus_hour_weekday_avg"])  
    baseline = (value_sum / value_count.where(value_count.ne(0.0))).fillna(0.0).rename("baseline_bus_hour_weekday_avg").reset_index()  
    baseline["baseline_bus_hour_weekday_avg"] = baseline["baseline_bus_hour_weekday_avg"].astype("float32")  
    baseline["he"] = baseline["he"].astype("int16")  
    baseline["dayofweek"] = baseline["dayofweek"].astype("int16")  
    return baseline[["bus_unique_id", "zone_name", "he", "dayofweek", "baseline_bus_hour_weekday_avg"]]  


def read_bus_day(data_dir: Path, date_value: pd.Timestamp) -> pd.DataFrame:  
    """Read one day of bus rows using only the columns required for forecasting and scoring."""
    date_str = date_value.strftime("%Y-%m-%d")  
    path = bus_path(data_dir, date_value.year)  
    return pd.read_parquet(  
        path,  
        filters=[("date", "==", date_str)],  
        columns=["bus_unique_id", "zone_name", "pd", "date", "he"],  
    )  


def source_share_for_day(  
    data_dir: Path,  
    source_date: pd.Timestamp,  
    share_column: str,  
    baseline_column: str,  
    imputer: BusPdImputer,  
) -> pd.DataFrame:  
    """Compute bus shares and source-day baselines from a historical source date.

The next-day baseline uses the previous week; the next-month baseline uses the previous year."""
    source = read_bus_day(data_dir, source_date)  
    source = apply_partial_missing_bus_imputation(source, source_date.year, imputer)  
    excluded_all_missing = all_missing_bus_set(imputer, source_date.year)  
    if excluded_all_missing:  
        source = source.loc[~source["bus_unique_id"].astype(str).isin(excluded_all_missing)].copy()  
    source[baseline_column] = source["history_pd"].fillna(0.0).astype("float32")  
    source = source.groupby(["bus_unique_id", "zone_name", "he"], observed=True, as_index=False)[baseline_column].sum()  
    zone_total = source.groupby(["zone_name", "he"], observed=True)[baseline_column].transform("sum")  
    source[share_column] = (source[baseline_column] / zone_total.where(zone_total.ne(0.0))).fillna(0.0).astype("float32")  
    return source[["bus_unique_id", "zone_name", "he", share_column, baseline_column]]  


def source_share_cache_root(output_dir: Path | None = None) -> Path:
    """Return the shared daily source-share cache directory."""
    base_dir = Path(__file__).resolve().parent if output_dir is None else Path(output_dir)
    return base_dir / SOURCE_SHARE_CACHE_NAME


def source_share_cache_path(source_date: pd.Timestamp, output_dir: Path | None = None) -> Path:
    """Return the partition-like cache path for one source date."""
    source_date = pd.Timestamp(source_date)
    return (
        source_share_cache_root(output_dir)
        / f"year={source_date.year:04d}"
        / f"month={source_date.month:02d}"
        / f"date={source_date.strftime('%Y-%m-%d')}.parquet"
    )


def build_source_share_cache_for_day(
    data_dir: Path,
    source_date: pd.Timestamp,
    imputer: BusPdImputer,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Read one raw bus day once, impute it, and cache source_pd/source_share for reuse."""
    source_date = pd.Timestamp(source_date)
    cache_path = source_share_cache_path(source_date, output_dir)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    source = read_bus_day(data_dir, source_date)
    source = apply_partial_missing_bus_imputation(source, source_date.year, imputer)
    excluded_all_missing = all_missing_bus_set(imputer, source_date.year)
    if excluded_all_missing:
        source = source.loc[~source["bus_unique_id"].astype(str).isin(excluded_all_missing)].copy()
    source["source_pd"] = source["history_pd"].fillna(0.0).astype("float32")
    grouped = source.groupby(["bus_unique_id", "zone_name", "he"], observed=True, as_index=False)["source_pd"].sum()
    zone_total = grouped.groupby(["zone_name", "he"], observed=True)["source_pd"].transform("sum")
    grouped["source_share"] = (grouped["source_pd"] / zone_total.where(zone_total.ne(0.0))).fillna(0.0).astype("float32")
    grouped.insert(0, "date", source_date.strftime("%Y-%m-%d"))
    grouped["he"] = grouped["he"].astype("int16")
    grouped = grouped[["date", "bus_unique_id", "zone_name", "he", "source_pd", "source_share"]]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.{time.time_ns()}.tmp.parquet")
    grouped.to_parquet(tmp_path, index=False, compression=FORECAST_COMPRESSION)
    if cache_path.exists():
        tmp_path.unlink(missing_ok=True)
    else:
        os.replace(tmp_path, cache_path)
    return pd.read_parquet(cache_path) if cache_path.exists() else grouped


def dynamic_share_for_dates(  
    data_dir: Path,  
    weighted_source_dates: list[tuple[pd.Timestamp, float]],  
    share_column: str,  
    baseline_column: str,  
    baseline_source_date: pd.Timestamp,  
    imputer: BusPdImputer,  
) -> pd.DataFrame:  
    """Compute dynamic weighted bus shares from multiple historical dates while preserving the single-source baseline value."""
    weighted_bus_sum = None  
    baseline_frame = None  
    baseline_key = pd.Timestamp(baseline_source_date).strftime("%Y-%m-%d")  
    for source_date, weight in weighted_source_dates:  
        source_date = pd.Timestamp(source_date)  
        source = build_source_share_cache_for_day(data_dir, source_date, imputer)
        grouped = source.set_index(["bus_unique_id", "zone_name", "he"])["source_pd"].astype("float64") * float(weight)
        weighted_bus_sum = grouped if weighted_bus_sum is None else weighted_bus_sum.add(grouped, fill_value=0.0)  
        if source_date.strftime("%Y-%m-%d") == baseline_key:  
            baseline_frame = source[["bus_unique_id", "zone_name", "he", "source_pd"]].copy()
            baseline_frame = baseline_frame.rename(columns={"source_pd": baseline_column})
    if weighted_bus_sum is None:  
        return pd.DataFrame(columns=["bus_unique_id", "zone_name", "he", share_column, baseline_column])  
    share = weighted_bus_sum.rename("_dynamic_share_pd").reset_index()  
    zone_total = share.groupby(["zone_name", "he"], observed=True)["_dynamic_share_pd"].transform("sum")  
    share[share_column] = (share["_dynamic_share_pd"] / zone_total.where(zone_total.ne(0.0))).fillna(0.0).astype("float32")  
    share = share.drop(columns=["_dynamic_share_pd"])  
    if baseline_frame is None:  
        baseline_frame = pd.DataFrame(columns=["bus_unique_id", "zone_name", "he", baseline_column])  
    out = share.merge(baseline_frame, on=["bus_unique_id", "zone_name", "he"], how="left")  
    out[baseline_column] = out[baseline_column].fillna(0.0).astype("float32")  
    return out[["bus_unique_id", "zone_name", "he", share_column, baseline_column]]  


def normalize_shares(frame: pd.DataFrame, raw_share_column: str, fallback: pd.DataFrame, output_column: str) -> pd.DataFrame:  
    """Merge raw and fallback shares, then normalize within each zone/hour group while keeping all-zero groups at zero."""
    out = frame.merge(fallback, on=["bus_unique_id", "zone_name", "he"], how="left")  
    out["share_raw"] = out[raw_share_column].fillna(out["fallback_share"]).fillna(0.0).astype("float32")  
    group_keys = ["zone_name", "he"]  
    denominator = out.groupby(group_keys, observed=True)["share_raw"].transform("sum")  
    out[f"{output_column}_zero_share_group"] = denominator.le(0.0)  
    out[output_column] = np.where(denominator.gt(0.0), out["share_raw"] / denominator, 0.0).astype("float32")  
    return out.drop(columns=["fallback_share", "share_raw"])  


def zone_prediction_lookup(zone_prediction: pd.DataFrame, date_str: str) -> pd.DataFrame:  
    """Select one date of zone-hour predictions for bus-level allocation."""
    return zone_prediction[zone_prediction["target_date_str"].eq(date_str)][  
        ["zone_name", "he", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"]  
    ].copy()  


def write_forecast_rows(  
    writer: pq.ParquetWriter,  
    frame: pd.DataFrame,  
    model_name: str,  
    prediction_column: str,  
    forecast_created_at_column: str,  
) -> None:  
    """Format one day of bus forecasts using the required README schema and append them to a parquet writer."""
    output = pd.DataFrame(  
        {  
            "model_name": model_name,  
            "forecast_created_at": frame[forecast_created_at_column].astype(str),  
            "target_date": frame["date"].astype(str),  
            "he": frame["he"].astype("int16"),  
            "bus_id": frame["bus_unique_id"].astype(str),  
            "zone_id": frame["zone_name"].astype(str),  
            "predict_pd": np.maximum(frame[prediction_column].astype("float32"), 0.0),  
        }  
    )  
    table = pa.Table.from_pandas(output[FORECAST_COLUMNS], schema=FORECAST_SCHEMA, preserve_index=False)  
    writer.write_table(table)  


def update_metric(metrics: dict[tuple[str, str, str], MetricAccumulator], key: tuple[str, str, str], actual: np.ndarray, predicted: np.ndarray) -> None:  
    """Update the streaming metric accumulator for a task, aggregation level, and model name."""
    if key not in metrics:  
        metrics[key] = MetricAccumulator()  
    metrics[key].update(actual, predicted)  


def metrics_need_historical_average_refresh(paths: dict[str, Path], data_dir: Path) -> bool:  
    """Refresh metrics and reports when complete forecast parquet files already exist but the third baseline is missing."""
    if not paths["metrics"].exists():  
        return False  
    if not forecast_files_are_complete(paths, data_dir):  
        return False  
    try:  
        metrics = pd.read_csv(paths["metrics"])  
        actual_keys = set(zip(metrics["task"], metrics["level"], metrics["model_name"]))  
        core_keys = {  
            ("next_day", "bus", TRANSFORMER_NEXT_DAY_MODEL_NAME),  
            ("next_day", "zone", TRANSFORMER_NEXT_DAY_MODEL_NAME),  
            ("next_month", "bus", TRANSFORMER_NEXT_MONTH_MODEL_NAME),  
            ("next_month", "zone", TRANSFORMER_NEXT_MONTH_MODEL_NAME),  
            ("next_day", "bus", "baseline_previous_week"),  
            ("next_day", "zone", "baseline_previous_week"),  
            ("next_month", "bus", "baseline_previous_year"),  
            ("next_month", "zone", "baseline_previous_year"),  
        }  
        historical_keys = {  
            ("next_day", "bus", "baseline_bus_hour_weekday_avg"),  # next-day bus-level historical-average baseline。
            ("next_day", "zone", "baseline_bus_hour_weekday_avg"),  # next-day zone-level historical-average baseline。
            ("next_month", "bus", "baseline_bus_hour_weekday_avg"),  # next-month bus-level historical-average baseline。
            ("next_month", "zone", "baseline_bus_hour_weekday_avg"),  # next-month zone-level historical-average baseline。
        }  
        return core_keys.issubset(actual_keys) and not historical_keys.issubset(actual_keys)  
    except Exception:  
        return False  


def refresh_historical_average_baseline_metrics(paths: dict[str, Path], data_dir: Path) -> pd.DataFrame:  
    """Compute bus/hour/weekday historical-average baseline metrics from existing forecast parquet files."""
    imputer = build_bus_pd_imputer(data_dir, paths["missingness"])  
    historical_average_baseline = compute_bus_hour_weekday_average_baseline(data_dir, imputer)  
    historical_average_lookup = historical_average_baseline.rename(columns={"dayofweek": "_dayofweek"})  
    metrics: dict[tuple[str, str, str], MetricAccumulator] = {}  
    for day_index, target_date in enumerate(pd.date_range("2025-01-01", "2025-12-31", freq="D"), start=1):  
        target = read_bus_day(data_dir, target_date)  
        target["actual_pd"] = target["pd"].fillna(0.0).astype("float32")  
        target["_dayofweek"] = np.int16(target_date.dayofweek)  
        target = target.merge(historical_average_lookup, on=["bus_unique_id", "zone_name", "he", "_dayofweek"], how="left")  
        target["baseline_bus_hour_weekday_avg"] = target["baseline_bus_hour_weekday_avg"].fillna(0.0).astype("float32")  
        update_metric(metrics, ("next_day", "bus", "baseline_bus_hour_weekday_avg"), target["actual_pd"].to_numpy(), target["baseline_bus_hour_weekday_avg"].to_numpy())  
        update_metric(metrics, ("next_month", "bus", "baseline_bus_hour_weekday_avg"), target["actual_pd"].to_numpy(), target["baseline_bus_hour_weekday_avg"].to_numpy())  
        zone_eval = target.groupby(["date", "he", "zone_name"], observed=True).agg(  
            actual_pd=("actual_pd", "sum"),  
            historical_avg_pd=("baseline_bus_hour_weekday_avg", "sum"),  
        )  
        update_metric(metrics, ("next_day", "zone", "baseline_bus_hour_weekday_avg"), zone_eval["actual_pd"].to_numpy(), zone_eval["historical_avg_pd"].to_numpy())  
        update_metric(metrics, ("next_month", "zone", "baseline_bus_hour_weekday_avg"), zone_eval["actual_pd"].to_numpy(), zone_eval["historical_avg_pd"].to_numpy())  
        if day_index % 30 == 0 or day_index in [1, 365]:  
            print(f"refreshed historical-average baseline metrics through {target_date:%Y-%m-%d}", flush=True)  
    rows = []  
    for (task, level, model_name), accumulator in sorted(metrics.items()):  
        row = {"task": task, "level": level, "model_name": model_name}  
        row.update(accumulator.as_dict())  
        rows.append(row)  
    historical_metrics = pd.DataFrame(rows)  
    existing_metrics = pd.read_csv(paths["metrics"])  
    existing_metrics = existing_metrics.loc[~existing_metrics["model_name"].eq("baseline_bus_hour_weekday_avg")].copy()  
    metrics_df = pd.concat([existing_metrics, historical_metrics], ignore_index=True)  
    metrics_df = metrics_df.sort_values(["task", "level", "model_name"]).reset_index(drop=True)  
    metrics_df.to_csv(paths["metrics"], index=False)  
    return metrics_df  


def process_bus_forecasts(  
    data_dir: Path,  
    output_dir: Path,  
    paths: dict[str, Path],  
    next_day_zone: pd.DataFrame,  
    next_month_zone: pd.DataFrame,  
    fallback: pd.DataFrame,  
    historical_average_baseline: pd.DataFrame,  
    imputer: BusPdImputer,  
    next_day_model_name: str,  
    next_month_model_name: str,  
) -> pd.DataFrame:  
    """Stream through 2025 daily bus rows, write next-day and next-month forecast parquet files, and accumulate bus- and zone-level metrics."""
    metrics: dict[tuple[str, str, str], MetricAccumulator] = {}  
    zone_metrics: dict[tuple[str, str, str], MetricAccumulator] = {}  # Per-zone zone-level metrics for diagnosing weak zones.
    all_missing_2025 = all_missing_bus_set(imputer, 2025)  
    active_history = historical_active_bus_set(imputer, (2022, 2023, 2024))  
    historical_average_lookup = historical_average_baseline.rename(columns={"dayofweek": "_dayofweek"})  
    audit_counts = {  
        ("next_day", "all_missing_2025_bus_rows"): 0,  
        ("next_month", "all_missing_2025_bus_rows"): 0,  
        ("next_day", "no_historical_nonzero_bus_rows"): 0,  
        ("next_month", "no_historical_nonzero_bus_rows"): 0,  
        ("next_day", "zero_share_group_rows"): 0,  
        ("next_month", "zero_share_group_rows"): 0,  
    }
    next_day_writer = pq.ParquetWriter(paths["next_day"], FORECAST_SCHEMA, compression=FORECAST_COMPRESSION, use_dictionary=FORECAST_USE_DICTIONARY)  
    next_month_writer = pq.ParquetWriter(paths["next_month"], FORECAST_SCHEMA, compression=FORECAST_COMPRESSION, use_dictionary=FORECAST_USE_DICTIONARY)  
    try:  
        for day_index, target_date in enumerate(pd.date_range("2025-01-01", "2025-12-31", freq="D"), start=1):  
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
            year_source = dynamic_share_for_dates(data_dir, next_month_share_sources, "next_month_raw_share", "baseline_next_month_pd", previous_year_date, imputer)  
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
            update_zone_metric_by_zone(zone_metrics, "next_day", next_day_model_name, day_zone_eval, "model_pd")
            update_zone_metric_by_zone(zone_metrics, "next_day", "baseline_previous_week", day_zone_eval, "baseline_pd")
            update_zone_metric_by_zone(zone_metrics, "next_day", "baseline_bus_hour_weekday_avg", day_zone_eval, "historical_avg_pd")

            month_zone_eval = month_frame.groupby(["date", "he", "zone_name"], observed=True).agg(  
                actual_pd=("actual_pd", "sum"),  
                model_pd=("next_month_predict_pd", "sum"),  
                baseline_pd=("baseline_next_month_pd", "sum"),  
                historical_avg_pd=("baseline_bus_hour_weekday_avg", "sum"),  
            )  
            update_metric(metrics, ("next_month", "zone", next_month_model_name), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["model_pd"].to_numpy())  
            update_metric(metrics, ("next_month", "zone", "baseline_previous_year"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["baseline_pd"].to_numpy())  
            update_metric(metrics, ("next_month", "zone", "baseline_bus_hour_weekday_avg"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["historical_avg_pd"].to_numpy())  
            update_zone_metric_by_zone(zone_metrics, "next_month", next_month_model_name, month_zone_eval, "model_pd")
            update_zone_metric_by_zone(zone_metrics, "next_month", "baseline_previous_year", month_zone_eval, "baseline_pd")
            update_zone_metric_by_zone(zone_metrics, "next_month", "baseline_bus_hour_weekday_avg", month_zone_eval, "historical_avg_pd")

            if day_index % 30 == 0 or day_index in [1, 365]:  
                print(f"processed {day_index}/365 days through {date_str}", flush=True)  
    finally:  
        next_day_writer.close()  
        next_month_writer.close()  

    rows = []  
    for (task, level, model_name), accumulator in sorted(metrics.items()):  
        row = {"task": task, "level": level, "model_name": model_name}  
        row.update(accumulator.as_dict())  
        rows.append(row)  
    metrics_df = pd.DataFrame(rows)  
    metrics_df.to_csv(paths["metrics"], index=False)  
    zone_metric_accumulators_to_frame(zone_metrics).to_csv(paths["zone_accuracy_by_zone"], index=False)
    audit_df = pd.DataFrame(  
        [{"task": task, "audit_item": item, "row_count": count} for (task, item), count in sorted(audit_counts.items())]
    )
    audit_df.to_csv(paths["missing_bus_audit"], index=False)  
    return metrics_df  


def merge_metric_accumulators(
    target: dict[tuple[str, str, str], MetricAccumulator],
    source: dict[tuple[str, str, str], MetricAccumulator],
) -> None:
    """Merge one worker's metric totals into the full metrics dictionary."""
    for key, accumulator in source.items():
        if key not in target:
            target[key] = MetricAccumulator()
        target[key].n += accumulator.n
        target[key].abs_error_sum += accumulator.abs_error_sum
        target[key].sq_error_sum += accumulator.sq_error_sum
        target[key].actual_sum += accumulator.actual_sum


def metric_accumulators_to_frame(metrics: dict[tuple[str, str, str], MetricAccumulator]) -> pd.DataFrame:
    """Convert streaming metric totals into the assignment metrics table."""
    rows = []
    for (task, level, model_name), accumulator in sorted(metrics.items()):
        row = {"task": task, "level": level, "model_name": model_name}
        row.update(accumulator.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def update_zone_metric_by_zone(
    zone_metrics: dict[tuple[str, str, str], MetricAccumulator],
    task: str,
    model_name: str,
    zone_eval: pd.DataFrame,
    predicted_column: str,
) -> None:
    """Accumulate zone-level metrics separately for each zone_id."""
    for zone_id, group in zone_eval.groupby("zone_name", observed=True):
        key = (task, str(zone_id), model_name)
        update_metric(zone_metrics, key, group["actual_pd"].to_numpy(), group[predicted_column].to_numpy())


def zone_metric_accumulators_to_frame(zone_metrics: dict[tuple[str, str, str], MetricAccumulator]) -> pd.DataFrame:
    """Convert per-zone metric totals into task, zone_id, model_name, n, mae, rmse, wmape, sum_actual rows."""
    rows = []
    for (task, zone_id, model_name), accumulator in sorted(zone_metrics.items()):
        row = {"task": task, "zone_id": zone_id, "model_name": model_name}
        row.update(accumulator.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def write_month_bus_forecast_parts(
    month_start: pd.Timestamp,
    data_dir: Path,
    part_dir: Path,
    next_day_zone: pd.DataFrame,
    next_month_zone: pd.DataFrame,
    fallback: pd.DataFrame,
    historical_average_lookup: pd.DataFrame,
    imputer: BusPdImputer,
    all_missing_2025: set[str],
    active_history: set[str],
    next_day_model_name: str,
    next_month_model_name: str,
) -> dict[str, object]:
    """Generate one month of Method 2 bus forecasts into independent parquet part files."""
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
            year_source = dynamic_share_for_dates(data_dir, next_month_share_sources, "next_month_raw_share", "baseline_next_month_pd", previous_year_date, imputer)
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
            update_zone_metric_by_zone(zone_metrics, "next_day", next_day_model_name, day_zone_eval, "model_pd")
            update_zone_metric_by_zone(zone_metrics, "next_day", "baseline_previous_week", day_zone_eval, "baseline_pd")
            update_zone_metric_by_zone(zone_metrics, "next_day", "baseline_bus_hour_weekday_avg", day_zone_eval, "historical_avg_pd")

            month_zone_eval = month_frame.groupby(["date", "he", "zone_name"], observed=True).agg(
                actual_pd=("actual_pd", "sum"),
                model_pd=("next_month_predict_pd", "sum"),
                baseline_pd=("baseline_next_month_pd", "sum"),
                historical_avg_pd=("baseline_bus_hour_weekday_avg", "sum"),
            )
            update_metric(metrics, ("next_month", "zone", next_month_model_name), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["model_pd"].to_numpy())
            update_metric(metrics, ("next_month", "zone", "baseline_previous_year"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["baseline_pd"].to_numpy())
            update_metric(metrics, ("next_month", "zone", "baseline_bus_hour_weekday_avg"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["historical_avg_pd"].to_numpy())
            update_zone_metric_by_zone(zone_metrics, "next_month", next_month_model_name, month_zone_eval, "model_pd")
            update_zone_metric_by_zone(zone_metrics, "next_month", "baseline_previous_year", month_zone_eval, "baseline_pd")
            update_zone_metric_by_zone(zone_metrics, "next_month", "baseline_bus_hour_weekday_avg", month_zone_eval, "historical_avg_pd")
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


def write_month_combined_bus_forecast_parts(
    month_start: pd.Timestamp,
    data_dir: Path,
    part_dir: Path,
    model_specs: list[dict[str, object]],
    fallback: pd.DataFrame,
    historical_average_lookup: pd.DataFrame,
    imputer: BusPdImputer,
    all_missing_2025: set[str],
    active_history: set[str],
) -> dict[str, object]:
    """Generate one month of shared bus-share outputs for multiple zone-model specs."""
    month_start = pd.Timestamp(month_start)
    month_label = month_start.strftime("%Y_%m")
    month_end = month_start + pd.offsets.MonthEnd(0)
    metrics_by_spec: dict[str, dict[tuple[str, str, str], MetricAccumulator]] = {}
    zone_metrics_by_spec: dict[str, dict[tuple[str, str, str], MetricAccumulator]] = {}
    writers: dict[tuple[str, str], pq.ParquetWriter] = {}
    part_paths: dict[str, dict[str, Path]] = {}
    audit_counts = {
        ("next_day", "all_missing_2025_bus_rows"): 0,
        ("next_month", "all_missing_2025_bus_rows"): 0,
        ("next_day", "no_historical_nonzero_bus_rows"): 0,
        ("next_month", "no_historical_nonzero_bus_rows"): 0,
        ("next_day", "zero_share_group_rows"): 0,
        ("next_month", "zero_share_group_rows"): 0,
    }
    for spec in model_specs:
        spec_name = str(spec["name"])
        metrics_by_spec[spec_name] = {}
        zone_metrics_by_spec[spec_name] = {}
        spec_dir = part_dir / spec_name
        next_day_part = spec_dir / "next_day" / f"part_{month_label}.parquet"
        next_month_part = spec_dir / "next_month" / f"part_{month_label}.parquet"
        next_day_part.parent.mkdir(parents=True, exist_ok=True)
        next_month_part.parent.mkdir(parents=True, exist_ok=True)
        for part_path in [next_day_part, next_month_part]:
            if part_path.exists():
                part_path.unlink()
        writers[(spec_name, "next_day")] = pq.ParquetWriter(next_day_part, FORECAST_SCHEMA, compression=FORECAST_COMPRESSION, use_dictionary=FORECAST_USE_DICTIONARY)
        writers[(spec_name, "next_month")] = pq.ParquetWriter(next_month_part, FORECAST_SCHEMA, compression=FORECAST_COMPRESSION, use_dictionary=FORECAST_USE_DICTIONARY)
        part_paths[spec_name] = {"next_day_part": next_day_part, "next_month_part": next_month_part}

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
            day_base = target.merge(lag_source, on=["bus_unique_id", "zone_name", "he"], how="left")
            day_base["baseline_next_day_pd"] = day_base["baseline_next_day_pd"].fillna(0.0).astype("float32")
            day_base = normalize_shares(day_base, "next_day_raw_share", fallback, "next_day_share")
            audit_counts[("next_day", "zero_share_group_rows")] += int(day_base["next_day_share_zero_share_group"].sum())

            previous_year_date = target_date - pd.DateOffset(years=1)
            next_month_share_sources = [
                (previous_year_date, 0.40),
                (previous_year_date - pd.Timedelta(days=7), 0.20),
                (previous_year_date + pd.Timedelta(days=7), 0.20),
                (previous_year_date - pd.Timedelta(days=14), 0.10),
                (previous_year_date + pd.Timedelta(days=14), 0.10),
            ]
            year_source = dynamic_share_for_dates(data_dir, next_month_share_sources, "next_month_raw_share", "baseline_next_month_pd", previous_year_date, imputer)
            month_base = target.merge(year_source, on=["bus_unique_id", "zone_name", "he"], how="left")
            month_base["baseline_next_month_pd"] = month_base["baseline_next_month_pd"].fillna(0.0).astype("float32")
            month_base = normalize_shares(month_base, "next_month_raw_share", fallback, "next_month_share")
            audit_counts[("next_month", "zero_share_group_rows")] += int(month_base["next_month_share_zero_share_group"].sum())

            for spec in model_specs:
                spec_name = str(spec["name"])
                metrics = metrics_by_spec[spec_name]
                zone_metrics = zone_metrics_by_spec[spec_name]

                day_zone = zone_prediction_lookup(spec["next_day_zone"], date_str).rename(
                    columns={
                        "predict_zone_pd": "next_day_zone_pred",
                        "baseline_zone_pd": "next_day_zone_baseline",
                        "forecast_created_at": "next_day_created_at",
                    }
                )
                day_frame = day_base.merge(day_zone, on=["zone_name", "he"], how="left")
                day_frame["next_day_predict_pd"] = (day_frame["next_day_zone_pred"].fillna(0.0) * day_frame["next_day_share"]).astype("float32")
                write_forecast_rows(writers[(spec_name, "next_day")], day_frame, str(spec["next_day_model_name"]), "next_day_predict_pd", "next_day_created_at")

                month_zone = zone_prediction_lookup(spec["next_month_zone"], date_str).rename(
                    columns={
                        "predict_zone_pd": "next_month_zone_pred",
                        "baseline_zone_pd": "next_month_zone_baseline",
                        "forecast_created_at": "next_month_created_at",
                    }
                )
                month_frame = month_base.merge(month_zone, on=["zone_name", "he"], how="left")
                month_frame["next_month_predict_pd"] = (month_frame["next_month_zone_pred"].fillna(0.0) * month_frame["next_month_share"]).astype("float32")
                write_forecast_rows(writers[(spec_name, "next_month")], month_frame, str(spec["next_month_model_name"]), "next_month_predict_pd", "next_month_created_at")

                update_metric(metrics, ("next_day", "bus", str(spec["next_day_model_name"])), day_frame["actual_pd"].to_numpy(), day_frame["next_day_predict_pd"].to_numpy())
                update_metric(metrics, ("next_day", "bus", "baseline_previous_week"), day_frame["actual_pd"].to_numpy(), day_frame["baseline_next_day_pd"].to_numpy())
                update_metric(metrics, ("next_day", "bus", "baseline_bus_hour_weekday_avg"), day_frame["actual_pd"].to_numpy(), day_frame["baseline_bus_hour_weekday_avg"].to_numpy())
                update_metric(metrics, ("next_month", "bus", str(spec["next_month_model_name"])), month_frame["actual_pd"].to_numpy(), month_frame["next_month_predict_pd"].to_numpy())
                update_metric(metrics, ("next_month", "bus", "baseline_previous_year"), month_frame["actual_pd"].to_numpy(), month_frame["baseline_next_month_pd"].to_numpy())
                update_metric(metrics, ("next_month", "bus", "baseline_bus_hour_weekday_avg"), month_frame["actual_pd"].to_numpy(), month_frame["baseline_bus_hour_weekday_avg"].to_numpy())

                day_zone_eval = day_frame.groupby(["date", "he", "zone_name"], observed=True).agg(
                    actual_pd=("actual_pd", "sum"),
                    model_pd=("next_day_predict_pd", "sum"),
                    baseline_pd=("baseline_next_day_pd", "sum"),
                    historical_avg_pd=("baseline_bus_hour_weekday_avg", "sum"),
                )
                update_metric(metrics, ("next_day", "zone", str(spec["next_day_model_name"])), day_zone_eval["actual_pd"].to_numpy(), day_zone_eval["model_pd"].to_numpy())
                update_metric(metrics, ("next_day", "zone", "baseline_previous_week"), day_zone_eval["actual_pd"].to_numpy(), day_zone_eval["baseline_pd"].to_numpy())
                update_metric(metrics, ("next_day", "zone", "baseline_bus_hour_weekday_avg"), day_zone_eval["actual_pd"].to_numpy(), day_zone_eval["historical_avg_pd"].to_numpy())
                update_zone_metric_by_zone(zone_metrics, "next_day", str(spec["next_day_model_name"]), day_zone_eval, "model_pd")
                update_zone_metric_by_zone(zone_metrics, "next_day", "baseline_previous_week", day_zone_eval, "baseline_pd")
                update_zone_metric_by_zone(zone_metrics, "next_day", "baseline_bus_hour_weekday_avg", day_zone_eval, "historical_avg_pd")

                month_zone_eval = month_frame.groupby(["date", "he", "zone_name"], observed=True).agg(
                    actual_pd=("actual_pd", "sum"),
                    model_pd=("next_month_predict_pd", "sum"),
                    baseline_pd=("baseline_next_month_pd", "sum"),
                    historical_avg_pd=("baseline_bus_hour_weekday_avg", "sum"),
                )
                update_metric(metrics, ("next_month", "zone", str(spec["next_month_model_name"])), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["model_pd"].to_numpy())
                update_metric(metrics, ("next_month", "zone", "baseline_previous_year"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["baseline_pd"].to_numpy())
                update_metric(metrics, ("next_month", "zone", "baseline_bus_hour_weekday_avg"), month_zone_eval["actual_pd"].to_numpy(), month_zone_eval["historical_avg_pd"].to_numpy())
                update_zone_metric_by_zone(zone_metrics, "next_month", str(spec["next_month_model_name"]), month_zone_eval, "model_pd")
                update_zone_metric_by_zone(zone_metrics, "next_month", "baseline_previous_year", month_zone_eval, "baseline_pd")
                update_zone_metric_by_zone(zone_metrics, "next_month", "baseline_bus_hour_weekday_avg", month_zone_eval, "historical_avg_pd")
    finally:
        for writer in writers.values():
            writer.close()

    return {
        "month": month_label,
        "part_paths": part_paths,
        "metrics_by_spec": metrics_by_spec,
        "zone_metrics_by_spec": zone_metrics_by_spec,
        "audit_counts": audit_counts,
    }


def append_parquet_parts(part_paths: list[Path], final_path: Path) -> int:
    """Append month part files into one required-schema parquet file."""
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


def process_bus_forecasts_parallel(
    data_dir: Path,
    output_dir: Path,
    paths: dict[str, Path],
    next_day_zone: pd.DataFrame,
    next_month_zone: pd.DataFrame,
    fallback: pd.DataFrame,
    historical_average_baseline: pd.DataFrame,
    imputer: BusPdImputer,
    next_day_model_name: str,
    next_month_model_name: str,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Generate bus forecasts with optional month-level CPU parallelism."""
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
    part_dir = output_dir / "assignment2_parallel_parts"
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

    print(f"Method 2 bus forecast parallel workers={worker_limit}", flush=True)
    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = [
            executor.submit(
                write_month_bus_forecast_parts,
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
            merge_metric_accumulators(metrics, result["metrics"])
            merge_metric_accumulators(zone_metrics, result["zone_metrics"])
            for key, count in result["audit_counts"].items():
                audit_counts[key] += int(count)
            print(f"Method 2 finished forecast part {index}/12 ({result['month']})", flush=True)

    results = sorted(results, key=lambda item: item["month"])
    print("Method 2 merging next-day parquet parts", flush=True)
    append_parquet_parts([result["next_day_part"] for result in results], paths["next_day"])
    print("Method 2 merging next-month parquet parts", flush=True)
    append_parquet_parts([result["next_month_part"] for result in results], paths["next_month"])

    metrics_df = metric_accumulators_to_frame(metrics)
    metrics_df.to_csv(paths["metrics"], index=False)
    zone_metric_accumulators_to_frame(zone_metrics).to_csv(paths["zone_accuracy_by_zone"], index=False)
    audit_df = pd.DataFrame(
        [{"task": task, "audit_item": item, "row_count": count} for (task, item), count in sorted(audit_counts.items())]
    )
    audit_df.to_csv(paths["missing_bus_audit"], index=False)
    shutil.rmtree(part_dir, ignore_errors=True)
    return metrics_df



def process_combined_bus_forecasts_parallel(
    data_dir: Path,
    output_dir: Path,
    model_specs: list[dict[str, object]],
    fallback: pd.DataFrame,
    historical_average_baseline: pd.DataFrame,
    imputer: BusPdImputer,
    max_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Generate multiple Method 2 bus-level forecast sets in one shared bus-share pass."""
    worker_limit = max(1, min(int(max_workers), os.cpu_count() or 1, 12))
    part_dir = output_dir / "assignment2_combined_parallel_parts"
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_spec: dict[str, dict[tuple[str, str, str], MetricAccumulator]] = {str(spec["name"]): {} for spec in model_specs}
    zone_metrics_by_spec: dict[str, dict[tuple[str, str, str], MetricAccumulator]] = {str(spec["name"]): {} for spec in model_specs}
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
    print(f"Combined Method 2 bus forecast workers={worker_limit}", flush=True)
    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = [
            executor.submit(
                write_month_combined_bus_forecast_parts,
                month_start,
                data_dir,
                part_dir,
                model_specs,
                fallback,
                historical_average_lookup,
                imputer,
                all_missing_2025,
                active_history,
            )
            for month_start in month_starts
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            for spec_name, spec_metrics in result["metrics_by_spec"].items():
                merge_metric_accumulators(metrics_by_spec[spec_name], spec_metrics)
            for spec_name, spec_zone_metrics in result["zone_metrics_by_spec"].items():
                merge_metric_accumulators(zone_metrics_by_spec[spec_name], spec_zone_metrics)
            for key, count in result["audit_counts"].items():
                audit_counts[key] += int(count)
            print(f"Combined Method 2 finished forecast part {index}/12 ({result['month']})", flush=True)

    results = sorted(results, key=lambda item: item["month"])
    audit_df = pd.DataFrame(
        [{"task": task, "audit_item": item, "row_count": count} for (task, item), count in sorted(audit_counts.items())]
    )
    output_metrics: dict[str, pd.DataFrame] = {}
    for spec in model_specs:
        spec_name = str(spec["name"])
        spec_paths = spec["paths"]
        print(f"Combined Method 2 merging parquet parts for {spec_name}", flush=True)
        append_parquet_parts([result["part_paths"][spec_name]["next_day_part"] for result in results], spec_paths["next_day"])
        append_parquet_parts([result["part_paths"][spec_name]["next_month_part"] for result in results], spec_paths["next_month"])
        metrics_df = metric_accumulators_to_frame(metrics_by_spec[spec_name])
        metrics_df.to_csv(spec_paths["metrics"], index=False)
        zone_metric_accumulators_to_frame(zone_metrics_by_spec[spec_name]).to_csv(spec_paths["zone_accuracy_by_zone"], index=False)
        if "missing_bus_audit" in spec_paths:
            audit_df.to_csv(spec_paths["missing_bus_audit"], index=False)
        output_metrics[spec_name] = metrics_df
    shutil.rmtree(part_dir, ignore_errors=True)
    return output_metrics
def write_sample_csv(parquet_path: Path, csv_path: Path, n: int = 1000) -> None:  
    """Write a small CSV sample from the first parquet row group for quick manual schema and prediction review."""
    pf = pq.ParquetFile(parquet_path)  
    if pf.num_row_groups == 0:  
        raise ValueError(f"No row groups found in {parquet_path}")  
    sample = pf.read_row_group(0).to_pandas().head(n)  
    sample.to_csv(csv_path, index=False)  


def write_report(paths: dict[str, Path], data_dir: Path, metrics_df: pd.DataFrame, elapsed_seconds: float) -> None:  
    """Generate the summary report from metrics, data-coverage notes, runtime, model description, leakage controls, and deliverables."""
    stage_paths = method2_zone_stage_paths(paths["next_day"].parent)
    next_day_model_name = infer_cached_zone_model_name(stage_paths, "next_day", TRANSFORMER_NEXT_DAY_MODEL_NAME)
    next_month_model_name = infer_cached_zone_model_name(stage_paths, "next_month", TRANSFORMER_NEXT_MONTH_MODEL_NAME)

    def primary_metric_model(task: str, fallback_model_name: str) -> str:
        matches = metrics_df[
            metrics_df["task"].eq(task)
            & metrics_df["level"].eq("bus")
            & ~metrics_df["model_name"].astype(str).str.startswith("baseline_")
        ]["model_name"].dropna().astype(str).unique().tolist()
        return matches[0] if matches else fallback_model_name

    next_day_model_name = primary_metric_model("next_day", next_day_model_name)
    next_month_model_name = primary_metric_model("next_month", next_month_model_name)

    def metric_line(task: str, level: str, model_name: str) -> str:  
        """Select and format one metrics row from metrics_df for the summary report."""
        matches = metrics_df[  # Select the requested metrics row for the report.
            metrics_df["task"].eq(task)
            & metrics_df["level"].eq(level)
            & metrics_df["model_name"].eq(model_name)
        ]
        if matches.empty:
            available = ", ".join(sorted(metrics_df.loc[metrics_df["task"].eq(task) & metrics_df["level"].eq(level), "model_name"].astype(str).unique()))
            return f"not available for {model_name}; available={available}"
        row = matches.iloc[0]
        return f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, WMAPE={row['wmape']:.4%}"

    expected_rows = expected_2025_bus_rows(data_dir)  
    missing_bus_audit = pd.read_csv(paths["missing_bus_audit"]) if paths.get("missing_bus_audit") and paths["missing_bus_audit"].exists() else pd.DataFrame(columns=["task", "audit_item", "row_count"])  

    def audit_count(task: str, item: str) -> int:  
        match = missing_bus_audit[missing_bus_audit["task"].eq(task) & missing_bus_audit["audit_item"].eq(item)]  
        return int(match["row_count"].iloc[0]) if not match.empty else 0  

    report = [  
        "# Assignment 2 Summary Report",  
        "",  
        "## Objective",  
        "Build a reproducible bus-level load forecasting pipeline for every 2025 bus-hour under next-day and next-month forecast settings.",  
        "",  
        "## Model Used",  
        "The primary model is a GPU-trained Transformer zone forecast plus bus-share allocation pipeline. Zone load is forecast with a PyTorch TransformerEncoder trained on CUDA, separately by zone. Each feature is treated as a token so the model can learn interactions among hour, calendar, lag, and historical-average signals. Zone forecasts are then distributed to all buses using historical bus shares.",  
        "",  
        "## Baselines",  
        "- Next-day baseline: same bus and hour from the previous week.",  
        "- Next-month baseline: same bus, calendar date, and hour from the previous year.",  
        "- Historical-average baseline: average historical load by bus, hour-ending, and weekday from 2022-2024.",  
        "",  
        "## Features",  
        "- Calendar features: hour-ending, day of week, month, day of year, weekend flag, and US federal holiday flag.",  
        "- Next-day zone features: 7-day, 14-day, 28-day, and 365-day lagged zone load plus historical zone-hour averages. The model does not use D-1 actual load under the strict `forecast_created_at = D-1 00:01:00` cutoff.",  
        "- Next-month zone features: previous-year load, configured month-lag features sourced only from data available before forecast creation, and historical month/day/hour averages available at forecast creation time.",  
        "- Bus allocation features: dynamic weighted bus shares. Next-day uses D-7/D-14/D-28 historical shares; next-month uses a previous-year same-season window at -14/-7/0/+7/+14 days. Missing shares fall back to 2024 bus-hour average share.",  
        "- Historical bus `pd` imputation: only partial-missing buses with at least one nonzero historical load are imputed. The fallback order is same bus + same HE + nearby historical days, same bus + same HE + same day-of-week mean, same bus + same HE mean, same zone + same HE + same day-of-week average multiplied by historical bus share, then 0.",  
        "- All-missing bus treatment: all-missing buses are not used to estimate bus shares; if a zone/hour has no usable share at all, the bus shares remain 0 rather than being averaged across buses.",  
        "",  
        "## Leakage Control",  
        "- The next-day zone model is trained on 2022-2024 and uses rolling lag features available before `forecast_created_at`; for target day D, this excludes D-1 actual load under the strict 00:01 cutoff.",  
        "- The next-month zone model is refit for each target month using rows before that forecast creation date; month-lag features are built from the same available-history cutoff, so unavailable prior-month actuals remain missing rather than being pulled from the future.",  
        "- Historical bus imputation uses prior-year non-null observations only: 2024 source rows use 2022-2023 history, and 2025 source rows use 2022-2024 history. Target-year `pd` is never used as a feature.",  
        "- 2025 target-day bus `pd` is used only as actuals for evaluation, not as a model feature.",  
        "- Missing bus `pd` values are filled with zero for modeling and scoring, as requested, so every 2025 bus row receives a forecast and participates in metrics.",  
        "- Interpretation caveat: a bus that is consistently zero or consistently missing may be a non-load bus, but intermittent missing `pd` values may represent either true zero load or unavailable measurements.",  
        "",  
        "## Data Coverage Note",  
        input_coverage_note(data_dir),  
        "",  
        "## Metrics",  
        f"- Forecast rows per task: {expected_rows:,}",  
        f"- Next-day rows from 2025 all-missing buses: {audit_count('next_day', 'all_missing_2025_bus_rows'):,}",  
        f"- Next-month rows from 2025 all-missing buses: {audit_count('next_month', 'all_missing_2025_bus_rows'):,}",  
        f"- Next-day rows with no 2022-2024 nonzero bus history: {audit_count('next_day', 'no_historical_nonzero_bus_rows'):,}",  
        f"- Next-month rows with no 2022-2024 nonzero bus history: {audit_count('next_month', 'no_historical_nonzero_bus_rows'):,}",  
        f"- Next-day rows in zero-share zone/hour groups: {audit_count('next_day', 'zero_share_group_rows'):,}",  
        f"- Next-month rows in zero-share zone/hour groups: {audit_count('next_month', 'zero_share_group_rows'):,}",  
        f"- Next-day Transformer bus-level ({next_day_model_name}): {metric_line('next_day', 'bus', next_day_model_name)}",  
        f"- Next-day baseline bus-level: {metric_line('next_day', 'bus', 'baseline_previous_week')}",  
        f"- Next-day historical-average baseline bus-level: {metric_line('next_day', 'bus', 'baseline_bus_hour_weekday_avg')}",  
        f"- Next-day Transformer zone-level ({next_day_model_name}): {metric_line('next_day', 'zone', next_day_model_name)}",  
        f"- Next-day baseline zone-level: {metric_line('next_day', 'zone', 'baseline_previous_week')}",  
        f"- Next-day historical-average baseline zone-level: {metric_line('next_day', 'zone', 'baseline_bus_hour_weekday_avg')}",  
        f"- Next-month Transformer bus-level ({next_month_model_name}): {metric_line('next_month', 'bus', next_month_model_name)}",  
        f"- Next-month baseline bus-level: {metric_line('next_month', 'bus', 'baseline_previous_year')}",  
        f"- Next-month historical-average baseline bus-level: {metric_line('next_month', 'bus', 'baseline_bus_hour_weekday_avg')}",  
        f"- Next-month Transformer zone-level ({next_month_model_name}): {metric_line('next_month', 'zone', next_month_model_name)}",  
        f"- Next-month baseline zone-level: {metric_line('next_month', 'zone', 'baseline_previous_year')}",  
        f"- Next-month historical-average baseline zone-level: {metric_line('next_month', 'zone', 'baseline_bus_hour_weekday_avg')}",  
        "",  
        "## Where It Works Well",  
        "The pipeline is strongest for stable load buses and stable zone shapes, because previous-week and previous-year bus shares preserve local distribution patterns while the zone model captures system-level seasonality.",  
        "",  
        "## Where It Performs Poorly",  
        "The model is weaker for newly appearing buses, buses with structural changes, buses with intermittent missing measurements, and rare calendar/weather conditions because it has no weather data and must fall back to historical share averages.",  
        "",  
        "## Improvements With More Time",  
        "- Add weather forecasts and weather normals by zone.",  
        "- Train direct bus-level models for high-load buses and keep share allocation for low-load or sparse buses.",  
        "- Add explicit treatment for topology changes and bus additions/removals.",  
        "- Tune model hyperparameters with rolling-origin validation.",  
        "",  
        "## Insights",  
        "- Zone-level modeling is computationally practical for this dataset because the zone table is small while the bus table has hundreds of millions of rows.",  
        "- The Transformer is used as the primary neural model because the assignment does not require MLP and because attention over feature tokens gives a stronger explanation for feature interaction modeling.",  
        "- Bus-share allocation keeps the full bus output schema tractable and preserves exact row coverage.",  
        "- Filling missing bus `pd` with zero allows all bus rows to be scored, but intermittent missing values should not be over-interpreted as proof that a node has no load.",  
        "- Historical imputation can increase bus-level error when target actuals are still scored as zero for missing `pd`; it is mainly a robustness improvement for partial-missing history rather than a guaranteed metric improvement under the zero-actual scoring convention.",  
        "",  
        "## Conclusions",  
        "This solution produces full-year 2025 forecasts for both required horizons, compares them with simple baselines, and reports both bus-level and zone-level accuracy without using target-day future load as a feature.",  
        "",  
        "## Deliverables",  
        f"- `{paths['next_day'].name}`",  
        f"- `{paths['next_month'].name}`",  
        f"- `{paths['next_day_sample'].name}`",  
        f"- `{paths['next_month_sample'].name}`",  
        f"- `{paths['metrics'].name}`",  
        f"- `{paths['missingness'].name}`",  
        f"- `{paths['missing_bus_audit'].name}`",  
        f"- `{paths['report'].name}`",  
        f"- `{paths['ai_log'].name}`",  
        "- `assignment2_load_forecasting.ipynb`",  
        "- `assignment2_method2_zone_share_transformer.py`",  
        f"- Runtime: {elapsed_seconds / 60:.1f} minutes",  
    ]  
    paths["report"].write_text("\n".join(report), encoding="utf-8")  


def write_ai_usage_log(paths: dict[str, Path]) -> None:  
    """Write the AI usage log that discloses where Codex/AI assistance was used in the assignment."""
    log = [  
        "# Assignment 2 AI Usage Log",  
        "",  
        "## Tool Used",  
        "Codex was used to read the assignment requirements, inspect the provided data, design the forecasting workflow, implement the Python notebook/pipeline, generate outputs, and validate schemas and metrics.",  
        "",  
        "## User Instructions",  
        "- Complete Assignment 2 according to `Summer2026-main/README.md`.",  
        "- Use local data path `2026 Summer Intern-20260524T050101Z-3-001/2026 Summer Intern`.",  
        "- Use Parquet for full forecast files and small CSV files for samples.",  
        "- Include all bus rows and treat missing `pd` as zero.",  
        "- Use a GPU Transformer as the primary model rather than an MLP.",  
        "- Impute historical `pd` only for partial-missing buses that also have nonzero historical load; all-missing, all-zero, and zero-or-missing-only buses stay unfilled and fall back to 0 downstream.",  
        "",  
        "## AI-Assisted Work",  
        "- Interpreted the README requirements and suggested approach.",  
        "- Verified input parquet schemas, row counts, date ranges, and bus/zone consistency.",  
        "- Implemented a CUDA PyTorch Transformer zone model plus bus-share allocation pipeline.",  
        "- Added non-leaking 5-level time-aware historical imputation for partial-missing active bus `pd` values used in training, shares, and baselines.",  
        "- Added previous-week, previous-year, and bus-hour-weekday historical-average baselines.",  
        "- Added leakage-control checks and summary reporting.",  
        "- Rebuilt the full forecast output with Transformer model names after confirming the machine has an RTX 3080.",  
        "",  
        "## Human Review Notes",  
        "The final forecasts, metrics, and report should be reviewed by the candidate before submission. The submitted chat history can be exported from this Codex session if the recruiting process requires the full transcript.",  
    ]  
    paths["ai_log"].write_text("\n".join(log), encoding="utf-8")  


def validate_outputs(paths: dict[str, Path], data_dir: Path) -> pd.DataFrame:  
    """Validate all main deliverables, including parquet schema, row counts, model_name, nonnegative predictions, metrics, audits, report, and AI log."""
    expected_rows = expected_2025_bus_rows(data_dir)  
    checks = []  
    stage_paths = method2_zone_stage_paths(paths["next_day"].parent)
    for key in ["next_day", "next_month"]:  
        pf = pq.ParquetFile(paths[key])  
        default_model = TRANSFORMER_NEXT_DAY_MODEL_NAME if key == "next_day" else TRANSFORMER_NEXT_MONTH_MODEL_NAME  
        expected_model = infer_cached_zone_model_name(stage_paths, key, default_model)
        checks.append({"check": f"{key}_exists", "passed": paths[key].exists(), "detail": str(paths[key])})  
        checks.append({"check": f"{key}_row_count", "passed": pf.metadata.num_rows == expected_rows, "detail": f"{pf.metadata.num_rows:,} rows"})  
        checks.append({"check": f"{key}_schema", "passed": pf.schema_arrow.names == FORECAST_COLUMNS, "detail": ",".join(pf.schema_arrow.names)})  
        model_names = set(pf.read_row_group(0, columns=["model_name"]).to_pandas()["model_name"].unique())  
        checks.append({"check": f"{key}_model_name", "passed": model_names == {expected_model}, "detail": ",".join(sorted(model_names))})  
        table = pf.read_row_group(0, columns=["predict_pd"])  
        min_prediction = float(table.column("predict_pd").to_pandas().min())  
        checks.append({"check": f"{key}_non_negative_sample", "passed": min_prediction >= 0.0, "detail": f"sample min={min_prediction:.6f}"})  
    metrics = pd.read_csv(paths["metrics"])  
    checks.append({"check": "metrics_rows", "passed": len(metrics) == 12, "detail": f"{len(metrics)} rows"})  
    checks.append({"check": "zone_accuracy_by_zone_exists", "passed": paths["zone_accuracy_by_zone"].exists(), "detail": paths["zone_accuracy_by_zone"].name})  # Verify the per-zone diagnostics table exists.
    checks.append({"check": "missingness_exists", "passed": paths["missingness"].exists(), "detail": paths["missingness"].name})  
    checks.append({"check": "missing_bus_audit_exists", "passed": paths["missing_bus_audit"].exists(), "detail": paths["missing_bus_audit"].name})  
    checks.append({"check": "report_exists", "passed": paths["report"].exists(), "detail": paths["report"].name})  
    checks.append({"check": "ai_log_exists", "passed": paths["ai_log"].exists(), "detail": paths["ai_log"].name})  
    return pd.DataFrame(checks)  


def run_pipeline(force_rebuild: bool = False, forecast_workers: int = 1) -> dict[str, object]:  
    """Main entry point for Assignment 2.

By default, the function validates existing outputs. If force_rebuild=True or outputs are incomplete,
it trains the Transformer, writes forecasts and reports, and then validates the deliverables."""
    start_time = time.time()  
    project_root = find_project_root()  
    output_dir = Path(__file__).resolve().parent  
    data_dir = find_assignment2_data_dir(project_root)  
    paths = output_paths(output_dir)  
    forecast_workers = max(1, min(int(forecast_workers), os.cpu_count() or 1, 12))  

    if force_rebuild:  
        remove_old_outputs(paths)  

    if outputs_are_complete(paths, data_dir):  
        validation = validate_outputs(paths, data_dir)  
        metrics = pd.read_csv(paths["metrics"])  
        return {  
            "project_root": project_root,  
            "data_dir": data_dir,  
            "output_dir": output_dir,  
            "paths": paths,  
            "metrics": metrics,  
            "validation": validation,  
            "elapsed_seconds": time.time() - start_time,  
            "rebuilt": False,  
        }  

    if metrics_need_historical_average_refresh(paths, data_dir):  
        print("Existing forecast parquet files are complete; refreshing only historical-average baseline metrics.", flush=True)  
        metrics = refresh_historical_average_baseline_metrics(paths, data_dir)  
        write_ai_usage_log(paths)  
        elapsed_seconds = time.time() - start_time  
        write_report(paths, data_dir, metrics, elapsed_seconds)  
        validation = validate_outputs(paths, data_dir)  
        if not validation["passed"].all():  
            raise RuntimeError("Output validation failed after metrics refresh:\n" + validation.to_string(index=False))  
        return {  
            "project_root": project_root,  
            "data_dir": data_dir,  
            "output_dir": output_dir,  
            "paths": paths,  
            "metrics": metrics,  
            "validation": validation,  
            "elapsed_seconds": elapsed_seconds,  
            "rebuilt": False,  
            "refreshed_historical_average_baseline": True,  
        }  

    if not torch.cuda.is_available():  
        raise RuntimeError(  
            "Existing outputs are missing or not generated by the GPU model, and CUDA is not available in this Python environment. "  
            "Use `conda run -n wentao_2025_10_13 python output\\assignment2_load_forecasting\\assignment2_method2_zone_share_transformer.py` "  
            "after installing the required parquet dependency in that environment."  
        )  

    remove_old_outputs(paths)  
    print(f"Project root: {project_root}", flush=True)  
    print(f"Data dir: {data_dir}", flush=True)  
    print(f"Output dir: {output_dir}", flush=True)  

    zone = load_zone_data(data_dir)  
    training_log_rows: list[dict[str, object]] = []  
    print("training next-day Transformer zone models", flush=True)  
    next_day_zone = build_next_day_transformer_zone_predictions(zone, training_log_rows=training_log_rows)  
    print("training rolling next-month Transformer zone models", flush=True)  
    next_month_zone = build_next_month_transformer_zone_predictions(zone, training_log_rows=training_log_rows)  
    pd.DataFrame(training_log_rows).to_csv(paths["training_log"], index=False)  
    print("building partial-missing bus pd imputer", flush=True)  
    imputer = build_bus_pd_imputer(data_dir, paths["missingness"])  
    print("computing 2024 fallback bus shares", flush=True)  
    fallback = compute_2024_bus_hour_share(data_dir, imputer)  
    print("computing bus-hour-weekday historical average baseline", flush=True)  
    historical_average_baseline = compute_bus_hour_weekday_average_baseline(data_dir, imputer)  
    print("streaming 2025 Transformer bus forecasts", flush=True)  
    metrics = process_bus_forecasts_parallel(  
        data_dir,  
        output_dir,  
        paths,  
        next_day_zone,  
        next_month_zone,  
        fallback,  
        historical_average_baseline,  
        imputer,  
        TRANSFORMER_NEXT_DAY_MODEL_NAME,  
        TRANSFORMER_NEXT_MONTH_MODEL_NAME,  
        max_workers=forecast_workers,  
    )  
    write_sample_csv(paths["next_day"], paths["next_day_sample"])  
    write_sample_csv(paths["next_month"], paths["next_month_sample"])  
    write_ai_usage_log(paths)  
    elapsed_seconds = time.time() - start_time  
    write_report(paths, data_dir, metrics, elapsed_seconds)  
    validation = validate_outputs(paths, data_dir)  
    if not validation["passed"].all():  
        raise RuntimeError("Output validation failed:\n" + validation.to_string(index=False))  
    return {  
        "project_root": project_root,  
        "data_dir": data_dir,  
        "output_dir": output_dir,  
        "paths": paths,  
        "metrics": metrics,  
        "validation": validation,  
        "elapsed_seconds": elapsed_seconds,  
        "forecast_workers": forecast_workers,  
        "rebuilt": True,  
    }  


def method2_zone_stage_paths(output_dir: Path) -> dict[str, Path]:
    """Return cached zone-level prediction paths for staged Method 2 notebook execution."""
    return {
        "next_day_zone": output_dir / "assignment2_next_day_zone_predictions.parquet",
        "next_month_zone": output_dir / "assignment2_next_month_zone_predictions.parquet",
        "training_log": output_dir / "assignment2_training_log.csv",
    }


def method2_zone_outputs_complete(stage_paths: dict[str, Path]) -> bool:
    """Check whether staged Method 2 zone forecasts and training log are available."""
    required = [stage_paths["next_day_zone"], stage_paths["next_month_zone"], stage_paths["training_log"]]
    if any(not path.exists() for path in required):
        return False
    try:
        next_day = pd.read_parquet(stage_paths["next_day_zone"], columns=["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"])
        next_month = pd.read_parquet(stage_paths["next_month_zone"], columns=["target_date_str", "he", "zone_name", "predict_zone_pd", "baseline_zone_pd", "forecast_created_at"])
    except Exception:
        return False
    return not next_day.empty and not next_month.empty


def method2_residual_zone_stage_paths(output_dir: Path) -> dict[str, Path]:
    """Return cached residual zone-level prediction paths for Method 2B."""
    return {
        "next_day_zone": output_dir / "assignment2_method2B_residual_next_day_zone_predictions.parquet",
        "next_month_zone": output_dir / "assignment2_method2B_residual_next_month_zone_predictions.parquet",
        "training_log": output_dir / "assignment2_method2B_residual_training_log.csv",
    }


def method2_residual_zone_outputs_complete(stage_paths: dict[str, Path]) -> bool:
    """Check whether residual Method 2B zone forecasts and training log are available."""
    return method2_zone_outputs_complete(stage_paths)


def infer_cached_zone_model_name(stage_paths: dict[str, Path], task: str, default_model_name: str) -> str:
    """Infer the cached zone model name from the training log."""
    training_log_path = stage_paths.get("training_log")
    if training_log_path is None or not training_log_path.exists():
        return default_model_name
    try:
        training_log = pd.read_csv(training_log_path, usecols=["task", "model_name"])
    except Exception:
        return default_model_name
    matches = (
        training_log.loc[training_log["task"].astype(str).eq(task), "model_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return matches[-1] if matches else default_model_name


def run_method2_residual_zone_forecasting_stage(
    force_rebuild: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Train/cache Method 2B residual zone forecasts; bus-share logic remains unchanged."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = output_paths(output_dir)
    stage_paths = method2_residual_zone_stage_paths(output_dir)
    if force_rebuild:
        for path in stage_paths.values():
            path.unlink(missing_ok=True)
    if method2_residual_zone_outputs_complete(stage_paths) and not force_rebuild:
        return {
            "method_name": "method2B_zone_residual_transformer",
            "project_root": project_root,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "paths": paths,
            "stage_paths": stage_paths,
            "next_day_zone": pd.read_parquet(stage_paths["next_day_zone"]),
            "next_month_zone": pd.read_parquet(stage_paths["next_month_zone"]),
            "training_log": pd.read_csv(stage_paths["training_log"]),
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Method 2B residual zone Transformer training.")
    zone = load_zone_data(data_dir)
    training_log_rows: list[dict[str, object]] = []
    print("Method 2B: training residual next-day zone Transformer", flush=True)
    next_day_zone = build_next_day_transformer_zone_predictions_residual(
        zone,
        training_log_rows=training_log_rows,
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
    )
    print("Method 2B: training residual rolling next-month zone Transformer", flush=True)
    next_month_zone = build_next_month_transformer_zone_predictions_residual(
        zone,
        training_log_rows=training_log_rows,
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
    next_day_zone.to_parquet(stage_paths["next_day_zone"], index=False)
    next_month_zone.to_parquet(stage_paths["next_month_zone"], index=False)
    training_log = pd.DataFrame(training_log_rows)
    training_log.to_csv(stage_paths["training_log"], index=False)
    return {
        "method_name": "method2B_zone_residual_transformer",
        "project_root": project_root,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "paths": paths,
        "stage_paths": stage_paths,
        "next_day_zone": next_day_zone,
        "next_month_zone": next_month_zone,
        "training_log": training_log,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": time.time() - start_time,
    }


def run_method2_zone_forecasting_stage(
    force_rebuild: bool = False,
    epochs: int = 140,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
    learning_rate: float = 0.003,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = False,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_lr: float = 1e-4,
    next_month_lookback_months: list[int] | tuple[int, ...] | None = None,
    use_residual: bool = True,
) -> dict[str, object]:
    """Stage 3.A: train/cache no-weather Method 2 zone forecasts; residual is the main formulation."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = output_paths(output_dir)
    stage_paths = method2_zone_stage_paths(output_dir)
    if force_rebuild:
        for path in stage_paths.values():
            path.unlink(missing_ok=True)
    expected_next_day_model = TRANSFORMER_RESIDUAL_NEXT_DAY_MODEL_NAME if use_residual else TRANSFORMER_NEXT_DAY_MODEL_NAME
    expected_next_month_model = TRANSFORMER_RESIDUAL_NEXT_MONTH_MODEL_NAME if use_residual else TRANSFORMER_NEXT_MONTH_MODEL_NAME
    cache_matches_formulation = (
        method2_zone_outputs_complete(stage_paths)
        and infer_cached_zone_model_name(stage_paths, "next_day", "") == expected_next_day_model
        and infer_cached_zone_model_name(stage_paths, "next_month", "") == expected_next_month_model
    )
    if cache_matches_formulation and not force_rebuild:
        return {
            "method_name": METHOD_NAME,
            "project_root": project_root,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "paths": paths,
            "stage_paths": stage_paths,
            "next_day_zone": pd.read_parquet(stage_paths["next_day_zone"]),
            "next_month_zone": pd.read_parquet(stage_paths["next_month_zone"]),
            "training_log": pd.read_csv(stage_paths["training_log"]),
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
            "hyperparameters": {
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
                "batch_size": None,
                "use_residual": use_residual,
            },
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Method 2 zone Transformer training.")
    zone = load_zone_data(data_dir)
    training_log_rows: list[dict[str, object]] = []
    next_day_builder = build_next_day_transformer_zone_predictions_residual if use_residual else build_next_day_transformer_zone_predictions
    next_month_builder = build_next_month_transformer_zone_predictions_residual if use_residual else build_next_month_transformer_zone_predictions
    formulation = "residual" if use_residual else "direct"
    print(f"Method 2 Stage 3.A: training {formulation} next-day zone Transformer", flush=True)
    next_day_zone = next_day_builder(
        zone,
        training_log_rows=training_log_rows,
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
    )
    print(f"Method 2 Stage 3.A: training {formulation} rolling next-month zone Transformer", flush=True)
    next_month_zone = next_month_builder(
        zone,
        training_log_rows=training_log_rows,
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
    next_day_zone.to_parquet(stage_paths["next_day_zone"], index=False)
    next_month_zone.to_parquet(stage_paths["next_month_zone"], index=False)
    training_log = pd.DataFrame(training_log_rows)
    training_log.to_csv(stage_paths["training_log"], index=False)
    return {
        "method_name": METHOD_NAME,
        "project_root": project_root,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "paths": paths,
        "stage_paths": stage_paths,
        "next_day_zone": next_day_zone,
        "next_month_zone": next_month_zone,
        "training_log": training_log,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": time.time() - start_time,
        "hyperparameters": {
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
            "batch_size": None,
            "use_residual": use_residual,
        },
    }


def remove_method2_bus_outputs(paths: dict[str, Path]) -> None:
    """Remove only Method 2 bus-level forecast/report outputs while preserving cached zone forecasts."""
    for key in ["next_day", "next_month", "next_day_sample", "next_month_sample", "metrics", "zone_accuracy_by_zone", "missingness", "missing_bus_audit", "report", "ai_log"]:
        path = paths[key]
        if path.exists() and path.is_file():
            path.unlink()
        elif path.exists() and path.is_dir():
            shutil.rmtree(path)


def run_method2_bus_share_stage(force_rebuild: bool = False, forecast_workers: int = 1) -> dict[str, object]:
    """Stage 3.B: allocate cached no-weather zone forecasts to bus-level outputs and metrics."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = output_paths(output_dir)
    stage_paths = method2_zone_stage_paths(output_dir)
    forecast_workers = max(1, min(int(forecast_workers), os.cpu_count() or 1, 12))
    if force_rebuild:
        remove_method2_bus_outputs(paths)
    if outputs_are_complete(paths, data_dir) and not force_rebuild:
        validation = validate_outputs(paths, data_dir)
        metrics = pd.read_csv(paths["metrics"])
        return {
            "method_name": METHOD_NAME,
            "project_root": project_root,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "paths": paths,
            "stage_paths": stage_paths,
            "metrics": metrics,
            "validation": validation,
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
            "forecast_workers": forecast_workers,
        }
    if not method2_zone_outputs_complete(stage_paths):
        raise FileNotFoundError("Method 2 zone forecasts are missing. Run run_method2_zone_forecasting_stage(...) first.")
    next_day_zone = pd.read_parquet(stage_paths["next_day_zone"])
    next_month_zone = pd.read_parquet(stage_paths["next_month_zone"])
    next_day_model_name = infer_cached_zone_model_name(stage_paths, "next_day", TRANSFORMER_NEXT_DAY_MODEL_NAME)
    next_month_model_name = infer_cached_zone_model_name(stage_paths, "next_month", TRANSFORMER_NEXT_MONTH_MODEL_NAME)
    print("Method 2 Stage 3.B: building partial-missing bus pd imputer", flush=True)
    imputer = build_bus_pd_imputer(data_dir, paths["missingness"])
    print("Method 2 Stage 3.B: computing 2024 fallback bus shares", flush=True)
    fallback = compute_2024_bus_hour_share(data_dir, imputer)
    print("Method 2 Stage 3.B: computing bus-hour-weekday historical average baseline", flush=True)
    historical_average_baseline = compute_bus_hour_weekday_average_baseline(data_dir, imputer)
    print("Method 2 Stage 3.B: streaming 2025 bus-level forecasts", flush=True)
    metrics = process_bus_forecasts_parallel(
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
    write_ai_usage_log(paths)
    elapsed_seconds = time.time() - start_time
    write_report(paths, data_dir, metrics, elapsed_seconds)
    validation = validate_outputs(paths, data_dir)
    if not validation["passed"].all():
        raise RuntimeError("Method 2 Stage 3.B output validation failed:\n" + validation.to_string(index=False))
    return {
        "method_name": METHOD_NAME,
        "project_root": project_root,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "paths": paths,
        "stage_paths": stage_paths,
        "metrics": metrics,
        "validation": validation,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": elapsed_seconds,
        "forecast_workers": forecast_workers,
    }



def run_combined_method2_bus_share_stage(
    force_rebuild: bool = False,
    forecast_workers: int = 1,
    include_weather: bool = True,
    use_observed_target_weather: bool = False,
) -> dict[str, object]:
    """Build no-weather and optional weather Method 2 bus outputs in one shared pass."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = output_paths(output_dir)
    stage_paths = method2_zone_stage_paths(output_dir)
    forecast_workers = max(1, min(int(forecast_workers), os.cpu_count() or 1, 12))
    weather_module = None
    weather_paths_dict = None
    if include_weather:
        import assignment2_weather as weather_module
        weather_paths_dict = weather_module.weather_model_paths(output_dir)

    if force_rebuild:
        remove_method2_bus_outputs(paths)
        if include_weather and weather_paths_dict is not None:
            weather_module.remove_weather_bus_outputs(weather_paths_dict)

    method2_complete = outputs_are_complete(paths, data_dir)
    weather_complete = True
    if include_weather and weather_paths_dict is not None:
        weather_complete = weather_module.weather_outputs_complete(weather_paths_dict, data_dir)
    if method2_complete and weather_complete and not force_rebuild:
        validation = validate_outputs(paths, data_dir)
        result = {
            "method_name": "combined_method2_bus_share",
            "paths": paths,
            "metrics": pd.read_csv(paths["metrics"]),
            "validation": validation,
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
            "forecast_workers": forecast_workers,
        }
        if include_weather and weather_paths_dict is not None:
            result["weather_paths"] = weather_paths_dict
            result["weather_metrics"] = pd.read_csv(weather_paths_dict["metrics"])
            result["weather_validation"] = weather_module.run_weather_validation_stage()["validation"]
        return result

    if not method2_zone_outputs_complete(stage_paths):
        raise FileNotFoundError("Method 2 zone forecasts are missing. Run run_method2_zone_forecasting_stage(...) first.")
    model_specs: list[dict[str, object]] = [
        {
            "name": "method2",
            "paths": paths,
            "next_day_zone": pd.read_parquet(stage_paths["next_day_zone"]),
            "next_month_zone": pd.read_parquet(stage_paths["next_month_zone"]),
            "next_day_model_name": infer_cached_zone_model_name(stage_paths, "next_day", TRANSFORMER_NEXT_DAY_MODEL_NAME),
            "next_month_model_name": infer_cached_zone_model_name(stage_paths, "next_month", TRANSFORMER_NEXT_MONTH_MODEL_NAME),
        }
    ]
    if include_weather and weather_paths_dict is not None:
        if not weather_module.weather_zone_outputs_complete(weather_paths_dict):
            raise FileNotFoundError("Weather zone forecasts are missing. Run run_weather_zone_training_stage(...) first, or set include_weather=False.")
        weather_default_next_day_model_name = weather_module.WEATHER_NEXT_DAY_MODEL_NAME
        weather_default_next_month_model_name = weather_module.WEATHER_NEXT_MONTH_MODEL_NAME
        weather_next_day_model_name = infer_cached_zone_model_name(weather_paths_dict, "next_day", weather_default_next_day_model_name)
        weather_next_month_model_name = infer_cached_zone_model_name(weather_paths_dict, "next_month", weather_default_next_month_model_name)
        model_specs.append(
            {
                "name": "weather",
                "paths": weather_paths_dict,
                "next_day_zone": pd.read_parquet(weather_paths_dict["next_day_zone"]),
                "next_month_zone": pd.read_parquet(weather_paths_dict["next_month_zone"]),
                "next_day_model_name": weather_next_day_model_name,
                "next_month_model_name": weather_next_month_model_name,
            }
        )

    print("Combined Method 2 bus-share stage: building partial-missing bus pd imputer", flush=True)
    imputer = build_bus_pd_imputer(data_dir, paths["missingness"])
    if include_weather and weather_paths_dict is not None and paths["missingness"].exists():
        shutil.copyfile(paths["missingness"], weather_paths_dict["missingness"])
    print("Combined Method 2 bus-share stage: computing 2024 fallback bus shares", flush=True)
    fallback = compute_2024_bus_hour_share(data_dir, imputer)
    print("Combined Method 2 bus-share stage: computing bus-hour-weekday historical average baseline", flush=True)
    historical_average_baseline = compute_bus_hour_weekday_average_baseline(data_dir, imputer)
    print("Combined Method 2 bus-share stage: streaming shared bus-level forecasts", flush=True)
    metrics_by_spec = process_combined_bus_forecasts_parallel(
        data_dir,
        output_dir,
        model_specs,
        fallback,
        historical_average_baseline,
        imputer,
        max_workers=forecast_workers,
    )

    write_sample_csv(paths["next_day"], paths["next_day_sample"])
    write_sample_csv(paths["next_month"], paths["next_month_sample"])
    write_ai_usage_log(paths)
    elapsed_seconds = time.time() - start_time
    write_report(paths, data_dir, metrics_by_spec["method2"], elapsed_seconds)
    validation = validate_outputs(paths, data_dir)
    if not validation["passed"].all():
        raise RuntimeError("Combined Method 2 output validation failed:\n" + validation.to_string(index=False))

    result = {
        "method_name": "combined_method2_bus_share",
        "paths": paths,
        "metrics": metrics_by_spec["method2"],
        "validation": validation,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": elapsed_seconds,
        "forecast_workers": forecast_workers,
    }
    if include_weather and weather_paths_dict is not None:
        write_sample_csv(weather_paths_dict["next_day"], weather_paths_dict["next_day_sample"])
        write_sample_csv(weather_paths_dict["next_month"], weather_paths_dict["next_month_sample"])
        result["weather_paths"] = weather_paths_dict
        result["weather_metrics"] = metrics_by_spec["weather"]
        result["weather_validation"] = weather_module.run_weather_validation_stage()["validation"]
    return result
def run_method2_validation_stage() -> dict[str, object]:
    """Stage 3.C: validate already-written no-weather Method 2 outputs without training or rewriting forecasts."""
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = output_paths(output_dir)
    validation = validate_outputs(paths, data_dir)
    metrics = pd.read_csv(paths["metrics"]) if paths["metrics"].exists() else pd.DataFrame()
    return {
        "method_name": METHOD_NAME,
        "project_root": project_root,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "paths": paths,
        "stage_paths": method2_zone_stage_paths(output_dir),
        "metrics": metrics,
        "validation": validation,
        "available": bool(validation["passed"].all()),
        "rebuilt": False,
    }


METHOD_NAME = "method2_zone_forecast_plus_bus_share_transformer"  # Notebook and summary report label for Method 2.


def run_zone_share_transformer(force_rebuild: bool = False, forecast_workers: int = 1) -> dict[str, object]:  # Method 2 wrapper used by the notebook.
    """Run Method 2: zone forecast plus bus-share Transformer pipeline."""
    zone_artifacts = run_method2_zone_forecasting_stage(force_rebuild=force_rebuild, use_residual=True)
    artifacts = run_method2_bus_share_stage(force_rebuild=force_rebuild, forecast_workers=forecast_workers)
    artifacts["method_name"] = METHOD_NAME
    artifacts["zone_artifacts"] = zone_artifacts
    return artifacts


if __name__ == "__main__":  # Allow this method file to run directly from the command line.
    artifacts = run_zone_share_transformer(force_rebuild=False)  # Reuse existing outputs when they are complete.
    print(artifacts["validation"].to_string(index=False))  # Print validation checks for manual review.
    print(artifacts["metrics"].to_string(index=False))  # Print metric rows for manual review.
