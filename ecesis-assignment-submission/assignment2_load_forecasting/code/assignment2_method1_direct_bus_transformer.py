from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import math
import json
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar

from assignment2_method2_zone_share_transformer import (
    FORECAST_COLUMNS,
    FORECAST_SCHEMA,
    MetricAccumulator,
    all_missing_bus_set,
    apply_partial_missing_bus_imputation,
    build_bus_pd_imputer,
    bus_path,
    expected_2025_bus_rows,
    find_assignment2_data_dir,
    find_project_root,
    historical_active_bus_set,
    update_metric,
    write_sample_csv,
)


METHOD_NAME = "method1_direct_bus_transformer"
DIRECT_NEXT_DAY_MODEL_NAME = "ecesis_method1_direct_bus_transformer_finetuned_next_day"
DIRECT_NEXT_MONTH_MODEL_NAME = "ecesis_method1_direct_bus_transformer_finetuned_next_month"
DEFAULT_METHOD1_TRAINING_YEARS = [2022, 2023, 2024]
DEFAULT_METHOD1_ROWS_PER_DATE = 20000
DEFAULT_METHOD1_BATCH_SIZE = 32768
DEFAULT_BUS_HASH_BUCKETS = 262_144
DIRECT_TARGET_MODE = "log1p_residual_vs_baseline"
DIRECT_SAMPLE_WEIGHT_MODE = "nonzero_and_large_load_weighted_mse"
BASE_FEATURE_COLUMNS = [
    "he_sin",
    "he_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "log_baseline_pd",
    "log_profile_pd",
]
MISSING_INDICATOR_COLUMNS = [
    "baseline_missing",
    "profile_missing",
]


def direct_feature_columns(use_missing_indicators: bool = True) -> list[str]:
    """Return direct-bus numeric feature columns, optionally including missing indicators."""
    return BASE_FEATURE_COLUMNS + (MISSING_INDICATOR_COLUMNS if use_missing_indicators else [])


@dataclass
class DirectBusTransformerBundle:
    """Container for the trained direct-bus Transformer and preprocessing metadata."""

    model: "DirectBusFeatureTransformer"
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_columns: list[str]
    zone_to_idx: dict[str, int]
    enabled_zone_indices: set[int]
    bus_hash_buckets: int


class DirectBusFeatureTransformer(torch.nn.Module):
    """Direct bus Transformer with zone and hashed-bus embeddings plus zone-specific heads."""

    def __init__(
        self,
        n_features: int,
        n_zones: int,
        n_bus_hash_buckets: int = DEFAULT_BUS_HASH_BUCKETS,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 96,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.model_config = {
            "n_features": int(n_features),
            "n_zones": int(n_zones),
            "n_bus_hash_buckets": int(n_bus_hash_buckets),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
            "target_mode": DIRECT_TARGET_MODE,
            "sample_weight_mode": DIRECT_SAMPLE_WEIGHT_MODE,
        }
        self.value_projection = torch.nn.Linear(1, d_model)
        self.feature_embedding = torch.nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.zone_embedding = torch.nn.Embedding(n_zones, d_model)
        self.bus_embedding = torch.nn.Embedding(n_bus_hash_buckets, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.shared_trunk = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.global_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, 1))
        self.zone_heads = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, 1)) for _ in range(n_zones)]
        )
        self.enabled_zone_indices: set[int] = set()

    def representation(self, x_numeric: torch.Tensor, zone_idx: torch.Tensor, bus_idx: torch.Tensor) -> torch.Tensor:
        feature_tokens = self.value_projection(x_numeric.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
        zone_token = self.zone_embedding(zone_idx).unsqueeze(1)
        bus_token = self.bus_embedding(bus_idx).unsqueeze(1)
        encoded = self.shared_trunk(torch.cat([zone_token, bus_token, feature_tokens], dim=1))
        return encoded.mean(dim=1)

    def forward(self, x_numeric: torch.Tensor, zone_idx: torch.Tensor, bus_idx: torch.Tensor, use_zone_heads: bool) -> torch.Tensor:
        hidden = self.representation(x_numeric, zone_idx, bus_idx)
        output = self.global_head(hidden)
        if not use_zone_heads or not self.enabled_zone_indices:
            return output
        for zone_value in torch.unique(zone_idx).tolist():
            zone_int = int(zone_value)
            if zone_int not in self.enabled_zone_indices:
                continue
            mask = zone_idx.eq(zone_int)
            output[mask] = self.zone_heads[zone_int](hidden[mask])
        return output


def direct_transformer_paths(output_dir: Path | None = None) -> dict[str, Path]:
    """Return Method 1 output paths."""
    output_dir = output_dir or Path(__file__).resolve().parent
    return {
        "next_day": output_dir / "assignment2_method1_direct_bus_next_day_forecast_2025.parquet",
        "next_month": output_dir / "assignment2_method1_direct_bus_next_month_forecast_2025.parquet",
        "next_day_sample": output_dir / "assignment2_method1_direct_bus_next_day_forecast_sample.csv",
        "next_month_sample": output_dir / "assignment2_method1_direct_bus_next_month_forecast_sample.csv",
        "metrics": output_dir / "assignment2_method1_direct_bus_metrics.csv",
        "training_log": output_dir / "assignment2_method1_direct_bus_training_log.csv",
        "missing_bus_audit": output_dir / "assignment2_method1_direct_bus_missing_bus_audit.csv",
    }


def direct_stage_paths(output_dir: Path | None = None) -> dict[str, Path]:
    """Return staged Method 1 intermediate paths for data preparation and model checkpoints."""
    output_dir = output_dir or Path(__file__).resolve().parent
    return {
        "data_manifest": output_dir / "assignment2_method1_direct_bus_data_manifest.json",
        "imputer": output_dir / "assignment2_method1_direct_bus_imputer.pkl",
        "zone_mapping": output_dir / "assignment2_method1_direct_bus_zone_mapping.json",
        "training_profile": output_dir / "assignment2_method1_direct_bus_training_profile.parquet",
        "forecast_profile": output_dir / "assignment2_method1_direct_bus_forecast_profile.parquet",
        "next_month_forecast_profile": output_dir / "assignment2_method1_direct_bus_next_month_forecast_profile.parquet",
        "next_day_train": output_dir / "assignment2_method1_direct_bus_next_day_train.parquet",
        "next_month_train": output_dir / "assignment2_method1_direct_bus_next_month_train.parquet",
        "next_day_model": output_dir / "assignment2_method1_direct_bus_next_day_model.pt",
        "next_month_model": output_dir / "assignment2_method1_direct_bus_next_month_model.pt",
    }


def direct_transformer_outputs_complete(paths: dict[str, Path]) -> bool:
    """Check whether Method 1 forecast and metrics files already exist."""
    required = [paths["next_day"], paths["next_month"], paths["metrics"], paths["missing_bus_audit"], paths["training_log"]]
    if any(not path.exists() for path in required):
        return False
    try:
        training_log = pd.read_csv(paths["training_log"])
        return (
            pq.ParquetFile(paths["next_day"]).schema_arrow.names == FORECAST_COLUMNS
            and pq.ParquetFile(paths["next_month"]).schema_arrow.names == FORECAST_COLUMNS
            and "loss_units" in training_log.columns
            and training_log["loss_units"].astype(str).eq("weighted_log1p_residual_mse").any()
        )
    except Exception:
        return False


def direct_data_stage_complete(
    stage_paths: dict[str, Path],
    use_missing_indicators: bool,
    rows_per_date: int | None = None,
    training_years: list[int] | None = None,
    bus_hash_buckets: int | None = None,
) -> bool:
    """Check whether Method 1 prepared data cache exists and matches the requested feature setting."""
    required = [
        "data_manifest",
        "imputer",
        "zone_mapping",
        "training_profile",
        "forecast_profile",
        "next_month_forecast_profile",
        "next_day_train",
        "next_month_train",
    ]
    if any(not stage_paths[key].exists() for key in required):
        return False
    try:
        manifest = json.loads(stage_paths["data_manifest"].read_text(encoding="utf-8"))
    except Exception:
        return False
    base_match = bool(
        manifest.get("use_missing_indicators") == bool(use_missing_indicators)
        and manifest.get("target_mode") == DIRECT_TARGET_MODE
        and manifest.get("sample_weight_mode") == DIRECT_SAMPLE_WEIGHT_MODE
        and int(manifest.get("bus_hash_buckets", 0)) > 0
        and "bus_idx" in manifest.get("feature_frame_columns", [])
        and "target_residual_log" in manifest.get("feature_frame_columns", [])
        and "sample_weight" in manifest.get("feature_frame_columns", [])
    )
    if not base_match:
        return False
    if rows_per_date is not None and int(manifest.get("rows_per_date", -1)) != int(rows_per_date):
        return False
    if training_years is not None and [int(year) for year in manifest.get("training_years", [])] != [int(year) for year in training_years]:
        return False
    if bus_hash_buckets is not None and int(manifest.get("bus_hash_buckets", -1)) != int(bus_hash_buckets):
        return False
    return True


def direct_training_stage_complete(stage_paths: dict[str, Path], paths: dict[str, Path]) -> bool:
    """Check whether Method 1 model checkpoints and training log exist."""
    if not (stage_paths["next_day_model"].exists() and stage_paths["next_month_model"].exists() and paths["training_log"].exists()):
        return False
    try:
        for key in ["next_day_model", "next_month_model"]:
            checkpoint = torch.load(stage_paths[key], map_location="cpu", weights_only=False)
            config = checkpoint.get("model_config", {})
            if config.get("target_mode") != DIRECT_TARGET_MODE:
                return False
            if config.get("sample_weight_mode") != DIRECT_SAMPLE_WEIGHT_MODE:
                return False
            if int(config.get("n_bus_hash_buckets", 0)) <= 0:
                return False
    except Exception:
        return False
    return True


def load_direct_data_stage(stage_paths: dict[str, Path]) -> dict[str, object]:
    """Load Method 1 prepared data cache written by run_direct_bus_data_stage."""
    with stage_paths["imputer"].open("rb") as handle:
        imputer = pickle.load(handle)
    zone_to_idx = json.loads(stage_paths["zone_mapping"].read_text(encoding="utf-8"))
    manifest = json.loads(stage_paths["data_manifest"].read_text(encoding="utf-8"))
    return {
        "imputer": imputer,
        "zone_to_idx": {str(key): int(value) for key, value in zone_to_idx.items()},
        "feature_columns": list(manifest["feature_columns"]),
        "training_profile": pd.read_parquet(stage_paths["training_profile"]),
        "forecast_profile": pd.read_parquet(stage_paths["forecast_profile"]),
        "next_month_forecast_profile": pd.read_parquet(stage_paths["next_month_forecast_profile"]),
        "next_day_train": pd.read_parquet(stage_paths["next_day_train"]),
        "next_month_train": pd.read_parquet(stage_paths["next_month_train"]),
        "manifest": manifest,
    }


def save_direct_bundle(path: Path, bundle: DirectBusTransformerBundle) -> None:
    """Save a trained direct-bus model bundle as a torch checkpoint."""
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in bundle.model.state_dict().items()},
        "feature_mean": bundle.feature_mean,
        "feature_std": bundle.feature_std,
        "feature_columns": bundle.feature_columns,
        "zone_to_idx": bundle.zone_to_idx,
        "enabled_zone_indices": sorted(bundle.enabled_zone_indices),
        "bus_hash_buckets": int(bundle.bus_hash_buckets),
        "model_config": dict(bundle.model.model_config),
    }
    torch.save(checkpoint, path)


def load_direct_bundle(path: Path) -> DirectBusTransformerBundle:
    """Load a trained direct-bus model bundle from a torch checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    feature_columns = list(checkpoint["feature_columns"])
    zone_to_idx = {str(key): int(value) for key, value in checkpoint["zone_to_idx"].items()}
    model_config = dict(checkpoint.get("model_config", {}))
    bus_hash_buckets = int(checkpoint.get("bus_hash_buckets", model_config.get("n_bus_hash_buckets", DEFAULT_BUS_HASH_BUCKETS)))
    model = DirectBusFeatureTransformer(
        len(feature_columns),
        len(zone_to_idx),
        n_bus_hash_buckets=bus_hash_buckets,
        d_model=int(model_config.get("d_model", 32)),
        nhead=int(model_config.get("nhead", 4)),
        num_layers=int(model_config.get("num_layers", 2)),
        dim_feedforward=int(model_config.get("dim_feedforward", 96)),
        dropout=float(model_config.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.enabled_zone_indices = set(int(value) for value in checkpoint.get("enabled_zone_indices", []))
    model.eval()
    return DirectBusTransformerBundle(
        model=model,
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype="float32"),
        feature_std=np.asarray(checkpoint["feature_std"], dtype="float32"),
        feature_columns=feature_columns,
        zone_to_idx=zone_to_idx,
        enabled_zone_indices=model.enabled_zone_indices,
        bus_hash_buckets=bus_hash_buckets,
    )


def run_direct_bus_validation_stage() -> dict[str, object]:
    """Validate Method 1 direct-bus forecast files, metrics, and audit outputs without training."""
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = direct_transformer_paths(output_dir)
    expected_rows = expected_2025_bus_rows(data_dir)
    checks: list[dict[str, object]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for key, expected_model in [
        ("next_day", DIRECT_NEXT_DAY_MODEL_NAME),
        ("next_month", DIRECT_NEXT_MONTH_MODEL_NAME),
    ]:
        path = paths[key]
        if not path.exists():
            add_check(f"{key}_forecast_exists", False, f"Missing {path.name}")
            continue
        try:
            pf = pq.ParquetFile(path)
            add_check(f"{key}_schema", pf.schema_arrow.names == FORECAST_COLUMNS, ",".join(pf.schema_arrow.names))
            add_check(f"{key}_row_count", pf.metadata.num_rows == expected_rows, f"{pf.metadata.num_rows:,} rows; expected {expected_rows:,}")
            sample = pf.read_row_group(0, columns=["model_name", "predict_pd"]).to_pandas()
            model_names = set(sample["model_name"].astype(str).unique())
            add_check(f"{key}_model_name", model_names == {expected_model}, ",".join(sorted(model_names)))
            add_check(f"{key}_nonnegative_sample", bool((sample["predict_pd"] >= 0).all()), "First row group checked")
        except Exception as exc:
            add_check(f"{key}_readable", False, repr(exc))

    if paths["metrics"].exists():
        metrics = pd.read_csv(paths["metrics"])
        required_metric_keys = {
            ("next_day", "bus", DIRECT_NEXT_DAY_MODEL_NAME),
            ("next_day", "zone", DIRECT_NEXT_DAY_MODEL_NAME),
            ("next_month", "bus", DIRECT_NEXT_MONTH_MODEL_NAME),
            ("next_month", "zone", DIRECT_NEXT_MONTH_MODEL_NAME),
        }
        actual_metric_keys = set(zip(metrics["task"], metrics["level"], metrics["model_name"]))
        missing = sorted(required_metric_keys - actual_metric_keys)
        add_check("metrics_required_rows", not missing, "missing=" + repr(missing))
    else:
        metrics = pd.DataFrame()
        add_check("metrics_exists", False, f"Missing {paths['metrics'].name}")

    add_check("training_log_exists", paths["training_log"].exists(), paths["training_log"].name)
    add_check("missing_bus_audit_exists", paths["missing_bus_audit"].exists(), paths["missing_bus_audit"].name)
    add_check("sample_csv_exists", paths["next_day_sample"].exists() and paths["next_month_sample"].exists(), "next-day and next-month sample CSV files")

    validation = pd.DataFrame(checks)
    return {
        "method_name": METHOD_NAME,
        "project_root": project_root,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "paths": paths,
        "metrics": metrics,
        "validation": validation,
        "available": bool(validation["passed"].all()),
        "rebuilt": False,
    }


def read_bus_day_for_direct(data_dir: Path, date_value: pd.Timestamp) -> pd.DataFrame:
    """Read one target/source day from the bus parquet table."""
    date_value = pd.Timestamp(date_value)
    path = bus_path(data_dir, date_value.year)
    if not path.exists():
        return pd.DataFrame(columns=["bus_unique_id", "zone_name", "pd", "date", "he"])
    return pd.read_parquet(
        path,
        filters=[("date", "==", date_value.strftime("%Y-%m-%d"))],
        columns=["bus_unique_id", "zone_name", "pd", "date", "he"],
    )


def build_zone_mapping(data_dir: Path) -> dict[str, int]:
    """Create a stable zone_name to integer index mapping for zone embeddings and heads."""
    zones: set[str] = set()
    pf = pq.ParquetFile(bus_path(data_dir, 2025))
    for row_group in range(pf.num_row_groups):
        chunk = pf.read_row_group(row_group, columns=["zone_name"]).to_pandas()
        zones.update(chunk["zone_name"].dropna().astype(str).unique())
    return {zone: idx for idx, zone in enumerate(sorted(zones))}


def build_bus_profile(data_dir: Path, imputer, years: list[int]) -> pd.DataFrame:
    """Build a same-bus same-HE average profile feature from the explicitly supplied historical years."""
    profile_sum = None
    profile_count = None
    if not years:
        return pd.DataFrame(columns=["bus_unique_id", "he", "profile_pd"])
    for year in years:
        excluded_all_missing = all_missing_bus_set(imputer, year)
        pf = pq.ParquetFile(bus_path(data_dir, year))
        for row_group in range(pf.num_row_groups):
            chunk = pf.read_row_group(row_group, columns=["bus_unique_id", "he", "pd"]).to_pandas()
            chunk = apply_partial_missing_bus_imputation(chunk, year, imputer)
            if excluded_all_missing:
                chunk = chunk.loc[~chunk["bus_unique_id"].astype(str).isin(excluded_all_missing)].copy()
            chunk["profile_value"] = chunk["history_pd"].fillna(0.0).astype("float64")
            grouped_sum = chunk.groupby(["bus_unique_id", "he"], observed=True)["profile_value"].sum()
            grouped_count = chunk.groupby(["bus_unique_id", "he"], observed=True)["profile_value"].size()
            profile_sum = grouped_sum if profile_sum is None else profile_sum.add(grouped_sum, fill_value=0.0)
            profile_count = grouped_count if profile_count is None else profile_count.add(grouped_count, fill_value=0.0)
    profile = (profile_sum / profile_count.replace(0, np.nan)).fillna(0.0).rename("profile_pd").reset_index()
    profile["profile_pd"] = profile["profile_pd"].astype("float32")
    return profile


def build_training_profiles_by_target_year(data_dir: Path, imputer, training_years: list[int]) -> pd.DataFrame:
    """Build target-year profiles that only use years earlier than each training target year."""
    profiles = []
    first_year = min(training_years)
    for target_year in training_years:
        historical_years = [year for year in training_years if first_year <= year < target_year]
        profile = build_bus_profile(data_dir, imputer, historical_years)
        profile["source_year"] = int(target_year)
        profiles.append(profile)
    if not profiles:
        return pd.DataFrame(columns=["source_year", "bus_unique_id", "he", "profile_pd"])
    return pd.concat(profiles, ignore_index=True)


def build_bus_profile_before(data_dir: Path, imputer, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Build a same-bus same-HE profile using only rows strictly before a forecast creation date."""
    cutoff_date = pd.Timestamp(cutoff_date)
    profile_sum = None
    profile_count = None
    for year in range(2022, cutoff_date.year + 1):
        excluded_all_missing = all_missing_bus_set(imputer, year)
        pf = pq.ParquetFile(bus_path(data_dir, year))
        for row_group in range(pf.num_row_groups):
            chunk = pf.read_row_group(row_group, columns=["bus_unique_id", "he", "pd", "date"]).to_pandas()
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk = chunk.loc[chunk["date"].lt(cutoff_date)].copy()
            if chunk.empty:
                continue
            chunk = apply_partial_missing_bus_imputation(chunk, year, imputer)
            if excluded_all_missing:
                chunk = chunk.loc[~chunk["bus_unique_id"].astype(str).isin(excluded_all_missing)].copy()
            chunk["profile_value"] = chunk["history_pd"].fillna(0.0).astype("float64")
            grouped_sum = chunk.groupby(["bus_unique_id", "he"], observed=True)["profile_value"].sum()
            grouped_count = chunk.groupby(["bus_unique_id", "he"], observed=True)["profile_value"].size()
            profile_sum = grouped_sum if profile_sum is None else profile_sum.add(grouped_sum, fill_value=0.0)
            profile_count = grouped_count if profile_count is None else profile_count.add(grouped_count, fill_value=0.0)
    if profile_sum is None or profile_count is None:
        return pd.DataFrame(columns=["bus_unique_id", "he", "profile_pd"])
    profile = (profile_sum / profile_count.replace(0, np.nan)).fillna(0.0).rename("profile_pd").reset_index()
    profile["profile_pd"] = profile["profile_pd"].astype("float32")
    return profile


def source_baseline_for_day(data_dir: Path, source_date: pd.Timestamp, baseline_column: str, imputer) -> pd.DataFrame:
    """Create a direct-bus historical baseline from previous week or previous year source day."""
    source_date = pd.Timestamp(source_date)
    if not bus_path(data_dir, source_date.year).exists():
        return pd.DataFrame(columns=["bus_unique_id", "zone_name", "he", baseline_column])
    source = read_bus_day_for_direct(data_dir, source_date)
    if source.empty:
        return pd.DataFrame(columns=["bus_unique_id", "zone_name", "he", baseline_column])
    source = apply_partial_missing_bus_imputation(source, source_date.year, imputer)
    excluded_all_missing = all_missing_bus_set(imputer, source_date.year)
    if excluded_all_missing:
        source = source.loc[~source["bus_unique_id"].astype(str).isin(excluded_all_missing)].copy()
    source[baseline_column] = source["history_pd"].fillna(0.0).astype("float32")
    return source.groupby(["bus_unique_id", "zone_name", "he"], observed=True, as_index=False)[baseline_column].sum()


def hashed_bus_index(bus_ids: pd.Series, n_buckets: int = DEFAULT_BUS_HASH_BUCKETS) -> np.ndarray:
    """Map bus ids to stable hashed embedding buckets."""
    n_buckets = max(1, int(n_buckets))
    hashed = pd.util.hash_pandas_object(bus_ids.astype(str), index=False).to_numpy(dtype="uint64")
    return (hashed % np.uint64(n_buckets)).astype("int64")


def direct_sample_weights(actual_pd: pd.Series | np.ndarray) -> np.ndarray:
    """Give nonzero and larger-load bus rows more influence in residual training."""
    actual = np.asarray(actual_pd, dtype="float32")
    actual = np.maximum(actual, 0.0)
    log_actual = np.log1p(actual)
    positive = log_actual[actual > 0.0]
    scale = float(np.percentile(positive, 95)) if positive.size else 1.0
    scale = max(scale, 1.0)
    weights = 0.5 + (actual > 0.0).astype("float32") + 2.0 * np.clip(log_actual / scale, 0.0, 3.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    return weights.astype("float32")


def add_direct_features(
    frame: pd.DataFrame,
    target_date: pd.Timestamp,
    zone_to_idx: dict[str, int],
    baseline_column: str,
    bus_hash_buckets: int = DEFAULT_BUS_HASH_BUCKETS,
) -> pd.DataFrame:
    """Add direct-bus Transformer numeric features and zone index."""
    out = frame.copy()
    target_date = pd.Timestamp(target_date)
    he_angle = 2.0 * math.pi * (out["he"].astype("float32") - 1.0) / 24.0
    dow_angle = 2.0 * math.pi * float(target_date.dayofweek) / 7.0
    month_angle = 2.0 * math.pi * float(target_date.month - 1) / 12.0
    holidays = USFederalHolidayCalendar().holidays(start=target_date, end=target_date)
    out["he_sin"] = np.sin(he_angle).astype("float32")
    out["he_cos"] = np.cos(he_angle).astype("float32")
    out["dow_sin"] = np.float32(math.sin(dow_angle))
    out["dow_cos"] = np.float32(math.cos(dow_angle))
    out["month_sin"] = np.float32(math.sin(month_angle))
    out["month_cos"] = np.float32(math.cos(month_angle))
    out["is_weekend"] = np.float32(target_date.dayofweek in [5, 6])
    out["is_holiday"] = np.float32(target_date.normalize() in pd.to_datetime(holidays))
    out["baseline_missing"] = out[baseline_column].isna().astype("float32")
    out["profile_missing"] = out["profile_pd"].isna().astype("float32")
    out[baseline_column] = out[baseline_column].fillna(0.0).astype("float32")
    out["profile_pd"] = out["profile_pd"].fillna(0.0).astype("float32")
    out["log_baseline_pd"] = np.log1p(np.maximum(out[baseline_column].to_numpy(dtype="float32"), 0.0))
    out["log_profile_pd"] = np.log1p(np.maximum(out["profile_pd"].to_numpy(dtype="float32"), 0.0))
    out["zone_idx"] = out["zone_name"].astype(str).map(zone_to_idx).fillna(0).astype("int64")
    out["bus_idx"] = hashed_bus_index(out["bus_unique_id"], bus_hash_buckets)
    return out


def make_training_frame(
    task: str,
    data_dir: Path,
    imputer,
    profile: pd.DataFrame,
    zone_to_idx: dict[str, int],
    feature_columns: list[str],
    rows_per_date: int = DEFAULT_METHOD1_ROWS_PER_DATE,
    training_years: list[int] | None = None,
    bus_hash_buckets: int = DEFAULT_BUS_HASH_BUCKETS,
) -> pd.DataFrame:
    """Sample all configured 2022-2024 training dates without using 2025 targets."""
    training_years = [int(year) for year in (training_years or DEFAULT_METHOD1_TRAINING_YEARS)]
    start_date = pd.Timestamp(min(training_years), 1, 1)
    end_date = pd.Timestamp(max(training_years), 12, 31)
    if task == "next_day":
        dates = pd.date_range(start_date, end_date, freq="D")
        source_date = lambda date: date - pd.Timedelta(days=7)
        baseline_column = "baseline_next_day_pd"
    else:
        dates = pd.date_range(start_date, end_date, freq="D")
        source_date = lambda date: date - pd.DateOffset(years=1)
        baseline_column = "baseline_next_month_pd"
    samples = []
    for date_index, target_date in enumerate(dates, start=1):
        target = read_bus_day_for_direct(data_dir, target_date)
        if target.empty:
            continue
        target = apply_partial_missing_bus_imputation(target, target_date.year, imputer)
        training_all_missing = all_missing_bus_set(imputer, target_date.year)
        if training_all_missing:
            target = target.loc[~target["bus_unique_id"].astype(str).isin(training_all_missing)].copy()
        target["actual_pd"] = target["history_pd"].fillna(0.0).astype("float32")
        source = source_baseline_for_day(data_dir, source_date(target_date), baseline_column, imputer)
        frame = target.merge(source, on=["bus_unique_id", "zone_name", "he"], how="left")
        if "source_year" in profile.columns:
            year_profile = profile.loc[profile["source_year"].eq(int(target_date.year)), ["bus_unique_id", "he", "profile_pd"]]
        else:
            year_profile = profile
        frame = frame.merge(year_profile, on=["bus_unique_id", "he"], how="left")
        frame = add_direct_features(frame, target_date, zone_to_idx, baseline_column, bus_hash_buckets=bus_hash_buckets)
        frame["target_residual_log"] = (
            np.log1p(np.maximum(frame["actual_pd"].to_numpy(dtype="float32"), 0.0)) - frame["log_baseline_pd"].to_numpy(dtype="float32")
        ).astype("float32")
        frame["sample_weight"] = direct_sample_weights(frame["actual_pd"])
        if len(frame) > rows_per_date:
            frame = frame.sample(rows_per_date, random_state=int(target_date.strftime("%Y%m%d")))
        frame["_target_date"] = target_date
        samples.append(frame[feature_columns + ["zone_idx", "bus_idx", "actual_pd", "target_residual_log", "sample_weight", "_target_date"]])
        if date_index % 50 == 0:
            print(f"Method 1 {task} sampled through {target_date:%Y-%m-%d}; rows={sum(len(sample) for sample in samples):,}", flush=True)
    return pd.concat(samples, ignore_index=True)


def write_direct_forecast(
    writer: pq.ParquetWriter,
    frame: pd.DataFrame,
    model_name: str,
    forecast_created_at: str,
    predict_pd: np.ndarray,
) -> None:
    """Write one day of direct-bus forecasts using the required README schema."""
    out = pd.DataFrame(
        {
            "model_name": model_name,
            "forecast_created_at": forecast_created_at,
            "target_date": frame["date"].astype(str),
            "he": frame["he"].astype("int16"),
            "bus_id": frame["bus_unique_id"].astype(str),
            "zone_id": frame["zone_name"].astype(str),
            "predict_pd": predict_pd.astype("float32"),
        }
    )
    writer.write_table(pa.Table.from_pandas(out[FORECAST_COLUMNS], schema=FORECAST_SCHEMA, preserve_index=False))


def train_direct_bus_transformer(
    train_frame: pd.DataFrame,
    n_zones: int,
    feature_columns: list[str],
    min_zone_samples: int = 2000,
    training_log_rows: list[dict[str, object]] | None = None,
    task: str = "unknown",
    shared_max_epochs: int = 20,
    head_max_epochs: int = 10,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 1e-4,
    shared_learning_rate: float = 0.003,
    head_learning_rate: float = 0.002,
    shared_weight_decay: float = 1e-4,
    head_weight_decay: float = 1e-4,
    batch_size: int = DEFAULT_METHOD1_BATCH_SIZE,
    bus_hash_buckets: int = DEFAULT_BUS_HASH_BUCKETS,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 96,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 3,
    lr_scheduler_min_lr: float = 1e-5,
) -> DirectBusTransformerBundle:
    """Train shared global Transformer first, then freeze the trunk and fine-tune zone-specific heads."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to train Method 1 direct-bus Transformer.")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    device = torch.device("cuda")
    if "_target_date" in train_frame.columns:
        ordered_frame = train_frame.sort_values("_target_date").reset_index(drop=True)
    else:
        ordered_frame = train_frame.sample(frac=1.0, random_state=2026).reset_index(drop=True)
    use_validation = bool(0.0 < validation_fraction < 0.5 and len(ordered_frame) >= 1000)
    if use_validation:
        split_index = int(len(ordered_frame) * (1.0 - validation_fraction))
        split_index = min(max(split_index, 1), len(ordered_frame) - 1)
        fit_frame = ordered_frame.iloc[:split_index].copy()
        validation_frame = ordered_frame.iloc[split_index:].copy()
    else:
        fit_frame = ordered_frame.copy()
        validation_frame = pd.DataFrame(columns=ordered_frame.columns)
    x_fit_np = fit_frame[feature_columns].to_numpy(dtype="float32")
    feature_mean = x_fit_np.mean(axis=0).astype("float32")
    feature_std = np.where(x_fit_np.std(axis=0) < 1e-6, 1.0, x_fit_np.std(axis=0)).astype("float32")
    x_fit_tensor = torch.tensor((x_fit_np - feature_mean) / feature_std, dtype=torch.float32, device=device)
    zone_fit_tensor = torch.tensor(fit_frame["zone_idx"].to_numpy(dtype="int64"), dtype=torch.long, device=device)
    bus_fit_tensor = torch.tensor(fit_frame["bus_idx"].to_numpy(dtype="int64"), dtype=torch.long, device=device)
    y_fit_np = (
        fit_frame["target_residual_log"].to_numpy(dtype="float32")
        if "target_residual_log" in fit_frame.columns
        else np.log1p(fit_frame["actual_pd"].to_numpy(dtype="float32")) - fit_frame["log_baseline_pd"].to_numpy(dtype="float32")
    )
    y_fit_tensor = torch.tensor(y_fit_np, dtype=torch.float32, device=device).view(-1, 1)
    weight_fit_np = fit_frame["sample_weight"].to_numpy(dtype="float32") if "sample_weight" in fit_frame.columns else direct_sample_weights(fit_frame["actual_pd"])
    weight_fit_tensor = torch.tensor(weight_fit_np, dtype=torch.float32, device=device).view(-1, 1)
    if use_validation:
        x_val_np = validation_frame[feature_columns].to_numpy(dtype="float32")
        x_val_tensor = torch.tensor((x_val_np - feature_mean) / feature_std, dtype=torch.float32, device=device)
        zone_val_tensor = torch.tensor(validation_frame["zone_idx"].to_numpy(dtype="int64"), dtype=torch.long, device=device)
        bus_val_tensor = torch.tensor(validation_frame["bus_idx"].to_numpy(dtype="int64"), dtype=torch.long, device=device)
        y_val_np = (
            validation_frame["target_residual_log"].to_numpy(dtype="float32")
            if "target_residual_log" in validation_frame.columns
            else np.log1p(validation_frame["actual_pd"].to_numpy(dtype="float32")) - validation_frame["log_baseline_pd"].to_numpy(dtype="float32")
        )
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32, device=device).view(-1, 1)
        weight_val_np = validation_frame["sample_weight"].to_numpy(dtype="float32") if "sample_weight" in validation_frame.columns else direct_sample_weights(validation_frame["actual_pd"])
        weight_val_tensor = torch.tensor(weight_val_np, dtype=torch.float32, device=device).view(-1, 1)
    else:
        x_val_tensor = None
        zone_val_tensor = None
        bus_val_tensor = None
        y_val_tensor = None
        weight_val_tensor = None
    batch_size = max(1, int(batch_size))
    model = DirectBusFeatureTransformer(
        len(feature_columns),
        n_zones,
        n_bus_hash_buckets=bus_hash_buckets,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=shared_learning_rate, weight_decay=shared_weight_decay)
    scheduler = (
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
    model_name = DIRECT_NEXT_DAY_MODEL_NAME if task == "next_day" else DIRECT_NEXT_MONTH_MODEL_NAME

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return ((pred - target).pow(2) * weight).sum() / weight.sum().clamp_min(1e-6)

    def evaluate_loss(
        x_eval: torch.Tensor | None,
        zone_eval: torch.Tensor | None,
        bus_eval: torch.Tensor | None,
        y_eval: torch.Tensor | None,
        weight_eval: torch.Tensor | None,
        use_zone_heads: bool,
        enabled_tensor: torch.Tensor | None = None,
    ) -> tuple[float, int]:
        if x_eval is None or zone_eval is None or bus_eval is None or y_eval is None or weight_eval is None or x_eval.shape[0] == 0:
            return np.nan, 0
        model.eval()
        loss_sum = 0.0
        weight_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for start in range(0, x_eval.shape[0], batch_size):
                x_batch = x_eval[start : start + batch_size]
                zone_batch = zone_eval[start : start + batch_size]
                bus_batch = bus_eval[start : start + batch_size]
                y_batch = y_eval[start : start + batch_size]
                weight_batch = weight_eval[start : start + batch_size]
                if enabled_tensor is not None:
                    mask = (zone_batch[:, None] == enabled_tensor[None, :]).any(dim=1)
                    if not bool(mask.any()):
                        continue
                    pred = model(x_batch, zone_batch, bus_batch, use_zone_heads=use_zone_heads)[mask]
                    y_batch = y_batch[mask]
                    weight_batch = weight_batch[mask]
                else:
                    pred = model(x_batch, zone_batch, bus_batch, use_zone_heads=use_zone_heads)
                batch_loss = weighted_mse(pred, y_batch, weight_batch)
                batch_count = int(y_batch.shape[0])
                batch_weight = float(weight_batch.sum().detach().cpu())
                loss_sum += float(batch_loss.detach().cpu()) * batch_weight
                weight_sum += batch_weight
                sample_count += batch_count
        return (loss_sum / weight_sum if weight_sum else np.nan), sample_count

    best_loss = float("inf")
    best_epoch = 0
    best_train_loss = np.nan
    best_val_loss = np.nan
    best_learning_rate = shared_learning_rate
    best_state = None
    stale_epochs = 0
    for epoch_index in range(shared_max_epochs):
        order = torch.randperm(x_fit_tensor.shape[0], device=device)
        epoch_loss_sum = 0.0
        epoch_weight_sum = 0.0
        epoch_sample_count = 0
        model.train()
        for start in range(0, x_fit_tensor.shape[0], batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(
                model(x_fit_tensor[idx], zone_fit_tensor[idx], bus_fit_tensor[idx], use_zone_heads=False),
                y_fit_tensor[idx],
                weight_fit_tensor[idx],
            )
            loss.backward()
            optimizer.step()
            batch_count = int(idx.numel())
            batch_weight = float(weight_fit_tensor[idx].sum().detach().cpu())
            epoch_loss_sum += float(loss.detach().cpu()) * batch_weight
            epoch_weight_sum += batch_weight
            epoch_sample_count += batch_count
        train_loss = epoch_loss_sum / epoch_weight_sum if epoch_weight_sum else np.nan
        val_loss, val_count = evaluate_loss(x_val_tensor, zone_val_tensor, bus_val_tensor, y_val_tensor, weight_val_tensor, use_zone_heads=False)
        monitored_loss = val_loss if not np.isnan(val_loss) else train_loss
        if scheduler is not None and not np.isnan(monitored_loss):
            scheduler.step(monitored_loss)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        if monitored_loss < best_loss - early_stopping_min_delta:
            improved = True
            best_loss = monitored_loss
            best_epoch = epoch_index + 1
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_learning_rate = current_learning_rate
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            improved = False
            stale_epochs += 1
        early_stopped = bool(early_stopping_patience and stale_epochs >= early_stopping_patience)
        print(
            f"{model_name}: phase=shared_trunk "
            f"epoch={epoch_index + 1}/{shared_max_epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"monitored_loss={monitored_loss:.6f} "
            f"lr={current_learning_rate:.6g} "
            f"stale={stale_epochs} "
            f"early_stopped={early_stopped}",
            flush=True,
        )
        if training_log_rows is not None:
            training_log_rows.append(
                {
                    "method": METHOD_NAME,
                    "task": task,
                    "phase": "shared_trunk",
                    "model_name": model_name,
                    "epoch": epoch_index + 1,
                    "max_epochs": shared_max_epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "monitored_loss": monitored_loss,
                    "loss_units": "weighted_log1p_residual_mse",
                    "learning_rate": current_learning_rate,
                    "initial_learning_rate": shared_learning_rate,
                    "weight_decay": shared_weight_decay,
                    "batch_size": batch_size,
                    "bus_hash_buckets": int(bus_hash_buckets),
                    "target_mode": DIRECT_TARGET_MODE,
                    "sample_weight_mode": DIRECT_SAMPLE_WEIGHT_MODE,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                    "lr_scheduler": "ReduceLROnPlateau" if scheduler is not None else "none",
                    "lr_scheduler_factor": lr_scheduler_factor if scheduler is not None else np.nan,
                    "lr_scheduler_patience": lr_scheduler_patience if scheduler is not None else np.nan,
                    "lr_scheduler_min_lr": lr_scheduler_min_lr if scheduler is not None else np.nan,
                    "n_train": int(x_fit_tensor.shape[0]),
                    "n_validation": int(val_count),
                    "validation_used": bool(use_validation),
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
        f"{model_name}: phase=shared_trunk "
        f"best_epoch={best_epoch}/{shared_max_epochs} "
        f"best_loss={best_loss:.6f} "
        f"best_train_loss={best_train_loss:.6f} "
        f"best_val_loss={best_val_loss:.6f} "
        f"best_lr={best_learning_rate:.6g}",
        flush=True,
    )

    zone_counts = train_frame["zone_idx"].value_counts()
    enabled = set(zone_counts.index[zone_counts.ge(min_zone_samples)].astype(int).tolist())
    model.enabled_zone_indices = enabled
    model.feature_embedding.requires_grad = False
    for module in [model.value_projection, model.zone_embedding, model.bus_embedding, model.shared_trunk, model.global_head]:
        for param in module.parameters():
            param.requires_grad = False
    if enabled:
        enabled_tensor = torch.tensor(sorted(enabled), dtype=torch.long, device=device)
        head_optimizer = torch.optim.AdamW(model.zone_heads.parameters(), lr=head_learning_rate, weight_decay=head_weight_decay)
        head_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                head_optimizer,
                mode="min",
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                min_lr=lr_scheduler_min_lr,
            )
            if use_reduce_lr_on_plateau
            else None
        )
        best_head_loss = float("inf")
        best_head_epoch = 0
        best_head_train_loss = np.nan
        best_head_val_loss = np.nan
        best_head_learning_rate = head_learning_rate
        best_head_state = None
        stale_head_epochs = 0
        for epoch_index in range(head_max_epochs):
            order = torch.randperm(x_fit_tensor.shape[0], device=device)
            epoch_loss_sum = 0.0
            epoch_weight_sum = 0.0
            epoch_sample_count = 0
            model.train()
            for start in range(0, x_fit_tensor.shape[0], batch_size):
                idx = order[start : start + batch_size]
                batch_zone = zone_fit_tensor[idx]
                batch_bus = bus_fit_tensor[idx]
                enabled_mask = (batch_zone[:, None] == enabled_tensor[None, :]).any(dim=1)
                if not bool(enabled_mask.any()):
                    continue
                head_optimizer.zero_grad(set_to_none=True)
                pred = model(x_fit_tensor[idx], batch_zone, batch_bus, use_zone_heads=True)
                loss = weighted_mse(pred[enabled_mask], y_fit_tensor[idx][enabled_mask], weight_fit_tensor[idx][enabled_mask])
                loss.backward()
                head_optimizer.step()
                batch_count = int(enabled_mask.sum().detach().cpu())
                batch_weight = float(weight_fit_tensor[idx][enabled_mask].sum().detach().cpu())
                epoch_loss_sum += float(loss.detach().cpu()) * batch_weight
                epoch_weight_sum += batch_weight
                epoch_sample_count += batch_count
            train_loss = epoch_loss_sum / epoch_weight_sum if epoch_weight_sum else np.nan
            val_loss, val_count = evaluate_loss(x_val_tensor, zone_val_tensor, bus_val_tensor, y_val_tensor, weight_val_tensor, use_zone_heads=True, enabled_tensor=enabled_tensor)
            monitored_loss = val_loss if not np.isnan(val_loss) else train_loss
            if head_scheduler is not None and not np.isnan(monitored_loss):
                head_scheduler.step(monitored_loss)
            current_head_learning_rate = float(head_optimizer.param_groups[0]["lr"])
            if monitored_loss < best_head_loss - early_stopping_min_delta:
                improved = True
                best_head_loss = monitored_loss
                best_head_epoch = epoch_index + 1
                best_head_train_loss = train_loss
                best_head_val_loss = val_loss
                best_head_learning_rate = current_head_learning_rate
                best_head_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_head_epochs = 0
            else:
                improved = False
                stale_head_epochs += 1
            early_stopped = bool(early_stopping_patience and stale_head_epochs >= early_stopping_patience)
            print(
                f"{model_name}: phase=zone_specific_heads "
                f"epoch={epoch_index + 1}/{head_max_epochs} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"monitored_loss={monitored_loss:.6f} "
                f"lr={current_head_learning_rate:.6g} "
                f"stale={stale_head_epochs} "
                f"early_stopped={early_stopped}",
                flush=True,
            )
            if training_log_rows is not None:
                training_log_rows.append(
                    {
                        "method": METHOD_NAME,
                        "task": task,
                        "phase": "zone_specific_heads",
                        "model_name": model_name,
                        "epoch": epoch_index + 1,
                        "max_epochs": head_max_epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "monitored_loss": monitored_loss,
                        "loss_units": "weighted_log1p_residual_mse",
                        "learning_rate": current_head_learning_rate,
                        "initial_learning_rate": head_learning_rate,
                        "weight_decay": head_weight_decay,
                        "batch_size": batch_size,
                        "bus_hash_buckets": int(bus_hash_buckets),
                        "target_mode": DIRECT_TARGET_MODE,
                        "sample_weight_mode": DIRECT_SAMPLE_WEIGHT_MODE,
                        "d_model": d_model,
                        "nhead": nhead,
                        "num_layers": num_layers,
                        "dim_feedforward": dim_feedforward,
                        "dropout": dropout,
                        "lr_scheduler": "ReduceLROnPlateau" if head_scheduler is not None else "none",
                        "lr_scheduler_factor": lr_scheduler_factor if head_scheduler is not None else np.nan,
                        "lr_scheduler_patience": lr_scheduler_patience if head_scheduler is not None else np.nan,
                        "lr_scheduler_min_lr": lr_scheduler_min_lr if head_scheduler is not None else np.nan,
                        "n_train": int(epoch_sample_count),
                        "n_validation": int(val_count),
                        "validation_used": bool(use_validation),
                        "improved": improved,
                        "stale_epochs": stale_head_epochs,
                        "best_epoch_so_far": best_head_epoch,
                        "best_loss_so_far": best_head_loss,
                        "best_train_loss_so_far": best_head_train_loss,
                        "best_val_loss_so_far": best_head_val_loss,
                        "best_learning_rate_so_far": best_head_learning_rate,
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                        "early_stopped": early_stopped,
                    }
                )
            if early_stopped:
                break
        if best_head_state is not None:
            model.load_state_dict(best_head_state)
        print(
            f"{model_name}: phase=zone_specific_heads "
            f"best_epoch={best_head_epoch}/{head_max_epochs} "
            f"best_loss={best_head_loss:.6f} "
            f"best_train_loss={best_head_train_loss:.6f} "
            f"best_val_loss={best_head_val_loss:.6f} "
            f"best_lr={best_head_learning_rate:.6g}",
            flush=True,
        )
    model.eval()
    return DirectBusTransformerBundle(model, feature_mean, feature_std, feature_columns, {}, enabled, int(bus_hash_buckets))


def predict_direct_bus_transformer(bundle: DirectBusTransformerBundle, frame: pd.DataFrame) -> np.ndarray:
    """Predict bus-level pd with the trained direct-bus Transformer."""
    if len(frame) == 0:
        return np.array([], dtype="float32")
    device = next(bundle.model.parameters()).device
    x_np = frame[bundle.feature_columns].to_numpy(dtype="float32")
    zone_np = frame["zone_idx"].to_numpy(dtype="int64")
    bus_np = frame["bus_idx"].to_numpy(dtype="int64")
    baseline_log_np = frame["log_baseline_pd"].to_numpy(dtype="float32")
    chunks = []
    with torch.no_grad():
        for start in range(0, len(frame), 8192):
            x_tensor = torch.tensor((x_np[start : start + 8192] - bundle.feature_mean) / bundle.feature_std, dtype=torch.float32, device=device)
            zone_tensor = torch.tensor(zone_np[start : start + 8192], dtype=torch.long, device=device)
            bus_tensor = torch.tensor(bus_np[start : start + 8192], dtype=torch.long, device=device)
            pred_residual = bundle.model(x_tensor, zone_tensor, bus_tensor, use_zone_heads=True).detach().cpu().numpy().reshape(-1)
            pred_log = baseline_log_np[start : start + 8192] + pred_residual
            chunks.append(np.maximum(np.expm1(pred_log), 0.0).astype("float32"))
    return np.concatenate(chunks)



def run_direct_bus_data_stage(
    force_rebuild: bool = False,
    use_missing_indicators: bool = True,
    rows_per_date: int = DEFAULT_METHOD1_ROWS_PER_DATE,
    training_years: list[int] | None = None,
    bus_hash_buckets: int = DEFAULT_BUS_HASH_BUCKETS,
) -> dict[str, object]:
    """Stage 2.B: prepare Method 1 imputer, profiles, zone mapping, and sampled training frames."""
    start_time = time.time()
    training_years = [int(year) for year in (training_years or DEFAULT_METHOD1_TRAINING_YEARS)]
    rows_per_date = int(rows_per_date)
    bus_hash_buckets = int(bus_hash_buckets)
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = direct_transformer_paths(output_dir)
    stage_paths = direct_stage_paths(output_dir)
    if force_rebuild:
        for key in ["data_manifest", "imputer", "zone_mapping", "training_profile", "forecast_profile", "next_month_forecast_profile", "next_day_train", "next_month_train"]:
            stage_paths[key].unlink(missing_ok=True)
    if direct_data_stage_complete(stage_paths, use_missing_indicators, rows_per_date, training_years, bus_hash_buckets) and not force_rebuild:
        cached = load_direct_data_stage(stage_paths)
        return {
            "method_name": METHOD_NAME,
            "paths": paths,
            "stage_paths": stage_paths,
            "available": True,
            "rebuilt": False,
            "use_missing_indicators": use_missing_indicators,
            "feature_columns": cached["feature_columns"],
            "training_years": cached["manifest"].get("training_years", training_years),
            "rows_per_date": int(cached["manifest"].get("rows_per_date", rows_per_date)),
            "bus_hash_buckets": int(cached["manifest"].get("bus_hash_buckets", bus_hash_buckets)),
            "next_day_train_rows": int(len(cached["next_day_train"])),
            "next_month_train_rows": int(len(cached["next_month_train"])),
            "elapsed_seconds": time.time() - start_time,
        }
    print("Method 1 Stage 2.B: building partial-missing bus imputer", flush=True)
    imputer = build_bus_pd_imputer(data_dir, None)
    print("Method 1 Stage 2.B: building zone mapping and feature columns", flush=True)
    zone_to_idx = build_zone_mapping(data_dir)
    feature_columns = direct_feature_columns(use_missing_indicators)
    print("Method 1 Stage 2.B: building train and forecast bus profiles", flush=True)
    training_profile = build_training_profiles_by_target_year(data_dir, imputer, training_years)
    forecast_profile = build_bus_profile(data_dir, imputer, [2022, 2023, 2024])
    next_month_forecast_profile = build_bus_profile_before(data_dir, imputer, pd.Timestamp("2024-12-01"))
    print("Method 1 Stage 2.B: sampling next-day direct-bus training rows", flush=True)
    next_day_train = make_training_frame(
        "next_day",
        data_dir,
        imputer,
        training_profile,
        zone_to_idx,
        feature_columns,
        rows_per_date=rows_per_date,
        training_years=training_years,
        bus_hash_buckets=bus_hash_buckets,
    )
    print("Method 1 Stage 2.B: sampling next-month direct-bus training rows", flush=True)
    next_month_train = make_training_frame(
        "next_month",
        data_dir,
        imputer,
        training_profile,
        zone_to_idx,
        feature_columns,
        rows_per_date=rows_per_date,
        training_years=training_years,
        bus_hash_buckets=bus_hash_buckets,
    )
    with stage_paths["imputer"].open("wb") as handle:
        pickle.dump(imputer, handle)
    stage_paths["zone_mapping"].write_text(json.dumps(zone_to_idx, indent=2), encoding="utf-8")
    training_profile.to_parquet(stage_paths["training_profile"], index=False)
    forecast_profile.to_parquet(stage_paths["forecast_profile"], index=False)
    next_month_forecast_profile.to_parquet(stage_paths["next_month_forecast_profile"], index=False)
    next_day_train.to_parquet(stage_paths["next_day_train"], index=False)
    next_month_train.to_parquet(stage_paths["next_month_train"], index=False)
    manifest = {
        "method_name": METHOD_NAME,
        "use_missing_indicators": bool(use_missing_indicators),
        "feature_columns": feature_columns,
        "feature_frame_columns": feature_columns + ["zone_idx", "bus_idx", "actual_pd", "target_residual_log", "sample_weight", "_target_date"],
        "training_years": training_years,
        "rows_per_date": rows_per_date,
        "bus_hash_buckets": bus_hash_buckets,
        "target_mode": DIRECT_TARGET_MODE,
        "sample_weight_mode": DIRECT_SAMPLE_WEIGHT_MODE,
        "next_day_train_rows": int(len(next_day_train)),
        "next_month_train_rows": int(len(next_month_train)),
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    stage_paths["data_manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "method_name": METHOD_NAME,
        "paths": paths,
        "stage_paths": stage_paths,
        "available": True,
        "rebuilt": True,
        "use_missing_indicators": use_missing_indicators,
        "feature_columns": feature_columns,
        "training_years": training_years,
        "rows_per_date": rows_per_date,
        "bus_hash_buckets": bus_hash_buckets,
        "next_day_train_rows": int(len(next_day_train)),
        "next_month_train_rows": int(len(next_month_train)),
        "elapsed_seconds": time.time() - start_time,
    }


def run_direct_bus_training_stage(
    force_rebuild: bool = False,
    use_missing_indicators: bool = True,
    shared_max_epochs: int = 20,
    head_max_epochs: int = 10,
    validation_fraction: float = 0.15,
    early_stopping_patience: int | None = 5,
    early_stopping_min_delta: float = 1e-4,
    shared_learning_rate: float = 0.003,
    head_learning_rate: float = 0.002,
    shared_weight_decay: float = 1e-4,
    head_weight_decay: float = 1e-4,
    batch_size: int = DEFAULT_METHOD1_BATCH_SIZE,
    min_zone_samples: int = 2000,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 96,
    dropout: float = 0.05,
    use_reduce_lr_on_plateau: bool = True,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 3,
    lr_scheduler_min_lr: float = 1e-5,
) -> dict[str, object]:
    """Stage 2.C: train Method 1 next-day and next-month direct-bus Transformer checkpoints."""
    start_time = time.time()
    output_dir = Path(__file__).resolve().parent
    paths = direct_transformer_paths(output_dir)
    stage_paths = direct_stage_paths(output_dir)
    if force_rebuild:
        for key in ["next_day_model", "next_month_model"]:
            stage_paths[key].unlink(missing_ok=True)
        paths["training_log"].unlink(missing_ok=True)
    if direct_training_stage_complete(stage_paths, paths) and not force_rebuild:
        training_log = pd.read_csv(paths["training_log"])
        return {
            "method_name": METHOD_NAME,
            "paths": paths,
            "stage_paths": stage_paths,
            "training_log": training_log,
            "available": True,
            "rebuilt": False,
            "elapsed_seconds": time.time() - start_time,
            "hyperparameters": {
                "shared_max_epochs": shared_max_epochs,
                "head_max_epochs": head_max_epochs,
                "validation_fraction": validation_fraction,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "shared_learning_rate": shared_learning_rate,
                "head_learning_rate": head_learning_rate,
                "shared_weight_decay": shared_weight_decay,
                "head_weight_decay": head_weight_decay,
                "batch_size": batch_size,
                "bus_hash_buckets": DEFAULT_BUS_HASH_BUCKETS,
                "target_mode": DIRECT_TARGET_MODE,
                "sample_weight_mode": DIRECT_SAMPLE_WEIGHT_MODE,
                "min_zone_samples": min_zone_samples,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "use_reduce_lr_on_plateau": use_reduce_lr_on_plateau,
                "lr_scheduler_factor": lr_scheduler_factor,
                "lr_scheduler_patience": lr_scheduler_patience,
                "lr_scheduler_min_lr": lr_scheduler_min_lr,
            },
        }
    if not direct_data_stage_complete(stage_paths, use_missing_indicators):
        raise FileNotFoundError("Method 1 prepared data cache is missing. Run run_direct_bus_data_stage(...) first.")
    cached = load_direct_data_stage(stage_paths)
    zone_to_idx = cached["zone_to_idx"]
    feature_columns = cached["feature_columns"]
    bus_hash_buckets = int(cached["manifest"].get("bus_hash_buckets", DEFAULT_BUS_HASH_BUCKETS))
    training_log_rows: list[dict[str, object]] = []
    print("Method 1 Stage 2.C: training next-day direct-bus Transformer", flush=True)
    next_day_bundle = train_direct_bus_transformer(
        cached["next_day_train"],
        len(zone_to_idx),
        feature_columns,
        min_zone_samples=min_zone_samples,
        training_log_rows=training_log_rows,
        task="next_day",
        shared_max_epochs=shared_max_epochs,
        head_max_epochs=head_max_epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        shared_learning_rate=shared_learning_rate,
        head_learning_rate=head_learning_rate,
        shared_weight_decay=shared_weight_decay,
        head_weight_decay=head_weight_decay,
        batch_size=batch_size,
        bus_hash_buckets=bus_hash_buckets,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
    )
    next_day_bundle.zone_to_idx = zone_to_idx
    print("Method 1 Stage 2.C: training next-month direct-bus Transformer", flush=True)
    next_month_bundle = train_direct_bus_transformer(
        cached["next_month_train"],
        len(zone_to_idx),
        feature_columns,
        min_zone_samples=min_zone_samples,
        training_log_rows=training_log_rows,
        task="next_month",
        shared_max_epochs=shared_max_epochs,
        head_max_epochs=head_max_epochs,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        shared_learning_rate=shared_learning_rate,
        head_learning_rate=head_learning_rate,
        shared_weight_decay=shared_weight_decay,
        head_weight_decay=head_weight_decay,
        batch_size=batch_size,
        bus_hash_buckets=bus_hash_buckets,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_reduce_lr_on_plateau=use_reduce_lr_on_plateau,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
    )
    next_month_bundle.zone_to_idx = zone_to_idx
    save_direct_bundle(stage_paths["next_day_model"], next_day_bundle)
    save_direct_bundle(stage_paths["next_month_model"], next_month_bundle)
    training_log = pd.DataFrame(training_log_rows)
    training_log.to_csv(paths["training_log"], index=False)
    return {
        "method_name": METHOD_NAME,
        "paths": paths,
        "stage_paths": stage_paths,
        "training_log": training_log,
        "available": True,
        "rebuilt": True,
        "elapsed_seconds": time.time() - start_time,
        "hyperparameters": {
            "shared_max_epochs": shared_max_epochs,
            "head_max_epochs": head_max_epochs,
            "validation_fraction": validation_fraction,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "shared_learning_rate": shared_learning_rate,
            "head_learning_rate": head_learning_rate,
            "shared_weight_decay": shared_weight_decay,
            "head_weight_decay": head_weight_decay,
            "batch_size": batch_size,
            "bus_hash_buckets": bus_hash_buckets,
            "target_mode": DIRECT_TARGET_MODE,
            "sample_weight_mode": DIRECT_SAMPLE_WEIGHT_MODE,
            "min_zone_samples": min_zone_samples,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "use_reduce_lr_on_plateau": use_reduce_lr_on_plateau,
            "lr_scheduler_factor": lr_scheduler_factor,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_min_lr": lr_scheduler_min_lr,
        },
    }


def run_direct_bus_forecast_stage(force_rebuild: bool = False, use_missing_indicators: bool = True) -> dict[str, object]:
    """Stage 2.D: load trained Method 1 checkpoints and write full 2025 bus forecast outputs."""
    start_time = time.time()
    project_root = find_project_root()
    data_dir = find_assignment2_data_dir(project_root)
    output_dir = Path(__file__).resolve().parent
    paths = direct_transformer_paths(output_dir)
    stage_paths = direct_stage_paths(output_dir)
    if direct_transformer_outputs_complete(paths) and not force_rebuild:
        metrics = pd.read_csv(paths["metrics"])
        if "method" in metrics.columns:
            metrics["method"] = METHOD_NAME
        else:
            metrics.insert(0, "method", METHOD_NAME)
        return {"method_name": METHOD_NAME, "paths": paths, "stage_paths": stage_paths, "metrics": metrics, "available": True, "rebuilt": False, "use_missing_indicators": use_missing_indicators}
    if not direct_training_stage_complete(stage_paths, paths):
        raise FileNotFoundError("Method 1 model checkpoints are missing. Run run_direct_bus_training_stage(...) first.")
    cached = load_direct_data_stage(stage_paths)
    imputer = cached["imputer"]
    zone_to_idx = cached["zone_to_idx"]
    forecast_profile = cached["forecast_profile"]
    next_month_forecast_profile = cached["next_month_forecast_profile"]
    active_history = historical_active_bus_set(imputer, (2022, 2023, 2024))
    all_missing_2025 = all_missing_bus_set(imputer, 2025)
    next_day_bundle = load_direct_bundle(stage_paths["next_day_model"])
    next_month_bundle = load_direct_bundle(stage_paths["next_month_model"])
    next_day_bundle.zone_to_idx = zone_to_idx
    next_month_bundle.zone_to_idx = zone_to_idx
    metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
    paths["next_day"].unlink(missing_ok=True)
    paths["next_month"].unlink(missing_ok=True)
    paths["missing_bus_audit"].unlink(missing_ok=True)

    audit_counts = {
        ("next_day", "all_missing_2025_bus_rows"): 0,
        ("next_month", "all_missing_2025_bus_rows"): 0,
        ("next_day", "no_historical_nonzero_bus_rows"): 0,
        ("next_month", "no_historical_nonzero_bus_rows"): 0,
        ("next_day", "rule_based_zero_forecast_rows"): 0,
        ("next_month", "rule_based_zero_forecast_rows"): 0,
    }

    with pq.ParquetWriter(paths["next_day"], FORECAST_SCHEMA, compression="snappy", use_dictionary=False) as next_day_writer, pq.ParquetWriter(paths["next_month"], FORECAST_SCHEMA, compression="snappy", use_dictionary=False) as next_month_writer:
        for target_date in pd.date_range("2025-01-01", "2025-12-31"):
            target = read_bus_day_for_direct(data_dir, target_date)
            if target.empty:
                continue
            target["actual_pd"] = target["pd"].fillna(0.0).astype("float32")
            target_bus_ids = target["bus_unique_id"].astype(str)
            all_missing_count = int(target_bus_ids.isin(all_missing_2025).sum())
            no_history_mask = ~target_bus_ids.isin(active_history)
            no_history_count = int(no_history_mask.sum())
            audit_counts[("next_day", "all_missing_2025_bus_rows")] += all_missing_count
            audit_counts[("next_month", "all_missing_2025_bus_rows")] += all_missing_count
            audit_counts[("next_day", "no_historical_nonzero_bus_rows")] += no_history_count
            audit_counts[("next_month", "no_historical_nonzero_bus_rows")] += no_history_count
            audit_counts[("next_day", "rule_based_zero_forecast_rows")] += no_history_count
            audit_counts[("next_month", "rule_based_zero_forecast_rows")] += no_history_count
            next_day_source = source_baseline_for_day(data_dir, target_date - pd.Timedelta(days=7), "baseline_next_day_pd", imputer)
            next_day_frame = target.merge(next_day_source, on=["bus_unique_id", "zone_name", "he"], how="left").merge(forecast_profile, on=["bus_unique_id", "he"], how="left")
            next_day_frame = add_direct_features(next_day_frame, target_date, zone_to_idx, "baseline_next_day_pd", bus_hash_buckets=next_day_bundle.bus_hash_buckets)
            next_day_pred = predict_direct_bus_transformer(next_day_bundle, next_day_frame)
            next_day_pred[no_history_mask.to_numpy()] = 0.0
            write_direct_forecast(next_day_writer, next_day_frame, DIRECT_NEXT_DAY_MODEL_NAME, (target_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:01:00"), next_day_pred)
            next_month_source = source_baseline_for_day(data_dir, target_date - pd.DateOffset(years=1), "baseline_next_month_pd", imputer)
            next_month_frame = target.merge(next_month_source, on=["bus_unique_id", "zone_name", "he"], how="left").merge(next_month_forecast_profile, on=["bus_unique_id", "he"], how="left")
            next_month_frame = add_direct_features(next_month_frame, target_date, zone_to_idx, "baseline_next_month_pd", bus_hash_buckets=next_month_bundle.bus_hash_buckets)
            next_month_pred = predict_direct_bus_transformer(next_month_bundle, next_month_frame)
            next_month_pred[no_history_mask.to_numpy()] = 0.0
            created_at = (pd.Timestamp(target_date.year, target_date.month, 1) - pd.DateOffset(months=1)).strftime("%Y-%m-01 00:01:00")
            write_direct_forecast(next_month_writer, next_month_frame, DIRECT_NEXT_MONTH_MODEL_NAME, created_at, next_month_pred)
            update_metric(metrics, ("next_day", "bus", DIRECT_NEXT_DAY_MODEL_NAME), target["actual_pd"].to_numpy(), next_day_pred)
            update_metric(metrics, ("next_month", "bus", DIRECT_NEXT_MONTH_MODEL_NAME), target["actual_pd"].to_numpy(), next_month_pred)
            next_day_zone = pd.DataFrame({"zone_name": target["zone_name"], "he": target["he"], "actual_pd": target["actual_pd"], "predict_pd": next_day_pred}).groupby(["zone_name", "he"], observed=True, as_index=False).sum()
            next_month_zone = pd.DataFrame({"zone_name": target["zone_name"], "he": target["he"], "actual_pd": target["actual_pd"], "predict_pd": next_month_pred}).groupby(["zone_name", "he"], observed=True, as_index=False).sum()
            update_metric(metrics, ("next_day", "zone", DIRECT_NEXT_DAY_MODEL_NAME), next_day_zone["actual_pd"].to_numpy(), next_day_zone["predict_pd"].to_numpy())
            update_metric(metrics, ("next_month", "zone", DIRECT_NEXT_MONTH_MODEL_NAME), next_month_zone["actual_pd"].to_numpy(), next_month_zone["predict_pd"].to_numpy())
    rows = []
    for (task, level, model_name), accumulator in sorted(metrics.items()):
        row = {"method": METHOD_NAME, "task": task, "level": level, "model_name": model_name}
        row.update(accumulator.as_dict())
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df["elapsed_minutes"] = (time.time() - start_time) / 60.0
    metrics_df["use_missing_indicators"] = bool(use_missing_indicators)
    metrics_df.to_csv(paths["metrics"], index=False)
    audit_df = pd.DataFrame(
        [{"task": task, "audit_item": item, "row_count": count, "use_missing_indicators": bool(use_missing_indicators)} for (task, item), count in sorted(audit_counts.items())]
    )
    audit_df.to_csv(paths["missing_bus_audit"], index=False)
    write_sample_csv(paths["next_day"], paths["next_day_sample"])
    write_sample_csv(paths["next_month"], paths["next_month_sample"])
    return {"method_name": METHOD_NAME, "paths": paths, "stage_paths": stage_paths, "metrics": metrics_df, "available": True, "rebuilt": True, "use_missing_indicators": use_missing_indicators}


def run_direct_bus_transformer(force_rebuild: bool = False, train_if_missing: bool = False, use_missing_indicators: bool = True) -> dict[str, object]:
    """Backward-compatible one-shot Method 1 runner built on the staged data/training/forecast functions."""
    output_dir = Path(__file__).resolve().parent
    paths = direct_transformer_paths(output_dir)
    if direct_transformer_outputs_complete(paths) and not force_rebuild:
        metrics = pd.read_csv(paths["metrics"])
        if "method" in metrics.columns:
            metrics["method"] = METHOD_NAME
        else:
            metrics.insert(0, "method", METHOD_NAME)
        return {"method_name": METHOD_NAME, "paths": paths, "metrics": metrics, "available": True, "rebuilt": False, "use_missing_indicators": use_missing_indicators}
    if not train_if_missing:
        empty = pd.DataFrame(columns=["method", "task", "level", "model_name", "n", "mae", "rmse", "wmape", "sum_actual"])
        return {
            "method_name": METHOD_NAME,
            "paths": paths,
            "metrics": empty,
            "available": False,
            "rebuilt": False,
            "message": "Method 1 direct-bus Transformer outputs are not present. Run staged cells or call with train_if_missing=True.",
            "use_missing_indicators": use_missing_indicators,
        }
    run_direct_bus_data_stage(force_rebuild=force_rebuild, use_missing_indicators=use_missing_indicators)
    run_direct_bus_training_stage(force_rebuild=force_rebuild, use_missing_indicators=use_missing_indicators)
    return run_direct_bus_forecast_stage(force_rebuild=force_rebuild, use_missing_indicators=use_missing_indicators)


if __name__ == "__main__":
    result = run_direct_bus_transformer(force_rebuild=False, train_if_missing=False)
    if result.get("available"):
        print(result["metrics"].to_string(index=False))
    else:
        print(result["message"])
