from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from phase1.model import _build_window_series, _make_sliding_windows

from .config import Phase3Config, TimeWindow


@dataclass
class NodeClientData:
    node_id: str
    node_type: str
    household_ids: list[str]
    household_count: int
    train_features: np.ndarray
    train_targets: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray
    eval_arrays: dict[str, dict[str, object]]
    train_mean: float
    train_std: float
    sample_count: int
    drift_score: float


@dataclass
class Phase3DataBundle:
    hourly_frame: pd.DataFrame
    node_assignments: pd.DataFrame
    node_clients: dict[str, NodeClientData]
    node_drift_scores: pd.DataFrame
    centralized_node_metrics: pd.DataFrame
    centralized_overall_metrics: pd.DataFrame


def _build_household_series_lookup(hourly_frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        household_id: household_frame.set_index("timestamp")["energy_kwh"].sort_index()
        for household_id, household_frame in hourly_frame.groupby("LCLid", sort=False)
    }


def _build_node_series_lookup(hourly_frame: pd.DataFrame, node_assignments: pd.DataFrame) -> dict[str, pd.Series]:
    node_frame = hourly_frame.merge(node_assignments[["LCLid", "node_id", "node_type"]], on="LCLid", how="inner")
    return {
        node_id: frame.groupby("timestamp")["energy_kwh"].mean().sort_index()
        for node_id, frame in node_frame.groupby("node_id", sort=False)
    }


def _build_split_arrays(
    source_series: pd.Series,
    config: Phase3Config,
    splits: tuple[TimeWindow, ...],
    train_mean: float | None = None,
    train_std: float | None = None,
    train_window_key: str = "local_train",
) -> dict[str, dict[str, object]]:
    split_lookup = {window.key: window for window in splits}
    if train_mean is None or train_std is None:
        train_series = _build_window_series(source_series, split_lookup[train_window_key])
        train_mean = float(train_series.mean())
        train_std = float(train_series.std())
        if train_std <= 1e-6:
            train_std = 1.0

    arrays: dict[str, dict[str, object]] = {}
    for split in splits:
        cleaned_series = _build_window_series(source_series, split)
        normalized_values = ((cleaned_series.to_numpy(dtype=np.float32) - train_mean) / train_std).astype(np.float32)
        stride = config.train_stride if split.key in {"local_train", "train_2011_to_2012q3"} else config.eval_stride
        features, targets = _make_sliding_windows(normalized_values, config.sequence_length, stride)
        arrays[split.key] = {
            "features": features[:, :, None],
            "targets": targets,
            "mean": train_mean,
            "std": train_std,
        }
    return arrays


def _build_local_train_arrays(
    household_series: list[pd.Series],
    config: Phase3Config,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if not household_series:
        raise ValueError("At least one household series is required to build a client training set.")

    stacked_train_values = np.concatenate(
        [
            _build_window_series(series, config.local_train_window).to_numpy(dtype=np.float32)
            for series in household_series
        ]
    )
    train_mean = float(stacked_train_values.mean())
    train_std = float(stacked_train_values.std())
    if train_std <= 1e-6:
        train_std = 1.0

    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    local_splits = (config.local_train_window, config.local_validation_window)

    for series in household_series:
        split_arrays = _build_split_arrays(
            source_series=series,
            config=config,
            splits=local_splits,
            train_mean=train_mean,
            train_std=train_std,
            train_window_key="local_train",
        )
        train_features = split_arrays["local_train"]["features"]
        train_targets = split_arrays["local_train"]["targets"]
        if len(train_features) == 0:
            continue
        feature_parts.append(train_features)
        target_parts.append(train_targets)

    if not feature_parts:
        raise ValueError("Local client training produced no samples.")

    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(target_parts, axis=0),
        train_mean,
        train_std,
    )


def _build_node_drift_scores(
    node_series_lookup: dict[str, pd.Series],
    node_assignments: pd.DataFrame,
    config: Phase3Config,
) -> pd.DataFrame:
    node_metadata = node_assignments.groupby("node_id", as_index=False).agg(
        node_type=("node_type", "first"),
        household_count=("LCLid", "nunique"),
    )
    rows: list[dict[str, object]] = []
    for metadata in node_metadata.itertuples(index=False):
        node_series = node_series_lookup[metadata.node_id]
        pre_series = _build_window_series(node_series, config.pre_drift_window)
        drift_series = _build_window_series(node_series, config.drift_window)
        ks_statistic, ks_p_value = ks_2samp(pre_series.to_numpy(dtype=np.float64), drift_series.to_numpy(dtype=np.float64))
        pre_mean = float(pre_series.mean())
        drift_mean = float(drift_series.mean())
        pre_std = float(pre_series.std())
        drift_std = float(drift_series.std())
        rows.append(
            {
                "node_id": metadata.node_id,
                "node_type": metadata.node_type,
                "household_count": metadata.household_count,
                "pre_mean_kwh": pre_mean,
                "drift_mean_kwh": drift_mean,
                "mean_shift_pct": ((drift_mean - pre_mean) / pre_mean * 100.0) if abs(pre_mean) > 1e-12 else np.nan,
                "pre_std_kwh": pre_std,
                "drift_std_kwh": drift_std,
                "std_shift_pct": ((drift_std - pre_std) / pre_std * 100.0) if abs(pre_std) > 1e-12 else np.nan,
                "ks_statistic": float(ks_statistic),
                "ks_p_value": float(ks_p_value),
            }
        )
    return pd.DataFrame(rows).sort_values(["node_type", "node_id"]).reset_index(drop=True)


def load_phase3_data(config: Phase3Config) -> Phase3DataBundle:
    phase1_config = config.load_phase1_config()
    if phase1_config["sequence_length"] != config.sequence_length:
        raise ValueError(
            f"Phase 1 sequence length is {phase1_config['sequence_length']} but Phase 3 requested {config.sequence_length}."
        )

    hourly_frame = pd.read_csv(config.phase1_data_dir / "hourly_subset.csv.gz", parse_dates=["timestamp"])
    node_assignments = pd.read_csv(config.phase1_data_dir / "node_assignments.csv")
    centralized_node_metrics = pd.read_csv(config.phase2_tables_dir / "attention_lstm_node_metrics.csv")
    centralized_overall_metrics = pd.read_csv(config.phase2_tables_dir / "table_2_centralized_model_comparison.csv")

    household_series_lookup = _build_household_series_lookup(hourly_frame)
    node_series_lookup = _build_node_series_lookup(hourly_frame, node_assignments)
    node_drift_scores = _build_node_drift_scores(node_series_lookup, node_assignments, config)
    drift_lookup = {
        row.node_id: float(row.ks_statistic)
        for row in node_drift_scores.itertuples(index=False)
    }

    node_clients: dict[str, NodeClientData] = {}
    node_metadata = node_assignments.groupby("node_id", as_index=False).agg(
        node_type=("node_type", "first"),
        household_count=("LCLid", "nunique"),
    )
    eval_splits = config.baseline_splits
    local_splits = (config.local_validation_window, *eval_splits[1:])

    for metadata in node_metadata.itertuples(index=False):
        household_ids = (
            node_assignments[node_assignments["node_id"] == metadata.node_id]["LCLid"]
            .drop_duplicates()
            .tolist()
        )
        household_series = [household_series_lookup[household_id] for household_id in household_ids]
        train_features, train_targets, train_mean, train_std = _build_local_train_arrays(household_series, config)

        node_eval_arrays = _build_split_arrays(
            source_series=node_series_lookup[metadata.node_id],
            config=config,
            splits=local_splits,
            train_mean=train_mean,
            train_std=train_std,
            train_window_key="local_validation",
        )

        node_clients[metadata.node_id] = NodeClientData(
            node_id=metadata.node_id,
            node_type=metadata.node_type,
            household_ids=household_ids,
            household_count=int(metadata.household_count),
            train_features=train_features,
            train_targets=train_targets,
            val_features=node_eval_arrays["local_validation"]["features"],
            val_targets=node_eval_arrays["local_validation"]["targets"],
            eval_arrays={split.key: node_eval_arrays[split.key] for split in eval_splits[1:]},
            train_mean=train_mean,
            train_std=train_std,
            sample_count=int(len(train_targets)),
            drift_score=drift_lookup[metadata.node_id],
        )

    return Phase3DataBundle(
        hourly_frame=hourly_frame,
        node_assignments=node_assignments,
        node_clients=node_clients,
        node_drift_scores=node_drift_scores,
        centralized_node_metrics=centralized_node_metrics,
        centralized_overall_metrics=centralized_overall_metrics,
    )
