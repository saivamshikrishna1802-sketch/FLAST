from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Phase1Config, TimeWindow


@dataclass
class BaselineRunArtifacts:
    metrics_tidy: pd.DataFrame
    metrics_table: pd.DataFrame
    training_history: pd.DataFrame


@dataclass
class NodeBaselineArtifacts:
    node_assignments: pd.DataFrame
    metrics_tidy: pd.DataFrame
    summary_table: pd.DataFrame
    training_history: pd.DataFrame


def _build_window_series(source_series: pd.Series, window: TimeWindow) -> pd.Series:
    full_index = pd.date_range(start=window.start, end=window.end, inclusive="left", freq="h")
    window_series = source_series.reindex(full_index)
    if window_series.notna().sum() == 0:
        return pd.Series(0.0, index=full_index)

    cleaned = window_series.interpolate(method="time", limit_direction="both")
    if cleaned.isna().any():
        fill_value = float(cleaned.median()) if pd.notna(cleaned.median()) else 0.0
        cleaned = cleaned.fillna(fill_value)
    return cleaned


def _make_sliding_windows(values: np.ndarray, sequence_length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= sequence_length:
        return np.empty((0, sequence_length), dtype=np.float32), np.empty((0,), dtype=np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(values, sequence_length + 1)[::stride]
    features = windows[:, :-1].astype(np.float32)
    targets = windows[:, -1].astype(np.float32)
    return features, targets


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the Phase 1 baseline. Install dependencies from requirements.txt."
        ) from exc

    return torch, nn, DataLoader, TensorDataset


def _fit_model_from_arrays(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    config: Phase1Config,
    seed_offset: int = 0,
) -> tuple[object, object, pd.DataFrame]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    seed = config.random_seed + seed_offset
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    class LSTMRegressor(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            projection_size = max(hidden_size // 2, 1)
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, projection_size),
                nn.ReLU(),
                nn.Linear(projection_size, 1),
            )

        def forward(self, inputs):
            outputs, _ = self.lstm(inputs)
            return self.head(outputs[:, -1, :]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(hidden_size=config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_targets),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    history_rows: list[dict[str, float]] = []
    for epoch in range(1, config.baseline_epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(features_batch)
            loss = loss_fn(predictions, targets_batch)
            loss.backward()
            optimizer.step()

            batch_size = len(targets_batch)
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / sample_count if sample_count else np.nan,
            }
        )

    return model, device, pd.DataFrame(history_rows)


def _evaluate_arrays(
    model: object,
    device: object,
    features: np.ndarray,
    targets: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
    batch_size: int,
) -> dict[str, float]:
    torch, _, DataLoader, TensorDataset = _require_torch()

    if len(features) == 0:
        return {"count": 0, "rmse": np.nan, "mae": np.nan}

    mean_array = np.full(len(targets), mean, dtype=np.float32) if np.isscalar(mean) else np.asarray(mean, dtype=np.float32)
    std_array = np.full(len(targets), std, dtype=np.float32) if np.isscalar(std) else np.asarray(std, dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
        torch.from_numpy(mean_array),
        torch.from_numpy(std_array),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_count = 0
    total_abs = 0.0
    total_sq = 0.0
    with torch.no_grad():
        for features_batch, targets_batch, mean_batch, std_batch in loader:
            predictions = model(features_batch.to(device)).cpu()
            predictions_original = predictions * std_batch + mean_batch
            targets_original = targets_batch + 0.0
            targets_original = targets_original * std_batch + mean_batch
            errors = predictions_original - targets_original
            total_count += len(targets_batch)
            total_abs += float(torch.abs(errors).sum().item())
            total_sq += float(errors.pow(2).sum().item())

    return {
        "count": total_count,
        "rmse": np.sqrt(total_sq / total_count) if total_count else np.nan,
        "mae": total_abs / total_count if total_count else np.nan,
    }


def _build_series_split_arrays(
    source_series: pd.Series,
    config: Phase1Config,
    train_mean: float | None = None,
    train_std: float | None = None,
) -> dict[str, dict[str, object]]:
    split_lookup = {window.key: window for window in config.baseline_splits}
    if train_mean is None or train_std is None:
        train_series = _build_window_series(source_series, split_lookup["train_2011_to_2012q3"])
        train_mean = float(train_series.mean())
        train_std = float(train_series.std())
        if train_std <= 1e-6:
            train_std = 1.0

    arrays: dict[str, dict[str, object]] = {}
    for split in config.baseline_splits:
        cleaned_series = _build_window_series(source_series, split)
        normalized_values = ((cleaned_series.to_numpy(dtype=np.float32) - train_mean) / train_std).astype(np.float32)
        stride = config.train_stride if split.key == "train_2011_to_2012q3" else config.eval_stride
        features, targets = _make_sliding_windows(normalized_values, config.sequence_length, stride)
        arrays[split.key] = {
            "features": features[:, :, None],
            "targets": targets,
            "mean": train_mean,
            "std": train_std,
        }
    return arrays


def _build_stacked_household_train_arrays(
    household_series: list[pd.Series],
    config: Phase1Config,
) -> tuple[dict[str, np.ndarray], float, float]:
    if not household_series:
        raise ValueError("At least one household series is required to build stacked node training arrays.")

    train_window = {window.key: window for window in config.baseline_splits}["train_2011_to_2012q3"]
    stacked_train_values = np.concatenate(
        [
            _build_window_series(series, train_window).to_numpy(dtype=np.float32)
            for series in household_series
        ]
    )
    train_mean = float(stacked_train_values.mean())
    train_std = float(stacked_train_values.std())
    if train_std <= 1e-6:
        train_std = 1.0

    train_feature_parts: list[np.ndarray] = []
    train_target_parts: list[np.ndarray] = []
    for series in household_series:
        household_arrays = _build_series_split_arrays(
            source_series=series,
            config=config,
            train_mean=train_mean,
            train_std=train_std,
        )
        train_features = household_arrays["train_2011_to_2012q3"]["features"]
        train_targets = household_arrays["train_2011_to_2012q3"]["targets"]
        if len(train_features) == 0:
            continue
        train_feature_parts.append(train_features)
        train_target_parts.append(train_targets)

    if not train_feature_parts:
        raise ValueError("Stacked household training produced no samples for the node.")

    stacked_arrays = {
        "features": np.concatenate(train_feature_parts, axis=0),
        "targets": np.concatenate(train_target_parts, axis=0),
    }
    return stacked_arrays, train_mean, train_std


def assign_nodes(selected_households: pd.DataFrame, config: Phase1Config) -> pd.DataFrame:
    if config.std_households % config.households_per_node != 0:
        raise ValueError("std_households must be divisible by households_per_node for node assignment.")
    if config.tou_households % config.households_per_node != 0:
        raise ValueError("tou_households must be divisible by households_per_node for node assignment.")

    rng = np.random.default_rng(config.random_seed)
    assignment_rows: list[dict[str, object]] = []

    for tariff, prefix in (("ToU", "node_tou"), ("Std", "node_std")):
        tariff_ids = selected_households[selected_households["stdorToU"] == tariff]["LCLid"].to_numpy()
        shuffled_ids = rng.permutation(tariff_ids)
        node_count = len(shuffled_ids) // config.households_per_node

        for node_index in range(node_count):
            start = node_index * config.households_per_node
            end = start + config.households_per_node
            for household_id in shuffled_ids[start:end]:
                assignment_rows.append(
                    {
                        "LCLid": household_id,
                        "node_id": f"{prefix}_{node_index}",
                        "node_type": tariff,
                        "node_index": node_index,
                        "node_household_count": config.households_per_node,
                    }
                )

    assignments = pd.DataFrame(assignment_rows)
    household_columns = [column for column in ["LCLid", "stdorToU", "Acorn_grouped", "file", "shift_score"] if column in selected_households.columns]
    assignments = assignments.merge(selected_households[household_columns], on="LCLid", how="left")
    return assignments.sort_values(["node_type", "node_index", "LCLid"]).reset_index(drop=True)


def run_lstm_baseline(hourly_frame: pd.DataFrame, selected_households: pd.DataFrame, config: Phase1Config) -> BaselineRunArtifacts:
    series_lookup = {
        household_id: household_frame.set_index("timestamp")["energy_kwh"].sort_index()
        for household_id, household_frame in hourly_frame.groupby("LCLid", sort=False)
    }

    split_lookup = {window.key: window for window in config.baseline_splits}
    split_samples: dict[str, list[dict[str, np.ndarray | float]]] = {window.key: [] for window in config.baseline_splits}

    for household in selected_households.itertuples(index=False):
        source_series = series_lookup[household.LCLid]
        train_series = _build_window_series(source_series, split_lookup["train_2011_to_2012q3"])
        train_mean = float(train_series.mean())
        train_std = float(train_series.std())
        if train_std <= 1e-6:
            train_std = 1.0

        for split in config.baseline_splits:
            cleaned_series = _build_window_series(source_series, split)
            normalized_values = ((cleaned_series.to_numpy(dtype=np.float32) - train_mean) / train_std).astype(np.float32)
            stride = config.train_stride if split.key == "train_2011_to_2012q3" else config.eval_stride
            features, targets = _make_sliding_windows(normalized_values, config.sequence_length, stride)
            if len(features) == 0:
                continue
            split_samples[split.key].append(
                {
                    "features": features[:, :, None],
                    "targets": targets,
                    "mean": np.full(len(targets), train_mean, dtype=np.float32),
                    "std": np.full(len(targets), train_std, dtype=np.float32),
                    "tariff": np.full(len(targets), 1 if household.stdorToU == "ToU" else 0, dtype=np.int8),
                }
            )

    def concatenate_split(split_key: str) -> dict[str, np.ndarray]:
        split_parts = split_samples[split_key]
        if not split_parts:
            raise ValueError(f"No samples were created for split {split_key}.")
        return {
            "features": np.concatenate([part["features"] for part in split_parts], axis=0),
            "targets": np.concatenate([part["targets"] for part in split_parts], axis=0),
            "mean": np.concatenate([part["mean"] for part in split_parts], axis=0),
            "std": np.concatenate([part["std"] for part in split_parts], axis=0),
            "tariff": np.concatenate([part["tariff"] for part in split_parts], axis=0),
        }

    train_arrays = concatenate_split("train_2011_to_2012q3")
    model, device, training_history = _fit_model_from_arrays(
        train_arrays["features"],
        train_arrays["targets"],
        config,
    )

    def evaluate_split(split_key: str, split_label: str) -> list[dict[str, object]]:
        split_arrays = concatenate_split(split_key)
        mask_lookup = {
            "Overall": np.ones(len(split_arrays["targets"]), dtype=bool),
            "Std": split_arrays["tariff"] == 0,
            "ToU": split_arrays["tariff"] == 1,
        }

        rows: list[dict[str, object]] = []
        for tariff_name, mask in mask_lookup.items():
            metrics = _evaluate_arrays(
                model=model,
                device=device,
                features=split_arrays["features"][mask],
                targets=split_arrays["targets"][mask],
                mean=split_arrays["mean"][mask],
                std=split_arrays["std"][mask],
                batch_size=config.batch_size,
            )
            rows.append(
                {
                    "split_key": split_key,
                    "split_label": split_label,
                    "tariff_group": tariff_name,
                    **metrics,
                }
            )
        return rows

    metric_rows: list[dict[str, object]] = []
    for split in config.baseline_splits[1:]:
        metric_rows.extend(evaluate_split(split.key, split.label))

    metrics_tidy = pd.DataFrame(metric_rows)
    pivot = metrics_tidy.pivot(index="split_key", columns="tariff_group", values=["rmse", "mae", "count"])
    pivot.columns = [f"{metric}_{group.lower()}" for metric, group in pivot.columns]
    metrics_table = pivot.reset_index().merge(
        pd.DataFrame(
            [{"split_key": split.key, "split_label": split.label} for split in config.baseline_splits[1:]]
        ),
        on="split_key",
        how="left",
    )

    baseline_2012 = float(
        metrics_tidy[
            (metrics_tidy["split_key"] == "test_2012q4") & (metrics_tidy["tariff_group"] == "Overall")
        ]["rmse"].iloc[0]
    )
    metrics_table["rmse_increase_vs_2012_pct"] = (
        (metrics_table["rmse_overall"] - baseline_2012) / baseline_2012 * 100.0
    )
    baseline_2012_mae = float(
        metrics_tidy[
            (metrics_tidy["split_key"] == "test_2012q4") & (metrics_tidy["tariff_group"] == "Overall")
        ]["mae"].iloc[0]
    )
    metrics_table["mae_increase_vs_2012_pct"] = (
        (metrics_table["mae_overall"] - baseline_2012_mae) / baseline_2012_mae * 100.0
    )

    return BaselineRunArtifacts(
        metrics_tidy=metrics_tidy.sort_values(["split_key", "tariff_group"]).reset_index(drop=True),
        metrics_table=metrics_table.sort_values("split_key").reset_index(drop=True),
        training_history=training_history,
    )


def run_node_lstm_analysis(
    hourly_frame: pd.DataFrame,
    node_assignments: pd.DataFrame,
    config: Phase1Config,
) -> NodeBaselineArtifacts:
    node_frame = hourly_frame.merge(node_assignments[["LCLid", "node_id", "node_type"]], on="LCLid", how="inner")
    household_series_lookup = {
        household_id: household_frame.set_index("timestamp")["energy_kwh"].sort_index()
        for household_id, household_frame in hourly_frame.groupby("LCLid", sort=False)
    }
    node_mean_series_lookup = {
        node_id: frame.groupby("timestamp")["energy_kwh"].mean().sort_index()
        for node_id, frame in node_frame.groupby("node_id", sort=False)
    }
    node_metadata = node_assignments.groupby("node_id", as_index=False).agg(
        node_type=("node_type", "first"),
        household_count=("LCLid", "nunique"),
    )

    metric_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    split_metadata = {window.key: window.label for window in config.baseline_splits}
    for seed_offset, metadata in enumerate(node_metadata.itertuples(index=False)):
        household_ids = (
            node_assignments[node_assignments["node_id"] == metadata.node_id]["LCLid"]
            .drop_duplicates()
            .tolist()
        )
        stacked_train_arrays, train_mean, train_std = _build_stacked_household_train_arrays(
            [household_series_lookup[household_id] for household_id in household_ids],
            config,
        )
        evaluation_arrays = _build_series_split_arrays(
            source_series=node_mean_series_lookup[metadata.node_id],
            config=config,
            train_mean=train_mean,
            train_std=train_std,
        )
        model, device, training_history = _fit_model_from_arrays(
            stacked_train_arrays["features"],
            stacked_train_arrays["targets"],
            config,
            seed_offset=seed_offset + 1,
        )
        history_parts.append(
            training_history.assign(
                node_id=metadata.node_id,
                node_type=metadata.node_type,
                household_count=metadata.household_count,
                train_sequence_count=len(stacked_train_arrays["targets"]),
            )
        )

        for split in config.baseline_splits[1:]:
            metrics = _evaluate_arrays(
                model=model,
                device=device,
                features=evaluation_arrays[split.key]["features"],
                targets=evaluation_arrays[split.key]["targets"],
                mean=evaluation_arrays[split.key]["mean"],
                std=evaluation_arrays[split.key]["std"],
                batch_size=config.batch_size,
            )
            metric_rows.append(
                {
                    "node_id": metadata.node_id,
                    "node_type": metadata.node_type,
                    "household_count": metadata.household_count,
                    "split_key": split.key,
                    "split_label": split_metadata[split.key],
                    **metrics,
                }
            )

    metrics_tidy = pd.DataFrame(metric_rows).sort_values(["node_type", "node_id", "split_key"]).reset_index(drop=True)
    summary_table = metrics_tidy.pivot(
        index=["node_id", "node_type", "household_count"],
        columns="split_key",
        values=["rmse", "mae", "count"],
    )
    summary_table.columns = [f"{metric}_{split_key}" for metric, split_key in summary_table.columns]
    summary_table = summary_table.reset_index()

    summary_table["rmse_increase_2013_vs_2012_pct"] = (
        (summary_table["rmse_test_2013"] - summary_table["rmse_test_2012q4"])
        / summary_table["rmse_test_2012q4"]
        * 100.0
    )
    summary_table["mae_increase_2013_vs_2012_pct"] = (
        (summary_table["mae_test_2013"] - summary_table["mae_test_2012q4"])
        / summary_table["mae_test_2012q4"]
        * 100.0
    )
    summary_table["exceeds_degradation_threshold"] = (
        summary_table["rmse_increase_2013_vs_2012_pct"] >= config.degradation_threshold_pct
    )
    summary_table = summary_table.sort_values(["node_type", "node_id"]).reset_index(drop=True)

    training_history = (
        pd.concat(history_parts, ignore_index=True)
        if history_parts
        else pd.DataFrame(columns=["epoch", "train_loss", "node_id", "node_type", "household_count"])
    )

    return NodeBaselineArtifacts(
        node_assignments=node_assignments,
        metrics_tidy=metrics_tidy,
        summary_table=summary_table,
        training_history=training_history,
    )
