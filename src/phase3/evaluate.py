from __future__ import annotations

from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from attention_model import AttentionLSTM
from phase2.pipeline import _evaluate_arrays

from .config import Phase3Config
from .data import NodeClientData
from .federated import FederatedRunArtifacts


def _build_model(config: Phase3Config) -> AttentionLSTM:
    return AttentionLSTM(hidden_size=config.hidden_size, dropout=config.dropout)


def _evaluate_state_for_client(
    state_dict: OrderedDict[str, torch.Tensor],
    client: NodeClientData,
    config: Phase3Config,
    split_key: str,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config)
    model.load_state_dict(state_dict)
    model = model.to(device)

    split_arrays = client.eval_arrays[split_key]
    metrics, _ = _evaluate_arrays(
        model=model,
        device=device,
        features=split_arrays["features"],
        targets=split_arrays["targets"],
        mean=float(split_arrays["mean"]),
        std=float(split_arrays["std"]),
        batch_size=config.batch_size,
    )
    return metrics


def evaluate_federated_model(
    method_name: str,
    node_clients: dict[str, NodeClientData],
    config: Phase3Config,
    global_state: OrderedDict[str, torch.Tensor],
    personalized_states: dict[str, OrderedDict[str, torch.Tensor]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    for node_id, client in sorted(node_clients.items()):
        state_dict = personalized_states[node_id] if personalized_states is not None else global_state
        for split in config.baseline_splits[1:]:
            metrics = _evaluate_state_for_client(
                state_dict=state_dict,
                client=client,
                config=config,
                split_key=split.key,
            )
            metric_rows.append(
                {
                    "model": method_name,
                    "node_id": node_id,
                    "node_type": client.node_type,
                    "household_count": client.household_count,
                    "sample_count": client.sample_count,
                    "split_key": split.key,
                    "split_label": split.label,
                    **metrics,
                }
            )

    metrics_tidy = pd.DataFrame(metric_rows).sort_values(["node_type", "node_id", "split_key"]).reset_index(drop=True)
    summary = metrics_tidy.pivot(
        index=["model", "node_id", "node_type", "household_count", "sample_count"],
        columns="split_key",
        values=["rmse", "mae", "count"],
    )
    summary.columns = [f"{metric}_{split_key}" for metric, split_key in summary.columns]
    summary = summary.reset_index().sort_values(["node_type", "node_id"]).reset_index(drop=True)
    summary["rmse_change_2013_vs_2012_pct"] = (
        (summary["rmse_test_2013"] - summary["rmse_test_2012q4"])
        / summary["rmse_test_2012q4"]
        * 100.0
    )
    return metrics_tidy, summary


def build_group_summary(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group_frame in [("Overall", metrics_summary), *metrics_summary.groupby("node_type")]:
        rmse_2012q4 = float(
            np.sqrt(np.average(group_frame["rmse_test_2012q4"] ** 2, weights=group_frame["count_test_2012q4"]))
        )
        rmse_2013 = float(
            np.sqrt(np.average(group_frame["rmse_test_2013"] ** 2, weights=group_frame["count_test_2013"]))
        )
        rmse_2014 = float(
            np.sqrt(np.average(group_frame["rmse_test_2014"] ** 2, weights=group_frame["count_test_2014"]))
        )
        mae_2012q4 = float(np.average(group_frame["mae_test_2012q4"], weights=group_frame["count_test_2012q4"]))
        mae_2013 = float(np.average(group_frame["mae_test_2013"], weights=group_frame["count_test_2013"]))
        mae_2014 = float(np.average(group_frame["mae_test_2014"], weights=group_frame["count_test_2014"]))
        rows.append(
            {
                "group": group_name,
                "node_count": int(len(group_frame)),
                "rmse_test_2012q4": rmse_2012q4,
                "rmse_test_2013": rmse_2013,
                "rmse_test_2014": rmse_2014,
                "mae_test_2012q4": mae_2012q4,
                "mae_test_2013": mae_2013,
                "mae_test_2014": mae_2014,
                "rmse_change_2013_vs_2012_pct": (rmse_2013 - rmse_2012q4) / rmse_2012q4 * 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_table_4_node_comparison(
    centralized_node_metrics: pd.DataFrame,
    fedavg_summary: pd.DataFrame,
    flast_summary: pd.DataFrame,
) -> pd.DataFrame:
    central = centralized_node_metrics.rename(
        columns={
            "rmse_test_2012q4": "centralized_rmse_2012q4",
            "rmse_test_2013": "centralized_rmse_2013",
            "rmse_test_2014": "centralized_rmse_2014",
            "rmse_change_2013_vs_2012_pct": "centralized_rmse_change_2013_vs_2012_pct",
        }
    )
    fedavg = fedavg_summary.rename(
        columns={
            "rmse_test_2012q4": "fedavg_rmse_2012q4",
            "rmse_test_2013": "fedavg_rmse_2013",
            "rmse_test_2014": "fedavg_rmse_2014",
            "rmse_change_2013_vs_2012_pct": "fedavg_rmse_change_2013_vs_2012_pct",
        }
    )
    flast = flast_summary.rename(
        columns={
            "rmse_test_2012q4": "flast_rmse_2012q4",
            "rmse_test_2013": "flast_rmse_2013",
            "rmse_test_2014": "flast_rmse_2014",
            "rmse_change_2013_vs_2012_pct": "flast_rmse_change_2013_vs_2012_pct",
        }
    )
    merged = (
        central[["node_id", "node_type", "household_count", "centralized_rmse_2012q4", "centralized_rmse_2013", "centralized_rmse_2014", "centralized_rmse_change_2013_vs_2012_pct"]]
        .merge(
            fedavg[["node_id", "fedavg_rmse_2012q4", "fedavg_rmse_2013", "fedavg_rmse_2014", "fedavg_rmse_change_2013_vs_2012_pct"]],
            on="node_id",
            how="inner",
        )
        .merge(
            flast[["node_id", "flast_rmse_2012q4", "flast_rmse_2013", "flast_rmse_2014", "flast_rmse_change_2013_vs_2012_pct"]],
            on="node_id",
            how="inner",
        )
    )
    merged["flast_vs_fedavg_improvement_2013_pct"] = (
        (merged["fedavg_rmse_2013"] - merged["flast_rmse_2013"]) / merged["fedavg_rmse_2013"] * 100.0
    )
    merged["flast_vs_centralized_improvement_2013_pct"] = (
        (merged["centralized_rmse_2013"] - merged["flast_rmse_2013"]) / merged["centralized_rmse_2013"] * 100.0
    )
    merged["fedavg_vs_centralized_improvement_2013_pct"] = (
        (merged["centralized_rmse_2013"] - merged["fedavg_rmse_2013"]) / merged["centralized_rmse_2013"] * 100.0
    )
    return merged.rename(
        columns={
            "node_id": "Node",
            "node_type": "Type",
            "household_count": "Households",
            "centralized_rmse_2012q4": "Centralized RMSE 2012Q4",
            "fedavg_rmse_2012q4": "FedAvg RMSE 2012Q4",
            "flast_rmse_2012q4": "FLAST RMSE 2012Q4",
            "centralized_rmse_2013": "Centralized RMSE 2013",
            "fedavg_rmse_2013": "FedAvg RMSE 2013",
            "flast_rmse_2013": "FLAST RMSE 2013",
            "centralized_rmse_2014": "Centralized RMSE 2014",
            "fedavg_rmse_2014": "FedAvg RMSE 2014",
            "flast_rmse_2014": "FLAST RMSE 2014",
            "flast_vs_fedavg_improvement_2013_pct": "FLAST vs FedAvg % (2013)",
            "flast_vs_centralized_improvement_2013_pct": "FLAST vs Centralized % (2013)",
            "fedavg_vs_centralized_improvement_2013_pct": "FedAvg vs Centralized % (2013)",
            "centralized_rmse_change_2013_vs_2012_pct": "Centralized % Change",
            "fedavg_rmse_change_2013_vs_2012_pct": "FedAvg % Change",
            "flast_rmse_change_2013_vs_2012_pct": "FLAST % Change",
        }
    ).sort_values(["Type", "Node"]).reset_index(drop=True)


def build_table_5_group_summary(
    centralized_node_metrics: pd.DataFrame,
    fedavg_summary: pd.DataFrame,
    flast_summary: pd.DataFrame,
) -> pd.DataFrame:
    central_group = build_group_summary(centralized_node_metrics).assign(model="Centralized Attention-LSTM")
    fedavg_group = build_group_summary(fedavg_summary).assign(model="FedAvg")
    flast_group = build_group_summary(flast_summary).assign(model="FLAST")
    return pd.concat([central_group, fedavg_group, flast_group], ignore_index=True)[
        ["model", "group", "node_count", "rmse_test_2012q4", "rmse_test_2013", "rmse_test_2014", "mae_test_2012q4", "mae_test_2013", "mae_test_2014", "rmse_change_2013_vs_2012_pct"]
    ]


def plot_federated_convergence(round_history: pd.DataFrame, figure_path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=round_history, x="round", y="mean_client_train_loss", hue="method", marker="o", ax=axes[0])
    axes[0].set_title("Mean local train loss by round")
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Loss")

    sns.lineplot(data=round_history, x="round", y="mean_personalized_val_loss", hue="method", marker="o", ax=axes[1])
    axes[1].set_title("Mean validation loss by round")
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Loss")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_drift_aware_weights(weight_history: pd.DataFrame, reference_weight: float, figure_path) -> None:
    flast_weights = (
        weight_history[weight_history["method"] == "FLAST"]
        .groupby(["node_id", "node_type"], as_index=False)["aggregation_weight"]
        .mean()
        .sort_values(["node_type", "node_id"])
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=flast_weights, x="node_id", y="aggregation_weight", hue="node_type", ax=ax)
    ax.axhline(reference_weight, color="black", linestyle="--", linewidth=1.5, label="FedAvg weight")
    ax.set_title("Average FLAST aggregation weights across rounds")
    ax.set_xlabel("Node")
    ax.set_ylabel("Aggregation weight")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_node_rmse_comparison(table_4: pd.DataFrame, figure_path) -> None:
    plot_frame = table_4.melt(
        id_vars=["Node", "Type"],
        value_vars=["Centralized RMSE 2013", "FedAvg RMSE 2013", "FLAST RMSE 2013"],
        var_name="Model",
        value_name="RMSE 2013",
    )
    order = table_4.sort_values("Centralized RMSE 2013", ascending=False)["Node"].tolist()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=plot_frame, x="Node", y="RMSE 2013", hue="Model", order=order, ax=ax)
    ax.set_title("2013 node-level RMSE: centralized vs FedAvg vs FLAST")
    ax.set_xlabel("Node")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trigger_activity(trigger_events: pd.DataFrame, figure_path) -> None:
    flast_events = trigger_events[trigger_events["method"] == "FLAST"].copy()
    if flast_events.empty:
        flast_events = pd.DataFrame(columns=["round", "node_id", "triggered"])
    heatmap_frame = flast_events.pivot(index="round", columns="node_id", values="triggered").fillna(False).astype(int)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(heatmap_frame, cmap="Blues", cbar=True, ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title("FLAST selective retraining trigger activity")
    ax.set_xlabel("Node")
    ax.set_ylabel("Communication round")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
