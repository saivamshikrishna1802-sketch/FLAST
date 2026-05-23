from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attention_model import AttentionLSTM

from .config import Phase3Config
from .data import NodeClientData
from .trigger import compute_trigger_decisions


@dataclass
class FederatedRunArtifacts:
    method_name: str
    final_global_state: OrderedDict[str, torch.Tensor]
    final_personalized_states: dict[str, OrderedDict[str, torch.Tensor]]
    round_history: pd.DataFrame
    validation_history: pd.DataFrame
    aggregation_weights: pd.DataFrame
    trigger_events: pd.DataFrame


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(config: Phase3Config) -> AttentionLSTM:
    return AttentionLSTM(hidden_size=config.hidden_size, dropout=config.dropout)


def _clone_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in state_dict.items())


def _train_from_state(
    initial_state: OrderedDict[str, torch.Tensor],
    client: NodeClientData,
    config: Phase3Config,
    epochs: int,
    seed_offset: int,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    _set_seed(config.random_seed + seed_offset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config)
    model.load_state_dict(initial_state)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(client.train_features), torch.from_numpy(client.train_targets))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    epoch_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features_batch, targets_batch in loader:
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
        epoch_losses.append(running_loss / sample_count if sample_count else np.nan)

    return _clone_state_dict(model.cpu().state_dict()), float(epoch_losses[-1]) if epoch_losses else np.nan


def _evaluate_validation_loss(
    state_dict: OrderedDict[str, torch.Tensor],
    client: NodeClientData,
    config: Phase3Config,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(client.val_features), torch.from_numpy(client.val_targets))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    loss_fn = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for features_batch, targets_batch in loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)
            predictions = model(features_batch)
            total_loss += float(loss_fn(predictions, targets_batch).item())
            total_count += len(targets_batch)
    return total_loss / total_count if total_count else np.nan


def _evaluate_validation_losses(
    state_dict: OrderedDict[str, torch.Tensor],
    node_clients: dict[str, NodeClientData],
    config: Phase3Config,
) -> dict[str, float]:
    return {
        node_id: _evaluate_validation_loss(state_dict, client, config)
        for node_id, client in node_clients.items()
    }


def _evaluate_personalized_validation_losses(
    state_by_node: dict[str, OrderedDict[str, torch.Tensor]],
    node_clients: dict[str, NodeClientData],
    config: Phase3Config,
) -> dict[str, float]:
    return {
        node_id: _evaluate_validation_loss(state_by_node[node_id], client, config)
        for node_id, client in node_clients.items()
    }


def _sample_weighted_average(
    client_states: dict[str, OrderedDict[str, torch.Tensor]],
    raw_weights: dict[str, float],
) -> OrderedDict[str, torch.Tensor]:
    total_weight = float(sum(raw_weights.values()))
    if total_weight <= 0.0:
        raise ValueError("Aggregation weights must sum to a positive value.")

    normalized_weights = {node_id: weight / total_weight for node_id, weight in raw_weights.items()}
    aggregated_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    state_keys = list(next(iter(client_states.values())).keys())
    for key in state_keys:
        aggregated_value = None
        for node_id, state_dict in client_states.items():
            weighted_value = state_dict[key].float() * normalized_weights[node_id]
            aggregated_value = weighted_value if aggregated_value is None else aggregated_value + weighted_value
        aggregated_state[key] = aggregated_value.detach().cpu()
    return aggregated_state


def _fedavg_weights(node_clients: dict[str, NodeClientData]) -> dict[str, float]:
    return {node_id: float(client.sample_count) for node_id, client in node_clients.items()}


def _drift_aware_weights(node_clients: dict[str, NodeClientData], config: Phase3Config) -> dict[str, float]:
    return {
        node_id: float(client.sample_count) / (float(client.drift_score) + config.drift_weight_epsilon)
        for node_id, client in node_clients.items()
    }


def run_federated_training(
    method_name: str,
    node_clients: dict[str, NodeClientData],
    config: Phase3Config,
    use_drift_aware_aggregation: bool,
    enable_selective_retraining: bool,
) -> FederatedRunArtifacts:
    _set_seed(config.random_seed)
    initial_model = _build_model(config)
    global_state = _clone_state_dict(initial_model.state_dict())
    round_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    trigger_rows: list[dict[str, object]] = []
    final_personalized_states = {node_id: _clone_state_dict(global_state) for node_id in node_clients}

    sorted_node_ids = sorted(node_clients)
    for round_idx in range(1, config.rounds + 1):
        client_states: dict[str, OrderedDict[str, torch.Tensor]] = {}
        client_train_losses: dict[str, float] = {}
        local_post_train_val_losses: dict[str, float] = {}
        for client_index, node_id in enumerate(sorted_node_ids):
            state_dict, train_loss = _train_from_state(
                initial_state=global_state,
                client=node_clients[node_id],
                config=config,
                epochs=config.local_epochs,
                seed_offset=round_idx * 100 + client_index,
            )
            client_states[node_id] = state_dict
            client_train_losses[node_id] = train_loss
            local_post_train_val_losses[node_id] = _evaluate_validation_loss(state_dict, node_clients[node_id], config)

        raw_weights = (
            _drift_aware_weights(node_clients, config)
            if use_drift_aware_aggregation
            else _fedavg_weights(node_clients)
        )
        global_state = _sample_weighted_average(client_states, raw_weights)

        total_weight = float(sum(raw_weights.values()))
        current_global_val_losses = _evaluate_validation_losses(global_state, node_clients, config)
        trigger_decisions = compute_trigger_decisions(
            current_val_losses=current_global_val_losses,
            previous_val_losses=local_post_train_val_losses,
            threshold=config.trigger_threshold,
        )

        personalized_states = {node_id: _clone_state_dict(global_state) for node_id in node_clients}
        triggered_node_count = 0
        if enable_selective_retraining:
            for client_index, decision in enumerate(trigger_decisions):
                node_id = str(decision["node_id"])
                if bool(decision["triggered"]):
                    personalized_state, _ = _train_from_state(
                        initial_state=global_state,
                        client=node_clients[node_id],
                        config=config,
                        epochs=config.extra_finetune_epochs,
                        seed_offset=round_idx * 1000 + client_index,
                    )
                    personalized_states[node_id] = personalized_state
                    triggered_node_count += 1

        current_personalized_val_losses = _evaluate_personalized_validation_losses(personalized_states, node_clients, config)
        final_personalized_states = personalized_states

        mean_train_loss = float(np.mean(list(client_train_losses.values())))
        mean_global_val_loss = float(np.mean(list(current_global_val_losses.values())))
        mean_personalized_val_loss = float(np.mean(list(current_personalized_val_losses.values())))

        round_rows.append(
            {
                "method": method_name,
                "round": round_idx,
                "mean_client_train_loss": mean_train_loss,
                "mean_global_val_loss": mean_global_val_loss,
                "mean_personalized_val_loss": mean_personalized_val_loss,
                "triggered_node_count": triggered_node_count,
            }
        )

        for node_id in sorted_node_ids:
            normalized_weight = raw_weights[node_id] / total_weight
            weight_rows.append(
                {
                    "method": method_name,
                    "round": round_idx,
                    "node_id": node_id,
                    "node_type": node_clients[node_id].node_type,
                    "sample_count": node_clients[node_id].sample_count,
                    "drift_score": node_clients[node_id].drift_score,
                    "aggregation_weight": normalized_weight,
                }
            )

        for decision in trigger_decisions:
            node_id = str(decision["node_id"])
            validation_rows.append(
                {
                    "method": method_name,
                    "round": round_idx,
                    "node_id": node_id,
                    "node_type": node_clients[node_id].node_type,
                    "train_loss": float(client_train_losses[node_id]),
                    "local_post_train_val_loss": float(decision["previous_val_loss"]),
                    "global_val_loss": float(decision["current_val_loss"]),
                    "personalized_val_loss": float(current_personalized_val_losses[node_id]),
                    "relative_change_pct": float(decision["relative_change_pct"]),
                    "triggered": bool(decision["triggered"]) if enable_selective_retraining else False,
                }
            )
            trigger_rows.append(
                {
                    "method": method_name,
                    "round": round_idx,
                    "node_id": node_id,
                    "node_type": node_clients[node_id].node_type,
                    "relative_change_pct": float(decision["relative_change_pct"]),
                    "triggered": bool(decision["triggered"]) if enable_selective_retraining else False,
                }
            )

    return FederatedRunArtifacts(
        method_name=method_name,
        final_global_state=global_state,
        final_personalized_states=final_personalized_states,
        round_history=pd.DataFrame(round_rows),
        validation_history=pd.DataFrame(validation_rows),
        aggregation_weights=pd.DataFrame(weight_rows),
        trigger_events=pd.DataFrame(trigger_rows),
    )
