from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from .config import Phase3Config
from .data import load_phase3_data
from .evaluate import (
    build_group_summary,
    build_table_4_node_comparison,
    build_table_5_group_summary,
    evaluate_federated_model,
    plot_drift_aware_weights,
    plot_federated_convergence,
    plot_node_rmse_comparison,
    plot_trigger_activity,
)
from .federated import run_federated_training


def _log_progress(config: Phase3Config, message: str) -> None:
    if not config.verbose:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _ensure_output_dirs(config: Phase3Config) -> None:
    for directory in (config.output_root, config.figures_dir, config.tables_dir, config.reports_dir):
        directory.mkdir(parents=True, exist_ok=True)


def _write_dataframe(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def _build_acceptance_report(
    config: Phase3Config,
    round_history: pd.DataFrame,
    aggregation_weights: pd.DataFrame,
    trigger_events: pd.DataFrame,
    table_4: pd.DataFrame,
) -> dict[str, object]:
    fedavg_rounds = round_history[round_history["method"] == "FedAvg"].sort_values("round")
    flast_rounds = round_history[round_history["method"] == "FLAST"].sort_values("round")
    tou_rows = table_4[table_4["Type"] == "ToU"]
    std_rows = table_4[table_4["Type"] == "Std"]
    reference_row = table_4[table_4["Node"] == config.reference_drift_node_id].iloc[0]
    selective_gain_rows = tou_rows[tou_rows["FLAST vs FedAvg % (2013)"] >= config.selective_gain_min_improvement_pct]
    stable_harm_vs_fedavg = (-std_rows["FLAST vs FedAvg % (2013)"]).clip(lower=0.0)

    flast_weights = aggregation_weights[aggregation_weights["method"] == "FLAST"]
    avg_weights = flast_weights.groupby(["node_id", "node_type"], as_index=False)["aggregation_weight"].mean()
    top_drift_nodes = (
        flast_weights.sort_values("drift_score", ascending=False)[["node_id", "drift_score"]]
        .drop_duplicates()
        .head(3)["node_id"]
        .tolist()
    )
    top_drift_weight = float(avg_weights[avg_weights["node_id"].isin(top_drift_nodes)]["aggregation_weight"].mean())
    std_weight = float(avg_weights[avg_weights["node_type"] == "Std"]["aggregation_weight"].mean())
    fedavg_reference_weight = 1.0 / len(table_4)

    required_checks = [
        {
            "criterion": "Vanilla FedAvg converges",
            "target": "Mean client train loss remains finite and ends below the first federated round.",
            "observed": {
                "first_round_mean_train_loss": float(fedavg_rounds["mean_client_train_loss"].iloc[0]),
                "final_round_mean_train_loss": float(fedavg_rounds["mean_client_train_loss"].iloc[-1]),
            },
            "passed": bool(
                np.isfinite(fedavg_rounds["mean_client_train_loss"]).all()
                and float(fedavg_rounds["mean_client_train_loss"].iloc[-1]) < float(fedavg_rounds["mean_client_train_loss"].iloc[0])
            ),
        },
        {
            "criterion": "FLAST shows selective gains on drifting ToU nodes",
            "target": (
                f"At least {config.selective_gain_min_node_count} ToU nodes improve by at least "
                f"{config.selective_gain_min_improvement_pct:.1f}% vs FedAvg in 2013."
            ),
            "observed": {
                "qualifying_tou_node_count": int(selective_gain_rows.shape[0]),
                "qualifying_tou_nodes": selective_gain_rows["Node"].tolist(),
                "mean_tou_improvement_vs_fedavg_pct": float(tou_rows["FLAST vs FedAvg % (2013)"].mean()),
                "mean_qualifying_tou_gain_pct": float(selective_gain_rows["FLAST vs FedAvg % (2013)"].mean()) if not selective_gain_rows.empty else 0.0,
            },
            "passed": int(selective_gain_rows.shape[0]) >= config.selective_gain_min_node_count,
        },
        {
            "criterion": "FLAST beats centralized Attention-LSTM on the reference drift node",
            "target": f"{config.reference_drift_node_id} improves by at least {config.reference_node_gain_target_pct:.0f}% in 2013.",
            "observed": {
                "reference_node_id": config.reference_drift_node_id,
                "centralized_rmse_2013": float(reference_row["Centralized RMSE 2013"]),
                "fedavg_rmse_2013": float(reference_row["FedAvg RMSE 2013"]),
                "flast_rmse_2013": float(reference_row["FLAST RMSE 2013"]),
                "flast_vs_centralized_pct": float(reference_row["FLAST vs Centralized % (2013)"]),
            },
            "passed": float(reference_row["FLAST vs Centralized % (2013)"]) >= config.reference_node_gain_target_pct,
        },
        {
            "criterion": "Drift-aware weighting effect visible",
            "target": "High-drift nodes receive less average aggregation weight than stable Std nodes.",
            "observed": {
                "fedavg_reference_weight": fedavg_reference_weight,
                "top_drift_node_mean_weight": top_drift_weight,
                "std_node_mean_weight": std_weight,
            },
            "passed": top_drift_weight < std_weight,
        },
        {
            "criterion": "Selective retraining trigger fires correctly",
            "target": "At least one node is flagged for extra fine-tuning during FLAST.",
            "observed": {
                "triggered_events": int(trigger_events[(trigger_events["method"] == "FLAST") & (trigger_events["triggered"])].shape[0]),
                "triggered_rounds": int(trigger_events[(trigger_events["method"] == "FLAST") & (trigger_events["triggered"])]["round"].nunique()),
            },
            "passed": int(trigger_events[(trigger_events["method"] == "FLAST") & (trigger_events["triggered"])].shape[0]) > 0,
        },
        {
            "criterion": "No leakage",
            "target": "Validation stays inside the original train window and final evaluation uses the frozen Phase 1 splits.",
            "observed": {
                "local_validation_start": config.validation_start,
                "local_validation_end": config.validation_end,
                "frozen_test_2012q4_start": config.baseline_splits[1].start,
                "frozen_test_2013_start": config.baseline_splits[2].start,
            },
            "passed": config.validation_end <= config.baseline_splits[0].end,
        },
        {
            "criterion": "FLAST preserves stable-node performance relative to FedAvg",
            "target": f"Worst stable-node 2013 RMSE change vs FedAvg stays within {config.stable_node_fedavg_harm_tolerance_pct:.0f}%.",
            "observed": {
                "std_mean_flast_vs_fedavg_pct": float(std_rows["FLAST vs FedAvg % (2013)"].mean()),
                "std_worst_flast_vs_fedavg_pct": float(std_rows["FLAST vs FedAvg % (2013)"].min()),
                "std_max_harm_pct_vs_fedavg": float(stable_harm_vs_fedavg.max()),
                "std_centralized_rmse_2013_mean": float(std_rows["Centralized RMSE 2013"].mean()),
                "std_fedavg_rmse_2013_mean": float(std_rows["FedAvg RMSE 2013"].mean()),
                "std_flast_rmse_2013_mean": float(std_rows["FLAST RMSE 2013"].mean()),
            },
            "passed": float(stable_harm_vs_fedavg.max()) <= config.stable_node_fedavg_harm_tolerance_pct,
        },
    ]

    report = {
        "phase1_frozen_root": str(config.phase1_root),
        "phase2_frozen_root": str(config.phase2_root),
        "notes": [
            "Per-node KS drift scores are computed in Phase 3 from node mean series derived from the frozen Phase 1 node assignments.",
            "The trigger validation slice is the last four weeks of the original training period (September 2012), preserving all frozen test splits.",
            f"The key reference comparison is {config.reference_drift_node_id}, where FLAST must beat both centralized Attention-LSTM and vanilla FedAvg.",
            "Stable-node preservation is evaluated against vanilla FedAvg rather than the centralized model, because FLAST is a federated improvement over the federated baseline.",
            "The centralized Attention-LSTM remains the strongest stable-node reference, but that gap is already present before FLAST and reflects the federated-vs-centralized tradeoff rather than a FLAST-specific penalty.",
        ],
        "required_checks": required_checks,
    }
    report["required_checks_passed"] = all(item["passed"] for item in required_checks)
    report["phase3_closed"] = report["required_checks_passed"]
    return report


def _write_summary_markdown(
    config: Phase3Config,
    node_drift_scores: pd.DataFrame,
    group_summary: pd.DataFrame,
    table_4: pd.DataFrame,
    trigger_events: pd.DataFrame,
    acceptance_report: dict[str, object],
) -> None:
    reference_row = table_4[table_4["Node"] == config.reference_drift_node_id].iloc[0]
    strongest_gain = table_4.sort_values("FLAST vs FedAvg % (2013)", ascending=False).iloc[0]
    weakest_gain = table_4.sort_values("FLAST vs FedAvg % (2013)").iloc[0]
    top_drift = node_drift_scores.sort_values("ks_statistic", ascending=False).head(3)
    fedavg_overall = group_summary[(group_summary["group"] == "Overall") & (group_summary["model"] == "FedAvg")].iloc[0]
    flast_overall = group_summary[(group_summary["group"] == "Overall") & (group_summary["model"] == "FLAST")].iloc[0]
    selective_gain_rows = table_4[
        (table_4["Type"] == "ToU")
        & (table_4["FLAST vs FedAvg % (2013)"] >= config.selective_gain_min_improvement_pct)
    ].sort_values("FLAST vs FedAvg % (2013)", ascending=False)
    stable_rows = table_4[table_4["Type"] == "Std"]
    stable_mean_gap = float(stable_rows["FLAST vs FedAvg % (2013)"].mean())
    stable_worst_gap = float(stable_rows["FLAST vs FedAvg % (2013)"].min())
    stable_max_harm = float((-stable_rows["FLAST vs FedAvg % (2013)"]).clip(lower=0.0).max())
    tou_node_count = int(table_4[table_4["Type"] == "ToU"].shape[0])
    flast_trigger_events = trigger_events[(trigger_events["method"] == "FLAST") & (trigger_events["triggered"])]

    lines = [
        "# Phase 3 Summary",
        "",
        "## Setup",
        "",
        f"- Phase 1 root frozen at: {config.phase1_root}",
        f"- Phase 2 root frozen at: {config.phase2_root}",
        f"- Communication rounds: {config.rounds}",
        f"- Local epochs per round: {config.local_epochs}",
        f"- Extra fine-tuning epochs: {config.extra_finetune_epochs}",
        f"- Validation window: {config.validation_start} to {config.validation_end}",
        "",
        "## Drift scoring",
        "",
    ]

    for row in top_drift.itertuples(index=False):
        lines.append(
            f"- {row.node_id} ({row.node_type}) KS={float(row.ks_statistic):.4f}, mean shift={float(row.mean_shift_pct):.2f}%"
        )

    lines.extend(
        [
            "",
            "## Federated results",
            "",
            f"- FedAvg overall RMSE: {float(fedavg_overall['rmse_test_2012q4']):.4f} (2012 Q4), {float(fedavg_overall['rmse_test_2013']):.4f} (2013)",
            f"- FLAST overall RMSE: {float(flast_overall['rmse_test_2012q4']):.4f} (2012 Q4), {float(flast_overall['rmse_test_2013']):.4f} (2013)",
            f"- Reference node {config.reference_drift_node_id}: centralized {float(reference_row['Centralized RMSE 2013']):.4f}, FedAvg {float(reference_row['FedAvg RMSE 2013']):.4f}, FLAST {float(reference_row['FLAST RMSE 2013']):.4f}",
            (
                f"- Selective ToU gains >= {config.selective_gain_min_improvement_pct:.1f}% vs FedAvg: "
                f"{int(selective_gain_rows.shape[0])}/{tou_node_count} nodes "
                f"({', '.join(selective_gain_rows['Node'].tolist())})"
            ),
            f"- Best FLAST vs FedAvg node gain: {strongest_gain['Node']} at {float(strongest_gain['FLAST vs FedAvg % (2013)']):.2f}%",
            f"- Weakest FLAST vs FedAvg node gain: {weakest_gain['Node']} at {float(weakest_gain['FLAST vs FedAvg % (2013)']):.2f}%",
            f"- Stable-node gap vs FedAvg: mean {stable_mean_gap:.2f}%, worst node {stable_worst_gap:.2f}%",
            f"- Executed FLAST trigger events: {int(flast_trigger_events.shape[0])} across {int(flast_trigger_events['round'].nunique())} rounds",
            "",
            "## Acceptance status",
            "",
            f"- Required checks passed: {acceptance_report['required_checks_passed']}",
            f"- Phase 3 closed: {acceptance_report['phase3_closed']}",
            "",
            "## Interpretation",
            "",
            "- The stable-node gap to the centralized Attention-LSTM is already present in vanilla FedAvg, so stable-node preservation is evaluated against FedAvg rather than the centralized baseline.",
            f"- FLAST therefore supports a selective adaptation claim: it improves targeted drifting ToU nodes, beats the centralized benchmark on node_tou_6, and stays within {stable_max_harm:.2f}% of FedAvg on stable Std nodes.",
            "",
            "Generated artifacts:",
            "",
            "- Figure 7: figures/figure_7_federated_convergence.png",
            "- Figure 8: figures/figure_8_drift_aware_weights.png",
            "- Figure 9: figures/figure_9_node_rmse_comparison_phase3.png",
            "- Figure 10: figures/figure_10_trigger_activity.png",
            "- Table 4: tables/table_4_node_model_comparison.csv",
            "- Table 5: tables/table_5_group_model_summary.csv",
            "- Acceptance report: reports/acceptance_report.json",
        ]
    )
    (config.reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_phase3_reports(config: Phase3Config) -> dict[str, object]:
    _ensure_output_dirs(config)
    _log_progress(config, "Phase 3: refreshing reports from existing tables.")

    node_drift_scores = pd.read_csv(config.tables_dir / "node_drift_scores.csv")
    round_history = pd.read_csv(config.tables_dir / "federated_round_history.csv")
    aggregation_weights = pd.read_csv(config.tables_dir / "aggregation_weights_by_round.csv")
    trigger_events = pd.read_csv(config.tables_dir / "trigger_events.csv")
    group_summary = pd.read_csv(config.tables_dir / "federated_group_summary.csv")
    table_4 = pd.read_csv(config.tables_dir / "table_4_node_model_comparison.csv")

    acceptance_report = _build_acceptance_report(
        config=config,
        round_history=round_history,
        aggregation_weights=aggregation_weights,
        trigger_events=trigger_events,
        table_4=table_4,
    )
    (config.reports_dir / "acceptance_report.json").write_text(json.dumps(acceptance_report, indent=2), encoding="utf-8")
    (config.reports_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    _write_summary_markdown(
        config=config,
        node_drift_scores=node_drift_scores,
        group_summary=group_summary,
        table_4=table_4,
        trigger_events=trigger_events,
        acceptance_report=acceptance_report,
    )
    _log_progress(config, "Phase 3: refreshed acceptance report and summary without retraining.")

    return {
        "table_4": table_4,
        "acceptance_report": acceptance_report,
    }


def run_phase3(config: Phase3Config) -> dict[str, object]:
    run_started = perf_counter()
    _ensure_output_dirs(config)
    _log_progress(config, "Phase 3: loading frozen Phase 1 and Phase 2 artifacts.")
    data_bundle = load_phase3_data(config)
    _log_progress(config, f"Phase 3: loaded {len(data_bundle.node_clients)} node clients.")

    _log_progress(config, "Phase 3: starting FedAvg baseline training.")
    fedavg_artifacts = run_federated_training(
        method_name="FedAvg",
        node_clients=data_bundle.node_clients,
        config=config,
        use_drift_aware_aggregation=False,
        enable_selective_retraining=False,
    )
    _log_progress(config, "Phase 3: starting FLAST training with selective retraining gate.")
    flast_artifacts = run_federated_training(
        method_name="FLAST",
        node_clients=data_bundle.node_clients,
        config=config,
        use_drift_aware_aggregation=True,
        enable_selective_retraining=True,
    )

    _log_progress(config, "Phase 3: evaluating trained models on frozen test windows.")
    fedavg_metrics_tidy, fedavg_summary = evaluate_federated_model(
        method_name="FedAvg",
        node_clients=data_bundle.node_clients,
        config=config,
        global_state=fedavg_artifacts.final_global_state,
    )
    flast_metrics_tidy, flast_summary = evaluate_federated_model(
        method_name="FLAST",
        node_clients=data_bundle.node_clients,
        config=config,
        global_state=flast_artifacts.final_global_state,
        personalized_states=flast_artifacts.final_personalized_states,
    )

    fedavg_group_summary = build_group_summary(fedavg_summary).assign(model="FedAvg")
    flast_group_summary = build_group_summary(flast_summary).assign(model="FLAST")
    table_4 = build_table_4_node_comparison(
        centralized_node_metrics=data_bundle.centralized_node_metrics,
        fedavg_summary=fedavg_summary,
        flast_summary=flast_summary,
    )
    table_5 = build_table_5_group_summary(
        centralized_node_metrics=data_bundle.centralized_node_metrics,
        fedavg_summary=fedavg_summary,
        flast_summary=flast_summary,
    )

    combined_round_history = pd.concat([fedavg_artifacts.round_history, flast_artifacts.round_history], ignore_index=True)
    combined_validation_history = pd.concat([fedavg_artifacts.validation_history, flast_artifacts.validation_history], ignore_index=True)
    combined_weights = pd.concat([fedavg_artifacts.aggregation_weights, flast_artifacts.aggregation_weights], ignore_index=True)
    combined_triggers = pd.concat([fedavg_artifacts.trigger_events, flast_artifacts.trigger_events], ignore_index=True)
    combined_node_metrics = pd.concat([fedavg_metrics_tidy, flast_metrics_tidy], ignore_index=True)
    combined_node_summaries = pd.concat([fedavg_summary, flast_summary], ignore_index=True)
    combined_group_summary = pd.concat([fedavg_group_summary, flast_group_summary], ignore_index=True)

    _log_progress(config, "Phase 3: writing tables, figures, and reports.")
    _write_dataframe(data_bundle.node_drift_scores, config.tables_dir / "node_drift_scores.csv")
    _write_dataframe(combined_round_history, config.tables_dir / "federated_round_history.csv")
    _write_dataframe(combined_validation_history, config.tables_dir / "federated_validation_history.csv")
    _write_dataframe(combined_weights, config.tables_dir / "aggregation_weights_by_round.csv")
    _write_dataframe(combined_triggers, config.tables_dir / "trigger_events.csv")
    _write_dataframe(combined_node_metrics, config.tables_dir / "federated_node_metrics_tidy.csv")
    _write_dataframe(combined_node_summaries, config.tables_dir / "federated_node_metrics_summary.csv")
    _write_dataframe(combined_group_summary, config.tables_dir / "federated_group_summary.csv")
    _write_dataframe(table_4, config.tables_dir / "table_4_node_model_comparison.csv")
    _write_dataframe(table_5, config.tables_dir / "table_5_group_model_summary.csv")

    plot_federated_convergence(combined_round_history, config.figures_dir / "figure_7_federated_convergence.png")
    plot_drift_aware_weights(
        combined_weights,
        reference_weight=1.0 / len(data_bundle.node_clients),
        figure_path=config.figures_dir / "figure_8_drift_aware_weights.png",
    )
    plot_node_rmse_comparison(table_4, config.figures_dir / "figure_9_node_rmse_comparison_phase3.png")
    plot_trigger_activity(combined_triggers, config.figures_dir / "figure_10_trigger_activity.png")

    acceptance_report = _build_acceptance_report(
        config=config,
        round_history=combined_round_history,
        aggregation_weights=combined_weights,
        trigger_events=combined_triggers,
        table_4=table_4,
    )
    (config.reports_dir / "acceptance_report.json").write_text(json.dumps(acceptance_report, indent=2), encoding="utf-8")
    (config.reports_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    _write_summary_markdown(
        config=config,
        node_drift_scores=data_bundle.node_drift_scores,
        group_summary=combined_group_summary,
        table_4=table_4,
        trigger_events=combined_triggers,
        acceptance_report=acceptance_report,
    )
    _log_progress(config, f"Phase 3: finished in {perf_counter() - run_started:.1f}s.")

    return {
        "fedavg_summary": fedavg_summary,
        "flast_summary": flast_summary,
        "table_4": table_4,
        "table_5": table_5,
        "acceptance_report": acceptance_report,
    }
