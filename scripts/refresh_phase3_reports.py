from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3 import Phase3Config, refresh_phase3_reports


def _configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh Phase 3 reports from existing experimental tables without retraining.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase3" / "reports" / "config.json",
        help="Existing Phase 3 config.json to load and reuse.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed refresh logs.",
    )
    return parser


def main() -> None:
    _configure_utf8_output()
    parser = build_parser()
    args = parser.parse_args()

    config_data = json.loads(args.config_path.read_text(encoding="utf-8"))
    config = Phase3Config.from_dict(config_data)
    config = Phase3Config(
        phase1_root=config.phase1_root,
        phase2_root=config.phase2_root,
        output_root=config.output_root,
        random_seed=config.random_seed,
        sequence_length=config.sequence_length,
        train_stride=config.train_stride,
        eval_stride=config.eval_stride,
        batch_size=config.batch_size,
        rounds=config.rounds,
        local_epochs=config.local_epochs,
        extra_finetune_epochs=config.extra_finetune_epochs,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        trigger_threshold=config.trigger_threshold,
        selective_retraining_node_type=config.selective_retraining_node_type,
        selective_retraining_min_drift_score=config.selective_retraining_min_drift_score,
        selective_gain_min_improvement_pct=config.selective_gain_min_improvement_pct,
        selective_gain_min_node_count=config.selective_gain_min_node_count,
        reference_node_gain_target_pct=config.reference_node_gain_target_pct,
        stable_node_fedavg_harm_tolerance_pct=config.stable_node_fedavg_harm_tolerance_pct,
        validation_start=config.validation_start,
        validation_end=config.validation_end,
        reference_drift_node_id=config.reference_drift_node_id,
        verbose=not args.quiet,
    )

    results = refresh_phase3_reports(config)
    acceptance = results["acceptance_report"]
    print("Phase 3 reports refreshed.")
    print(f"Required checks passed: {acceptance['required_checks_passed']}")
    print(f"Phase 3 closed: {acceptance['phase3_closed']}")


if __name__ == "__main__":
    main()
