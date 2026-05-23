from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3 import Phase3Config, run_phase3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 FLAST federated training on frozen Phase 1 and Phase 2 artifacts.")
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase1",
        help="Frozen Phase 1 output directory used as the input artifact set.",
    )
    parser.add_argument(
        "--phase2-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase2",
        help="Frozen Phase 2 output directory used as the centralized benchmark.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase3",
        help="Directory where Phase 3 figures, tables, and reports will be written.",
    )
    parser.add_argument("--random-seed", type=int, default=17, help="Random seed for federated training reproducibility.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Lookback window length in hours.")
    parser.add_argument("--train-stride", type=int, default=3, help="Stride for local training windows.")
    parser.add_argument("--eval-stride", type=int, default=1, help="Stride for evaluation windows.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for local client updates.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated communication rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client and round.")
    parser.add_argument("--extra-finetune-epochs", type=int, default=1, help="Extra local fine-tuning epochs for triggered nodes.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for the Attention-LSTM.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used in the Attention-LSTM.")
    parser.add_argument("--trigger-threshold", type=float, default=0.05, help="Relative validation-loss increase required to trigger selective retraining.")
    parser.add_argument("--reference-drift-node-id", type=str, default="node_tou_6", help="Reference drift node used in the acceptance report.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = Phase3Config(
        phase1_root=args.phase1_root,
        phase2_root=args.phase2_root,
        output_root=args.output_root,
        random_seed=args.random_seed,
        sequence_length=args.sequence_length,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        batch_size=args.batch_size,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        extra_finetune_epochs=args.extra_finetune_epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        trigger_threshold=args.trigger_threshold,
        reference_drift_node_id=args.reference_drift_node_id,
    )
    results = run_phase3(config)
    acceptance = results["acceptance_report"]
    print("Phase 3 completed.")
    print(f"Required checks passed: {acceptance['required_checks_passed']}")
    print(f"Phase 3 closed: {acceptance['phase3_closed']}")
    try:
        display_root = config.output_root.relative_to(REPO_ROOT)
    except ValueError:
        display_root = config.output_root
    print(f"Outputs written to: {display_root}")


if __name__ == "__main__":
    main()
