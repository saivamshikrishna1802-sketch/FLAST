from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2 import Phase2Config, run_phase2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 centralized Attention-LSTM benchmark.")
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase1",
        help="Frozen Phase 1 output directory used as the input artifact set.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase2",
        help="Directory where Phase 2 figures, tables, and reports will be written.",
    )
    parser.add_argument("--random-seed", type=int, default=17, help="Random seed for training reproducibility.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Lookback window length in hours.")
    parser.add_argument("--train-stride", type=int, default=3, help="Stride for training windows.")
    parser.add_argument("--eval-stride", type=int, default=1, help="Stride for evaluation windows.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for centralized model training.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for both centralized models.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for both centralized models.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used in the regression head.")
    parser.add_argument(
        "--improvement-target-pct",
        type=float,
        default=3.0,
        help="Required RMSE improvement target for Attention-LSTM on the 2012 Q4 split.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = Phase2Config(
        phase1_root=args.phase1_root,
        output_root=args.output_root,
        random_seed=args.random_seed,
        sequence_length=args.sequence_length,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        improvement_target_pct=args.improvement_target_pct,
    )
    results = run_phase2(config)
    acceptance = results["acceptance_report"]
    print("Phase 2 completed.")
    print(f"Required checks passed: {acceptance['required_checks_passed']}")
    print(f"Phase 2 closed: {acceptance['phase2_closed']}")
    try:
        display_root = config.output_root.relative_to(REPO_ROOT)
    except ValueError:
        display_root = config.output_root
    print(f"Outputs written to: {display_root}")


if __name__ == "__main__":
    main()
