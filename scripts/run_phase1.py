from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase1 import Phase1Config, run_phase1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 dataset validation for the conference paper.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "Main_dataset",
        help="Path to the London smart-meter dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase1",
        help="Directory where figures, tables, and reports will be written.",
    )
    parser.add_argument("--std-households", type=int, default=60, help="Number of flat-rate households to sample.")
    parser.add_argument("--tou-households", type=int, default=140, help="Number of ToU households to sample.")
    parser.add_argument("--random-seed", type=int, default=17, help="Random seed used for sampling and training.")
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "drift_targeted"],
        default="drift_targeted",
        help="Household selection strategy. 'drift_targeted' ranks ToU households by shift strength and Std households by stability.",
    )
    parser.add_argument(
        "--households-per-node",
        type=int,
        default=20,
        help="Number of households assigned to each simulated federated node.",
    )
    parser.add_argument(
        "--degradation-threshold-pct",
        type=float,
        default=10.0,
        help="RMSE degradation threshold used in the node-level acceptance check.",
    )
    parser.add_argument("--baseline-epochs", type=int, default=20, help="Number of epochs for the simple LSTM baseline.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for the simple LSTM baseline.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Lookback window length in hours for the LSTM.")
    parser.add_argument("--train-stride", type=int, default=3, help="Stride for train sliding windows.")
    parser.add_argument("--eval-stride", type=int, default=1, help="Stride for evaluation sliding windows.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for the LSTM baseline.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = Phase1Config(
        data_root=args.data_root,
        output_root=args.output_root,
        random_seed=args.random_seed,
        std_households=args.std_households,
        tou_households=args.tou_households,
        sampling_strategy=args.sampling_strategy,
        households_per_node=args.households_per_node,
        degradation_threshold_pct=args.degradation_threshold_pct,
        sequence_length=args.sequence_length,
        baseline_epochs=args.baseline_epochs,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        hidden_size=args.hidden_size,
    )
    if config.total_households <= 0:
        parser.error("At least one household must be sampled.")
    if config.std_households % config.households_per_node != 0:
        parser.error("--std-households must be divisible by --households-per-node.")
    if config.tou_households % config.households_per_node != 0:
        parser.error("--tou-households must be divisible by --households-per-node.")

    results = run_phase1(config)
    acceptance = results["acceptance_report"]
    print(f"Phase 1 completed for {config.total_households} households.")
    print(f"All acceptance checks passed: {acceptance['all_checks_passed']}")
    try:
        display_root = config.output_root.relative_to(REPO_ROOT)
    except ValueError:
        display_root = config.output_root
    print(f"Outputs written to: {display_root}")


if __name__ == "__main__":
    main()
