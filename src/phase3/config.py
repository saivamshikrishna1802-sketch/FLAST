from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeWindow:
    key: str
    label: str
    start: str
    end: str


@dataclass(frozen=True)
class Phase3Config:
    phase1_root: Path
    phase2_root: Path
    output_root: Path
    random_seed: int = 17
    sequence_length: int = 48
    train_stride: int = 3
    eval_stride: int = 1
    batch_size: int = 512
    rounds: int = 10
    local_epochs: int = 1
    extra_finetune_epochs: int = 1
    learning_rate: float = 1e-3
    hidden_size: int = 64
    dropout: float = 0.2
    trigger_threshold: float = 0.05
    drift_weight_epsilon: float = 1e-6
    drifting_node_gain_target_pct: float = 3.0
    reference_node_gain_target_pct: float = 2.0
    stable_node_harm_tolerance_pct: float = 2.0
    validation_start: str = "2012-09-01 00:00:00"
    validation_end: str = "2012-10-01 00:00:00"
    reference_drift_node_id: str = "node_tou_6"

    @property
    def phase1_reports_dir(self) -> Path:
        return self.phase1_root / "reports"

    @property
    def phase1_tables_dir(self) -> Path:
        return self.phase1_root / "tables"

    @property
    def phase1_data_dir(self) -> Path:
        return self.phase1_root / "data"

    @property
    def phase2_reports_dir(self) -> Path:
        return self.phase2_root / "reports"

    @property
    def phase2_tables_dir(self) -> Path:
        return self.phase2_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    def load_phase1_config(self) -> dict[str, object]:
        return json.loads((self.phase1_reports_dir / "config.json").read_text(encoding="utf-8"))

    @property
    def baseline_splits(self) -> tuple[TimeWindow, ...]:
        config = self.load_phase1_config()
        return tuple(TimeWindow(**window) for window in config["baseline_splits"])

    @property
    def local_train_window(self) -> TimeWindow:
        baseline_train = self.baseline_splits[0]
        return TimeWindow(
            key="local_train",
            label="Local train: 2011-11 to 2012-08",
            start=baseline_train.start,
            end=self.validation_start,
        )

    @property
    def local_validation_window(self) -> TimeWindow:
        return TimeWindow(
            key="local_validation",
            label="Local validation: 2012-09",
            start=self.validation_start,
            end=self.validation_end,
        )

    @property
    def pre_drift_window(self) -> TimeWindow:
        return TimeWindow(
            key="pre_2011_2012",
            label="2011-2012 pre-drift",
            start="2011-11-23 00:00:00",
            end="2013-01-01 00:00:00",
        )

    @property
    def drift_window(self) -> TimeWindow:
        return TimeWindow(
            key="drift_2013",
            label="2013 tariff-shift period",
            start="2013-01-01 00:00:00",
            end="2014-01-01 00:00:00",
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["phase1_root"] = str(self.phase1_root)
        data["phase2_root"] = str(self.phase2_root)
        data["output_root"] = str(self.output_root)
        data["baseline_splits"] = [asdict(window) for window in self.baseline_splits]
        data["local_train_window"] = asdict(self.local_train_window)
        data["local_validation_window"] = asdict(self.local_validation_window)
        data["pre_drift_window"] = asdict(self.pre_drift_window)
        data["drift_window"] = asdict(self.drift_window)
        return data
