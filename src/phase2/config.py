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
class Phase2Config:
    phase1_root: Path
    output_root: Path
    random_seed: int = 17
    sequence_length: int = 48
    train_stride: int = 3
    eval_stride: int = 1
    batch_size: int = 512
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 64
    dropout: float = 0.2
    improvement_target_pct: float = 3.0
    attention_heatmap_sequences: int = 48
    drift_node_id: str | None = None
    stable_node_id: str | None = None

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

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["phase1_root"] = str(self.phase1_root)
        data["output_root"] = str(self.output_root)
        data["baseline_splits"] = [asdict(window) for window in self.baseline_splits]
        return data
