from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeWindow:
    key: str
    label: str
    start: str
    end: str


@dataclass(frozen=True)
class Phase1Config:
    data_root: Path
    output_root: Path
    random_seed: int = 17
    std_households: int = 60
    tou_households: int = 140
    sampling_strategy: str = "drift_targeted"
    households_per_node: int = 20
    degradation_threshold_pct: float = 10.0
    min_pre_days: int = 60
    min_drift_days: int = 300
    min_post_days: int = 45
    adf_alpha: float = 0.05
    acf_lags: int = 72
    max_kde_points: int = 50_000
    sequence_length: int = 48
    train_stride: int = 3
    eval_stride: int = 1
    batch_size: int = 512
    baseline_epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 64

    @property
    def total_households(self) -> int:
        return self.std_households + self.tou_households

    @property
    def std_node_count(self) -> int:
        return self.std_households // self.households_per_node

    @property
    def tou_node_count(self) -> int:
        return self.tou_households // self.households_per_node

    @property
    def total_node_count(self) -> int:
        return self.std_node_count + self.tou_node_count

    @property
    def info_path(self) -> Path:
        return self.data_root / "informations_households.csv"

    @property
    def daily_summary_path(self) -> Path:
        return self.data_root / "daily_dataset.csv"

    @property
    def hhblock_root(self) -> Path:
        return self.data_root / "hhblock_dataset" / "hhblock_dataset"

    @property
    def data_dir(self) -> Path:
        return self.output_root / "data"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def analysis_windows(self) -> tuple[TimeWindow, ...]:
        return (
            TimeWindow(
                key="pre_2011_2012",
                label="2011-2012 pre-drift",
                start="2011-11-23 00:00:00",
                end="2013-01-01 00:00:00",
            ),
            TimeWindow(
                key="drift_2013",
                label="2013 tariff-shift period",
                start="2013-01-01 00:00:00",
                end="2014-01-01 00:00:00",
            ),
            TimeWindow(
                key="post_2014",
                label="2014 post-drift (Jan-Feb only)",
                start="2014-01-01 00:00:00",
                end="2014-03-01 00:00:00",
            ),
        )

    @property
    def baseline_splits(self) -> tuple[TimeWindow, ...]:
        return (
            TimeWindow(
                key="train_2011_to_2012q3",
                label="Train: 2011-11 to 2012-09",
                start="2011-11-23 00:00:00",
                end="2012-10-01 00:00:00",
            ),
            TimeWindow(
                key="test_2012q4",
                label="Test: 2012 Q4",
                start="2012-10-01 00:00:00",
                end="2013-01-01 00:00:00",
            ),
            TimeWindow(
                key="test_2013",
                label="Test: 2013 drift year",
                start="2013-01-01 00:00:00",
                end="2014-01-01 00:00:00",
            ),
            TimeWindow(
                key="test_2014",
                label="Test: 2014 post-drift",
                start="2014-01-01 00:00:00",
                end="2014-03-01 00:00:00",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["data_root"] = str(self.data_root)
        data["output_root"] = str(self.output_root)
        data["analysis_windows"] = [asdict(window) for window in self.analysis_windows]
        data["baseline_splits"] = [asdict(window) for window in self.baseline_splits]
        data["total_households"] = self.total_households
        data["std_node_count"] = self.std_node_count
        data["tou_node_count"] = self.tou_node_count
        data["total_node_count"] = self.total_node_count
        return data
