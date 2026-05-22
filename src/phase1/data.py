from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Phase1Config, TimeWindow

HALF_HOUR_COLUMNS = [f"hh_{index}" for index in range(48)]


@dataclass(frozen=True)
class HourlyLoadResult:
    hourly_frame: pd.DataFrame
    coverage_frame: pd.DataFrame
    quality_summary: pd.DataFrame


def load_household_info(info_path: Path) -> pd.DataFrame:
    info = pd.read_csv(info_path)
    info["stdorToU"] = info["stdorToU"].str.strip()
    info["file"] = info["file"].str.strip()
    return info


def build_coverage_frame(config: Phase1Config) -> pd.DataFrame:
    windows = {
        "pre_2011_2012": ("2011-11-23", "2013-01-01"),
        "drift_2013": ("2013-01-01", "2014-01-01"),
        "post_2014": ("2014-01-01", "2014-03-01"),
    }
    coverage: dict[str, dict[str, int]] = {}

    with config.daily_summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            household_id = row["LCLid"]
            day = row["day"]
            household_coverage = coverage.setdefault(
                household_id,
                {
                    "pre_2011_2012_days": 0,
                    "drift_2013_days": 0,
                    "post_2014_days": 0,
                },
            )

            for key, (start, end) in windows.items():
                if start <= day < end:
                    household_coverage[f"{key}_days"] += 1

    return (
        pd.DataFrame.from_dict(coverage, orient="index")
        .rename_axis("LCLid")
        .reset_index()
        .sort_values("LCLid")
        .reset_index(drop=True)
    )


def build_shift_proxy_frame(config: Phase1Config) -> pd.DataFrame:
    daily = pd.read_csv(config.daily_summary_path, usecols=["LCLid", "day", "energy_mean"])
    daily["day"] = pd.to_datetime(daily["day"])

    pre = (
        daily[(daily["day"] >= "2011-11-23") & (daily["day"] < "2013-01-01")]
        .groupby("LCLid", as_index=False)["energy_mean"]
        .agg(pre_mean="mean", pre_std="std")
    )
    drift = (
        daily[(daily["day"] >= "2013-01-01") & (daily["day"] < "2014-01-01")]
        .groupby("LCLid", as_index=False)["energy_mean"]
        .agg(drift_mean="mean", drift_std="std")
    )

    shift_frame = pre.merge(drift, on="LCLid", how="inner")
    shift_frame["mean_shift_pct"] = (
        (shift_frame["drift_mean"] - shift_frame["pre_mean"]).abs() / shift_frame["pre_mean"].clip(lower=1e-6) * 100.0
    )
    shift_frame["std_shift_pct"] = (
        (shift_frame["drift_std"] - shift_frame["pre_std"]).abs() / shift_frame["pre_std"].clip(lower=1e-6) * 100.0
    )
    shift_frame["shift_score"] = shift_frame["mean_shift_pct"] + shift_frame["std_shift_pct"]
    return shift_frame


def select_households(info_frame: pd.DataFrame, coverage_frame: pd.DataFrame, config: Phase1Config) -> pd.DataFrame:
    sampled = info_frame.merge(coverage_frame, on="LCLid", how="inner")
    eligible = sampled[
        (sampled["pre_2011_2012_days"] >= config.min_pre_days)
        & (sampled["drift_2013_days"] >= config.min_drift_days)
        & (sampled["post_2014_days"] >= config.min_post_days)
    ].copy()
    eligible = eligible.merge(build_shift_proxy_frame(config), on="LCLid", how="left")

    rng = np.random.default_rng(config.random_seed)
    selected_parts: list[pd.DataFrame] = []

    for tariff, sample_size in (("Std", config.std_households), ("ToU", config.tou_households)):
        tariff_frame = eligible[eligible["stdorToU"] == tariff].copy()
        if len(tariff_frame) < sample_size:
            raise ValueError(
                f"Requested {sample_size} households for {tariff}, but only {len(tariff_frame)} are eligible."
            )
        if config.sampling_strategy == "random":
            chosen_index = rng.choice(tariff_frame.index.to_numpy(), size=sample_size, replace=False)
            selected_parts.append(tariff_frame.loc[chosen_index])
            continue

        if config.sampling_strategy == "drift_targeted":
            ascending = tariff == "Std"
            ranked = tariff_frame.sort_values(
                ["shift_score", "mean_shift_pct", "std_shift_pct", "LCLid"],
                ascending=[ascending, ascending, ascending, True],
            )
            selected_parts.append(ranked.head(sample_size))
            continue

        raise ValueError(
            f"Unknown sampling strategy '{config.sampling_strategy}'. Use 'random' or 'drift_targeted'."
        )

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values(["stdorToU", "LCLid"])
        .reset_index(drop=True)
    )
    return selected


def _block_to_hourly(block_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    numeric_halfhours = block_frame[HALF_HOUR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric_halfhours.to_numpy(dtype=np.float32)

    raw_halfhour_missing = int(np.isnan(values).sum())
    raw_halfhour_total = int(values.size)

    pair_values = values.reshape(len(block_frame), 24, 2)
    hourly_missing_mask = np.isnan(pair_values).any(axis=2)
    hourly_values = pair_values.sum(axis=2, dtype=np.float32)
    hourly_values[hourly_missing_mask] = np.nan

    base_days = pd.to_datetime(block_frame["day"]).to_numpy(dtype="datetime64[h]")[:, None]
    hour_offsets = np.arange(24).astype("timedelta64[h]")
    timestamps = (base_days + hour_offsets).reshape(-1)

    hourly_frame = pd.DataFrame(
        {
            "LCLid": np.repeat(block_frame["LCLid"].to_numpy(), 24),
            "timestamp": pd.to_datetime(timestamps),
            "energy_kwh": hourly_values.reshape(-1),
        }
    )

    quality = {
        "raw_halfhour_missing": raw_halfhour_missing,
        "raw_halfhour_total": raw_halfhour_total,
        "raw_halfhour_missing_rate": raw_halfhour_missing / raw_halfhour_total if raw_halfhour_total else 0.0,
        "hourly_missing_before_fill": int(hourly_missing_mask.sum()),
        "hourly_total": int(hourly_values.size),
        "hourly_missing_rate_before_fill": float(hourly_missing_mask.mean()) if hourly_values.size else 0.0,
    }
    return hourly_frame, quality


def load_hourly_subset(selected_households: pd.DataFrame, config: Phase1Config) -> HourlyLoadResult:
    selected_lookup = selected_households[["LCLid", "stdorToU", "Acorn_grouped", "file"]].copy()
    selected_blocks = selected_lookup.groupby("file")["LCLid"].apply(list)

    hourly_parts: list[pd.DataFrame] = []
    quality_rows: list[dict[str, float | str]] = []

    for block_name, household_ids in selected_blocks.items():
        block_path = config.hhblock_root / f"{block_name}.csv"
        block_frame = pd.read_csv(block_path, usecols=["LCLid", "day", *HALF_HOUR_COLUMNS])
        block_frame = block_frame[block_frame["LCLid"].isin(household_ids)].copy()
        if block_frame.empty:
            continue

        hourly_frame, quality = _block_to_hourly(block_frame)
        hourly_parts.append(hourly_frame)
        quality_rows.append({"block": block_name, "households_loaded": len(household_ids), **quality})

    if not hourly_parts:
        raise ValueError("No hourly data was loaded for the sampled households.")

    hourly_frame = (
        pd.concat(hourly_parts, ignore_index=True)
        .merge(selected_lookup, on="LCLid", how="left")
        .sort_values(["LCLid", "timestamp"])
        .reset_index(drop=True)
    )

    coverage_frame = (
        hourly_frame.groupby(["LCLid", "stdorToU"], as_index=False)
        .agg(
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            observed_hours=("energy_kwh", "count"),
            missing_hours=("energy_kwh", lambda values: int(values.isna().sum())),
        )
        .sort_values(["stdorToU", "LCLid"])
        .reset_index(drop=True)
    )

    quality_summary = pd.DataFrame(quality_rows).sort_values("block").reset_index(drop=True)
    return HourlyLoadResult(hourly_frame=hourly_frame, coverage_frame=coverage_frame, quality_summary=quality_summary)


def build_clean_window_frame(
    hourly_frame: pd.DataFrame,
    selected_households: pd.DataFrame,
    window: TimeWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(window.start)
    end = pd.Timestamp(window.end)
    full_index = pd.date_range(start=start, end=end, inclusive="left", freq="h")

    series_lookup = {
        household_id: household_frame.set_index("timestamp")["energy_kwh"].sort_index()
        for household_id, household_frame in hourly_frame.groupby("LCLid", sort=False)
    }

    cleaned_rows: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []

    for household in selected_households.itertuples(index=False):
        source_series = series_lookup[household.LCLid]
        window_series = source_series.reindex(full_index)
        observed_hours = int(window_series.notna().sum())
        raw_missing_rate = float(window_series.isna().mean()) if len(window_series) else 0.0

        if observed_hours == 0:
            filled_series = pd.Series(0.0, index=full_index)
        else:
            filled_series = window_series.interpolate(method="time", limit_direction="both")
            if filled_series.isna().any():
                fill_value = float(filled_series.median()) if pd.notna(filled_series.median()) else 0.0
                filled_series = filled_series.fillna(fill_value)

        cleaned_rows.append(
            pd.DataFrame(
                {
                    "LCLid": household.LCLid,
                    "timestamp": full_index,
                    "energy_kwh": filled_series.to_numpy(dtype=np.float32),
                    "stdorToU": household.stdorToU,
                    "window_key": window.key,
                    "window_label": window.label,
                }
            )
        )
        quality_rows.append(
            {
                "LCLid": household.LCLid,
                "stdorToU": household.stdorToU,
                "window_key": window.key,
                "window_label": window.label,
                "observed_hours_before_fill": observed_hours,
                "expected_hours": len(full_index),
                "raw_missing_rate": raw_missing_rate,
                "filled_missing_rate": 0.0,
            }
        )

    return pd.concat(cleaned_rows, ignore_index=True), pd.DataFrame(quality_rows)
