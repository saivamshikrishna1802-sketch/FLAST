from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf, adfuller

from .config import Phase1Config

WINDOW_COLORS = {
    "pre_2011_2012": "#1b9e77",
    "drift_2013": "#d95f02",
    "post_2014": "#7570b3",
}


def run_adf_analysis(cleaned_window_frame: pd.DataFrame, config: Phase1Config, figure_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for (window_key, window_label, household_id, tariff), frame in cleaned_window_frame.groupby(
        ["window_key", "window_label", "LCLid", "stdorToU"],
        sort=False,
    ):
        values = frame.sort_values("timestamp")["energy_kwh"].to_numpy(dtype=np.float64)
        if len(values) < 48 or np.nanstd(values) <= 1e-8:
            p_value = np.nan
            statistic = np.nan
            adf_failed = True
        else:
            try:
                statistic, p_value, *_ = adfuller(values, autolag="AIC")
                adf_failed = bool(p_value > config.adf_alpha)
            except ValueError:
                p_value = np.nan
                statistic = np.nan
                adf_failed = True

        records.append(
            {
                "window_key": window_key,
                "window_label": window_label,
                "LCLid": household_id,
                "stdorToU": tariff,
                "adf_statistic": statistic,
                "p_value": p_value,
                "failed_adf": adf_failed,
            }
        )

    results = pd.DataFrame(records)
    grouped = (
        results.groupby(["window_key", "window_label", "stdorToU"], as_index=False)["failed_adf"]
        .mean()
        .rename(columns={"failed_adf": "failure_rate"})
    )
    overall = (
        results.groupby(["window_key", "window_label"], as_index=False)["failed_adf"]
        .mean()
        .assign(stdorToU="Overall")
        .rename(columns={"failed_adf": "failure_rate"})
    )
    summary = (
        pd.concat([grouped, overall], ignore_index=True)
        .assign(failure_rate_pct=lambda frame: frame["failure_rate"] * 100.0)
        .sort_values(["window_key", "stdorToU"])
        .reset_index(drop=True)
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_order = [window.key for window in config.analysis_windows]
    hue_order = ["Overall", "Std", "ToU"]
    sns.barplot(
        data=summary,
        x="window_key",
        y="failure_rate_pct",
        hue="stdorToU",
        order=plot_order,
        hue_order=hue_order,
        palette=["#4c566a", "#5e81ac", "#bf616a"],
        ax=ax,
    )
    ax.set_xlabel("Temporal window")
    ax.set_ylabel("Households failing ADF (%)")
    ax.set_title("ADF failure rate across temporal windows")
    ax.set_xticks(ax.get_xticks(), labels=[window.label for window in config.analysis_windows], rotation=12, ha="right")
    ax.legend(title="Tariff group")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return results, summary


def run_distribution_analysis(
    cleaned_window_frame: pd.DataFrame,
    config: Phase1Config,
    figure_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        cleaned_window_frame.groupby(["window_key", "window_label", "stdorToU"], as_index=False)
        .agg(
            mean_kwh=("energy_kwh", "mean"),
            std_kwh=("energy_kwh", "std"),
            median_kwh=("energy_kwh", "median"),
            min_kwh=("energy_kwh", "min"),
            max_kwh=("energy_kwh", "max"),
            sample_count=("energy_kwh", "count"),
        )
    )
    rng = np.random.default_rng(config.random_seed)

    shift_records: list[dict[str, object]] = []
    for tariff in ("Std", "ToU"):
        pre_values = cleaned_window_frame[
            (cleaned_window_frame["window_key"] == "pre_2011_2012") & (cleaned_window_frame["stdorToU"] == tariff)
        ]["energy_kwh"].to_numpy(dtype=np.float64)
        drift_values = cleaned_window_frame[
            (cleaned_window_frame["window_key"] == "drift_2013") & (cleaned_window_frame["stdorToU"] == tariff)
        ]["energy_kwh"].to_numpy(dtype=np.float64)
        ks_result = ks_2samp(pre_values, drift_values, alternative="two-sided", method="asymp")
        pre_mean = float(np.mean(pre_values))
        drift_mean = float(np.mean(drift_values))
        pre_std = float(np.std(pre_values))
        drift_std = float(np.std(drift_values))
        shift_records.append(
            {
                "stdorToU": tariff,
                "pre_mean_kwh": pre_mean,
                "drift_mean_kwh": drift_mean,
                "mean_shift_pct": abs(drift_mean - pre_mean) / max(abs(pre_mean), 1e-6) * 100.0,
                "pre_std_kwh": pre_std,
                "drift_std_kwh": drift_std,
                "std_shift_pct": abs(drift_std - pre_std) / max(abs(pre_std), 1e-6) * 100.0,
                "ks_statistic": float(ks_result.statistic),
                "ks_p_value": float(ks_result.pvalue),
            }
        )
    shift_summary = pd.DataFrame(shift_records)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, tariff in zip(axes, ("Std", "ToU"), strict=True):
        for window in config.analysis_windows:
            values = cleaned_window_frame[
                (cleaned_window_frame["window_key"] == window.key) & (cleaned_window_frame["stdorToU"] == tariff)
            ]["energy_kwh"].to_numpy(dtype=np.float64)
            if len(values) > config.max_kde_points:
                values = rng.choice(values, size=config.max_kde_points, replace=False)
            sns.kdeplot(
                values,
                ax=axis,
                label=window.label,
                color=WINDOW_COLORS[window.key],
                linewidth=2,
                bw_adjust=0.9,
                clip=(0, None),
            )
        axis.set_title(f"{tariff} tariff load distribution by window")
        axis.set_ylabel("Density")
        axis.legend()
    axes[-1].set_xlabel("Hourly load (kWh)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return summary, shift_summary


def run_acf_analysis(cleaned_window_frame: pd.DataFrame, config: Phase1Config, figure_path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for tariff in ("Std", "ToU"):
        for window in config.analysis_windows:
            curves: list[np.ndarray] = []
            tariff_window = cleaned_window_frame[
                (cleaned_window_frame["stdorToU"] == tariff) & (cleaned_window_frame["window_key"] == window.key)
            ]
            for _, household_frame in tariff_window.groupby("LCLid", sort=False):
                values = household_frame.sort_values("timestamp")["energy_kwh"].to_numpy(dtype=np.float64)
                if len(values) < config.acf_lags + 2 or np.std(values) <= 1e-8:
                    continue
                curves.append(acf(values, nlags=config.acf_lags, fft=True))

            if not curves:
                continue

            curve_matrix = np.vstack(curves)
            mean_curve = curve_matrix.mean(axis=0)
            std_curve = curve_matrix.std(axis=0)
            for lag, (mean_value, std_value) in enumerate(zip(mean_curve, std_curve, strict=True)):
                records.append(
                    {
                        "stdorToU": tariff,
                        "window_key": window.key,
                        "window_label": window.label,
                        "lag_hours": lag,
                        "mean_acf": mean_value,
                        "std_acf": std_value,
                        "household_count": curve_matrix.shape[0],
                    }
                )

    summary = pd.DataFrame(records)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, tariff in zip(axes, ("Std", "ToU"), strict=True):
        tariff_summary = summary[summary["stdorToU"] == tariff]
        for window in config.analysis_windows:
            curve = tariff_summary[tariff_summary["window_key"] == window.key]
            if curve.empty:
                continue
            axis.plot(
                curve["lag_hours"],
                curve["mean_acf"],
                label=window.label,
                color=WINDOW_COLORS[window.key],
                linewidth=2,
            )
        axis.set_title(f"{tariff} tariff autocorrelation")
        axis.set_xlabel("Lag (hours)")
        axis.legend()
    axes[0].set_ylabel("Average ACF")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return summary


def plot_node_degradation(
    node_summary: pd.DataFrame,
    degradation_threshold_pct: float,
    figure_path: Path,
) -> None:
    plot_frame = node_summary.sort_values("rmse_increase_2013_vs_2012_pct", ascending=False).copy()
    plot_frame["node_label"] = plot_frame["node_id"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(
        data=plot_frame,
        x="node_label",
        y="rmse_increase_2013_vs_2012_pct",
        hue="node_type",
        palette={"Std": "#5e81ac", "ToU": "#bf616a"},
        dodge=False,
        ax=ax,
    )
    ax.axhline(
        degradation_threshold_pct,
        color="#2e3440",
        linestyle="--",
        linewidth=1.5,
        label=f"{degradation_threshold_pct:.0f}% threshold",
    )
    ax.set_xlabel("Federated node")
    ax.set_ylabel("RMSE increase in 2013 vs 2012 Q4 (%)")
    ax.set_title("Per-node baseline degradation under tariff drift")
    ax.set_xticks(ax.get_xticks(), labels=plot_frame["node_label"], rotation=20, ha="right")
    ax.legend(title="Node type")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
