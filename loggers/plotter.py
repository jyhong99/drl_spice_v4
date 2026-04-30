"""Plotting helpers for structured experiment logs.

This module provides utilities for generating static matplotlib plots from
structured CSV logs saved during reinforcement-learning experiments. It can
plot full metric grids for each structured dataset and standalone highlighted
metrics configured through default, file-based, or in-memory plot settings.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loggers.reader import load_structured_csv


DEFAULT_PLOT_CONFIG = {
    "window": 50,
    "include": {},
    "exclude": {},
    "highlights": {
        "eval": ["max_ep_ret", "mean_ep_ret", "max_ep_len", "mean_ep_len"],
        "learner": [
            "actor_loss",
            "critic_loss",
            "entropy",
            "alpha",
            "approx_kl",
            "actor_grad_norm",
            "critic_grad_norm",
            "mean_target_q",
        ],
        "scheduler": [
            "effective_utd",
            "buffer_fill_ratio",
            "buffer_size",
            "chunk_updates",
        ],
        "rollout_steps": [
            "fom",
            "reward",
            "viol",
            "reward_perf",
            "reward_viol",
            "info_is_feasible",
            "info_is_non_convergent",
            "info_is_non_stable",
        ],
        "simulation_profiles": [
            "simulation_total_ms",
            "simulation_parse_ms",
            "simulation_cache_hit",
        ],
        "reset_events": [
            "reset_reset_profile_attempts",
            "reset_reset_profile_total_ms",
        ],
    },
}


def _save_figure(fig, path: Path) -> None:
    """Persist a matplotlib figure to disk and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    path : pathlib.Path
        Output image path. Parent directories are created automatically.

    Returns
    -------
    None
        The figure is saved to ``path`` and then closed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas series to numeric values when possible.

    Parameters
    ----------
    series : pandas.Series
        Input series to convert.

    Returns
    -------
    pandas.Series
        Numeric series. Values that cannot be converted are replaced with
        ``NaN``.
    """

    return pd.to_numeric(series, errors="coerce")


def _is_binary_series(series: pd.Series) -> bool:
    """Check whether a series contains only binary values.

    Parameters
    ----------
    series : pandas.Series
        Input series to inspect.

    Returns
    -------
    bool
        ``True`` if the non-missing numeric values are all contained in
        ``{0, 1}``; otherwise ``False``.
    """

    numeric = _coerce_numeric(series).dropna()

    if numeric.empty:
        return False

    return set(numeric.unique()).issubset({0, 1})


def _is_cumulative_metric(metric: str, series: pd.Series) -> bool:
    """Identify binary rollout flags that should be plotted cumulatively.

    Parameters
    ----------
    metric : str
        Metric column name.
    series : pandas.Series
        Metric values associated with ``metric``.

    Returns
    -------
    bool
        ``True`` if the series is binary and the metric name corresponds to an
        event-like validity flag; otherwise ``False``.
    """

    if not _is_binary_series(series):
        return False

    return metric.endswith(
        (
            "is_feasible",
            "is_non_convergent",
            "is_non_stable",
            "is_invalid",
        )
    )


def _merge_plot_config(base_config, override_config):
    """Recursively merge two plot-configuration dictionaries.

    Values from ``override_config`` take precedence over values from
    ``base_config``. Nested dictionaries are merged recursively.

    Parameters
    ----------
    base_config : dict[str, object]
        Base plot configuration.
    override_config : dict[str, object] or None
        Override configuration. If ``None``, ``base_config`` is effectively
        copied unchanged.

    Returns
    -------
    dict[str, object]
        Merged plot configuration.
    """

    merged = deepcopy(base_config)

    for key, value in (override_config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_plot_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def _load_plot_config(project_dir: Path, plot_config=None):
    """Load plot configuration from defaults, file, and call overrides.

    Configuration precedence is:

    1. ``DEFAULT_PLOT_CONFIG``
    2. ``plot_config.json`` under ``project_dir``
    3. explicit ``plot_config`` argument

    Parameters
    ----------
    project_dir : pathlib.Path
        Project log directory.
    plot_config : dict[str, object] or None, optional
        In-memory configuration overrides. The default is ``None``.

    Returns
    -------
    dict[str, object]
        Final merged plot configuration.
    """

    config = deepcopy(DEFAULT_PLOT_CONFIG)
    config_path = project_dir / "plot_config.json"

    if config_path.exists() and config_path.stat().st_size > 0:
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
        config = _merge_plot_config(config, file_config)

    if plot_config is not None:
        config = _merge_plot_config(config, plot_config)

    return config


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    """Infer the x-axis column used for plotting a structured dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Structured log dataset.

    Returns
    -------
    str or None
        Name of the inferred time column, or ``None`` if no known time column
        is present.
    """

    candidates = [
        "learner_step",
        "timesteps",
        "chunk_time_steps",
        "time_steps",
        "agent_timesteps",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def _numeric_metric_columns(
    df: pd.DataFrame,
    time_column: Optional[str],
) -> list[str]:
    """Return plottable numeric metric columns for a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Structured log dataset.
    time_column : str or None
        Column selected for the x-axis. This column is excluded from the
        returned metric columns.

    Returns
    -------
    list[str]
        Names of columns that contain at least one numeric value and are not
        excluded metadata or time columns.
    """

    excluded = {
        time_column,
        "number_of_eps",
        "runner_name",
        "runner_policy_version",
        "runner_env_steps_total",
        "runner_learner_updates_total",
        "chunk_time_steps",
        "agent_timesteps",
        "time_steps",
        "timesteps",
        "learner_step",
    }

    columns = []

    for column in df.columns:
        if column in excluded:
            continue

        numeric = _coerce_numeric(df[column])
        if numeric.notna().any():
            columns.append(column)

    return columns


def _select_metric_columns(
    df: pd.DataFrame,
    dataset_name: str,
    time_column: Optional[str],
    config,
) -> list[str]:
    """Apply include and exclude plot configuration to metric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Structured log dataset.
    dataset_name : str
        Name used to look up dataset-specific include and exclude settings.
    time_column : str or None
        Column selected for the x-axis.
    config : dict[str, object]
        Plot configuration dictionary.

    Returns
    -------
    list[str]
        Metric columns selected for plotting.
    """

    columns = _numeric_metric_columns(df, time_column)

    include_map = config.get("include", {})
    exclude_map = config.get("exclude", {})

    include = include_map.get(dataset_name)
    if include:
        columns = [column for column in columns if column in include]

    exclude = set(exclude_map.get(dataset_name, []))
    if exclude:
        columns = [column for column in columns if column not in exclude]

    return columns


def _rolling_window_size(length: int, default: int = 50) -> int:
    """Choose a rolling-window size scaled to dataset length.

    Parameters
    ----------
    length : int
        Number of valid samples in the metric series.
    default : int, optional
        Maximum preferred rolling-window size. The default is ``50``.

    Returns
    -------
    int
        Rolling-window size. The value is at least ``1`` and is reduced for
        short datasets.
    """

    if length <= 1:
        return 1

    return max(2, min(default, length // 5 if length >= 10 else length))


def _plot_metric_grid(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    config,
    max_timesteps: Optional[int] = None,
) -> None:
    """Render a two-column plot grid for selected metrics in a dataset.

    The left column shows raw metric values or cumulative event counts. The
    right column shows rolling mean and rolling standard deviation. Empty
    datasets, datasets without a recognized time column, and datasets without
    selected numeric metrics are skipped.

    Parameters
    ----------
    df : pandas.DataFrame
        Structured log dataset to plot.
    dataset_name : str
        Dataset name used in output filenames and plot titles.
    output_dir : pathlib.Path
        Directory where the generated metric-grid figure is saved.
    config : dict[str, object]
        Plot configuration dictionary.
    max_timesteps : int or None, optional
        Optional upper bound applied to the inferred time column. The default
        is ``None``.

    Returns
    -------
    None
        A figure is saved under ``output_dir`` if plottable metrics exist.
    """

    if df.empty:
        return

    time_column = _find_time_column(df)
    if time_column is None:
        return

    working_df = df.copy()
    working_df[time_column] = _coerce_numeric(working_df[time_column])
    working_df = working_df.dropna(subset=[time_column]).sort_values(time_column)

    if max_timesteps is not None:
        working_df = working_df[working_df[time_column] <= max_timesteps]

    if working_df.empty:
        return

    metric_columns = _select_metric_columns(
        working_df,
        dataset_name,
        time_column,
        config,
    )

    if not metric_columns:
        return

    ncols = 2
    nrows = len(metric_columns)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(16, max(4 * nrows, 6)),
    )

    if nrows == 1:
        axs = np.array([axs])

    for idx, column in enumerate(metric_columns):
        metric_df = working_df[[time_column, column]].copy()
        metric_df[column] = _coerce_numeric(metric_df[column])
        metric_df = metric_df.dropna(subset=[column])

        if metric_df.empty:
            axs[idx][0].set_visible(False)
            axs[idx][1].set_visible(False)
            continue

        x = metric_df[time_column]
        y = metric_df[column]
        cumulative_metric = _is_cumulative_metric(column, y)

        if cumulative_metric:
            axs[idx][0].plot(x, y.cumsum(), alpha=0.8, linewidth=1.4)
            axs[idx][0].set_title(f"{dataset_name}: {column} cumulative count")
        else:
            axs[idx][0].plot(x, y, alpha=0.7, linewidth=1.2)
            axs[idx][0].set_title(f"{dataset_name}: {column}")

        axs[idx][0].set_xlabel(time_column)
        axs[idx][0].grid(axis="y")

        window = _rolling_window_size(
            len(metric_df),
            default=int(config.get("window", 50)),
        )
        rolling_mean = y.rolling(window=window, min_periods=1).mean()
        rolling_std = y.rolling(window=window, min_periods=1).std().fillna(0.0)

        axs[idx][1].plot(
            x,
            rolling_mean,
            color="orange",
            linewidth=1.4,
        )
        axs[idx][1].fill_between(
            x,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.15,
        )

        if cumulative_metric:
            axs[idx][1].set_title(f"{dataset_name}: {column} rolling event rate")
        else:
            axs[idx][1].set_title(f"{dataset_name}: {column} rolling mean/std")

        axs[idx][1].set_xlabel(time_column)
        axs[idx][1].grid(axis="y")

    _save_figure(fig, output_dir / f"{dataset_name}_metrics.png")


def _plot_single_metric(
    df: pd.DataFrame,
    dataset_name: str,
    metric: str,
    output_dir: Path,
    max_timesteps: Optional[int] = None,
) -> None:
    """Render one highlighted metric as a standalone figure.

    Parameters
    ----------
    df : pandas.DataFrame
        Structured log dataset containing the metric.
    dataset_name : str
        Dataset name used in output filename and plot title.
    metric : str
        Metric column to plot.
    output_dir : pathlib.Path
        Directory where the standalone figure is saved.
    max_timesteps : int or None, optional
        Optional upper bound applied to the inferred time column. The default
        is ``None``.

    Returns
    -------
    None
        A figure is saved under ``output_dir`` if the metric and time column
        are available.
    """

    if df.empty or metric not in df.columns:
        return

    time_column = _find_time_column(df)
    if time_column is None:
        return

    working_df = df[[time_column, metric]].copy()
    working_df[time_column] = _coerce_numeric(working_df[time_column])
    working_df[metric] = _coerce_numeric(working_df[metric])
    working_df = working_df.dropna().sort_values(time_column)

    if max_timesteps is not None:
        working_df = working_df[working_df[time_column] <= max_timesteps]

    if working_df.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if _is_cumulative_metric(metric, working_df[metric]):
        ax.plot(
            working_df[time_column],
            working_df[metric].cumsum(),
            linewidth=1.5,
        )
        ax.set_title(f"{dataset_name}: {metric} cumulative count")
    else:
        ax.plot(
            working_df[time_column],
            working_df[metric],
            linewidth=1.5,
        )
        ax.set_title(f"{dataset_name}: {metric}")

    ax.set_xlabel(time_column)
    ax.grid(axis="y")

    _save_figure(fig, output_dir / f"{dataset_name}_{metric}.png")


def plot_saved_training_logs(
    project_name: str,
    max_timesteps: Optional[int] = None,
    plot_config=None,
) -> None:
    """Generate plot artifacts from structured logs for one project.

    This function loads structured CSV logs for a project, generates metric
    grid plots for each available dataset, and saves standalone highlighted
    plots for configured metrics.

    Parameters
    ----------
    project_name : str
        Experiment project name. Logs are expected under
        ``./log/{project_name}``.
    max_timesteps : int or None, optional
        Optional upper bound applied to each dataset's inferred time column.
        The default is ``None``.
    plot_config : dict[str, object] or None, optional
        In-memory plot-configuration overrides merged after default and
        file-based configuration. The default is ``None``.

    Returns
    -------
    None
        Plot image files are saved under ``./log/{project_name}/plots``.
    """

    project_dir = Path(f"./log/{project_name}")
    structured_dir = project_dir / "structured"

    if not structured_dir.exists():
        print(f"Structured log directory does not exist: {structured_dir}")
        return

    plots_dir = project_dir / "plots"
    config = _load_plot_config(project_dir, plot_config=plot_config)

    datasets = {
        "eval": load_structured_csv(project_name, "eval_metrics"),
        "learner": load_structured_csv(project_name, "learner_metrics"),
        "scheduler": load_structured_csv(project_name, "scheduler_metrics"),
        "rollout_chunks": load_structured_csv(project_name, "rollout_chunks"),
        "rollout_steps": load_structured_csv(project_name, "rollout_steps"),
        "simulation_profiles": load_structured_csv(
            project_name,
            "simulation_profiles",
        ),
        "reset_events": load_structured_csv(project_name, "reset_events"),
    }

    for dataset_name, df in datasets.items():
        _plot_metric_grid(
            df=df,
            dataset_name=dataset_name,
            output_dir=plots_dir,
            config=config,
            max_timesteps=max_timesteps,
        )

    for dataset_name, metrics in config.get("highlights", {}).items():
        df = datasets.get(dataset_name, pd.DataFrame())

        for metric in metrics:
            _plot_single_metric(
                df=df,
                dataset_name=dataset_name,
                metric=metric,
                output_dir=plots_dir / "highlights",
                max_timesteps=max_timesteps,
            )

    print(f"Plots have been saved under: {plots_dir}")