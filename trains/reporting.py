"""Reporting and log-saving helpers for training runs.

This module provides utility functions for printing training configuration,
formatting metric values, building final log records, computing elapsed time,
printing final circuit summaries, saving structured logs, and generating plot
artifacts from saved logs.
"""

import datetime
import time

from loggers.plotter import plot_saved_training_logs


def format_metric_value(value):
    """Format a numeric metric value for human-readable console output.

    Values with very small or very large magnitude are formatted in scientific
    notation. Other values are rounded to four decimal places.

    Parameters
    ----------
    value : int or float
        Numeric value to format.

    Returns
    -------
    str
        Formatted metric value.
    """

    numeric_value = float(value)
    abs_value = abs(numeric_value)

    if abs_value != 0 and (abs_value < 1e-3 or abs_value >= 1e4):
        return f"{numeric_value:.4e}"

    return str(round(numeric_value, 4))


def print_training_start(init_log, *, distributed=False):
    """Print the training-start configuration summary.

    Parameters
    ----------
    init_log : dict[str, object]
        Initial training log record containing environment, agent, seed,
        project, iteration, evaluation, checkpoint, and loading metadata.
    distributed : bool, optional
        Whether to print distributed-training fields such as runner count,
        runner iterations, UTD ratio, and checkpoint intervals. The default is
        ``False``.

    Returns
    -------
    None
        The formatted training-start summary is printed to standard output.
    """

    print(f"================================================================================================")
    print(f"TRAINING STARTING TIME      | {init_log['start_time']}")
    print(f"PROJECT NAME                | {init_log['project_name']}")

    if distributed:
        print(f"ENVIRONMENT                 | {init_log['circuit_type']}")
    else:
        print(f"ENVIRONMENT                 | {init_log['env']}")

    print(f"EVALUATION ENVIRONMENT      | {init_log['eval_env']}")
    print(f"AGENT                       | {init_log['agent']}")
    print(f"SEED                        | {init_log['seed']}")
    print(f"TOTAL TRAINING TIMESTEPS    | {init_log['max_iters']}")

    if distributed:
        print(f"THE NUMBER OF RUNNERS       | {init_log['n_runners']}")
        print(f"ITERATIONS PER A RUNNER     | {init_log['runner_iters']}")
        print(f"UTD RATIO                   | {init_log['utd_ratio']}")

    print(f"EVALUATION MODE             | {init_log['eval_mode']}")
    print(f"EVALUATION INTERVALS        | {init_log['eval_intervals']}")

    if distributed:
        print(f"CHECKPOINT INTERVALS        | {init_log['checkpoint_intervals']}")

    print(f"EVALUATION ITERATIONS       | {init_log['eval_iters']}")
    print(f"LOAD PATH                   | {init_log['load_path']}")
    print(f"================================================================================================")


def build_end_log(
    *,
    best_performances,
    best_parameters,
    best_fom,
    end_time_now,
    time_elapse,
    global_stats=None,
):
    """Build the final training log record.

    Parameters
    ----------
    best_performances : list or sequence
        Best performance vector observed during training.
    best_parameters : list or sequence
        Parameter vector associated with the best observed performance.
    best_fom : float
        Best figure of merit observed during training.
    end_time_now : str
        Training end timestamp as a formatted string.
    time_elapse : str
        Human-readable training elapsed time.
    global_stats : dict[str, object] or None, optional
        Optional global episode statistics. If provided, values are appended
        to the final record. The default is ``None``.

    Returns
    -------
    dict[str, object]
        Final structured training log record.
    """

    record = {
        "end_time": end_time_now,
        "tims_elapse": time_elapse,
        "best_performances": best_performances,
        "best_parameters": best_parameters,
        "best_fom": best_fom,
    }

    if global_stats is not None:
        record.update(
            {
                "total_episodes": global_stats["ep_count"],
                "global_max_len": global_stats["max_len"],
                "global_max_ret": global_stats["max_ret"],
                "global_mean_len": global_stats["mean_len"],
                "global_mean_ret": global_stats["mean_ret"],
            }
        )

    return record


def finalize_timing(start_time):
    """Compute final timestamp and elapsed training time.

    Parameters
    ----------
    start_time : float
        Start time returned by ``time.time()``.

    Returns
    -------
    end_time_now : str
        Current timestamp formatted as ``YYYY-MM-DD HH:MM:SS``.
    time_elapse : str
        Human-readable elapsed time formatted as ``"<hours>h <minutes>m
        <seconds>s"``.
    """

    end_time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
    total_seconds = int(elapsed_time.total_seconds())

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_elapse = f"{hours}h {minutes}m {seconds}s"

    return end_time_now, time_elapse


def print_circuit_summary(
    *,
    circuit_type,
    end_time_now,
    time_elapse,
    best_performances,
    best_parameters,
    best_fom,
    global_stats=None,
):
    """Print the final best-circuit summary.

    The summary includes training finish time, elapsed time, optional global
    episode statistics, best performance metrics, fixed values, design
    parameters, and best FOM.

    Parameters
    ----------
    circuit_type : {"CGCS", "CS"}
        Circuit family identifier used to choose metric and parameter labels.
    end_time_now : str
        Training end timestamp.
    time_elapse : str
        Human-readable training elapsed time.
    best_performances : sequence
        Best performance vector observed during training.
    best_parameters : sequence
        Parameter vector associated with the best observed design. The first
        fixed-value entries are printed separately from tunable parameters.
    best_fom : float
        Best figure of merit observed during training.
    global_stats : dict[str, object] or None, optional
        Optional global episode statistics to print in the header. The default
        is ``None``.

    Returns
    -------
    None
        The formatted circuit summary is printed to standard output. If the
        best result is empty or ``circuit_type`` is unsupported, the function
        returns without printing a full summary.
    """

    if not best_performances or not best_parameters:
        return

    include_iip3 = len(best_performances) >= 6

    header_lines = [
        ("TRINING FINISHING TIME", end_time_now),
        ("TOTAL TRAINIMG ELAPSE", time_elapse),
    ]

    if global_stats is not None:
        header_lines.extend(
            [
                ("TOTAL NUMBER OF EPISODES", global_stats["ep_count"]),
                ("GLOBAL MAX EPISODE LENGTH", global_stats["max_len"]),
                ("GLOBAL MAX EPISODE RETURN", round(global_stats["max_ret"], 4)),
                ("GLOBAL MEAN EPISODE LENGTH", round(global_stats["mean_len"], 4)),
                ("GLOBAL MEAN EPISODE RETURN", round(global_stats["mean_ret"], 4)),
            ]
        )

    print(f"================================================================================================")

    for key, value in header_lines:
        print(f"{key:<28} | {value}")

    print(f"------------------------------------------------------------------------------------------------")

    if circuit_type == "CGCS":
        perf_labels = [
            ("S11", "dB"),
            ("S21", "dB"),
            ("S22", "dB"),
            ("NF", "dB"),
            ("PD", "mW"),
        ]

        if include_iip3:
            perf_labels.append(("IIP3", "dBm"))

        fixed_labels = [
            ("V_DD", "V"),
            ("R_b", "Ω"),
            ("C_1", "F"),
            ("l_m", "μm"),
        ]
        param_labels = [
            ("V_b1", "V"),
            ("V_b2", "V"),
            ("V_b3", "V"),
            ("V_b4", "V"),
            ("R_D1", "Ω"),
            ("R_D4", "Ω"),
            ("R_S5", "Ω"),
            ("C_D1", "F"),
            ("C_D4", "F"),
            ("C_S3", "F"),
            ("C_S4", "F"),
            ("XM1", "μm"),
            ("XM2", "μm"),
            ("XM3", "μm"),
            ("XM4", "μm"),
            ("XM5", "μm"),
        ]

    elif circuit_type == "CS":
        perf_labels = [
            ("S11", "dB"),
            ("S21", "dB"),
            ("S22", "dB"),
            ("NF", "dB"),
            ("PD", "mW"),
        ]

        if include_iip3:
            perf_labels.append(("IIP3", "dBm"))

        fixed_labels = [
            ("V_DD", "V"),
            ("R_b", "Ω"),
            ("C_1", "F"),
            ("l_m", "μm"),
        ]
        param_labels = [
            ("V_b", "V"),
            ("R_D", "Ω"),
            ("L_D", "H"),
            ("L_G", "H"),
            ("L_S", "H"),
            ("C_D", "F"),
            ("C_ex", "F"),
            ("XM1", "μm"),
            ("XM2", "μm"),
        ]

    else:
        return

    print_metric_block(
        "BEST PERFORMANCES",
        perf_labels,
        best_performances[: len(perf_labels)],
    )
    print_metric_block(
        "FIXED VALUES",
        fixed_labels,
        best_parameters[: len(fixed_labels)],
    )
    print_metric_block(
        "BEST PARAMETERS",
        param_labels,
        best_parameters[
            len(fixed_labels) : len(fixed_labels) + len(param_labels)
        ],
    )

    print(f"BEST FOM                     | {round(best_fom, 4)}")
    print(f"================================================================================================")


def print_metric_block(title, labels, values):
    """Print a labeled metric block.

    Parameters
    ----------
    title : str
        Block title printed in the first row.
    labels : sequence[tuple[str, str]]
        Sequence of ``(label, unit)`` pairs.
    values : sequence
        Metric values corresponding to ``labels``.

    Returns
    -------
    None
        The formatted metric block is printed to standard output.
    """

    for idx, ((label, unit), value) in enumerate(zip(labels, values)):
        title_cell = title if idx == 0 else ""
        print(f"{title_cell:<28} | {label:<4} | {format_metric_value(value)} {unit}")

    print(f"------------------------------------------------------------------------------------------------")


def save_logs(logger):
    """Save all structured logs managed by a logger.

    Parameters
    ----------
    logger : TrainingLogger
        Logger object exposing ``save_all``.

    Returns
    -------
    None
        All managed logs are written to disk.
    """

    logger.save_all()


def save_logs_and_plot(*, logger, project_name, max_timesteps):
    """Save structured logs and generate training plots.

    Parameters
    ----------
    logger : TrainingLogger
        Logger object exposing ``save_all``.
    project_name : str
        Experiment project name used to locate saved structured logs.
    max_timesteps : int
        Maximum timestep used as an upper plotting bound.

    Returns
    -------
    None
        Structured logs are saved and plot artifacts are generated.
    """

    logger.save_all()
    plot_saved_training_logs(
        project_name,
        max_timesteps=max_timesteps,
    )