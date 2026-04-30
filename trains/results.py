"""Training-result aggregation and logging helpers.

This module provides small utility functions for tracking the best observed
circuit result, building rollout and learner log records, formatting elapsed
time, and aggregating episode-level evaluation statistics.
"""

import datetime
import time

import numpy as np


def update_best_result(
    best_fom,
    best_performances,
    best_parameters,
    *,
    performances,
    parameters,
    fom,
):
    """Update the best observed result when a better FOM is found.

    Parameters
    ----------
    best_fom : float
        Current best figure of merit.
    best_performances : sequence
        Performance vector associated with the current best FOM.
    best_parameters : sequence
        Parameter vector associated with the current best FOM.
    performances : sequence
        Candidate performance vector.
    parameters : sequence
        Candidate parameter vector.
    fom : float
        Candidate figure of merit.

    Returns
    -------
    best_performances : sequence
        Updated best performance vector if ``fom`` improves on
        ``best_fom``; otherwise the previous best performance vector.
    best_parameters : sequence
        Updated best parameter vector if ``fom`` improves on ``best_fom``;
        otherwise the previous best parameter vector.
    best_fom : float
        Updated best FOM value.
    """

    if fom > best_fom:
        return performances, parameters, fom

    return best_performances, best_parameters, best_fom


def extract_best_from_runner_results(
    runner_results,
    best_fom,
    best_performances,
    best_parameters,
):
    """Extract and apply the best result from rollout step records.

    Parameters
    ----------
    runner_results : sequence[dict]
        Per-step rollout records returned by a runner. Each record is expected
        to contain ``"fom"``, ``"performances"``, and ``"parameters"``.
    best_fom : float
        Current global best figure of merit.
    best_performances : sequence
        Performance vector associated with the current global best FOM.
    best_parameters : sequence
        Parameter vector associated with the current global best FOM.

    Returns
    -------
    best_performances : sequence
        Updated best performance vector.
    best_parameters : sequence
        Updated best parameter vector.
    best_fom : float
        Updated best FOM value.
    """

    if not runner_results:
        return best_performances, best_parameters, best_fom

    best_obs = max(runner_results, key=lambda x: x["fom"])

    return update_best_result(
        best_fom,
        best_performances,
        best_parameters,
        performances=best_obs["performances"],
        parameters=best_obs["parameters"],
        fom=best_obs["fom"],
    )


def make_rollout_chunk_log(
    *,
    timesteps,
    runner_name,
    time_per_run,
    ep_per_run,
    ep_ret,
    ep_len,
    runner_results,
    elapse,
    policy_version,
    env_steps_total,
    learner_updates_total,
    effective_utd,
):
    """Build a structured rollout-chunk log record.

    Parameters
    ----------
    timesteps : int
        Global timestep count before or at the rollout chunk.
    runner_name : int or str
        Runner identifier that produced the rollout chunk.
    time_per_run : int
        Number of environment steps collected by the runner.
    ep_per_run : int
        Number of completed episodes in the rollout chunk.
    ep_ret : sequence[float]
        Episode returns completed in the rollout chunk.
    ep_len : sequence[int]
        Episode lengths completed in the rollout chunk.
    runner_results : sequence[dict]
        Per-step rollout records collected by the runner.
    elapse : sequence[int]
        Elapsed wall-clock time represented as ``[hours, minutes, seconds]``.
    policy_version : int
        Policy version used for the rollout chunk.
    env_steps_total : int
        Total environment steps reported by the scheduler.
    learner_updates_total : int
        Total learner updates completed before this chunk.
    effective_utd : float
        Effective updates-to-data ratio associated with this chunk.

    Returns
    -------
    dict[str, object]
        Structured rollout-chunk log record.
    """

    return {
        "time_steps": timesteps,
        "runner_name": runner_name,
        "time_per_run": time_per_run,
        "ep_per_run": ep_per_run,
        "ep_ret": ep_ret,
        "ep_len": ep_len,
        "results": runner_results,
        "elapse": elapse,
        "policy_version": policy_version,
        "env_steps_total": env_steps_total,
        "learner_updates_total": learner_updates_total,
        "effective_utd": effective_utd,
    }


def append_learner_result(epoch_logger, *, timesteps, result, learner_step=None):
    """Append one learner-update result to the epoch logger.

    Parameters
    ----------
    epoch_logger : list-like
        Epoch-level log stream.
    timesteps : int
        Global environment timestep associated with the learner result.
    result : dict or None
        Learner result payload. If ``None``, no log entry is appended.
    learner_step : int or None, optional
        Optional learner-update index. The default is ``None``.

    Returns
    -------
    None
        A learner-result record is appended to ``epoch_logger`` when
        ``result`` is not ``None``.
    """

    if result is None:
        return

    record = {
        "timesteps": timesteps,
        "result": result,
    }

    if learner_step is not None:
        record["learner_step"] = learner_step

    epoch_logger.append(record)


def append_scheduler_metrics(epoch_logger, **metrics):
    """Append scheduler metrics to the epoch logger.

    Parameters
    ----------
    epoch_logger : list-like
        Epoch-level log stream.
    **metrics : dict[str, object]
        Scheduler metrics to record.

    Returns
    -------
    None
        A copy of ``metrics`` is appended to ``epoch_logger``.
    """

    epoch_logger.append(dict(metrics))


def format_elapsed_components(start_time):
    """Compute elapsed wall-clock time components.

    Parameters
    ----------
    start_time : float
        Start timestamp returned by ``time.time()``.

    Returns
    -------
    elapsed_time : datetime.timedelta
        Full elapsed wall-clock duration.
    components : list[int]
        Elapsed time represented as ``[hours, minutes, seconds]``.
    """

    elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
    total_seconds = int(elapsed_time.total_seconds())

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return elapsed_time, [hours, minutes, seconds]


def aggregate_global_stats(
    global_stats,
    *,
    max_ep_ret,
    max_ep_len,
    mean_ep_ret,
    mean_ep_len,
    num_eps,
):
    """Accumulate evaluation statistics into global summary fields.

    Parameters
    ----------
    global_stats : dict[str, float or int]
        Running global statistics dictionary. Expected keys are ``"max_ret"``,
        ``"max_len"``, ``"mean_ret"``, ``"mean_len"``, ``"ep_count"``, and
        ``"eval_count"``.
    max_ep_ret : float
        Maximum episode return from the latest evaluation batch.
    max_ep_len : int or float
        Maximum episode length from the latest evaluation batch.
    mean_ep_ret : float
        Mean episode return from the latest evaluation batch.
    mean_ep_len : float
        Mean episode length from the latest evaluation batch.
    num_eps : int
        Number of training episodes associated with the logging window.

    Returns
    -------
    dict[str, float or int]
        Updated global statistics dictionary.
    """

    global_stats["max_ret"] = max(global_stats["max_ret"], max_ep_ret)
    global_stats["max_len"] = max(global_stats["max_len"], max_ep_len)
    global_stats["mean_ret"] += mean_ep_ret
    global_stats["mean_len"] += mean_ep_len
    global_stats["ep_count"] += num_eps
    global_stats["eval_count"] += 1

    return global_stats


def finalize_global_stats(global_stats):
    """Finalize averaged global evaluation statistics.

    Parameters
    ----------
    global_stats : dict[str, float or int]
        Running global statistics dictionary.

    Returns
    -------
    dict[str, float or int]
        Finalized statistics dictionary. Mean return and mean length are
        divided by ``eval_count`` when at least one evaluation batch exists.
    """

    finalized = dict(global_stats)

    if finalized["eval_count"] > 0:
        finalized["mean_len"] /= finalized["eval_count"]
        finalized["mean_ret"] /= finalized["eval_count"]

    return finalized