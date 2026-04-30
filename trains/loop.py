"""Distributed training-loop helpers for runner and evaluation completion.

This module provides utility functions used by the distributed trainer to
process completed rollout tasks, update rollout metadata, start or collect
evaluation batches, and drain any active evaluation tasks during shutdown.
"""

import numpy as np
import ray

from trains.results import (
    extract_best_from_runner_results,
    make_rollout_chunk_log,
)
from trains.scheduler import (
    _flatten_eval_results,
    collect_completed_eval_batch,
    start_next_eval_batch,
)


def process_runner_completion(
    *,
    done_ref,
    active_runner_tasks,
    obs_logger,
    timesteps,
    train_iters,
    learner_updates_total,
    policy_version,
    best_fom,
    best_performances,
    best_parameters,
):
    """Process one completed rollout runner task.

    This function removes the completed Ray task from the active-task mapping,
    retrieves its rollout result, appends a structured rollout-chunk log, and
    updates the best observed circuit result from the returned step records.

    Parameters
    ----------
    done_ref : ray.ObjectRef
        Completed Ray object reference returned by a runner ``run`` call.
    active_runner_tasks : dict[ray.ObjectRef, ray.actor.ActorHandle]
        Mapping from active runner task references to runner actors. The
        completed task is removed from this mapping.
    obs_logger : list-like
        Observation log stream. The generated rollout-chunk record is appended
        to this logger.
    timesteps : int
        Global environment timestep count before this rollout chunk is applied.
    train_iters : int
        Number of training steps already counted for the current scheduler
        iteration.
    learner_updates_total : int
        Total number of learner updates completed before this rollout chunk.
    policy_version : int
        Policy version used by the completed runner task.
    best_fom : float
        Best figure of merit observed so far.
    best_performances : list
        Performance vector associated with the current best figure of merit.
    best_parameters : list
        Parameter vector associated with the current best figure of merit.

    Returns
    -------
    dict[str, object]
        Completion summary containing the runner actor, rollout statistics,
        raw runner step records, chunk-level best FOM, and updated global best
        result fields.
    """

    runner = active_runner_tasks.pop(done_ref)
    result = ray.get(done_ref)

    name, time_per_run, ep_per_run, ep_ret, ep_len, runner_results, elapse = result

    obs_logger.append(
        make_rollout_chunk_log(
            timesteps=timesteps,
            runner_name=name,
            time_per_run=time_per_run,
            ep_per_run=ep_per_run,
            ep_ret=ep_ret,
            ep_len=ep_len,
            runner_results=runner_results,
            elapse=elapse,
            policy_version=policy_version,
            env_steps_total=timesteps + train_iters,
            learner_updates_total=learner_updates_total,
            effective_utd=0.0,
        )
    )

    best_performances, best_parameters, best_fom = (
        extract_best_from_runner_results(
            runner_results,
            best_fom,
            best_performances,
            best_parameters,
        )
    )

    return {
        "runner": runner,
        "name": name,
        "time_per_run": time_per_run,
        "ep_per_run": ep_per_run,
        "ep_ret": ep_ret,
        "ep_len": ep_len,
        "runner_results": runner_results,
        "chunk_best_fom": max(
            (item["fom"] for item in runner_results),
            default=None,
        ),
        "best_performances": best_performances,
        "best_parameters": best_parameters,
        "best_fom": best_fom,
    }


def update_last_rollout_chunk_effective_utd(obs_logger, effective_utd):
    """Update the effective UTD value of the most recent rollout chunk.

    Parameters
    ----------
    obs_logger : list-like
        Observation log stream containing rollout-chunk records.
    effective_utd : float
        Effective updates-to-data ratio to attach to the latest rollout chunk.

    Returns
    -------
    None
        The latest rollout-chunk log is updated in place when available.
    """

    if not obs_logger:
        return

    last = obs_logger[-1]

    if isinstance(last, dict) and "results" in last:
        last["effective_utd"] = effective_utd


def maybe_collect_active_eval(epoch_logger, active_eval_batch, num_eps):
    """Collect a completed evaluation batch if one is ready.

    Parameters
    ----------
    epoch_logger : list-like
        Epoch-level log stream. A completed evaluation log is appended when
        available.
    active_eval_batch : dict[str, object] or None
        Currently active evaluation batch descriptor.
    num_eps : int
        Number of training episodes completed since the last relevant logging
        point.

    Returns
    -------
    dict[str, object] or None
        Updated active evaluation batch. Returns ``None`` when the previous
        batch has been collected.
    """

    eval_log, active_eval_batch = collect_completed_eval_batch(
        active_eval_batch,
        num_eps,
    )

    if eval_log is not None:
        epoch_logger.append(eval_log)

    return active_eval_batch


def maybe_start_eval_batch(eval_runners, pending_eval_queue, active_eval_batch):
    """Start the next pending evaluation batch when no batch is active.

    Parameters
    ----------
    eval_runners : list[ray.actor.ActorHandle] or None
        Evaluation runner actors.
    pending_eval_queue : list[dict]
        Queue of pending evaluation requests.
    active_eval_batch : dict[str, object] or None
        Currently active evaluation batch descriptor.

    Returns
    -------
    dict[str, object] or None
        Active evaluation batch after attempting to start the next queued
        request.
    """

    return start_next_eval_batch(
        eval_runners,
        pending_eval_queue,
        active_eval_batch,
    )


def drain_active_eval_batch(epoch_logger, active_eval_batch, timesteps, num_eps):
    """Synchronously collect an active evaluation batch during shutdown.

    This helper is used in cleanup paths to avoid losing a completed or nearly
    completed evaluation batch. Exceptions are suppressed so that shutdown and
    checkpoint cleanup can proceed safely.

    Parameters
    ----------
    epoch_logger : list-like
        Epoch-level log stream. A drained evaluation log is appended when
        valid results are available.
    active_eval_batch : dict[str, object] or None
        Active evaluation batch descriptor containing Ray task references and
        metadata.
    timesteps : int
        Current global timestep used as a fallback logging timestep.
    num_eps : int
        Number of training episodes completed since the last relevant logging
        point.

    Returns
    -------
    None
        Evaluation results are appended to ``epoch_logger`` when available.
    """

    if active_eval_batch is None:
        return

    try:
        results = ray.get(active_eval_batch["tasks"])
        total_ep_ret, total_ep_len = _flatten_eval_results(results)

        if total_ep_ret and total_ep_len:
            epoch_logger.append(
                {
                    "timesteps": active_eval_batch.get("timesteps", timesteps),
                    "number_of_eps": num_eps,
                    "max_ep_ret": np.max(total_ep_ret),
                    "max_ep_len": np.max(total_ep_len),
                    "mean_ep_ret": np.mean(total_ep_ret),
                    "mean_ep_len": np.mean(total_ep_len),
                    "eval_policy_version": active_eval_batch.get(
                        "policy_version"
                    ),
                }
            )

    except Exception:
        pass