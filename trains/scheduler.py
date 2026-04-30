"""Scheduling helpers for distributed training and evaluation.

This module provides utility functions for updates-to-data scheduling,
on-policy batch finalization, effective UTD computation, and asynchronous
evaluation request management in Ray-based distributed training.
"""

import numpy as np
import ray


def _flatten_eval_results(results):
    """Flatten nested evaluation results into return and length lists.

    Parameters
    ----------
    results : sequence
        Evaluation results returned by one or more evaluation runners. Each
        element may be either a single ``(episode_return, episode_length)``
        tuple or an iterable of such tuples.

    Returns
    -------
    total_ep_ret : list[float]
        Flattened list of episode returns.
    total_ep_len : list[int]
        Flattened list of episode lengths.
    """

    total_ep_ret, total_ep_len = [], []

    for eval_result in results:
        if isinstance(eval_result, tuple) and len(eval_result) == 2:
            ep_ret, ep_len = eval_result
            total_ep_ret.append(ep_ret)
            total_ep_len.append(ep_len)
            continue

        for episode_result in eval_result:
            ep_ret, ep_len = episode_result
            total_ep_ret.append(ep_ret)
            total_ep_len.append(ep_len)

    return total_ep_ret, total_ep_len


def compute_utd_update_budget(chunk_env_steps, utd_ratio, residual_credit=0.0):
    """Compute the learner-update budget from UTD credit.

    The update budget is computed by accumulating fractional update credit from
    the number of newly collected environment steps and the target UTD ratio.
    Integer credit is spent as learner updates, while the fractional residual
    is carried to the next scheduling step.

    Parameters
    ----------
    chunk_env_steps : int
        Number of environment steps collected in the latest rollout chunk.
    utd_ratio : float
        Target updates-to-data ratio.
    residual_credit : float, optional
        Fractional update credit carried over from previous chunks. The
        default is ``0.0``.

    Returns
    -------
    chunk_updates : int
        Number of learner updates to run for the current chunk.
    next_residual : float
        Fractional update credit carried forward.

    Raises
    ------
    ValueError
        If ``chunk_env_steps``, ``utd_ratio``, or ``residual_credit`` is
        negative.
    """

    if chunk_env_steps < 0:
        raise ValueError("chunk_env_steps must be non-negative")

    if utd_ratio < 0:
        raise ValueError("utd_ratio must be non-negative")

    if residual_credit < 0:
        raise ValueError("residual_credit must be non-negative")

    credit = float(residual_credit) + (float(chunk_env_steps) * float(utd_ratio))
    chunk_updates = int(credit)
    next_residual = credit - chunk_updates

    return chunk_updates, next_residual


def compute_effective_utd(learner_updates_total, env_steps_total):
    """Compute the realized updates-to-data ratio.

    Parameters
    ----------
    learner_updates_total : int
        Total number of learner updates completed.
    env_steps_total : int
        Total number of environment steps collected.

    Returns
    -------
    float
        Effective UTD ratio, computed as learner updates divided by
        environment steps. The denominator is clamped to at least ``1``.
    """

    return float(learner_updates_total) / max(int(env_steps_total), 1)


def should_stop_on_policy_rescheduling(
    *,
    buffer_size,
    update_after,
    timesteps,
    max_iters,
):
    """Check whether on-policy rollout rescheduling should stop.

    Parameters
    ----------
    buffer_size : int
        Current number of transitions stored in the rollout buffer.
    update_after : int
        Minimum rollout size required before an on-policy learner update.
    timesteps : int
        Current global environment timestep count.
    max_iters : int
        Maximum allowed environment timesteps.

    Returns
    -------
    bool
        ``True`` if the rollout buffer is ready for update or training has
        reached the maximum timestep budget.
    """

    return int(buffer_size) >= int(update_after) or int(timesteps) >= int(max_iters)


def should_finalize_on_policy_batch(
    *,
    active_runner_tasks,
    buffer_size,
    update_after,
    timesteps,
    max_iters,
):
    """Check whether the current on-policy rollout batch should be updated.

    Parameters
    ----------
    active_runner_tasks : dict
        Mapping of active rollout Ray task references to runner actors.
    buffer_size : int
        Current number of transitions stored in the rollout buffer.
    update_after : int
        Minimum rollout size required before an on-policy learner update.
    timesteps : int
        Current global environment timestep count.
    max_iters : int
        Maximum allowed environment timesteps.

    Returns
    -------
    bool
        ``True`` when no rollout tasks are still active, the buffer contains
        data, and either the update threshold or timestep limit has been
        reached.
    """

    return (not active_runner_tasks) and int(buffer_size) > 0 and (
        int(buffer_size) >= int(update_after) or int(timesteps) >= int(max_iters)
    )


def enqueue_due_eval_requests(
    current_timesteps,
    next_eval_timestep,
    eval_intervals,
    pending_eval_queue,
    policy_state_ref,
    policy_version,
):
    """Enqueue all evaluation requests due at the current timestep.

    Parameters
    ----------
    current_timesteps : int
        Current global environment timestep count.
    next_eval_timestep : int
        Next scheduled evaluation timestep.
    eval_intervals : int
        Number of timesteps between evaluation requests.
    pending_eval_queue : list[dict]
        Queue receiving newly due evaluation request dictionaries.
    policy_state_ref : ray.ObjectRef or object
        Policy state reference or object associated with the queued evaluation.
    policy_version : int
        Policy version associated with the queued evaluation.

    Returns
    -------
    int
        Updated next evaluation timestep after all due requests have been
        enqueued.

    Raises
    ------
    ValueError
        If ``eval_intervals`` is not positive.
    """

    if eval_intervals <= 0:
        raise ValueError("eval_intervals must be positive")

    while current_timesteps >= next_eval_timestep:
        pending_eval_queue.append(
            {
                "timesteps": int(next_eval_timestep),
                "policy_state_ref": policy_state_ref,
                "policy_version": policy_version,
            }
        )
        next_eval_timestep += eval_intervals

    return next_eval_timestep


def start_next_eval_batch(eval_runners, pending_eval_queue, active_eval_batch):
    """Start the next pending evaluation batch if possible.

    Parameters
    ----------
    eval_runners : sequence[ray.actor.ActorHandle]
        Evaluation runner actors used to execute evaluation requests.
    pending_eval_queue : list[dict]
        Queue of pending evaluation request dictionaries.
    active_eval_batch : dict or None
        Currently running evaluation batch. If not ``None``, no new batch is
        started.

    Returns
    -------
    dict or None
        Active evaluation batch. Returns the existing batch if one is already
        active, a newly started request if the queue is non-empty, or ``None``
        when no evaluation is pending.
    """

    if active_eval_batch is not None or not pending_eval_queue:
        return active_eval_batch

    request = pending_eval_queue.pop(0)
    request["tasks"] = [
        eval_runner.evaluate.remote(
            policy_state_ref=request["policy_state_ref"],
            policy_version=request["policy_version"],
            eval_timesteps=request["timesteps"],
        )
        for eval_runner in eval_runners
    ]

    return request


def collect_completed_eval_batch(active_eval_batch, num_eps):
    """Collect an active evaluation batch when all tasks are complete.

    Parameters
    ----------
    active_eval_batch : dict or None
        Active evaluation batch containing ``"tasks"``, ``"timesteps"``,
        and optional ``"policy_version"`` fields.
    num_eps : int
        Number of training episodes completed during the corresponding
        logging window.

    Returns
    -------
    eval_log : dict or None
        Evaluation log record when all evaluation tasks are complete and valid
        results are available; otherwise ``None``.
    active_eval_batch : dict or None
        Updated active evaluation batch. Returns the original batch if tasks
        are still running, or ``None`` when the batch has been consumed.

    Raises
    ------
    ray.exceptions.RayError
        Propagated if ``ray.get`` fails after all tasks are reported ready.
    """

    if active_eval_batch is None:
        return None, None

    tasks = active_eval_batch["tasks"]
    ready_eval_refs, _ = ray.wait(tasks, num_returns=len(tasks), timeout=0)

    if len(ready_eval_refs) != len(tasks):
        return None, active_eval_batch

    results = ray.get(tasks)
    total_ep_ret, total_ep_len = _flatten_eval_results(results)

    if not total_ep_ret or not total_ep_len:
        return None, None

    eval_log = {
        "timesteps": active_eval_batch["timesteps"],
        "number_of_eps": num_eps,
        "max_ep_ret": np.max(total_ep_ret),
        "max_ep_len": np.max(total_ep_len),
        "mean_ep_ret": np.mean(total_ep_ret),
        "mean_ep_len": np.mean(total_ep_len),
        "eval_policy_version": active_eval_batch.get("policy_version"),
    }

    return eval_log, None