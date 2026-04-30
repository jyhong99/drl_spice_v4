"""Learner update helpers for distributed training.

This module defines helper functions used by the distributed trainer to run
learner updates from Ray-backed shared buffers. It supports on-policy updates,
uniform off-policy replay updates, prioritized replay updates, and optional
enrichment of learner results with buffer statistics.
"""

import numpy as np
import ray


def enrich_with_buffer_stats(result, buffer_stats):
    """Attach shared-buffer statistics to a learner result dictionary.

    Parameters
    ----------
    result : dict or None
        Learner result dictionary to enrich. If ``None``, the function returns
        ``None`` immediately.
    buffer_stats : dict[str, int or float]
        Buffer statistics dictionary returned by the shared buffer actor. The
        expected keys include ``"size"``, ``"fill_ratio"``,
        ``"store_calls"``, ``"sample_calls"``, ``"stored_transitions"``, and
        ``"overwrite_count"``. Prioritized buffers may also provide
        ``"max_priority"``.

    Returns
    -------
    dict or None
        Enriched learner result dictionary. Returns ``None`` if ``result`` is
        ``None``.
    """

    if result is None:
        return None

    result["buffer_size"] = int(buffer_stats["size"])
    result["buffer_fill_ratio"] = float(buffer_stats["fill_ratio"])
    result["buffer_store_calls"] = int(buffer_stats["store_calls"])
    result["buffer_sample_calls"] = int(buffer_stats["sample_calls"])
    result["buffer_stored_transitions"] = int(buffer_stats["stored_transitions"])
    result["buffer_overwrite_count"] = int(buffer_stats["overwrite_count"])

    if "max_priority" in buffer_stats:
        result["buffer_max_priority"] = float(buffer_stats["max_priority"])

    return result


def run_distributed_on_policy_update(*, buffer, learner, to_device_batch):
    """Run one distributed on-policy learner update.

    This function samples the full rollout from a shared rollout buffer,
    converts the sampled arrays to device tensors, and delegates the update to
    the learner.

    Parameters
    ----------
    buffer : ray.actor.ActorHandle
        Shared rollout buffer actor exposing ``sample.remote``.
    learner : object
        On-policy learner object exposing a ``learn`` method.
    to_device_batch : callable
        Function that converts sampled NumPy arrays into tensors on the target
        learner device.

    Returns
    -------
    dict or object
        Learner update result returned by ``learner.learn``.
    """

    states, actions, rewards, next_states, dones, truncateds = to_device_batch(
        ray.get(buffer.sample.remote())
    )

    return learner.learn(
        states,
        actions,
        rewards,
        next_states,
        dones,
        truncateds,
    )


def run_distributed_off_policy_uniform_update(
    *,
    buffer,
    learner,
    to_device_batch,
    learner_step,
):
    """Run one distributed off-policy update using uniform replay sampling.

    This function samples a mini-batch from a shared uniform replay buffer,
    converts it to device tensors, and performs one learner update without
    importance-sampling weights.

    Parameters
    ----------
    buffer : ray.actor.ActorHandle
        Shared replay buffer actor exposing ``sample.remote``.
    learner : object
        Off-policy learner object exposing a ``learn`` method.
    to_device_batch : callable
        Function that converts sampled NumPy arrays into tensors on the target
        learner device.
    learner_step : int
        Global learner update index forwarded to the learner as
        ``global_timesteps``.

    Returns
    -------
    dict or object
        Learner update result returned by ``learner.learn``.
    """

    states, actions, rewards, next_states, dones, truncateds = to_device_batch(
        ray.get(buffer.sample.remote())
    )

    return learner.learn(
        states,
        actions,
        rewards,
        next_states,
        dones,
        truncateds,
        weights=None,
        global_timesteps=learner_step,
    )


def run_distributed_off_policy_prioritized_update(
    *,
    buffer,
    learner,
    to_device_batch,
    prio_beta,
    prio_eps,
    learner_step,
):
    """Run one distributed off-policy update using prioritized replay.

    This function samples a prioritized mini-batch from a shared prioritized
    replay buffer, converts tensor fields to the learner device while keeping
    replay indices unchanged, performs one learner update with
    importance-sampling weights, and updates replay priorities using the
    resulting TD errors.

    Parameters
    ----------
    buffer : ray.actor.ActorHandle
        Shared prioritized replay buffer actor exposing ``sample.remote`` and
        ``update_priorities.remote``.
    learner : object
        Off-policy learner object exposing a ``learn`` method. The learner
        result is expected to include ``"td_error"`` when priority updates are
        available.
    to_device_batch : callable
        Function that converts sampled arrays into tensors. It must support
        ``has_weights`` and ``has_indices`` keyword arguments.
    prio_beta : float
        Importance-sampling correction exponent used during prioritized
        replay sampling.
    prio_eps : float
        Small positive constant added to absolute TD errors when computing new
        priorities.
    learner_step : int
        Global learner update index forwarded to the learner.

    Returns
    -------
    dict
        Learner update result enriched with prioritized replay diagnostics,
        including ``"prio_beta"``, ``"mean_is_weight"``,
        ``"max_is_weight"``, and, when TD errors are available,
        ``"mean_abs_td_error"`` and ``"max_abs_td_error"``.
    """

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        truncateds,
        weights,
        idxs,
    ) = to_device_batch(
        ray.get(buffer.sample.remote(prio_beta)),
        has_weights=True,
        has_indices=True,
    )

    result = learner.learn(
        states,
        actions,
        rewards,
        next_states,
        dones,
        truncateds,
        weights,
        learner_step,
    )

    if result["td_error"] is not None:
        td_error = result["td_error"].detach().cpu().abs().numpy().flatten()
        new_prios = td_error + prio_eps

        buffer.update_priorities.remote(idxs, new_prios)

        result["mean_abs_td_error"] = float(np.mean(td_error))
        result["max_abs_td_error"] = float(np.max(td_error))

    result["prio_beta"] = float(prio_beta)
    result["mean_is_weight"] = float(weights.mean().item())
    result["max_is_weight"] = float(weights.max().item())

    return result