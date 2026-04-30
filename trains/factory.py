"""Factory helpers for distributed training actors and shared buffers.

This module provides construction helpers for Ray-based training components.
It creates shared buffer actors, rollout runner actors, evaluation runner
actors, and active rollout-collection tasks.
"""

from buffers import (
    SharedPrioritizedReplayBuffer,
    SharedReplayBuffer,
    SharedRolloutBuffer,
)
from trains.actors import EvalRunner, Runner


def create_shared_buffer_actor(
    *,
    policy_type,
    prioritized_mode,
    state_dim,
    action_dim,
    action_storage_shape,
    buffer_size,
    batch_size,
    device,
    prio_alpha,
    seed,
):
    """Create a shared Ray buffer actor for the selected training mode.

    Parameters
    ----------
    policy_type : {"on_policy", "off_policy"}
        Policy/update family. On-policy training uses a rollout buffer, while
        off-policy training uses a replay buffer.
    prioritized_mode : bool
        Whether to create a prioritized replay buffer for off-policy training.
        This argument is ignored for on-policy training.
    state_dim : int or tuple[int, ...]
        Observation dimension or shape.
    action_dim : int or tuple[int, ...]
        Action dimension or shape.
    action_storage_shape : int or tuple[int, ...]
        Internal storage shape used for actions.
    buffer_size : int
        Maximum number of transitions stored in the shared buffer.
    batch_size : int or None
        Number of transitions sampled per learner update. On-policy rollout
        buffers may receive ``None``.
    device : str or torch.device
        Device argument forwarded to the buffer actor for interface
        compatibility.
    prio_alpha : float or None
        Prioritization exponent used by prioritized replay. Ignored when
        ``prioritized_mode`` is ``False``.
    seed : int
        Base random seed. Different deterministic offsets are used for each
        buffer type.

    Returns
    -------
    ray.actor.ActorHandle
        Remote shared buffer actor.
    """

    if policy_type == "on_policy":
        return SharedRolloutBuffer.remote(
            state_dim=state_dim,
            action_dim=action_dim,
            action_storage_shape=action_storage_shape,
            buffer_size=buffer_size,
            device=device,
            seed=seed + 101,
        )

    if prioritized_mode:
        return SharedPrioritizedReplayBuffer.remote(
            state_dim=state_dim,
            action_dim=action_dim,
            action_storage_shape=action_storage_shape,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            alpha=prio_alpha,
            seed=seed + 211,
        )

    return SharedReplayBuffer.remote(
        state_dim=state_dim,
        action_dim=action_dim,
        action_storage_shape=action_storage_shape,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device,
        seed=seed + 151,
    )


def create_runner_actors(
    *,
    n_runners,
    env,
    learner,
    max_iters,
    policy_type,
    load_path,
    circuit_type,
    seed,
    project_name,
    run_id,
    run_root,
    collection_batch_size,
):
    """Create distributed rollout runner actors.

    Parameters
    ----------
    n_runners : int
        Number of rollout runner actors to create.
    env : gym.Env
        Environment object passed to each runner actor.
    learner : object
        Learner object used to create a process-local remote-copy policy.
    max_iters : int
        Maximum training iteration count forwarded to each runner.
    policy_type : {"on_policy", "off_policy"}
        Policy/update family used by the runners.
    load_path : str or pathlib.Path or None
        Optional learner checkpoint path forwarded to each runner.
    circuit_type : str
        Circuit family identifier used for runner workspace naming.
    seed : int
        Base random seed forwarded to runner actors.
    project_name : str
        Project namespace used for simulator workspaces.
    run_id : str
        Shared experiment run identifier.
    run_root : str or pathlib.Path
        Shared experiment run-root directory.
    collection_batch_size : int
        Number of transitions each runner buffers locally before sending them
        to the shared buffer actor.

    Returns
    -------
    list[ray.actor.ActorHandle]
        List of remote rollout runner actors.
    """

    remote_learner = learner.make_remote_copy()

    return [
        Runner.remote(
            name=runner_name,
            env=env,
            learner=remote_learner,
            max_iters=max_iters,
            policy_type=policy_type,
            load_path=load_path,
            circuit_type=circuit_type,
            seed=seed,
            project_name=project_name,
            run_id=run_id,
            run_root=str(run_root),
            collection_batch_size=collection_batch_size,
        )
        for runner_name in range(n_runners)
    ]


def create_eval_runners(
    *,
    eval_iters,
    env,
    learner,
    seed,
    project_name,
    run_id,
    run_root,
):
    """Create evaluation runner actors.

    Parameters
    ----------
    eval_iters : int
        Number of evaluation episodes executed by the evaluation runner.
    env : gym.Env
        Evaluation environment object passed to the actor.
    learner : object
        Learner object used to create a process-local remote-copy policy.
    seed : int
        Base random seed forwarded to the evaluation actor.
    project_name : str
        Project namespace used for simulator workspaces.
    run_id : str
        Shared experiment run identifier.
    run_root : str or pathlib.Path
        Shared experiment run-root directory.

    Returns
    -------
    list[ray.actor.ActorHandle]
        List containing one remote evaluation runner actor.
    """

    remote_learner = learner.make_remote_copy()

    return [
        EvalRunner.remote(
            name=0,
            env=env,
            learner=remote_learner,
            seed=seed,
            project_name=project_name,
            run_id=run_id,
            run_root=str(run_root),
            eval_episodes=eval_iters,
        )
    ]


def create_active_runner_tasks(
    runners,
    *,
    buffer,
    runner_iters,
    best_fom,
    policy_state_ref,
    policy_version,
    progress_tracker=None,
):
    """Schedule one rollout task for each active runner.

    Parameters
    ----------
    runners : iterable[ray.actor.ActorHandle]
        Rollout runner actors to schedule.
    buffer : ray.actor.ActorHandle
        Shared buffer actor passed to each rollout task.
    runner_iters : int
        Maximum number of timesteps collected by each runner task.
    best_fom : float
        Current best figure of merit forwarded to runner environments.
    policy_state_ref : ray.ObjectRef or object or None
        Policy state reference or object used for runner policy
        synchronization.
    policy_version : int or None
        Monotonic policy version forwarded to runners.
    progress_tracker : ray.actor.ActorHandle or None, optional
        Optional progress tracker actor passed to each runner task. The default
        is ``None``.

    Returns
    -------
    dict[ray.ObjectRef, ray.actor.ActorHandle]
        Mapping from scheduled rollout task references to their corresponding
        runner actors.
    """

    return {
        runner.run.remote(
            buffer=buffer,
            runner_iters=runner_iters,
            best_fom=best_fom,
            policy_state_ref=policy_state_ref,
            policy_version=policy_version,
            progress_tracker=progress_tracker,
        ): runner
        for runner in runners
    }