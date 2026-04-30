"""Algorithm runtime adapters for local and distributed training.

This module defines runtime adapters that connect algorithm implementations
to local or distributed trainers. Runtime objects manage replay or rollout
buffers, choose trainer backends, configure algorithm runtime options, and
perform per-step learner updates for local training.
"""

from copy import deepcopy

import numpy as np

from buffers import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
)
from trains.distributed import DistributedTrainer
from trains.engine import Trainer


class AlgorithmRuntimeBase:
    """Base runtime adapter for algorithm training.

    A runtime adapter owns an algorithm and a buffer, prepares training
    configuration, selects either local or distributed training, and provides
    extension hooks for algorithm-family-specific runtime behavior.

    Parameters
    ----------
    algorithm : object
        Algorithm or agent instance managed by this runtime.
    buffer : object
        Replay or rollout buffer used for local training.

    Attributes
    ----------
    algorithm : object
        Managed algorithm instance.
    buffer : object
        Local buffer associated with the runtime.
    trainer : Trainer or DistributedTrainer or None
        Trainer instance created when training is launched.
    """

    def __init__(self, algorithm, buffer):
        """Initialize the runtime adapter.

        Parameters
        ----------
        algorithm : object
            Algorithm or agent instance.
        buffer : object
            Local replay or rollout buffer.

        Returns
        -------
        None
            Algorithm, buffer, and trainer placeholder are initialized in
            place.
        """

        self.algorithm = algorithm
        self.buffer = buffer
        self.trainer = None

    def launch_training(self, project_name, **config):
        """Launch local or distributed training for the managed algorithm.

        The method copies the training environment for evaluation, applies
        runtime configuration values to the algorithm, delegates
        algorithm-family-specific configuration to :meth:`configure_runtime`,
        and selects a distributed trainer when ``n_runners > 1``.

        Parameters
        ----------
        project_name : str
            Default experiment project name.
        **config : dict[str, object]
            Training configuration overrides. Supported keys include
            ``project_name``, ``load_path``, ``seed``, ``max_iters``,
            ``n_runners``, ``runner_iters``, ``eval_mode``,
            ``eval_intervals``, ``eval_iters``, and ``circuit_type``.
            Subclasses may consume additional keys.

        Returns
        -------
        object
            Return value produced by the selected trainer's ``train`` method.
        """

        algo = self.algorithm
        algo.eval_env = deepcopy(algo.env)

        algo.project_name = config.get("project_name", project_name)
        algo.load_path = config.get("load_path", None)
        algo.seed = config.get("seed", 0)
        algo.max_iters = config.get("max_iters", 10000)
        algo.n_runners = config.get("n_runners", self.default_n_runners())
        algo.runner_iters = config.get("runner_iters", self.default_runner_iters())
        algo.eval_mode = config.get("eval_mode", False)
        algo.eval_intervals = config.get(
            "eval_intervals",
            self.default_eval_intervals(),
        )
        algo.eval_iters = config.get("eval_iters", 10)
        algo.circuit_type = config.get("circuit_type", "CGCS")

        self.configure_runtime(algo, config)

        if algo.n_runners > 1:
            self.trainer = DistributedTrainer(
                env=algo.env,
                eval_env=algo.eval_env,
                agent=algo,
                seed=algo.seed,
            )
            return self.run_distributed(algo)

        self.trainer = Trainer(
            env=algo.env,
            eval_env=algo.eval_env,
            agent=algo,
            seed=algo.seed,
        )
        return self.run_local(algo)

    def configure_runtime(self, algorithm, config):
        """Configure algorithm-family-specific runtime settings.

        Parameters
        ----------
        algorithm : object
            Algorithm instance being configured.
        config : dict[str, object]
            Runtime configuration dictionary.

        Returns
        -------
        None
            Base implementation performs no configuration.
        """

        return None

    def default_n_runners(self):
        """Return the default number of rollout runners.

        Returns
        -------
        int
            Default runner count. The base implementation returns ``1``.
        """

        return 1

    def default_runner_iters(self):
        """Return the default number of timesteps per runner task.

        Returns
        -------
        int
            Default runner iteration count. The base implementation returns
            ``10``.
        """

        return 10

    def default_eval_intervals(self):
        """Return the default evaluation interval.

        Returns
        -------
        int
            Default number of timesteps between evaluations.
        """

        return 200

    def run_local(self, algo):
        """Run training through the local trainer.

        Parameters
        ----------
        algo : object
            Configured algorithm instance.

        Returns
        -------
        object
            Return value produced by :class:`Trainer`.
        """

        return self.trainer.train(
            project_name=algo.project_name,
            load_path=algo.load_path,
            max_iters=algo.max_iters,
            eval_mode=algo.eval_mode,
            eval_intervals=algo.eval_intervals,
            eval_iters=algo.eval_iters,
            circuit_type=algo.circuit_type,
        )

    def run_distributed(self, algo):
        """Run training through the distributed trainer.

        Parameters
        ----------
        algo : object
            Configured algorithm instance.

        Returns
        -------
        object
            Return value produced by :class:`DistributedTrainer`.
        """

        return self.trainer.train(
            project_name=algo.project_name,
            load_path=algo.load_path,
            max_iters=algo.max_iters,
            n_runners=algo.n_runners,
            runner_iters=algo.runner_iters,
            eval_mode=algo.eval_mode,
            eval_intervals=algo.eval_intervals,
            eval_iters=algo.eval_iters,
            policy_type=algo.policy_type,
            action_type=algo.action_type,
            circuit_type=algo.circuit_type,
            **self.distributed_extra_kwargs(algo),
        )

    def distributed_extra_kwargs(self, algo):
        """Return additional distributed-trainer keyword arguments.

        Parameters
        ----------
        algo : object
            Configured algorithm instance.

        Returns
        -------
        dict[str, object]
            Additional keyword arguments passed to
            :meth:`DistributedTrainer.train`.
        """

        return {}

    def step(self, *args, **kwargs):
        """Run one local-training runtime step.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed by the local trainer.
        **kwargs : dict[str, object]
            Keyword arguments passed by the local trainer.

        Returns
        -------
        object
            Algorithm-family-specific update result.

        Raises
        ------
        NotImplementedError
            Always raised by the base implementation.
        """

        raise NotImplementedError()


class OnPolicyRuntime(AlgorithmRuntimeBase):
    """Runtime adapter for on-policy algorithms.

    This runtime stores transitions in a rollout buffer and performs a learner
    update whenever the rollout buffer reaches the algorithm's
    ``update_after`` threshold.
    """

    def default_n_runners(self):
        """Return the default number of on-policy rollout runners.

        Returns
        -------
        int
            Default runner count for on-policy training.
        """

        return 1

    def default_runner_iters(self):
        """Return the default on-policy runner iteration count.

        Returns
        -------
        int
            Default number of timesteps per runner task.
        """

        return 10

    def default_eval_intervals(self):
        """Return the default on-policy evaluation interval.

        Returns
        -------
        int
            Default number of timesteps between evaluations.
        """

        return 200

    def step(self, state, action, reward, next_state, terminated, truncated=False):
        """Store one transition and update an on-policy learner if ready.

        Parameters
        ----------
        state : array-like
            Current observation.
        action : array-like or int
            Action selected by the policy.
        reward : float
            Reward received from the environment.
        next_state : array-like
            Next observation.
        terminated : bool
            Whether the episode terminated.
        truncated : bool, optional
            Whether the episode was truncated. The default is ``False``.

        Returns
        -------
        dict or None
            Learner update result enriched with buffer statistics when an
            update occurs; otherwise ``None``.
        """

        algo = self.algorithm
        algo.timesteps += 1
        result = None

        self.buffer.store(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )

        if self.buffer.size >= algo.update_after:
            states, actions, rewards, next_states, terminateds, truncateds = (
                self.buffer.sample()
            )
            result = algo.learn(
                states,
                actions,
                rewards,
                next_states,
                terminateds,
                truncateds,
            )

            if result is not None:
                buffer_stats = self.buffer.stats()
                result["buffer_size"] = int(buffer_stats["size"])
                result["buffer_fill_ratio"] = float(buffer_stats["fill_ratio"])
                result["buffer_store_calls"] = int(buffer_stats["store_calls"])
                result["buffer_sample_calls"] = int(buffer_stats["sample_calls"])
                result["buffer_stored_transitions"] = int(
                    buffer_stats["stored_transitions"]
                )
                result["buffer_overwrite_count"] = int(
                    buffer_stats["overwrite_count"]
                )

        return result


class OffPolicyRuntime(AlgorithmRuntimeBase):
    """Runtime adapter for off-policy algorithms.

    This runtime stores transitions in replay memory and performs learner
    updates once the replay buffer reaches the algorithm's ``update_after``
    threshold. It supports both uniform replay and prioritized replay.
    """

    def configure_runtime(self, algorithm, config):
        """Configure off-policy runtime settings.

        Parameters
        ----------
        algorithm : object
            Off-policy algorithm instance.
        config : dict[str, object]
            Training configuration dictionary.

        Returns
        -------
        None
            Off-policy UTD ratio and checkpoint interval are written into the
            algorithm instance.
        """

        algorithm.utd_ratio = config.get("utd_ratio", 1.0)
        algorithm.checkpoint_intervals = config.get(
            "checkpoint_intervals",
            algorithm.eval_intervals,
        )

    def default_n_runners(self):
        """Return the default number of off-policy rollout runners.

        Returns
        -------
        int
            Default runner count for off-policy training.
        """

        return 10

    def default_runner_iters(self):
        """Return the default off-policy runner iteration count.

        Returns
        -------
        int
            Default number of timesteps per runner task.
        """

        return 4

    def default_eval_intervals(self):
        """Return the default off-policy evaluation interval.

        Returns
        -------
        int
            Default number of timesteps between evaluations.
        """

        return 100

    def distributed_extra_kwargs(self, algo):
        """Return off-policy distributed-trainer keyword arguments.

        Parameters
        ----------
        algo : object
            Configured off-policy algorithm instance.

        Returns
        -------
        dict[str, object]
            Off-policy-specific distributed trainer options.
        """

        return {
            "utd_ratio": algo.utd_ratio,
            "checkpoint_intervals": algo.checkpoint_intervals,
        }

    def step(self, state, action, reward, next_state, terminated, truncated=False):
        """Store one transition and update an off-policy learner if ready.

        Parameters
        ----------
        state : array-like
            Current observation.
        action : array-like or int
            Action selected by the policy.
        reward : float
            Reward received from the environment.
        next_state : array-like
            Next observation.
        terminated : bool
            Whether the episode terminated.
        truncated : bool, optional
            Whether the episode was truncated. The default is ``False``.

        Returns
        -------
        dict or None
            Learner update result enriched with replay-buffer statistics and,
            for prioritized replay, priority diagnostics. Returns ``None`` if
            the replay buffer is not ready for learning.
        """

        algo = self.algorithm
        algo.timesteps += 1
        result = None

        self.buffer.store(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )

        if algo.prioritized_mode:
            fraction = min(algo.timesteps / algo.max_iters, 1.0)
            algo.prio_beta = (
                algo.prio_beta_start
                + fraction * (1.0 - algo.prio_beta_start)
            )

            if self.buffer.size >= algo.update_after:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminateds,
                    truncateds,
                    weights,
                    idxs,
                ) = self.buffer.sample(algo.prio_beta)

                result = algo.learn(
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminateds,
                    truncateds,
                    weights,
                )

                if result["td_error"] is not None:
                    td_error = result["td_error"].detach().cpu().abs().numpy().flatten()
                    new_prios = td_error + algo.prio_eps
                    self.buffer.update_priorities(idxs, new_prios)
                    result["mean_abs_td_error"] = float(np.mean(td_error))
                    result["max_abs_td_error"] = float(np.max(td_error))

                result["prio_beta"] = float(algo.prio_beta)
                result["mean_is_weight"] = float(weights.mean().item())
                result["max_is_weight"] = float(weights.max().item())

        else:
            if self.buffer.size >= algo.update_after:
                states, actions, rewards, next_states, terminateds, truncateds = (
                    self.buffer.sample()
                )
                result = algo.learn(
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminateds,
                    truncateds,
                )

        if result is not None:
            buffer_stats = self.buffer.stats()
            result["buffer_size"] = int(buffer_stats["size"])
            result["buffer_fill_ratio"] = float(buffer_stats["fill_ratio"])
            result["buffer_store_calls"] = int(buffer_stats["store_calls"])
            result["buffer_sample_calls"] = int(buffer_stats["sample_calls"])
            result["buffer_stored_transitions"] = int(
                buffer_stats["stored_transitions"]
            )
            result["buffer_overwrite_count"] = int(buffer_stats["overwrite_count"])

        return result


def create_on_policy_runtime(
    algorithm,
    *,
    state_dim,
    action_dim,
    action_storage_shape,
    buffer_size,
    device,
):
    """Create an on-policy runtime adapter.

    Parameters
    ----------
    algorithm : object
        On-policy algorithm instance.
    state_dim : int or tuple[int, ...]
        Observation dimension or shape.
    action_dim : int or tuple[int, ...]
        Action dimension or shape.
    action_storage_shape : int or tuple[int, ...]
        Internal action storage shape.
    buffer_size : int
        Rollout buffer capacity.
    device : str or torch.device
        Device on which sampled rollout tensors are allocated.

    Returns
    -------
    OnPolicyRuntime
        Runtime adapter with an initialized :class:`RolloutBuffer`.
    """

    buffer = RolloutBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        action_storage_shape=action_storage_shape,
        buffer_size=buffer_size,
        device=device,
    )

    return OnPolicyRuntime(algorithm, buffer)


def create_off_policy_runtime(
    algorithm,
    *,
    state_dim,
    action_dim,
    action_storage_shape,
    buffer_size,
    batch_size,
    device,
    prioritized_mode,
    prio_alpha,
):
    """Create an off-policy runtime adapter.

    Parameters
    ----------
    algorithm : object
        Off-policy algorithm instance.
    state_dim : int or tuple[int, ...]
        Observation dimension or shape.
    action_dim : int or tuple[int, ...]
        Action dimension or shape.
    action_storage_shape : int or tuple[int, ...]
        Internal action storage shape.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Mini-batch size used for replay sampling.
    device : str or torch.device
        Device on which sampled replay tensors are allocated.
    prioritized_mode : bool
        Whether to construct a prioritized replay buffer.
    prio_alpha : float
        Prioritization exponent used when ``prioritized_mode`` is ``True``.

    Returns
    -------
    OffPolicyRuntime
        Runtime adapter with an initialized replay buffer.
    """

    if prioritized_mode:
        buffer = PrioritizedReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            action_storage_shape=action_storage_shape,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            alpha=prio_alpha,
        )
    else:
        buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            action_storage_shape=action_storage_shape,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
        )

    return OffPolicyRuntime(algorithm, buffer)