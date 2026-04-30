"""Ray-based rollout and evaluation runners for distributed training.

This module defines Ray actor classes used to collect environment transitions
and evaluate policies in parallel. Runners manage process-local environment
workspaces, policy-state synchronization, batched replay-buffer insertion,
episode resets, and structured rollout records.
"""

import datetime
import time

import ray
import torch

from simulator.ngspice.workspace import create_experiment_run_root
from trains.utils import (
    get_next_step,
    get_reset_result,
    get_reset_state,
    seed_all,
    select_reset_mode,
)


def _resolve_policy_state(policy_state_or_ref):
    """Resolve a policy state object or Ray object reference.

    Parameters
    ----------
    policy_state_or_ref : object or ray.ObjectRef or None
        Policy state object, Ray object reference containing a policy state,
        or ``None``.

    Returns
    -------
    object or None
        Resolved policy state. Returns ``None`` when ``policy_state_or_ref`` is
        ``None``.
    """

    if policy_state_or_ref is None:
        return None

    if isinstance(policy_state_or_ref, ray.ObjectRef):
        return ray.get(policy_state_or_ref)

    return policy_state_or_ref


@ray.remote(num_gpus=0)
class Runner:
    """Ray actor that collects training transitions from one environment.

    The runner owns a process-local environment and learner copy. It collects
    transitions using the current policy, stores them into a shared Ray buffer,
    records detailed rollout diagnostics, and periodically synchronizes policy
    weights from the learner.

    Parameters
    ----------
    name : int
        Runner identifier. Used for worker naming and seed offsetting.
    env : gym.Env
        Environment instance assigned to this runner.
    learner : object
        Agent or policy object used to select actions and optionally compute
        TD errors for prioritized replay.
    max_iters : int
        Maximum number of iterations associated with the runner. Stored for
        runtime compatibility.
    policy_type : {"on_policy", "off_policy"}
        Training policy type. On-policy runners stop collecting when the
        shared buffer reaches ``runner.update_after``.
    load_path : str or pathlib.Path or None
        Optional checkpoint path loaded when no policy-state reference is
        provided.
    circuit_type : str
        Circuit family identifier used for default project naming.
    seed : int or None, optional
        Base random seed. The runner adds ``name`` to this value. The default
        is ``None``.
    project_name : str or None, optional
        Project namespace used for simulator workspaces. If ``None``, a
        default name is built from ``circuit_type``. The default is ``None``.
    run_id : str or None, optional
        Shared experiment run identifier. If missing, a new run root is
        created. The default is ``None``.
    run_root : str or pathlib.Path or None, optional
        Shared experiment run-root directory. If missing, a new run root is
        created. The default is ``None``.
    collection_batch_size : int, optional
        Number of transitions buffered locally before remote insertion into
        the shared replay buffer. The default is ``16``.

    Attributes
    ----------
    train_env : gym.Env
        Process-local training environment.
    runner : object
        Process-local learner or policy object.
    seed : int
        Runner-specific random seed.
    project_name : str
        Project namespace used for workspace creation.
    run_id : str
        Experiment run identifier.
    run_root : str
        Experiment run-root directory.
    collection_batch_size : int
        Local transition batch size for remote buffer insertion.
    state : numpy.ndarray
        Current environment state.
    ep_ret : float
        Current episode return accumulator.
    ep_len : int
        Current episode length accumulator.
    """

    def __init__(
        self,
        name,
        env,
        learner,
        max_iters,
        policy_type,
        load_path,
        circuit_type,
        seed=None,
        project_name=None,
        run_id=None,
        run_root=None,
        collection_batch_size=16,
    ):
        """Initialize a distributed rollout runner.

        Parameters
        ----------
        name : int
            Runner identifier.
        env : gym.Env
            Environment instance assigned to this runner.
        learner : object
            Agent or policy object used by the runner.
        max_iters : int
            Maximum iteration count stored for compatibility.
        policy_type : {"on_policy", "off_policy"}
            Training policy type.
        load_path : str or pathlib.Path or None
            Optional checkpoint path used when no policy-state reference is
            supplied.
        circuit_type : str
            Circuit family identifier used for default project naming.
        seed : int or None, optional
            Base random seed. The default is ``None``.
        project_name : str or None, optional
            Project namespace. The default is ``None``.
        run_id : str or None, optional
            Experiment run identifier. The default is ``None``.
        run_root : str or pathlib.Path or None, optional
            Experiment run-root directory. The default is ``None``.
        collection_batch_size : int, optional
            Number of locally buffered transitions before remote storage. The
            default is ``16``.

        Returns
        -------
        None
            The runner configures its workspace, seeds RNGs, resets the
            environment, and initializes episode counters in place.
        """

        self.name = name
        self.train_env = env
        self.runner = learner
        self.max_iters = max_iters
        self.policy_type = policy_type
        self.load_path = load_path

        base_seed = 0 if seed is None else int(seed)
        self.seed = base_seed + self.name

        self.circuit_type = circuit_type
        self.project_name = project_name or f"{self.circuit_type}_runner_job"
        self.run_id = run_id
        self.run_root = run_root
        self.collection_batch_size = int(collection_batch_size)

        self.train_env.gamma = self.runner.gamma

        if self.run_id is None or self.run_root is None:
            self.run_id, actual_run_root = create_experiment_run_root(
                self.project_name
            )
            self.run_root = str(actual_run_root)

        self.train_env.configure_workspace(
            project_name=self.project_name,
            run_id=self.run_id,
            run_root=self.run_root,
            worker_name=self.name,
            scope="train",
            clean=True,
        )

        seed_all(self.seed)

        self.state, _ = get_reset_result(
            self.train_env,
            self.seed,
            options={"reset_mode": "random"},
        )
        self.ep_ret, self.ep_len = 0, 0

    def _flush_transition_batch(
        self,
        buffer,
        pending_transitions,
        pending_store_refs,
    ):
        """Submit locally buffered transitions to a shared Ray buffer.

        Parameters
        ----------
        buffer : ray.actor.ActorHandle
            Remote replay or rollout buffer exposing ``store_many.remote``.
        pending_transitions : list[tuple]
            Locally accumulated transition tuples waiting to be stored.
        pending_store_refs : list[ray.ObjectRef]
            List to which the asynchronous remote store reference is appended.

        Returns
        -------
        int
            Number of transitions flushed to the shared buffer.
        """

        if not pending_transitions:
            return 0

        pending_store_refs.append(
            buffer.store_many.remote(list(pending_transitions))
        )
        flushed = len(pending_transitions)
        pending_transitions.clear()

        return flushed

    @staticmethod
    def _format_action_tensor(action, device):
        """Convert an action into a batched float tensor.

        Parameters
        ----------
        action : array-like or scalar
            Action selected by the policy.
        device : torch.device or str
            Device on which the tensor should be allocated.

        Returns
        -------
        torch.Tensor
            Batched action tensor with shape ``(1, action_dim)``. Scalar
            actions are converted to shape ``(1, 1)``.
        """

        action_array = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=device,
        )

        if action_array.ndim == 0:
            action_array = action_array.unsqueeze(0)

        return action_array.unsqueeze(0)

    def run(
        self,
        buffer,
        runner_iters,
        best_fom,
        policy_state_ref=None,
        policy_version=None,
        progress_tracker=None,
    ):
        """Collect transitions and write them to a shared buffer.

        Parameters
        ----------
        buffer : ray.actor.ActorHandle
            Shared replay or rollout buffer actor.
        runner_iters : int
            Maximum number of environment steps to collect in this call.
        best_fom : float
            Best figure of merit observed by the scheduler or learner. This is
            copied to the environment for logging or reward logic.
        policy_state_ref : object or ray.ObjectRef or None, optional
            Policy state or Ray object reference containing policy state. If
            provided, the runner synchronizes its local policy before
            collection. The default is ``None``.
        policy_version : int or None, optional
            Monotonic policy version used to avoid redundant synchronization.
            The default is ``None``.
        progress_tracker : ray.actor.ActorHandle or None, optional
            Optional actor exposing ``increment.remote`` for progress
            reporting. The default is ``None``.

        Returns
        -------
        tuple
            Rollout summary containing:

            ``name`` : int
                Runner identifier.
            ``timesteps`` : int
                Number of environment steps collected.
            ``ep_counter`` : int
                Number of completed episodes.
            ``total_ep_ret`` : list[float]
                Episode returns completed during this collection call.
            ``total_ep_len`` : list[int]
                Episode lengths completed during this collection call.
            ``results`` : list[dict]
                Per-step structured rollout records.
            ``elapse`` : list[int]
                Elapsed wall-clock time as ``[hours, minutes, seconds]``.
        """

        ep_counter, timesteps = 0, 0
        total_ep_ret, total_ep_len = [], []

        self._sync_policy_state(policy_state_ref, policy_version)
        self.train_env.best_fom = best_fom

        results = []
        pending_transitions = []
        pending_store_refs = []
        start_time = time.time()

        buffer_size = ray.get(buffer.size.remote())

        while timesteps < runner_iters:
            if self.policy_type == "on_policy":
                if buffer_size >= self.runner.update_after:
                    break

            timesteps += 1

            if progress_tracker is not None:
                progress_tracker.increment.remote(self.name, 1)

            effective_buffer_size = buffer_size + len(pending_transitions)
            action = self.runner.act(
                self.state,
                global_buffer_size=effective_buffer_size,
            )
            next_state, reward, terminated, truncated, info = get_next_step(
                self.train_env,
                action,
            )

            if self.policy_type == "off_policy":
                if self.runner.prioritized_mode:
                    state_tensor = torch.tensor(
                        [self.state],
                        dtype=torch.float32,
                        device=self.runner.device,
                    )
                    action_tensor = self._format_action_tensor(
                        action,
                        self.runner.device,
                    )
                    reward_tensor = torch.tensor(
                        [reward],
                        dtype=torch.float32,
                        device=self.runner.device,
                    )
                    next_state_tensor = torch.tensor(
                        [next_state],
                        dtype=torch.float32,
                        device=self.runner.device,
                    )
                    terminated_tensor = torch.tensor(
                        [terminated],
                        dtype=torch.float32,
                        device=self.runner.device,
                    )

                    td_error = self.runner.calculate_td_error(
                        state_tensor,
                        action_tensor,
                        reward_tensor,
                        next_state_tensor,
                        terminated_tensor,
                    )
                    td_error = (
                        td_error.detach()
                        .cpu()
                        .abs()
                        .numpy()
                        .flatten()
                        .item()
                    )
                    new_prio = td_error + self.runner.prio_eps

                    pending_transitions.append(
                        (
                            self.state,
                            action,
                            reward,
                            next_state,
                            terminated,
                            truncated,
                            new_prio,
                        )
                    )
                else:
                    pending_transitions.append(
                        (
                            self.state,
                            action,
                            reward,
                            next_state,
                            terminated,
                            truncated,
                        )
                    )
            else:
                pending_transitions.append(
                    (
                        self.state,
                        action,
                        reward,
                        next_state,
                        terminated,
                        truncated,
                    )
                )

            if len(pending_transitions) >= self.collection_batch_size:
                buffer_size += self._flush_transition_batch(
                    buffer,
                    pending_transitions,
                    pending_store_refs,
                )

            step_record = {
                "agent_timesteps": timesteps,
                "performances": self.train_env.performances.tolist(),
                "parameters": list(self.train_env.design_variables_config.values()),
                "action": list(self.train_env.action),
                "action_info": dict(
                    getattr(self.runner, "last_action_info", {}) or {}
                ),
                "viol": self.train_env.viol,
                "perf": self.train_env.perf,
                "fom": self.train_env.fom,
                "var": self.train_env.var,
                "pbrs_viol": self.train_env.pbrs_viol,
                "pbrs_perf": self.train_env.pbrs_perf,
                "reward_viol": self.train_env.reward_viol,
                "reward_perf": self.train_env.reward_perf,
                "reward": self.train_env.reward,
                "info": info,
                "simulation_profile": dict(
                    getattr(self.train_env, "last_simulation_profile", {}) or {}
                ),
                "reset_info": None,
            }
            results.append(step_record)

            self.ep_ret += reward
            self.ep_len += 1
            self.state = next_state

            finished = terminated or truncated

            if finished:
                ep_counter += 1
                total_ep_ret.append(self.ep_ret)
                total_ep_len.append(self.ep_len)

                self.ep_ret, self.ep_len = 0, 0

                reset_mode = select_reset_mode(self.train_env, info)
                self.state, reset_info = get_reset_result(
                    self.train_env,
                    options={"reset_mode": reset_mode},
                )
                step_record["reset_info"] = reset_info

        buffer_size += self._flush_transition_batch(
            buffer,
            pending_transitions,
            pending_store_refs,
        )

        if pending_store_refs:
            ray.get(pending_store_refs)

        elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
        total_seconds = int(elapsed_time.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapse = [hours, minutes, seconds]

        return (
            self.name,
            timesteps,
            ep_counter,
            total_ep_ret,
            total_ep_len,
            results,
            elapse,
        )

    def _sync_policy_state(self, policy_state_ref=None, policy_version=None):
        """Synchronize the local policy state when needed.

        Parameters
        ----------
        policy_state_ref : object or ray.ObjectRef or None, optional
            Policy state or Ray object reference containing policy state. If
            ``None``, the runner optionally loads ``self.load_path``.
            The default is ``None``.
        policy_version : int or None, optional
            Monotonic policy version. If the supplied version is not newer than
            the current local version, synchronization is skipped. The default
            is ``None``.

        Returns
        -------
        None
            The local runner policy is updated in place when synchronization
            is required.
        """

        if policy_state_ref is None:
            if self.load_path:
                self.runner.load(self.load_path)
            return

        current_version = getattr(self, "_policy_version", -1)

        if policy_version is not None and policy_version <= current_version:
            return

        policy_state = _resolve_policy_state(policy_state_ref)
        self.runner.set_policy_state(policy_state)
        self._policy_version = (
            policy_version
            if policy_version is not None
            else current_version + 1
        )


@ray.remote(num_gpus=0)
class EvalRunner:
    """Ray actor that evaluates a policy in a separate environment.

    Parameters
    ----------
    name : int
        Evaluation runner identifier.
    env : gym.Env
        Environment instance assigned to the evaluation actor.
    learner : object
        Agent or policy object used for deterministic evaluation actions.
    seed : int or None, optional
        Base random seed. The evaluation runner offsets it by ``10000`` and
        ``name``. The default is ``None``.
    project_name : str or None, optional
        Project namespace used for evaluation workspaces. The default is
        ``None``.
    run_id : str or None, optional
        Experiment run identifier. If missing, a new run root is created. The
        default is ``None``.
    run_root : str or pathlib.Path or None, optional
        Experiment run-root directory. If missing, a new run root is created.
        The default is ``None``.
    eval_episodes : int, optional
        Number of episodes evaluated per call. The default is ``1``.
    """

    def __init__(
        self,
        name,
        env,
        learner,
        seed=None,
        project_name=None,
        run_id=None,
        run_root=None,
        eval_episodes=1,
    ):
        """Initialize an evaluation runner.

        Parameters
        ----------
        name : int
            Evaluation runner identifier.
        env : gym.Env
            Environment instance assigned to this actor.
        learner : object
            Agent or policy object used for evaluation.
        seed : int or None, optional
            Base random seed. The default is ``None``.
        project_name : str or None, optional
            Project namespace. The default is ``None``.
        run_id : str or None, optional
            Experiment run identifier. The default is ``None``.
        run_root : str or pathlib.Path or None, optional
            Experiment run-root directory. The default is ``None``.
        eval_episodes : int, optional
            Number of evaluation episodes per call. The default is ``1``.

        Returns
        -------
        None
            The runner configures its evaluation workspace and seeds RNGs in
            place.
        """

        self.name = name
        self.eval_env = env
        self.runner = learner

        base_seed = 0 if seed is None else int(seed)
        self.seed = base_seed + 10000 + int(name)

        self.eval_episodes = max(1, int(eval_episodes))
        self.project_name = project_name or "eval_job"
        self.run_id = run_id
        self.run_root = run_root

        if self.run_id is None or self.run_root is None:
            self.run_id, actual_run_root = create_experiment_run_root(
                self.project_name
            )
            self.run_root = str(actual_run_root)

        self.eval_env.configure_workspace(
            project_name=self.project_name,
            run_id=self.run_id,
            run_root=self.run_root,
            worker_name=f"eval_{self.name}",
            scope="eval",
            clean=True,
        )

        seed_all(self.seed)

    def evaluate(
        self,
        policy_state_ref=None,
        policy_version=None,
        eval_timesteps=None,
    ):
        """Evaluate the current policy for one or more episodes.

        Parameters
        ----------
        policy_state_ref : object or ray.ObjectRef or None, optional
            Policy state or Ray object reference containing policy state. If
            provided, the evaluation policy is synchronized before evaluation.
            The default is ``None``.
        policy_version : int or None, optional
            Monotonic policy version used to skip redundant synchronization.
            The default is ``None``.
        eval_timesteps : int or None, optional
            Optional global timestep used to shift the evaluation seed. The
            default is ``None``.

        Returns
        -------
        list[tuple[float, int]]
            Episode-level evaluation results. Each tuple contains
            ``(episode_return, episode_length)``.
        """

        self._sync_policy_state(policy_state_ref, policy_version)

        results = []
        base_eval_seed = (
            self.seed
            if eval_timesteps is None
            else self.seed + int(eval_timesteps)
        )

        for episode_idx in range(self.eval_episodes):
            ep_ret, ep_len, finished = 0, 0, False
            state = get_reset_state(
                self.eval_env,
                seed=base_eval_seed + episode_idx,
                options={"reset_mode": "random"},
            )

            while not finished:
                action = self.runner.act(state, training=False)
                next_state, reward, terminated, truncated, _ = get_next_step(
                    self.eval_env,
                    action,
                )

                ep_ret += reward
                ep_len += 1
                state = next_state
                finished = terminated or truncated

            results.append((ep_ret, ep_len))

        return results

    def _sync_policy_state(self, policy_state_ref=None, policy_version=None):
        """Synchronize the local evaluation policy state when needed.

        Parameters
        ----------
        policy_state_ref : object or ray.ObjectRef or None, optional
            Policy state or Ray object reference containing policy state. If
            ``None``, synchronization is skipped. The default is ``None``.
        policy_version : int or None, optional
            Monotonic policy version. If the supplied version is not newer than
            the current local version, synchronization is skipped. The default
            is ``None``.

        Returns
        -------
        None
            The local evaluation policy is updated in place when needed.
        """

        if policy_state_ref is None:
            return

        current_version = getattr(self, "_policy_version", -1)

        if policy_version is not None and policy_version <= current_version:
            return

        policy_state = _resolve_policy_state(policy_state_ref)
        self.runner.set_policy_state(policy_state)
        self._policy_version = (
            policy_version
            if policy_version is not None
            else current_version + 1
        )