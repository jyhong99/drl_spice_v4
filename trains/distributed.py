"""Distributed training coordinator for Ray-based reinforcement learning.

This module defines the high-level distributed trainer used to coordinate
parallel rollout workers, shared replay or rollout buffers, learner updates,
evaluation actors, checkpointing, structured logging, progress reporting, and
simulator workspace cleanup.
"""

import datetime
import os
import time

import numpy as np
import ray
import torch

from loggers.writer import TrainingLogger
from simulator.ngspice.workspace import (
    cleanup_experiment_run_root,
    create_experiment_run_root,
)
from trains.checkpoint import (
    get_checkpoint_paths,
    load_buffer_checkpoint,
    load_learner_checkpoint,
    save_buffer_checkpoint,
    save_learner_checkpoint,
)
from trains.factory import (
    create_active_runner_tasks,
    create_eval_runners,
    create_runner_actors,
    create_shared_buffer_actor,
)
from trains.reporting import (
    build_end_log,
    finalize_timing,
    print_circuit_summary,
    print_training_start,
    save_logs,
    save_logs_and_plot,
)
from trains.learners import (
    enrich_with_buffer_stats,
    run_distributed_off_policy_prioritized_update,
    run_distributed_off_policy_uniform_update,
    run_distributed_on_policy_update,
)
from trains.loop import (
    drain_active_eval_batch,
    maybe_collect_active_eval,
    maybe_start_eval_batch,
    process_runner_completion,
    update_last_rollout_chunk_effective_utd,
)
from trains.progress import TrainingProgressBars, WorkerProgressTracker
from trains.results import (
    append_learner_result,
    append_scheduler_metrics,
)
from trains.scheduler import (
    compute_effective_utd,
    compute_utd_update_budget,
    enqueue_due_eval_requests,
    should_finalize_on_policy_batch,
    should_stop_on_policy_rescheduling,
)


class DistributedTrainer(object):
    """Coordinator for distributed Ray-based RL training.

    This class manages a distributed training loop with multiple rollout
    actors, a shared buffer actor, optional evaluation actors, learner updates,
    checkpointing, logging, and progress reporting. It supports both
    on-policy and off-policy training flows, including prioritized replay for
    compatible off-policy learners.

    Parameters
    ----------
    env : gym.Env
        Training environment used as the template environment for rollout
        workers.
    eval_env : gym.Env
        Evaluation environment used for coordinator-side and evaluation-runner
        workspaces.
    agent : object
        Learner or RL agent object. The object is expected to expose methods
        such as ``act``, ``save``, ``load``, ``get_policy_state``, and learner
        update-related attributes.
    seed : int
        Base random seed for training, rollout workers, and evaluation
        workers.

    Attributes
    ----------
    train_env : gym.Env
        Training environment template.
    eval_env : gym.Env
        Evaluation environment template.
    learner : object
        Learner or RL agent managed by the trainer.
    seed : int
        Base random seed.
    device : torch.device
        Device used for converting learner batches.
    """

    def __init__(self, env, eval_env, agent, seed):
        """Initialize the distributed trainer.

        Parameters
        ----------
        env : gym.Env
            Training environment.
        eval_env : gym.Env
            Evaluation environment.
        agent : object
            Learner or RL agent.
        seed : int
            Base random seed.

        Returns
        -------
        None
            The trainer stores core objects, selects the compute device, and
            configures Ray-related environment variables in place.
        """

        self.train_env = env
        self.eval_env = eval_env
        self.learner = agent
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        os.environ["RAY_verbose_spill_logs"] = "0"
        os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

    @staticmethod
    def _refresh_worker_progress(progress, progress_tracker, pending_progress_ref):
        """Refresh progress bars from asynchronous worker counters.

        Parameters
        ----------
        progress : TrainingProgressBars
            Progress-bar manager used to display global and worker progress.
        progress_tracker : ray.actor.ActorHandle
            Remote progress tracker exposing ``snapshot.remote``.
        pending_progress_ref : ray.ObjectRef or None
            Pending Ray reference for a progress snapshot. If ``None``, a new
            snapshot request is issued.

        Returns
        -------
        ray.ObjectRef
            Ray reference for the next or still-pending progress snapshot.
        """

        if pending_progress_ref is None:
            return progress_tracker.snapshot.remote()

        ready_refs, _ = ray.wait([pending_progress_ref], num_returns=1, timeout=0)

        if not ready_refs:
            return pending_progress_ref

        progress.sync_worker_counts(ray.get(ready_refs[0]))
        progress.refresh_global_from_workers()

        return progress_tracker.snapshot.remote()

    def train(
        self,
        project_name,
        load_path,
        max_iters,
        n_runners,
        runner_iters,
        utd_ratio=1.0,
        eval_mode=False,
        eval_intervals=100,
        checkpoint_intervals=None,
        eval_iters=10,
        policy_type=None,
        action_type=None,
        circuit_type=None,
    ):
        """Run distributed reinforcement-learning training.

        This method initializes logging, workspaces, Ray actors, shared
        buffers, checkpoint paths, rollout workers, optional evaluation
        workers, and the main distributed training loop. It schedules rollout
        collection, learner updates, checkpoint saves, evaluation requests,
        structured logging, progress-bar updates, and final cleanup.

        Parameters
        ----------
        project_name : str
            Experiment project name used for logs, checkpoints, and simulator
            workspaces.
        load_path : str or os.PathLike or None
            Optional directory containing previous ``model.pth`` and
            ``buffer.pkl`` checkpoints. If falsy, training starts from the
            current learner state.
        max_iters : int
            Maximum total environment timesteps to collect.
        n_runners : int
            Number of distributed rollout runner actors.
        runner_iters : int
            Maximum number of timesteps each runner actor collects per
            scheduled rollout task.
        utd_ratio : float or None, optional
            Target updates-to-data ratio for off-policy learner updates. If
            ``None``, it is treated as ``1.0``. The default is ``1.0``.
        eval_mode : bool, optional
            Whether to create evaluation actors and run periodic evaluations.
            The default is ``False``.
        eval_intervals : int, optional
            Timestep interval between evaluation requests. The default is
            ``100``.
        checkpoint_intervals : int or None, optional
            Timestep interval between checkpoint saves. If ``None``, it is set
            to ``eval_intervals``. The default is ``None``.
        eval_iters : int, optional
            Number of evaluation episodes or evaluation tasks, depending on
            the evaluation-runner factory implementation. The default is
            ``10``.
        policy_type : {"on_policy", "off_policy"} or None, optional
            Policy/update family used by the learner. The default is ``None``.
        action_type : {"continuous", "discrete", "multidiscrete"} or None, optional
            Action-space type used to infer action dimensions. The default is
            ``None``.
        circuit_type : str or None, optional
            Circuit family identifier used for logging and worker workspace
            naming. The default is ``None``.

        Returns
        -------
        None
            Training artifacts, checkpoints, logs, and plots are written to
            disk. The method returns normally when no valid best result is
            available and no exception is pending.

        Raises
        ------
        ValueError
            If ``utd_ratio`` is negative or if ``checkpoint_intervals`` is not
            positive.
        Exception
            Re-raises unexpected exceptions after recording cleanup state.
        """

        self.project_name = project_name
        self.load_path = load_path
        self.max_iters = max_iters
        self.n_runners = n_runners
        self.runner_iters = runner_iters

        self.utd_ratio = float(1.0 if utd_ratio is None else utd_ratio)
        if self.utd_ratio < 0:
            raise ValueError("utd_ratio must be non-negative")

        self.eval_mode = eval_mode
        self.eval_intervals = eval_intervals

        if checkpoint_intervals is None:
            checkpoint_intervals = eval_intervals

        self.checkpoint_intervals = int(checkpoint_intervals)
        if self.checkpoint_intervals <= 0:
            raise ValueError("checkpoint_intervals must be positive")

        self.eval_iters = eval_iters
        self.circuit_type = circuit_type

        self.policy_type = policy_type
        self.action_type = action_type

        self.start_time = time.time()
        self.start_time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.logger = TrainingLogger(self.project_name)
        self.train_logger = self.logger.train_logs
        self.epoch_logger = self.logger.epoch_logs

        self.state_dim = self.train_env.observation_space.shape[0]

        if self.action_type == "continuous":
            self.action_dim = self.train_env.action_space.shape[0]
        elif self.action_type == "discrete":
            self.action_dim = self.train_env.action_space.n
        elif self.action_type == "multidiscrete":
            self.action_dim = self.train_env.action_space.nvec

        self.action_storage_shape = self.learner.action_storage_shape

        if self.policy_type == "off_policy":
            self.prioritized_mode = self.learner.prioritized_mode
            self.batch_size = self.learner.batch_size
            self.prio_alpha = self.learner.prio_alpha
            self.prio_beta = self.learner.prio_beta
            self.prio_eps = self.learner.prio_eps
            self.collection_batch_size = int(
                self.learner.config.get("collection_batch_size", 16)
            )
        else:
            self.collection_batch_size = 16

        self.init_log = {
            "start_time": self.start_time_now,
            "env": self.train_env,
            "eval_env": self.eval_env,
            "agent": self.learner,
            "agent_config": self.learner.config,
            "seed": self.seed,
            "project_name": self.project_name,
            "run_id": None,
            "run_root": None,
            "load_path": self.load_path,
            "max_iters": self.max_iters,
            "n_runners": self.n_runners,
            "runner_iters": self.runner_iters,
            "utd_ratio": self.utd_ratio,
            "eval_mode": self.eval_mode,
            "eval_intervals": self.eval_intervals,
            "checkpoint_intervals": self.checkpoint_intervals,
            "eval_iters": self.eval_iters,
            "circuit_type": self.circuit_type,
        }

        self.run_id, self.run_root = create_experiment_run_root(self.project_name)
        self.init_log["run_id"] = self.run_id
        self.init_log["run_root"] = str(self.run_root)
        self.train_logger.append(self.init_log)

        print_training_start(self.init_log, distributed=True)

        self.init_log2 = {
            "target_spec": self.train_env.target_spec,
            "target_bound": self.train_env.bound,
            "fixed_values": self.train_env.fixed_values,
            "max_steps": self.train_env.max_steps,
            "n_variables": self.train_env.n_variables,
        }
        self.train_logger.append(self.init_log2)

        self.eval_env.configure_workspace(
            project_name=self.project_name,
            run_id=self.run_id,
            run_root=self.run_root,
            worker_name="coordinator",
            scope="eval",
            clean=True,
        )

        self.obs_logger = self.logger.obs_logs
        self.best_performances = []
        self.best_parameters = []
        self.best_fom = -np.inf

        model_path = None
        buffer_path = None
        buffer = None
        ray_started = False
        pending_exception = None

        ray.init()
        ray_started = True

        try:
            self.buffer_size = self.learner.buffer_size

            buffer = create_shared_buffer_actor(
                policy_type=self.policy_type,
                prioritized_mode=getattr(self, "prioritized_mode", False),
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                action_storage_shape=self.action_storage_shape,
                buffer_size=self.buffer_size,
                batch_size=getattr(self, "batch_size", None),
                device=self.device,
                prio_alpha=getattr(self, "prio_alpha", None),
                seed=self.seed,
            )

            checkpoint_paths = get_checkpoint_paths(self.project_name)
            model_path = checkpoint_paths["model"]
            buffer_path = checkpoint_paths["buffer"]

            if load_path:
                loaded_model_path = load_learner_checkpoint(self.learner, load_path)
                if loaded_model_path is not None:
                    print(f"LEARNER HAS BEEN LOADED FROM: {loaded_model_path}")

                loaded_buffer_path = load_buffer_checkpoint(buffer, load_path)
                if loaded_buffer_path is not None:
                    print(f"BUFFER HAS BEEN LOADED FROM: {loaded_buffer_path}")
                else:
                    print("BUFFER HAS BEEN FAILED TO BE LOADED.")

            runners = create_runner_actors(
                n_runners=self.n_runners,
                env=self.train_env,
                learner=self.learner,
                max_iters=self.max_iters,
                policy_type=self.policy_type,
                load_path=model_path,
                circuit_type=self.circuit_type,
                seed=self.seed,
                project_name=self.project_name,
                run_id=self.run_id,
                run_root=self.run_root,
                collection_batch_size=self.collection_batch_size,
            )

            eval_runners = None
            if self.eval_mode:
                eval_runners = create_eval_runners(
                    eval_iters=self.eval_iters,
                    env=self.eval_env,
                    learner=self.learner,
                    seed=self.seed,
                    project_name=self.project_name,
                    run_id=self.run_id,
                    run_root=self.run_root,
                )

            save_learner_checkpoint(self.learner, model_path)

            policy_version = 0
            policy_state_ref = ray.put(self.learner.get_policy_state())

            timesteps, num_eps, train_iters = 0, 0, 0
            learner_updates_total = 0
            utd_credit = 0.0
            checkpoint_counter = 0
            next_eval_timestep = self.eval_intervals
            pending_eval_queue = []
            active_eval_batch = None

            global_ep_ret_sum = 0.0
            global_ep_ret_count = 0
            latest_global_ret = None

            progress = TrainingProgressBars(
                max_iters=self.max_iters,
                n_runners=self.n_runners,
                runner_iters=self.runner_iters,
            )
            progress.setup()

            progress_tracker = WorkerProgressTracker.remote(self.n_runners)
            pending_progress_ref = progress_tracker.snapshot.remote()

            active_runner_tasks = create_active_runner_tasks(
                runners,
                buffer=buffer,
                runner_iters=self.runner_iters,
                best_fom=self.best_fom,
                policy_state_ref=policy_state_ref,
                policy_version=policy_version,
                progress_tracker=progress_tracker,
            )
            on_policy_batch_closed = False

            try:
                while timesteps < max_iters and active_runner_tasks:
                    train_iters = 0

                    if self.eval_mode:
                        active_eval_batch = maybe_collect_active_eval(
                            self.epoch_logger,
                            active_eval_batch,
                            num_eps,
                        )
                        active_eval_batch = maybe_start_eval_batch(
                            eval_runners,
                            pending_eval_queue,
                            active_eval_batch,
                        )

                    done_refs, _ = ray.wait(
                        list(active_runner_tasks.keys()),
                        num_returns=1,
                        timeout=0.1,
                    )
                    pending_progress_ref = self._refresh_worker_progress(
                        progress,
                        progress_tracker,
                        pending_progress_ref,
                    )

                    if not done_refs:
                        continue

                    done_ref = done_refs[0]
                    completion = process_runner_completion(
                        done_ref=done_ref,
                        active_runner_tasks=active_runner_tasks,
                        obs_logger=self.obs_logger,
                        timesteps=timesteps,
                        train_iters=train_iters,
                        learner_updates_total=learner_updates_total,
                        policy_version=policy_version,
                        best_fom=self.best_fom,
                        best_performances=self.best_performances,
                        best_parameters=self.best_parameters,
                    )

                    runner = completion["runner"]
                    name = completion["name"]
                    train_iters = completion["time_per_run"]
                    num_eps += completion["ep_per_run"]
                    runner_results = completion["runner_results"]

                    self.best_performances = completion["best_performances"]
                    self.best_parameters = completion["best_parameters"]
                    self.best_fom = completion["best_fom"]

                    global_ep_ret_sum += float(sum(completion["ep_ret"]))
                    global_ep_ret_count += int(len(completion["ep_ret"]))

                    if completion["ep_ret"]:
                        latest_global_ret = float(completion["ep_ret"][-1])

                    progress.update_worker_stats(
                        name,
                        ep_ret=completion["ep_ret"],
                        best_fom=completion["chunk_best_fom"],
                    )

                    timesteps += train_iters
                    checkpoint_counter += train_iters

                    buffer_stats = ray.get(buffer.stats.remote())
                    buffer_size = int(buffer_stats["size"])

                    learner_updated = False
                    chunk_updates = 0
                    result = None

                    if self.policy_type == "on_policy":
                        on_policy_batch_closed = should_stop_on_policy_rescheduling(
                            buffer_size=buffer_size,
                            update_after=self.learner.update_after,
                            timesteps=timesteps,
                            max_iters=max_iters,
                        )

                    elif buffer_size >= self.learner.update_after:
                        if self.policy_type == "off_policy":
                            chunk_updates, utd_credit = compute_utd_update_budget(
                                train_iters,
                                self.utd_ratio,
                                utd_credit,
                            )

                            for t in range(1, chunk_updates + 1):
                                if self.prioritized_mode:
                                    fraction = min(timesteps / max_iters, 1.0)
                                    self.prio_beta = (
                                        self.learner.prio_beta_start
                                        + fraction
                                        * (1.0 - self.learner.prio_beta_start)
                                    )
                                    result = (
                                        run_distributed_off_policy_prioritized_update(
                                            buffer=buffer,
                                            learner=self.learner,
                                            to_device_batch=self._to_device_batch,
                                            prio_beta=self.prio_beta,
                                            prio_eps=self.prio_eps,
                                            learner_step=learner_updates_total + t,
                                        )
                                    )
                                else:
                                    result = run_distributed_off_policy_uniform_update(
                                        buffer=buffer,
                                        learner=self.learner,
                                        to_device_batch=self._to_device_batch,
                                        learner_step=learner_updates_total + t,
                                    )

                                if result is not None:
                                    result = enrich_with_buffer_stats(
                                        result,
                                        buffer_stats,
                                    )
                                    append_learner_result(
                                        self.epoch_logger,
                                        timesteps=timesteps,
                                        learner_step=learner_updates_total + t,
                                        result=result,
                                    )

                            learner_updates_total += chunk_updates
                            learner_updated = chunk_updates > 0

                    if learner_updated:
                        policy_version += 1
                        policy_state_ref = ray.put(self.learner.get_policy_state())

                        if checkpoint_counter >= self.checkpoint_intervals:
                            save_learner_checkpoint(self.learner, model_path)
                            save_buffer_checkpoint(buffer, buffer_path, wait=False)
                            checkpoint_counter = 0

                    effective_utd = compute_effective_utd(
                        learner_updates_total,
                        timesteps,
                    )
                    update_last_rollout_chunk_effective_utd(
                        self.obs_logger,
                        effective_utd,
                    )
                    append_scheduler_metrics(
                        self.epoch_logger,
                        timesteps=timesteps,
                        learner_updates_total=learner_updates_total,
                        chunk_env_steps=train_iters,
                        chunk_updates=chunk_updates,
                        utd_ratio_target=self.utd_ratio,
                        utd_credit_residual=utd_credit,
                        effective_utd=effective_utd,
                        buffer_size=int(buffer_stats["size"]),
                        buffer_fill_ratio=float(buffer_stats["fill_ratio"]),
                        buffer_store_calls=int(buffer_stats["store_calls"]),
                        buffer_sample_calls=int(buffer_stats["sample_calls"]),
                        buffer_stored_transitions=int(
                            buffer_stats["stored_transitions"]
                        ),
                        buffer_overwrite_count=int(buffer_stats["overwrite_count"]),
                    )

                    progress.update_global(
                        timesteps=timesteps,
                        ep=global_ep_ret_count,
                        ret=latest_global_ret,
                        best_fom=self.best_fom,
                    )

                    if self.policy_type == "on_policy":
                        if (not on_policy_batch_closed) and timesteps < max_iters:
                            active_runner_tasks[
                                runner.run.remote(
                                    buffer=buffer,
                                    runner_iters=self.runner_iters,
                                    best_fom=self.best_fom,
                                    policy_state_ref=policy_state_ref,
                                    policy_version=policy_version,
                                    progress_tracker=progress_tracker,
                                )
                            ] = runner

                        if should_finalize_on_policy_batch(
                            active_runner_tasks=active_runner_tasks,
                            buffer_size=buffer_size,
                            update_after=self.learner.update_after,
                            timesteps=timesteps,
                            max_iters=max_iters,
                        ):
                            result = run_distributed_on_policy_update(
                                buffer=buffer,
                                learner=self.learner,
                                to_device_batch=self._to_device_batch,
                            )
                            append_learner_result(
                                self.epoch_logger,
                                timesteps=timesteps,
                                result=result,
                            )

                            learner_updated = True
                            policy_version += 1
                            policy_state_ref = ray.put(self.learner.get_policy_state())
                            on_policy_batch_closed = False

                            if checkpoint_counter >= self.checkpoint_intervals:
                                save_learner_checkpoint(self.learner, model_path)
                                save_buffer_checkpoint(
                                    buffer,
                                    buffer_path,
                                    wait=False,
                                )
                                checkpoint_counter = 0

                            if timesteps < max_iters:
                                active_runner_tasks = create_active_runner_tasks(
                                    runners,
                                    buffer=buffer,
                                    runner_iters=self.runner_iters,
                                    best_fom=self.best_fom,
                                    policy_state_ref=policy_state_ref,
                                    policy_version=policy_version,
                                    progress_tracker=progress_tracker,
                                )

                    elif timesteps < max_iters:
                        active_runner_tasks[
                            runner.run.remote(
                                buffer=buffer,
                                runner_iters=self.runner_iters,
                                best_fom=self.best_fom,
                                policy_state_ref=policy_state_ref,
                                policy_version=policy_version,
                                progress_tracker=progress_tracker,
                            )
                        ] = runner

                    if self.eval_mode:
                        next_eval_timestep = enqueue_due_eval_requests(
                            current_timesteps=timesteps,
                            next_eval_timestep=next_eval_timestep,
                            eval_intervals=self.eval_intervals,
                            pending_eval_queue=pending_eval_queue,
                            policy_state_ref=policy_state_ref,
                            policy_version=policy_version,
                        )
                        active_eval_batch = maybe_start_eval_batch(
                            eval_runners,
                            pending_eval_queue,
                            active_eval_batch,
                        )

                    self._save_train_logs()
                    self._save_obs_logs()

            finally:
                progress.close()

        except KeyboardInterrupt:
            print(
                "\n================================== TRAINING HAS BEEN SHUTDOWN ==================================\n"
            )

        except Exception as exc:
            pending_exception = exc
            raise

        finally:
            if "active_eval_batch" in locals():
                drain_active_eval_batch(
                    self.epoch_logger,
                    active_eval_batch,
                    timesteps,
                    num_eps,
                )

            self.train_env.close()
            self.eval_env.close()

            if model_path is not None:
                save_learner_checkpoint(self.learner, model_path)

            if buffer is not None and buffer_path is not None:
                save_buffer_checkpoint(buffer, buffer_path, wait=True)

            if ray_started:
                ray.shutdown()

            self.end_time_now, self.time_elapse = finalize_timing(self.start_time)
            end_log = build_end_log(
                best_performances=self.best_performances,
                best_parameters=self.best_parameters,
                best_fom=self.best_fom,
                end_time_now=self.end_time_now,
                time_elapse=self.time_elapse,
            )
            self.train_logger.append(end_log)

            if len(self.best_performances) == 0 or len(self.best_parameters) == 0:
                self._save_train_logs()
                self._save_obs_logs()
                cleanup_experiment_run_root(self.run_root)

                if pending_exception is None:
                    return

            else:
                print_circuit_summary(
                    circuit_type=self.circuit_type,
                    end_time_now=self.end_time_now,
                    time_elapse=self.time_elapse,
                    best_performances=self.best_performances,
                    best_parameters=self.best_parameters,
                    best_fom=self.best_fom,
                )

                save_logs_and_plot(
                    logger=self.logger,
                    project_name=project_name,
                    max_timesteps=self.max_iters,
                )
                cleanup_experiment_run_root(self.run_root)

    def get_logger(self):
        """Return managed training log streams.

        Returns
        -------
        tuple
            Tuple returned by ``self.logger.get_logs()``, typically containing
            train, epoch, and observation log streams.
        """

        return self.logger.get_logs()

    def _save_train_logs(self):
        """Persist current training logs.

        Returns
        -------
        None
            Current logger contents are saved through the reporting helper.
        """

        save_logs(self.logger)

    def _save_obs_logs(self):
        """Persist current observation logs.

        Returns
        -------
        None
            Current logger contents are saved through the reporting helper.
        """

        save_logs(self.logger)

    def _to_device_tensor(self, item):
        """Convert one batch item to a float tensor on the trainer device.

        Parameters
        ----------
        item : array-like or numpy.ndarray or torch.Tensor
            Batch item to convert. Non-writeable NumPy arrays are copied before
            conversion to avoid PyTorch warnings or unsafe views.

        Returns
        -------
        torch.Tensor
            Float tensor allocated on ``self.device``.
        """

        if isinstance(item, np.ndarray) and not item.flags.writeable:
            item = np.array(item, copy=True)

        return torch.as_tensor(
            item,
            dtype=torch.float32,
            device=self.device,
        )

    def _to_device_batch(self, batch, has_weights=False, has_indices=False):
        """Convert a sampled batch to device tensors.

        Parameters
        ----------
        batch : tuple
            Batch returned by a shared buffer actor.
        has_weights : bool, optional
            Whether the batch contains importance-sampling weights. This
            argument is accepted for interface compatibility. The default is
            ``False``.
        has_indices : bool, optional
            Whether the final batch element contains replay-buffer indices that
            should not be converted to tensors. The default is ``False``.

        Returns
        -------
        tuple
            Converted batch. If ``has_indices`` is ``True``, all elements
            except the final index payload are converted to tensors and the
            indices are returned unchanged.
        """

        if has_indices:
            *tensor_items, idxs = batch
            converted = [
                self._to_device_tensor(item)
                for item in tensor_items
            ]
            return (*converted, idxs)

        return tuple(
            self._to_device_tensor(item)
            for item in batch
        )


__all__ = ["DistributedTrainer"]
"""list[str]: Public symbols exported by this module."""