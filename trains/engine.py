"""Single-process training loop for reinforcement-learning agents.

This module defines a non-distributed trainer that runs one environment and
one learner in a local process. It handles simulator workspace setup,
training-step execution, periodic evaluation, checkpointing, structured
logging, progress reporting, final result summarization, and workspace
cleanup.
"""

import datetime
import os
import time

import numpy as np

from loggers.writer import TrainingLogger
from simulator.ngspice.workspace import (
    cleanup_experiment_run_root,
    create_experiment_run_root,
)
from trains.actors import (
    evaluate,
    get_next_step,
    get_reset_result,
)
from trains.checkpoint import (
    get_checkpoint_paths,
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from trains.progress import TrainingProgressBars
from trains.reporting import (
    build_end_log,
    finalize_timing,
    print_circuit_summary,
    print_training_start,
    save_logs,
    save_logs_and_plot,
)
from trains.results import (
    aggregate_global_stats,
    append_learner_result,
    finalize_global_stats,
    update_best_result,
)
from trains.utils import select_reset_mode


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Trainer(object):
    """Single-process reinforcement-learning trainer.

    This trainer executes environment interaction and learner updates in the
    same Python process. It is intended for local training workflows or
    debugging runs where Ray-based distributed rollout collection is not
    required.

    Parameters
    ----------
    env : gym.Env
        Training environment.
    eval_env : gym.Env
        Evaluation environment.
    agent : object
        Reinforcement-learning agent. The object is expected to expose
        ``act``, ``step``, ``save``, ``load``, ``buffer``, and ``config``
        attributes or methods.
    seed : int
        Random seed used for environment resets and evaluation.

    Attributes
    ----------
    train_env : gym.Env
        Training environment.
    eval_env : gym.Env
        Evaluation environment.
    seed : int
        Random seed.
    agent : object
        Reinforcement-learning agent.
    """

    def __init__(self, env, eval_env, agent, seed):
        """Initialize the local trainer.

        Parameters
        ----------
        env : gym.Env
            Training environment.
        eval_env : gym.Env
            Evaluation environment.
        agent : object
            Reinforcement-learning agent.
        seed : int
            Random seed.

        Returns
        -------
        None
            The trainer stores the training environment, evaluation
            environment, agent, and seed in place.
        """

        self.train_env = env
        self.eval_env = eval_env
        self.seed = seed
        self.agent = agent

    def train(
        self,
        project_name,
        load_path,
        max_iters,
        eval_mode,
        eval_intervals,
        eval_iters,
        circuit_type,
    ):
        """Run the single-process training loop.

        The method configures simulator workspaces, initializes logging and
        checkpoints, resets the environment, collects transitions, updates the
        agent, performs periodic evaluation and checkpointing, saves structured
        logs, generates plots, prints a final circuit summary, and removes the
        temporary experiment workspace.

        Parameters
        ----------
        project_name : str
            Experiment project name used for logs, checkpoints, and simulator
            workspaces.
        load_path : str or os.PathLike or None
            Optional directory containing a previous ``model.pth`` checkpoint
            and optionally ``buffer.pkl``. If falsy, training starts from the
            current agent state.
        max_iters : int
            Maximum number of environment steps to run.
        eval_mode : bool
            Whether to run periodic evaluation.
        eval_intervals : int
            Number of training timesteps between checkpoint/evaluation events.
        eval_iters : int
            Number of evaluation episodes or iterations passed to
            :func:`evaluate`.
        circuit_type : str
            Circuit family identifier used in final reporting.

        Returns
        -------
        None
            Training outputs, checkpoints, logs, and plots are written to disk.
        """

        self.project_name = project_name
        self.load_path = load_path
        self.max_iters = max_iters
        self.eval_mode = eval_mode
        self.eval_intervals = eval_intervals
        self.eval_iters = eval_iters

        self.circuit_type = circuit_type
        self.start_time = time.time()
        self.start_time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.logger = TrainingLogger(self.project_name)
        self.train_logger = self.logger.train_logs
        self.epoch_logger = self.logger.epoch_logs

        self.init_log = {
            "start_time": self.start_time_now,
            "env": self.train_env,
            "eval_env": self.eval_env,
            "agent": self.agent,
            "agent_config": self.agent.config,
            "seed": self.seed,
            "project_name": self.project_name,
            "run_id": None,
            "run_root": None,
            "load_path": self.load_path,
            "max_iters": self.max_iters,
            "eval_mode": self.eval_mode,
            "eval_intervals": self.eval_intervals,
            "eval_iters": self.eval_iters,
        }

        self.run_id, self.run_root = create_experiment_run_root(self.project_name)
        self.init_log["run_id"] = self.run_id
        self.init_log["run_root"] = str(self.run_root)

        self.train_env.configure_workspace(
            project_name=self.project_name,
            run_id=self.run_id,
            run_root=self.run_root,
            worker_name="trainer",
            scope="train",
            clean=True,
        )
        self.eval_env.configure_workspace(
            project_name=self.project_name,
            run_id=self.run_id,
            run_root=self.run_root,
            worker_name="trainer",
            scope="eval",
            clean=True,
        )

        self.train_logger.append(self.init_log)

        print_training_start(self.init_log, distributed=False)

        self.obs_logger = self.logger.obs_logs
        self.best_performances = []
        self.best_parameters = []
        self.best_fom = -np.inf

        checkpoint_paths = get_checkpoint_paths(self.project_name)
        model_path = checkpoint_paths["model"]
        buffer_path = checkpoint_paths["buffer"]

        if load_path:
            loaded_model_path = load_learner_checkpoint(self.agent, load_path)

            if loaded_model_path is not None:
                loaded_buffer_path = os.path.join(load_path, "buffer.pkl")

                if os.path.exists(loaded_buffer_path):
                    self.agent.buffer.load(loaded_buffer_path)

                print(f"LEARNER HAS BEEN LOADED FROM: {load_path}")

        global_stats = {
            "max_ret": -np.inf,
            "max_len": 0,
            "mean_ret": 0,
            "mean_len": 0,
            "ep_count": 0,
            "eval_count": 0,
        }

        total_ep_ret, total_ep_len = [], []
        num_eps, ep_ret, ep_len = 0, 0, 0

        global_ep_ret_sum = 0.0
        global_ep_ret_count = 0
        latest_global_ret = None

        state, _ = get_reset_result(
            self.train_env,
            seed=self.seed,
            options={"reset_mode": "random"},
        )

        progress = TrainingProgressBars(
            max_iters=self.max_iters,
            n_runners=1,
        )
        progress.setup()

        try:
            for timesteps in range(self.max_iters):
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = get_next_step(
                    self.train_env,
                    action,
                )
                result = self.agent.step(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    truncated,
                )

                (
                    self.best_performances,
                    self.best_parameters,
                    self.best_fom,
                ) = update_best_result(
                    self.best_fom,
                    self.best_performances,
                    self.best_parameters,
                    performances=self.train_env.performances.tolist(),
                    parameters=list(self.train_env.design_variables_config.values()),
                    fom=self.train_env.fom,
                )

                step_record = {
                    "time_steps": timesteps,
                    "performances": self.train_env.performances.tolist(),
                    "parameters": list(self.train_env.design_variables_config.values()),
                    "action": list(self.train_env.action),
                    "action_info": dict(
                        getattr(self.agent, "last_action_info", {}) or {}
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
                self.obs_logger.append(step_record)

                self.train_env.best_fom = self.best_fom

                ep_ret += reward
                ep_len += 1
                state = next_state
                finished = terminated or truncated

                if result is not None:
                    append_learner_result(
                        self.epoch_logger,
                        timesteps=timesteps,
                        result=result,
                    )

                if finished:
                    num_eps += 1
                    total_ep_ret.append(ep_ret)
                    total_ep_len.append(ep_len)

                    global_ep_ret_sum += float(ep_ret)
                    global_ep_ret_count += 1
                    latest_global_ret = float(ep_ret)

                    ep_ret, ep_len = 0, 0

                    reset_mode = select_reset_mode(self.train_env, info)
                    state, reset_info = get_reset_result(
                        self.train_env,
                        options={"reset_mode": reset_mode},
                    )
                    step_record["reset_info"] = reset_info

                progress.update_global(
                    timesteps=timesteps + 1,
                    ep=global_ep_ret_count,
                    ret=latest_global_ret,
                    best_fom=self.best_fom,
                )

                if (timesteps + 1) % self.eval_intervals == 0:
                    if self.eval_mode is True:
                        total_ep_ret, total_ep_len = evaluate(
                            self.eval_env,
                            self.agent,
                            self.seed,
                            self.eval_iters,
                        )

                    if self.eval_mode is True and total_ep_ret != [] and total_ep_len != []:
                        max_ep_ret = np.max(total_ep_ret)
                        max_ep_len = np.max(total_ep_len)
                        mean_ep_ret = np.mean(total_ep_ret)
                        mean_ep_len = np.mean(total_ep_len)

                        self.epoch_logger.append(
                            {
                                "timesteps": timesteps,
                                "number_of_eps": num_eps,
                                "max_ep_ret": max_ep_ret,
                                "max_ep_len": max_ep_len,
                                "mean_ep_ret": mean_ep_ret,
                                "mean_ep_len": mean_ep_len,
                            }
                        )

                        global_stats = aggregate_global_stats(
                            global_stats,
                            max_ep_ret=max_ep_ret,
                            max_ep_len=max_ep_len,
                            mean_ep_ret=mean_ep_ret,
                            mean_ep_len=mean_ep_len,
                            num_eps=num_eps,
                        )

                    save_learner_checkpoint(self.agent, model_path)
                    self.agent.buffer.save(buffer_path)

                    self._save_train_logs()
                    self._save_obs_logs()

                    num_eps = 0
                    total_ep_ret, total_ep_len = [], []

        finally:
            progress.close()

        self.train_env.close()
        self.eval_env.close()

        self.end_time_now, self.time_elapse = finalize_timing(self.start_time)
        global_stats = finalize_global_stats(global_stats)

        end_log = build_end_log(
            best_performances=self.best_performances,
            best_parameters=self.best_parameters,
            best_fom=self.best_fom,
            end_time_now=self.end_time_now,
            time_elapse=self.time_elapse,
            global_stats=global_stats,
        )
        self.train_logger.append(end_log)

        save_learner_checkpoint(self.agent, model_path)
        self.agent.buffer.save(buffer_path)

        self._save_train_logs()
        self._save_obs_logs()

        save_logs_and_plot(
            logger=self.logger,
            project_name=project_name,
            max_timesteps=self.max_iters,
        )

        print_circuit_summary(
            circuit_type=self.circuit_type,
            end_time_now=self.end_time_now,
            time_elapse=self.time_elapse,
            best_performances=self.best_performances,
            best_parameters=self.best_parameters,
            best_fom=self.best_fom,
            global_stats=global_stats,
        )

        cleanup_experiment_run_root(self.run_root)

    def get_logs(self):
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