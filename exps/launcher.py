"""Helpers for building project names and launching configured agents."""

from copy import deepcopy

from exps.registry import get_agent_registry
from trains.checkpoint import load_learner_checkpoint


def build_project_name(*, circuit_type, env, agent, suffix, max_iters, sweep_tag=None):
    """Build a legacy-compatible experiment name.

    Parameters
    ----------
    circuit_type : str
        Circuit identifier such as ``"CS"`` or ``"CGCS"``.
    env : object
        Environment instance whose string representation is included.
    agent : object
        Agent instance whose string representation is included.
    suffix : str
        Agent-specific suffix such as ``"default"`` or ``"prior"``.
    max_iters : int
        Total training-step budget.
    sweep_tag : str, optional
        Additional suffix used to distinguish experiment variants.

    Returns
    -------
    str
        Project name used for logs, checkpoints, and workspaces.
    """

    project_name = f"{circuit_type}_{env.__str__()}_{agent.__str__()}_{suffix}_{max_iters}"
    if sweep_tag:
        project_name = f"{project_name}_{sweep_tag}"
    return project_name


def launch_enabled_agents(
    env,
    *,
    max_iters,
    n_runners,
    circuit_type,
    seed=None,
    load_path=None,
    eval_mode=False,
    eval_iters=None,
    **kwargs,
):
    """Instantiate and launch every agent enabled in ``kwargs``.

    Parameters
    ----------
    env : gym.Env
        Prototype environment copied once per launched agent.
    max_iters : int
        Total training-step budget for each run.
    n_runners : int
        Number of rollout workers requested by the trainer.
    circuit_type : str
        Circuit identifier forwarded into project naming and training.
    seed : int, optional
        Experiment seed.
    load_path : str, optional
        Checkpoint directory from which model state should be restored.
    eval_mode : bool, default=False
        Whether periodic evaluation is enabled.
    eval_iters : int, optional
        Number of evaluation episodes per evaluation pass.
    **kwargs
        Mixed launch configuration containing agent enable flags, per-agent
        overrides such as ``ddpg_kwargs``, and trainer/runtime settings.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of launched agent keys and their project names.
    """

    agents = get_agent_registry()
    launched_agents = []
    agent_flag_keys = set(agents.keys())
    agent_config_keys = {f"{agent_key}_kwargs" for agent_key in agent_flag_keys}
    reserved_keys = {"sweep_tag", *agent_config_keys}
    train_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in agent_flag_keys and key not in reserved_keys
    }

    for agent_key, (agent_class, prioritized_mode, suffix) in agents.items():
        if not kwargs.get(agent_key, False):
            continue

        agent_env = deepcopy(env)
        agent_overrides = dict(kwargs.get(f"{agent_key}_kwargs", {}))
        agent = agent_class(agent_env, prioritized_mode=prioritized_mode, **agent_overrides)
        if agent_key == "random":
            agent.update_after = max_iters

        project_name = build_project_name(
            circuit_type=circuit_type,
            env=agent_env,
            agent=agent,
            suffix=suffix,
            max_iters=max_iters,
            sweep_tag=kwargs.get("sweep_tag"),
        )

        if load_path is not None:
            load_learner_checkpoint(agent, load_path)

        agent.train(
            project_name=project_name,
            load_path=load_path,
            max_iters=max_iters,
            n_runners=n_runners,
            seed=seed,
            eval_mode=eval_mode,
            eval_iters=eval_iters if eval_iters is not None else n_runners,
            circuit_type=circuit_type,
            **train_kwargs,
        )
        launched_agents.append((agent_key, project_name))

    return launched_agents


def run_experiment(
    env,
    max_iters,
    n_runners,
    circuit_type,
    seed=None,
    load_path=None,
    eval_mode=False,
    **kwargs,
):
    """Compatibility wrapper around :func:`launch_enabled_agents`.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of launched agent keys and project names.
    """

    return launch_enabled_agents(
        env,
        max_iters=max_iters,
        n_runners=n_runners,
        circuit_type=circuit_type,
        seed=seed,
        load_path=load_path,
        eval_mode=eval_mode,
        **kwargs,
    )


test_module = run_experiment
test_module.__test__ = False
