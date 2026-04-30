"""Preset experiment configuration for the common-source LNA."""

from exps.configs.agent_defaults import build_agent_launch_defaults_10k
from exps.configs.lna_defaults import build_lna_env_defaults


def build_cs_experiment_config():
    """Build the default CS experiment configuration bundle.

    Returns
    -------
    dict
        Configuration dictionary containing circuit metadata, environment
        kwargs, seeds, and launch kwargs.
    """

    agent_launch_defaults = build_agent_launch_defaults_10k()
    env_kwargs = build_lna_env_defaults()
    env_kwargs.update({
        "target_spec": {
            "S11": -10.0,
            "S21": 20.0,
            "S22": -10.0,
            "NF": 2.0,
            "PD": 5.0,
            "IIP3": 0.0,
        },
        "references": {
            "S11": -5.0,
            "S21": 10.0,
            "S22": -5.0,
            "NF": 4.0,
            "PD": 10.0,
            "IIP3": -5.0,
        },
        "bound": {
            "V_b": [0.7, 1.0],
            "R_D": [10, 1000],
            "L_D": [1e-10, 2e-8],
            "L_G": [1e-10, 2e-8],
            "L_S": [1e-11, 2e-9],
            "C_D": [5e-14, 5e-12],
            "C_ex": [5e-15, 5e-13],
            "XM1": [1, 100],
            "XM2": [1, 100],
        },
        "bound_decode_mode": {
            "V_b": "lin",
        },
        "fixed_values": {
            "V_dd": 1.8,
            "R_b": 1e4,
            "C_1": 1e-11,
            "l_m": 0.15,
        },
        "freq_range": [2.4e9, 2.4e9],
        "enable_iip3": False,
    })
    return {
        "circuit_type": "CS",
        "env_name": "modular",
        "seeds": [100, 200, 300, 400, 500],
        "env_kwargs": env_kwargs,
        "launch_kwargs": {
            "load_path": None,
            "max_iters": 10000,
            "n_runners": 10,
            "runner_iters": 5,
            "eval_mode": False,
            "eval_intervals": 100,
            "utd_ratio": 2.0,
            "checkpoint_intervals": 100,
            **agent_launch_defaults,
            "ppo": True,
            "ddpg": True,
            "td3": True,
            "sac": True,
        },
    }
