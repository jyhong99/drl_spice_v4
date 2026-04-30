"""Preset experiment configuration for the CGCS LNA."""

from exps.configs.agent_defaults import build_agent_launch_defaults_10k
from exps.configs.lna_defaults import build_lna_env_defaults


def build_cgcs_experiment_config():
    """Build the default CGCS experiment configuration bundle.

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
            "S21": 16.0,
            "S22": -10.0,
            "NF": 5.0,
            "PD": 7.0,
            "IIP3": 5.0,
        },
        "references": {
            "S11": -5.0,
            "S21": 8.0,
            "S22": -5.0,
            "NF": 10.0,
            "PD": 14.0,
            "IIP3": 0.0,
        },
        "bound": {
            "V_b1": [0.6, 1.2],
            "V_b2": [0.6, 1.2],
            "V_b3": [0.6, 1.2],
            "V_b4": [0.6, 1.2],
            "R_D1": [10, 1000],
            "R_D4": [10, 1000],
            "R_S5": [10, 1000],
            "C_D1": [1e-13, 1e-11],
            "C_D4": [1e-13, 1e-11],
            "C_S3": [1e-13, 1e-11],
            "C_S4": [1e-13, 1e-11],
            "XM1": [1, 100],
            "XM2": [1, 100],
            "XM3": [1, 100],
            "XM4": [1, 100],
            "XM5": [1, 100],
        },
        "bound_decode_mode": {
            "V_b1": "lin",
            "V_b2": "lin",
            "V_b3": "lin",
            "V_b4": "lin",
        },
        "fixed_values": {
            "V_dd": 1.8,
            "R_b": 1e4,
            "C_1": 1e-11,
            "l_m": 0.15,
        },
        "freq_range": [1.0e9, 3.0e9],
        "enable_iip3": False,
    })
    return {
        "circuit_type": "CGCS",
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
