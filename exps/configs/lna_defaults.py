"""Default environment hyperparameter presets for LNA experiments."""

def build_lna_env_defaults():
    """Return baseline LNA environment keyword arguments.

    Returns
    -------
    dict
        Default keyword arguments shared across LNA experiment presets.
    """

    return {
        "enable_iip3": True,
        "max_steps": 20,
        "max_param": 1.0,
        "n_restricted": 2,
        "p": 2.0,
        "beta": 0.2,
        "gamma": 0.99,
        "eta": None,
        "reset_probability": 0.0,
        "allow_reset_fallback": False,
        "reset_fallback_after": 10,
        "penal_viol": 2.0,
        "penal_perf": 20.0,
        "lmda_viol": 1.0,
        "lmda_perf": 1.0,
        "lmda_var": 0.1,
        "reward_name": "default",
    }
