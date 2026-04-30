"""Environment factory helpers used by experiment presets."""

from envs.lna import LNAEnvBase


def get_lna_env_registry():
    """Return the registry of supported LNA environment presets.

    Returns
    -------
    dict[str, type]
        Mapping of environment preset names to environment classes.
    """

    return {
        "modular": LNAEnvBase,
        "default": LNAEnvBase,
    }


def build_lna_env(*, circuit_type, env_kwargs, env_name=None):
    """Instantiate an LNA environment from a preset name and kwargs.

    Parameters
    ----------
    circuit_type : str
        Circuit identifier such as ``"CS"`` or ``"CGCS"``.
    env_kwargs : dict
        Keyword arguments forwarded into the environment constructor.
    env_name : str, optional
        Registered environment preset name. Defaults to ``"modular"``.

    Returns
    -------
    LNAEnvBase
        Instantiated environment.
    """

    registry = get_lna_env_registry()
    key = str(env_name or "modular").lower()
    if key not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unsupported env_name: {env_name}. Supported envs: {supported}")

    kwargs = dict(env_kwargs)
    env_class = registry[key]
    return env_class(circuit_type=circuit_type, **kwargs)
