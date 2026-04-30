"""Lazy exports for LNA environment implementations and reward helpers.

This package-level module exposes the main LNA environment classes and reward
helper through lazy imports. Lazy resolution avoids importing simulator-heavy
modules until the corresponding symbol is actually requested.
"""

__all__ = [
    "LNA_Environment_Base",
    "LNAEnvBase",
    "LNA_Modular_base",
    "Reward",
]
"""list[str]: Public symbols exported when using ``from envs.lna import *``.

The exported names are:

- ``LNA_Environment_Base``: Base Gym environment for LNA optimization.
- ``LNAEnvBase``: Concrete modular LNA optimization environment.
- ``LNA_Modular_base``: Backward-compatible alias for ``LNAEnvBase``.
- ``Reward``: Default reward strategy class.
"""


def __getattr__(name):
    """Lazily resolve exported environment symbols.

    Parameters
    ----------
    name : str
        Name of the exported attribute requested from the package.

    Returns
    -------
    object
        Resolved class or helper object corresponding to ``name``.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the supported lazy-export symbols.
    """

    if name == "LNA_Environment_Base":
        from envs.lna.base import LNA_Environment_Base

        return LNA_Environment_Base

    if name in {"LNAEnvBase", "LNA_Modular_base"}:
        from envs.lna.modular import LNAEnvBase, LNA_Modular_base

        return {
            "LNAEnvBase": LNAEnvBase,
            "LNA_Modular_base": LNA_Modular_base,
        }[name]

    if name == "Reward":
        from envs.lna.reward import Reward

        return Reward

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")