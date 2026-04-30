"""Lazy package exports for training implementations.

This package-level module exposes the local and distributed trainer classes
through lazy imports. Lazy resolution avoids importing trainer dependencies,
such as Ray or simulator-related modules, until the corresponding class is
actually requested.
"""

__all__ = ["Trainer", "DistributedTrainer"]
"""list[str]: Public symbols exported when using ``from trains import *``.

The exported names are:

- ``Trainer``: Single-process training loop implementation.
- ``DistributedTrainer``: Ray-based distributed training coordinator.
"""


def __getattr__(name):
    """Lazily resolve exported trainer symbols.

    Parameters
    ----------
    name : str
        Name of the exported attribute requested from the ``trains`` package.

    Returns
    -------
    object
        Resolved trainer class corresponding to ``name``.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the supported lazy-export symbols.
    """

    if name == "Trainer":
        from trains.engine import Trainer

        return Trainer

    if name == "DistributedTrainer":
        from trains.distributed import DistributedTrainer

        return DistributedTrainer

    raise AttributeError(f"module 'trains' has no attribute {name!r}")