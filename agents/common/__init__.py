"""Common agent abstractions shared across algorithm implementations.

This package-level module re-exports the common base classes used by
reinforcement-learning agent implementations. Importing from this module
allows algorithm-specific modules to access the shared on-policy and
off-policy abstractions through a shorter and more stable import path.

Examples
--------
Import the shared base classes directly from the package:

>>> from agents.common import OnPolicyAlgorithm, OffPolicyAlgorithm
"""

from agents.common.policy import OnPolicyAlgorithm, OffPolicyAlgorithm

__all__ = ["OnPolicyAlgorithm", "OffPolicyAlgorithm"]
"""list[str]: Public symbols exported when using ``from agents.common import *``.

The exported names are:

- ``OnPolicyAlgorithm``: Base abstraction for on-policy RL algorithms.
- ``OffPolicyAlgorithm``: Base abstraction for off-policy RL algorithms.
"""