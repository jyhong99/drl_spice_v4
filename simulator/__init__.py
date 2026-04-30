"""Simulator backend abstractions and result containers.

This package-level module re-exports the common simulator backend interface
and structured simulation result type. It provides a compact import location
for environment and service code that depends on simulator abstractions rather
than simulator-specific implementations.

Examples
--------
Import simulator abstractions directly from the package:

>>> from simulator import SimulatorBackend, SimulationResult
"""

from simulator.base import SimulatorBackend
from simulator.result import SimulationResult


__all__ = ["SimulatorBackend", "SimulationResult"]
"""list[str]: Public symbols exported when using ``from simulator import *``.

The exported names are:

- ``SimulatorBackend``: Abstract interface for simulator backends.
- ``SimulationResult``: Structured return type for simulator evaluations.
"""