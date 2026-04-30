"""Convenience exports for ngspice reader implementations.

This package-level module re-exports ngspice reader classes and related
exceptions from the reader registry. It provides a stable import location for
code that needs access to specific reader implementations or the generic
``Reader`` factory.

Examples
--------
Import reader implementations directly from the package:

>>> from simulator.ngspice import Reader, SParamReader, NoiseReader
"""

from simulator.ngspice.readers import (
    ACSimulationReader,
    BaseReader,
    DCReader,
    LinearityReader,
    NoiseReader,
    Reader,
    SParamReader,
    StabilityReader,
    StaleOutputError,
    TransientReader,
)


__all__ = [
    "StaleOutputError",
    "BaseReader",
    "TransientReader",
    "DCReader",
    "SParamReader",
    "StabilityReader",
    "NoiseReader",
    "ACSimulationReader",
    "LinearityReader",
    "Reader",
]
"""list[str]: Public symbols exported when using ``from simulator.ngspice import *``.

The exported names are:

- ``StaleOutputError``: Exception raised for missing, empty, or stale outputs.
- ``BaseReader``: Base class for ngspice output readers.
- ``TransientReader``: Reader for transient-analysis outputs.
- ``DCReader``: Reader for DC operating-point outputs.
- ``SParamReader``: Reader for S-parameter outputs.
- ``StabilityReader``: Reader for stability-factor metrics.
- ``NoiseReader``: Reader for noise-analysis outputs.
- ``ACSimulationReader``: Reader for generic AC simulation outputs.
- ``LinearityReader``: Reader for FFT-based IIP3 estimation outputs.
- ``Reader``: Factory wrapper for analysis-specific reader dispatch.
"""