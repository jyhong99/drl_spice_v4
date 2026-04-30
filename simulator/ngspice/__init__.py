"""ngspice-backed simulator implementation and helpers.

This package-level module re-exports the main ngspice simulator components,
including artifact management, circuit parsing, netlist rewriting, phase
execution, process backends, simulation pipelines, reader implementations,
workspace management, and the high-level simulation service.

Examples
--------
Import core ngspice simulator utilities directly from the package:

>>> from simulator.ngspice import NgSpiceSimulationService, Circuit
>>> from simulator.ngspice import create_spice_backend, prepare_worker_workspace
"""

from simulator.ngspice.artifacts import OutputArtifactManager
from simulator.ngspice.circuit import Circuit
from simulator.ngspice.designer import Designer
from simulator.ngspice.executor import PhaseExecutor
from simulator.ngspice.kernel import (
    BaseSpiceBackend,
    BatchSpiceBackend,
    SessionPreparingSpiceBackend,
    SpiceKernel,
    SpiceRunResult,
    create_spice_backend,
)
from simulator.ngspice.pipeline import run_simulation_pipeline
from simulator.ngspice.readers import (
    ACSimulationReader,
    DCReader,
    LinearityReader,
    NoiseReader,
    Reader,
    SParamReader,
    StabilityReader,
    StaleOutputError,
    TransientReader,
)
from simulator.ngspice.service import NgSpiceSimulationService
from simulator.ngspice.templates import (
    CircuitTemplate,
    get_circuit_template,
    list_circuit_templates,
)
from simulator.ngspice.workspace import (
    NgSpiceWorkspace,
    cleanup_experiment_run_root,
    cleanup_workspace_outputs,
    create_experiment_run_root,
    create_run_id,
    prepare_worker_workspace,
)


__all__ = [
    "OutputArtifactManager",
    "Circuit",
    "Designer",
    "PhaseExecutor",
    "BaseSpiceBackend",
    "BatchSpiceBackend",
    "SessionPreparingSpiceBackend",
    "SpiceKernel",
    "SpiceRunResult",
    "create_spice_backend",
    "run_simulation_pipeline",
    "TransientReader",
    "DCReader",
    "SParamReader",
    "StabilityReader",
    "NoiseReader",
    "ACSimulationReader",
    "LinearityReader",
    "Reader",
    "StaleOutputError",
    "NgSpiceSimulationService",
    "CircuitTemplate",
    "get_circuit_template",
    "list_circuit_templates",
    "NgSpiceWorkspace",
    "cleanup_experiment_run_root",
    "cleanup_workspace_outputs",
    "create_experiment_run_root",
    "create_run_id",
    "prepare_worker_workspace",
]
"""list[str]: Public symbols exported when using ``from simulator.ngspice import *``.

The exported names are:

- ``OutputArtifactManager``: Manager for unique simulator output artifact paths.
- ``Circuit``: Parsed annotated ngspice netlist wrapper.
- ``Designer``: Netlist design-variable and output-path rewriter.
- ``PhaseExecutor``: Executor for one designed ngspice simulation phase.
- ``BaseSpiceBackend``: Abstract base class for ngspice execution backends.
- ``BatchSpiceBackend``: Subprocess backend using one fresh ngspice process per run.
- ``SessionPreparingSpiceBackend``: Persistent-session backend with batch fallback.
- ``SpiceKernel``: Backward-compatible alias for the batch backend.
- ``SpiceRunResult``: Structured result object for one ngspice invocation.
- ``create_spice_backend``: Factory for constructing ngspice execution backends.
- ``run_simulation_pipeline``: End-to-end simulator pipeline for one design.
- ``TransientReader``: Reader for transient-analysis outputs.
- ``DCReader``: Reader for DC operating-point outputs.
- ``SParamReader``: Reader for S-parameter outputs.
- ``StabilityReader``: Reader for stability-factor metrics.
- ``NoiseReader``: Reader for noise-analysis outputs.
- ``ACSimulationReader``: Reader for generic AC simulation outputs.
- ``LinearityReader``: Reader for FFT-based IIP3 estimation outputs.
- ``Reader``: Factory wrapper for analysis-specific reader dispatch.
- ``StaleOutputError``: Exception raised for missing, empty, or stale outputs.
- ``NgSpiceSimulationService``: High-level simulation service used by environments.
- ``CircuitTemplate``: Immutable template metadata for annotated netlists.
- ``get_circuit_template``: Lookup helper for circuit template metadata.
- ``list_circuit_templates``: Helper returning supported template names.
- ``NgSpiceWorkspace``: Dataclass describing a prepared worker workspace.
- ``cleanup_experiment_run_root``: Helper for deleting managed experiment roots.
- ``cleanup_workspace_outputs``: Helper for deleting workspace output artifacts.
- ``create_experiment_run_root``: Helper for creating managed experiment roots.
- ``create_run_id``: Helper for creating unique simulator run identifiers.
- ``prepare_worker_workspace``: Helper for creating and populating worker workspaces.
"""