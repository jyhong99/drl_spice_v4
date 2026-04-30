"""Phase execution helpers for ngspice simulation workflows.

This module defines a small execution wrapper that applies design-variable
updates to a circuit netlist, runs ngspice through a configured kernel, and
records timing metadata for the simulation phase.
"""

import time


class PhaseExecutor:
    """Execute one designed ngspice simulation phase.

    This class coordinates three operations:

    1. Build a designer for the target circuit.
    2. Rewrite the circuit netlist using decoded design variables.
    3. Run the ngspice backend and collect phase-level profiling metadata.

    Parameters
    ----------
    spice_kernel : object
        Backend object that exposes a ``run`` method for executing ngspice.
    simulation_timeout_s : float
        Maximum allowed simulation time in seconds.
    designer_factory : callable
        Factory function that receives a circuit object and returns a designer
        object with a ``design_circuit`` method.

    Attributes
    ----------
    spice_kernel : object
        Configured ngspice execution backend.
    simulation_timeout_s : float
        Simulation timeout in seconds.
    _designer_factory : callable
        Factory used to create designer instances.
    """

    def __init__(self, *, spice_kernel, simulation_timeout_s, designer_factory):
        """Initialize the phase executor.

        Parameters
        ----------
        spice_kernel : object
            Backend object used to execute ngspice simulations.
        simulation_timeout_s : float
            Maximum allowed simulation time in seconds.
        designer_factory : callable
            Factory function used to create designer objects for circuits.

        Returns
        -------
        None
            The executor stores backend, timeout, and designer-factory
            references in place.
        """

        self.spice_kernel = spice_kernel
        self.simulation_timeout_s = float(simulation_timeout_s)
        self._designer_factory = designer_factory

    def execute(
        self,
        circuit,
        design_variables_config,
        *,
        phase_name,
        run_id,
        output_path_map,
    ):
        """Design and execute one ngspice simulation phase.

        Parameters
        ----------
        circuit : Circuit
            Circuit object whose netlist is rewritten and executed.
        design_variables_config : dict[str, object]
            Mapping from design-variable names to values written into the
            circuit netlist.
        phase_name : str
            Human-readable phase label stored in the returned profiling
            dictionary.
        run_id : str
            Unique run identifier forwarded to the ngspice backend.
        output_path_map : dict[str, str]
            Mapping from output filenames to concrete output paths used when
            rewriting netlist ``write`` commands and running the backend.

        Returns
        -------
        result : object
            Result object returned by ``self.spice_kernel.run``.
        profile : dict[str, object]
            Timing and status metadata for the phase. The dictionary contains
            design time, ngspice execution time, total time, backend name,
            success flag, and timeout flag.

        Raises
        ------
        RuntimeError
            If the ngspice backend reports an unsuccessful run.
        """

        t0 = time.perf_counter_ns()

        designer = self._designer_factory(circuit)
        designer.design_circuit(
            circuit,
            design_variables_config,
            output_path_map=output_path_map,
        )

        t1 = time.perf_counter_ns()

        result = self.spice_kernel.run(
            circuit,
            run_id=run_id,
            output_paths=output_path_map,
            timeout_s=self.simulation_timeout_s,
        )

        t2 = time.perf_counter_ns()

        profile = {
            "phase": phase_name,
            "run_id": run_id,
            "design_ms": (t1 - t0) / 1e6,
            "ngspice_ms": (t2 - t1) / 1e6,
            "total_ms": (t2 - t0) / 1e6,
            "backend": result.backend,
            "ok": result.ok,
            "timed_out": result.timed_out,
        }

        if not result.ok:
            raise RuntimeError(
                f"NGSPICE run failed for {circuit.netlist_path}: "
                f"{result.error}. stderr={result.stderr_tail}"
            )

        return result, profile