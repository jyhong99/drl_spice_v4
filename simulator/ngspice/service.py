"""High-level ngspice simulation service used by environments.

This module defines the high-level ngspice simulation service used by
simulator-backed LNA environments. The service coordinates workspace
preparation, netlist design, ngspice execution, result parsing, simulation
caching, artifact cleanup, and profiling metadata.
"""

from pathlib import Path

import numpy as np

from simulator.base import SimulatorBackend
from simulator.cache import SimulationCache
from simulator.ngspice.artifacts import OutputArtifactManager
from simulator.ngspice.circuit import Circuit
from simulator.ngspice.designer import Designer
from simulator.ngspice.executor import PhaseExecutor
from simulator.ngspice.kernel import create_spice_backend
from simulator.ngspice.pipeline import run_simulation_pipeline
from simulator.ngspice.reader import (
    DCReader,
    LinearityReader,
    NoiseReader,
    SParamReader,
    StabilityReader,
)
from simulator.ngspice.workspace import prepare_worker_workspace
from simulator.result import SimulationResult


class NgSpiceSimulationService(SimulatorBackend):
    """Coordinator for ngspice execution, parsing, caching, and workspaces.

    This service evaluates decoded circuit design configurations by rewriting
    ngspice netlists, executing the required simulation phases, parsing scalar
    performance metrics, and returning structured simulation results. It also
    supports workspace reconfiguration, in-memory simulation caching, profiling
    history, and cleanup of generated output artifacts.

    Parameters
    ----------
    circuit_type : str
        Circuit family identifier used to select circuit templates and
        workspaces.
    enable_iip3 : bool
        Whether IIP3 simulation and parsing are enabled.
    fixed_values : dict[str, float]
        Fixed circuit values used by the simulator and derived metric
        computation.
    freq_range : sequence[float]
        Frequency range in Hz used when extracting S-parameter and
        noise-figure metrics.
    num_design_variables : int
        Expected number of design variables in the circuit netlists.
    workspace_root : str or pathlib.Path
        Root directory used to create simulator workspaces.
    worker_name : str or int
        Worker label used to isolate workspace files.
    kernel_backend : str
        ngspice execution backend identifier.
    project_name : str
        Project or experiment namespace.
    run_id : str
        Experiment run identifier.
    profile_history_limit : int, optional
        Maximum number of simulation profiles retained in memory. The default
        is ``256``.
    simulation_timeout_s : float, optional
        Maximum runtime for each ngspice phase in seconds. The default is
        ``60.0``.
    simulation_cache_enabled : bool, optional
        Whether in-memory simulation caching is enabled. The default is
        ``True``.
    simulation_cache_maxsize : int, optional
        Maximum number of cached simulation results. The default is ``256``.
    s_param_netlist_path : str or pathlib.Path or None, optional
        Explicit S-parameter netlist path. If explicit paths are provided, the
        service uses them instead of preparing a workspace. The default is
        ``None``.
    nf_netlist_path : str or pathlib.Path or None, optional
        Explicit noise-figure netlist path. The default is ``None``.
    iip3_netlist_path : str or pathlib.Path or None, optional
        Explicit IIP3 netlist path. The default is ``None``.
    dc_op_result_path : str or pathlib.Path or None, optional
        Explicit DC operating-point output path. The default is ``None``.
    s_param_bandwidth_result_path : str or pathlib.Path or None, optional
        Explicit S-parameter bandwidth output path. The default is ``None``.
    nf_result_path : str or pathlib.Path or None, optional
        Explicit noise-figure output path. The default is ``None``.
    iip3_result_path : str or pathlib.Path or None, optional
        Explicit IIP3 output path. The default is ``None``.

    Attributes
    ----------
    circuit_type : str
        Circuit family identifier.
    enable_iip3 : bool
        Whether IIP3 simulation is enabled.
    fixed_values : dict[str, float]
        Fixed circuit values.
    freq_range : tuple[float, float]
        Frequency range used for metric extraction.
    num_design_variables : int
        Expected design-variable count.
    project_name : str
        Project or experiment namespace.
    experiment_run_id : str
        Experiment run identifier.
    kernel_backend : str
        Configured ngspice backend identifier.
    spice_kernel : BaseSpiceBackend
        ngspice execution backend.
    executor : PhaseExecutor
        Phase executor used to design and run simulation phases.
    dc_reader : DCReader
        Reader for DC operating-point metrics.
    s_param_reader : SParamReader
        Reader for S-parameter metrics.
    stability_reader : StabilityReader
        Reader for stability metrics.
    nf_reader : NoiseReader
        Reader for noise-figure metrics.
    iip3_reader : LinearityReader or None
        Reader for IIP3 metrics when enabled.
    simulation_cache : dict
        Public reference to the cache storage.
    simulation_cache_stats : dict
        Public reference to cache statistics.
    last_simulation_error : str or None
        Last simulation error message.
    last_simulation_profile : dict
        Most recent simulation profile.
    simulation_profiles : list[dict]
        Recent simulation profiles.
    """

    def __init__(
        self,
        *,
        circuit_type,
        enable_iip3,
        fixed_values,
        freq_range,
        num_design_variables,
        workspace_root,
        worker_name,
        kernel_backend,
        project_name,
        run_id,
        profile_history_limit=256,
        simulation_timeout_s=60.0,
        simulation_cache_enabled=True,
        simulation_cache_maxsize=256,
        s_param_netlist_path=None,
        nf_netlist_path=None,
        iip3_netlist_path=None,
        dc_op_result_path=None,
        s_param_bandwidth_result_path=None,
        nf_result_path=None,
        iip3_result_path=None,
    ):
        """Initialize the ngspice simulation service.

        Parameters
        ----------
        circuit_type : str
            Circuit family identifier.
        enable_iip3 : bool
            Whether IIP3 simulation and parsing are enabled.
        fixed_values : dict[str, float]
            Fixed circuit values.
        freq_range : sequence[float]
            Frequency range in Hz used for metric extraction.
        num_design_variables : int
            Expected number of design variables in the netlists.
        workspace_root : str or pathlib.Path
            Root directory for prepared or scratch workspaces.
        worker_name : str or int
            Worker label used in workspace naming.
        kernel_backend : str
            ngspice execution backend identifier.
        project_name : str
            Project or experiment namespace.
        run_id : str
            Experiment run identifier.
        profile_history_limit : int, optional
            Maximum number of simulation profiles retained. The default is
            ``256``.
        simulation_timeout_s : float, optional
            Maximum runtime for each ngspice phase in seconds. The default is
            ``60.0``.
        simulation_cache_enabled : bool, optional
            Whether simulation caching is enabled. The default is ``True``.
        simulation_cache_maxsize : int, optional
            Maximum number of cached simulation results. The default is
            ``256``.
        s_param_netlist_path : str or pathlib.Path or None, optional
            Explicit S-parameter netlist path. The default is ``None``.
        nf_netlist_path : str or pathlib.Path or None, optional
            Explicit noise-figure netlist path. The default is ``None``.
        iip3_netlist_path : str or pathlib.Path or None, optional
            Explicit IIP3 netlist path. The default is ``None``.
        dc_op_result_path : str or pathlib.Path or None, optional
            Explicit DC output path. The default is ``None``.
        s_param_bandwidth_result_path : str or pathlib.Path or None, optional
            Explicit S-parameter output path. The default is ``None``.
        nf_result_path : str or pathlib.Path or None, optional
            Explicit noise-figure output path. The default is ``None``.
        iip3_result_path : str or pathlib.Path or None, optional
            Explicit IIP3 output path. The default is ``None``.

        Returns
        -------
        None
            The constructor initializes backend, readers, workspace or explicit
            netlist paths, cache state, and profiling fields in place.
        """

        self.circuit_type = circuit_type
        self.enable_iip3 = enable_iip3
        self.fixed_values = fixed_values
        self.freq_range = tuple(float(value) for value in freq_range)
        self.num_design_variables = num_design_variables

        self.project_name = str(project_name)
        self.experiment_run_id = str(run_id)
        self.kernel_backend = str(kernel_backend)

        self.profile_history_limit = int(profile_history_limit)
        self.simulation_timeout_s = float(simulation_timeout_s)
        self.simulation_cache_enabled = bool(simulation_cache_enabled)
        self.simulation_cache_maxsize = int(simulation_cache_maxsize)

        self.workspace = None
        self.spice_kernel = create_spice_backend(self.kernel_backend)
        self.executor = PhaseExecutor(
            spice_kernel=self.spice_kernel,
            simulation_timeout_s=self.simulation_timeout_s,
            designer_factory=self._get_designer,
        )
        self._designers = {}

        self.dc_reader = DCReader()
        self.s_param_reader = SParamReader()
        self.stability_reader = StabilityReader()
        self.nf_reader = NoiseReader()
        self.iip3_reader = LinearityReader() if self.enable_iip3 else None

        self._run_sequence = 0
        self.last_simulation_error = None
        self.last_simulation_profile = {}
        self.simulation_profiles = []
        self.artifacts = None

        self._simulation_cache = SimulationCache(
            enabled=self.simulation_cache_enabled,
            maxsize=self.simulation_cache_maxsize,
        )
        self.simulation_cache = self._simulation_cache.storage
        self.simulation_cache_stats = self._simulation_cache.stats

        default_root = Path(workspace_root)
        explicit_paths = any(
            path is not None
            for path in (
                s_param_netlist_path,
                nf_netlist_path,
                iip3_netlist_path,
                dc_op_result_path,
                s_param_bandwidth_result_path,
                nf_result_path,
                iip3_result_path,
            )
        )

        if explicit_paths:
            self.s_param_circuit = Circuit(s_param_netlist_path, self.circuit_type)
            self.nf_circuit = Circuit(nf_netlist_path, self.circuit_type)
            self.iip3_circuit = (
                Circuit(iip3_netlist_path, self.circuit_type)
                if self.enable_iip3
                else None
            )

            self.dc_op_result_path = dc_op_result_path
            self.s_param_bandwidth_result_path = s_param_bandwidth_result_path
            self.nf_result_path = nf_result_path
            self.iip3_result_path = iip3_result_path

            self.run_root = default_root / "scratch"
            self.run_root.mkdir(parents=True, exist_ok=True)
            self.artifacts = OutputArtifactManager(self.run_root)
        else:
            self.configure_workspace(
                project_name=self.project_name,
                run_id=self.experiment_run_id,
                run_root=default_root,
                worker_name=worker_name,
                scope="env",
                clean=True,
            )

    def evaluate(self, design_variables_config):
        """Evaluate one decoded design configuration through ngspice.

        The method first checks the in-memory simulation cache. On a cache
        miss, it runs the full simulation pipeline, parses performance metrics,
        stores the result in the cache, and returns a structured
        :class:`SimulationResult`.

        Parameters
        ----------
        design_variables_config : dict[str, float]
            Decoded simulator parameter mapping for one candidate design.

        Returns
        -------
        SimulationResult
            Successful simulation result containing the performance vector,
            stability factor, and profile metadata.

        Raises
        ------
        RuntimeError
            If the simulation pipeline or ngspice execution fails.
        """

        self.last_simulation_error = None

        cache_key = self.make_simulation_cache_key(design_variables_config)
        cached_result = self.get_cached_simulation(cache_key)

        if cached_result is not None:
            performances, stability_factor = cached_result
            self.last_simulation_profile = {
                "total_ms": 0.0,
                "parse_ms": 0.0,
                "phases": [],
                "enable_iip3": self.enable_iip3,
                "kernel_backend": self.kernel_backend,
                "cache_hit": True,
                "cache_stats": dict(self.simulation_cache_stats),
            }
            self.simulation_profiles.append(self.last_simulation_profile)
            self.simulation_profiles = self.simulation_profiles[
                -self.profile_history_limit :
            ]

            return SimulationResult(
                status="ok",
                performances=np.array(performances, copy=True),
                stability_factor=float(stability_factor),
                metadata={"profile": dict(self.last_simulation_profile)},
            )

        cleanup_paths = []

        try:
            pipeline_result = run_simulation_pipeline(
                design_variables_config=design_variables_config,
                artifacts=self.artifacts,
                executor=self.executor,
                readers={
                    "dc": self.dc_reader,
                    "s_param": self.s_param_reader,
                    "stability": self.stability_reader,
                    "nf": self.nf_reader,
                    "iip3": self.iip3_reader,
                },
                circuits={
                    "s_param": self.s_param_circuit,
                    "nf": self.nf_circuit,
                    "iip3": self.iip3_circuit,
                },
                output_paths={
                    "dc_op": self.dc_op_result_path,
                    "s_param_bandwidth": self.s_param_bandwidth_result_path,
                    "nf": self.nf_result_path,
                    "iip3": self.iip3_result_path,
                },
                fixed_values=self.fixed_values,
                freq_range=self.freq_range,
                enable_iip3=self.enable_iip3,
                kernel_backend=self.kernel_backend,
                simulation_cache_stats=self.simulation_cache_stats,
            )
            cleanup_paths = pipeline_result["cleanup_paths"]

        except RuntimeError as exc:
            self.last_simulation_error = str(exc)
            raise

        finally:
            self.artifacts.cleanup_outputs(cleanup_paths)

        performances = np.array(pipeline_result["performances"])
        stability_factor = float(pipeline_result["stability_factor"])

        self.last_simulation_profile = pipeline_result["profile"]
        self.simulation_profiles.append(self.last_simulation_profile)
        self.simulation_profiles = self.simulation_profiles[
            -self.profile_history_limit :
        ]

        self.store_cached_simulation(cache_key, performances, stability_factor)

        return SimulationResult(
            status="ok",
            performances=performances,
            stability_factor=stability_factor,
            metadata={"profile": dict(self.last_simulation_profile)},
        )

    def close(self):
        """Close the underlying spice backend.

        Returns
        -------
        None
            Backend resources are released when possible. Backend close errors
            are ignored.
        """

        if self.spice_kernel is not None:
            try:
                self.spice_kernel.close()
            except Exception:
                pass

    def reset_simulation_cache(self):
        """Clear the in-memory simulation cache and reset cache stats.

        Returns
        -------
        None
            Cache storage and statistics are reset in place.
        """

        self._ensure_simulation_cache()
        self._simulation_cache.reset()
        self.simulation_cache = self._simulation_cache.storage
        self.simulation_cache_stats = self._simulation_cache.stats

    def configure_workspace(
        self,
        *,
        project_name,
        run_id,
        run_root,
        worker_name,
        scope="train",
        clean=False,
    ):
        """Bind the service to a prepared worker workspace.

        This method prepares or refreshes the ngspice workspace, constructs
        circuit objects for each enabled simulation phase, resets simulation
        cache state, and configures the output artifact manager.

        Parameters
        ----------
        project_name : str
            Project or experiment namespace.
        run_id : str
            Experiment run identifier.
        run_root : str or pathlib.Path
            Root directory under which worker workspaces are prepared.
        worker_name : str or int
            Worker label used to isolate workspace files.
        scope : str, optional
            Workspace scope label. The default is ``"train"``.
        clean : bool, optional
            Whether to clean the workspace before use. The default is
            ``False``.

        Returns
        -------
        None
            Workspace, circuit objects, result paths, cache, and artifact
            manager are updated in place.
        """

        self.project_name = project_name
        self.experiment_run_id = run_id

        self.workspace = prepare_worker_workspace(
            project_name=project_name,
            run_id=run_id,
            run_root=Path(run_root),
            circuit_type=self.circuit_type,
            worker_name=worker_name,
            scope=scope,
            enable_iip3=self.enable_iip3,
            clean=clean,
        )

        self.s_param_circuit = Circuit(
            str(self.workspace.s_param_netlist_path),
            self.circuit_type,
        )
        self.nf_circuit = Circuit(
            str(self.workspace.nf_netlist_path),
            self.circuit_type,
        )
        self.iip3_circuit = (
            Circuit(str(self.workspace.iip3_netlist_path), self.circuit_type)
            if self.enable_iip3 and self.workspace.iip3_netlist_path is not None
            else None
        )

        self._designers = {}
        self.reset_simulation_cache()

        self.dc_op_result_path = str(self.workspace.dc_op_result_path)
        self.s_param_bandwidth_result_path = str(
            self.workspace.s_param_bandwidth_result_path
        )
        self.nf_result_path = str(self.workspace.nf_result_path)
        self.iip3_result_path = (
            str(self.workspace.iip3_result_path)
            if self.workspace.iip3_result_path is not None
            else None
        )

        self.run_root = self.workspace.scratch_dir / "sim_runs"
        self.run_root.mkdir(parents=True, exist_ok=True)

        if self.artifacts is None:
            self.artifacts = OutputArtifactManager(self.run_root)
        else:
            self.artifacts.set_run_root(self.run_root)

    def make_simulation_cache_key(self, design_variables_config):
        """Build a stable cache key for one decoded design configuration.

        Parameters
        ----------
        design_variables_config : dict[str, float]
            Decoded simulator parameter mapping.

        Returns
        -------
        tuple
            Hashable cache key containing circuit context, backend settings,
            frequency range, netlist fingerprints, and sorted design-variable
            values.
        """

        netlist_fingerprint = (
            str(getattr(self.s_param_circuit, "netlist_path", "")),
            str(getattr(self.nf_circuit, "netlist_path", "")),
            (
                str(getattr(self.iip3_circuit, "netlist_path", ""))
                if getattr(self, "iip3_circuit", None) is not None
                else None
            ),
        )

        return (
            self.circuit_type,
            self.kernel_backend,
            self.enable_iip3,
            self.freq_range,
            netlist_fingerprint,
            tuple(
                sorted(
                    (name, float(value))
                    for name, value in design_variables_config.items()
                )
            ),
        )

    def get_cached_simulation(self, cache_key):
        """Return a cached simulation result if available.

        Parameters
        ----------
        cache_key : tuple
            Cache key produced by :meth:`make_simulation_cache_key`.

        Returns
        -------
        tuple[numpy.ndarray, float] or None
            Cached ``(performances, stability_factor)`` pair if present;
            otherwise ``None``.
        """

        self._ensure_simulation_cache()
        cached = self._simulation_cache.get(cache_key)
        self.simulation_cache = self._simulation_cache.storage
        self.simulation_cache_stats = self._simulation_cache.stats
        return cached

    def store_cached_simulation(self, cache_key, performances, stability_factor):
        """Store a simulation result in the in-memory cache.

        Parameters
        ----------
        cache_key : tuple
            Cache key associated with the evaluated design.
        performances : array-like or numpy.ndarray
            Performance vector to cache.
        stability_factor : float
            Stability factor to cache.

        Returns
        -------
        None
            The cache storage and public cache references are updated in
            place.
        """

        self._ensure_simulation_cache()
        self._simulation_cache.put(cache_key, performances, stability_factor)
        self.simulation_cache = self._simulation_cache.storage
        self.simulation_cache_stats = self._simulation_cache.stats

    def _ensure_simulation_cache(self):
        """Lazily reconstruct the in-memory simulation cache wrapper.

        This method is used after deserialization or state restoration when
        the private ``_simulation_cache`` wrapper may be missing but public
        cache storage and statistics are still present.

        Returns
        -------
        None
            ``_simulation_cache``, ``simulation_cache``, and
            ``simulation_cache_stats`` are guaranteed to exist after this
            method returns.
        """

        if hasattr(self, "_simulation_cache"):
            return

        cache = SimulationCache(
            enabled=getattr(self, "simulation_cache_enabled", True),
            maxsize=getattr(self, "simulation_cache_maxsize", 256),
        )

        existing_storage = getattr(self, "simulation_cache", None)

        if existing_storage:
            for key, value in existing_storage.items():
                performances, stability_factor = value
                cache.put(key, performances, stability_factor)

            cache._stats = dict(
                getattr(self, "simulation_cache_stats", cache.stats)
            )

        self._simulation_cache = cache
        self.simulation_cache = cache.storage
        self.simulation_cache_stats = cache.stats

    def _get_designer(self, circuit):
        """Return or create the netlist designer associated with a circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit object for which a cached designer is requested.

        Returns
        -------
        Designer
            Cached or newly created designer for ``circuit.netlist_path``.

        Raises
        ------
        ValueError
            If the circuit design-variable count does not match
            ``self.num_design_variables``.
        """

        key = str(circuit.netlist_path)
        designer = self._designers.get(key)

        if designer is None:
            designer = Designer(
                circuit,
                num_design_variables=self.num_design_variables,
            )
            designer._check_params(circuit)
            self._designers[key] = designer

        return designer