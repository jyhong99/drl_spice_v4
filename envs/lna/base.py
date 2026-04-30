"""Base environment abstractions for low-noise-amplifier optimization.

This module defines the base Gym environment used for SPICE-in-the-loop
low-noise-amplifier design optimization. It provides common configuration
normalization, simulator workspace management, parameter decoding, simulation
caching, and Gym space definitions shared by concrete LNA environments.
"""

import gym
import logging
import os
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import torch
from gym import spaces

from envs.lna.config import (
    canonicalize_name,
    get_bound_names,
    normalize_bound_array,
    normalize_bound_decode_mode,
    normalize_fixed_values,
    normalize_freq_range,
    normalize_named_array,
)
from envs.lna.decoder import make_design_variables_config, restore_params
from envs.lna.simulation import (
    configure_workspace as bridge_configure_workspace,
    get_cached_simulation as bridge_get_cached_simulation,
    make_simulation_cache_key as bridge_make_simulation_cache_key,
    reset_simulation_cache as bridge_reset_simulation_cache,
    simulate as bridge_simulate,
    store_cached_simulation as bridge_store_cached_simulation,
    sync_simulator_state,
)
from simulator.ngspice.workspace import create_run_id
from simulator.ngspice.service import NgSpiceSimulationService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LNA_Environment_Base(gym.Env):
    """Base Gym environment for LNA design optimization.

    This class defines the shared infrastructure for LNA optimization
    environments. It normalizes circuit specifications, bounds, fixed values,
    and frequency ranges; constructs simulator services; defines Gym
    observation and action spaces; and provides helper methods for parameter
    decoding, simulation execution, workspace configuration, and simulation
    caching.

    Concrete subclasses must implement :meth:`step` and :meth:`reset`.

    Parameters
    ----------
    circuit_type : str
        Supported circuit family identifier, such as ``"CS"`` or ``"CGCS"``.
    target_spec : sequence or mapping
        Target performance specification. If a sequence is provided, it must
        follow the order of active performance names. If a mapping is provided,
        keys are matched to performance names.
    bound : sequence or mapping
        Per-parameter lower and upper bounds. The normalized result has shape
        ``(num_design_parameters, 2)``.
    fixed_values : mapping
        Fixed circuit values required by the simulator netlists.
    freq_range : sequence[float]
        Simulation frequency range in Hz, usually ``[f_start, f_stop]``.
    max_steps : int
        Maximum number of environment steps before truncation.
    max_param : float
        Upper bound of the normalized parameter representation used during
        decoding.
    bound_decode_mode : mapping or sequence or None, optional
        Per-parameter decoding mode. Supported modes are typically ``"log"``
        and ``"lin"``. If ``None``, default decoding modes are used.
    enable_iip3 : bool, optional
        Whether IIP3 simulation and IIP3-related performance terms are
        enabled. The default is ``True``.
    n_restricted : int, optional
        Number of invalidity or restriction flags appended to the observation.
        The default is ``0``.
    s_param_netlist_path : str or os.PathLike or None, optional
        Explicit path to the S-parameter netlist. The default is ``None``.
    nf_netlist_path : str or os.PathLike or None, optional
        Explicit path to the noise-figure netlist. The default is ``None``.
    iip3_netlist_path : str or os.PathLike or None, optional
        Explicit path to the IIP3 netlist. The default is ``None``.
    dc_op_result_path : str or os.PathLike or None, optional
        Explicit path to the DC operating-point result file. The default is
        ``None``.
    s_param_bandwidth_result_path : str or os.PathLike or None, optional
        Explicit path to the S-parameter bandwidth result file. The default is
        ``None``.
    nf_result_path : str or os.PathLike or None, optional
        Explicit path to the noise-figure result file. The default is
        ``None``.
    iip3_result_path : str or os.PathLike or None, optional
        Explicit path to the IIP3 result file. The default is ``None``.
    workspace_root : str or os.PathLike or None, optional
        Root directory used by the ngspice workspace manager. If ``None``, a
        temporary workspace root is created. The default is ``None``.
    worker_name : str or int, optional
        Worker label used in workspace naming. The default is
        ``"standalone"``.
    kernel_backend : str, optional
        ngspice execution backend identifier. The default is ``"batch"``.

    Attributes
    ----------
    circuit_type : str
        Circuit family identifier.
    enable_iip3 : bool
        Whether IIP3-related simulation and metrics are enabled.
    performance_names : list[str]
        Full ordered list of supported performance names.
    active_performance_names : list[str]
        Ordered list of performance names used by the current environment.
    bound_names : list[str]
        Ordered list of tunable circuit parameter names.
    fixed_value_names : list[str]
        Ordered list of fixed circuit value names.
    target_spec : numpy.ndarray
        Ordered target performance vector.
    bound : numpy.ndarray
        Parameter bound array with shape ``(N, 2)``.
    bound_decode_mode : list[str]
        Ordered list of decoding modes for design parameters.
    fixed_values : dict[str, float]
        Normalized fixed simulator values.
    freq_range : tuple[float, float]
        Validated simulation frequency range.
    max_steps : int
        Maximum episode length.
    n_spec : int
        Number of active performance specifications.
    max_param : float
        Upper bound of the normalized design parameter domain.
    simulator : NgSpiceSimulationService
        Simulator service used to evaluate candidate designs.
    n_variables : int
        Number of tunable design variables.
    N : int
        Alias for the number of tunable design variables.
    M : int
        Number of active performance metrics.
    K : int
        Number of restriction flags appended to the observation.
    n_fixed_values : int
        Number of fixed simulator values.
    num_design_variables : int
        Total number of tunable and fixed design variables.
    observation_dim : int
        Dimension of the Gym observation vector.
    action_dim : int
        Dimension of the Gym action vector.
    observation_space : gym.spaces.Box
        Continuous observation space.
    action_space : gym.spaces.Box
        Normalized continuous action space.
    """

    MIN_FREQUENCY_HZ = 1e6
    MAX_FREQUENCY_HZ = 8e9

    def __init__(
        self,
        circuit_type,
        target_spec,
        bound,
        fixed_values,
        freq_range,
        max_steps,
        max_param,
        bound_decode_mode=None,
        enable_iip3=True,
        n_restricted=0,
        s_param_netlist_path=None,
        nf_netlist_path=None,
        iip3_netlist_path=None,
        dc_op_result_path=None,
        s_param_bandwidth_result_path=None,
        nf_result_path=None,
        iip3_result_path=None,
        workspace_root=None,
        worker_name="standalone",
        kernel_backend="batch",
    ):
        """Initialize the base LNA optimization environment.

        Parameters
        ----------
        circuit_type : str
            Supported circuit family identifier, such as ``"CS"`` or
            ``"CGCS"``.
        target_spec : sequence or mapping
            Target performance specification ordered by active performance
            names or keyed by performance names.
        bound : sequence or mapping
            Per-parameter lower and upper bounds.
        fixed_values : mapping
            Fixed circuit values required by the simulator.
        freq_range : sequence[float]
            Simulation frequency range in Hz.
        max_steps : int
            Maximum number of steps per episode.
        max_param : float
            Upper bound of the normalized design parameter domain.
        bound_decode_mode : mapping or sequence or None, optional
            Per-parameter decode modes. The default is ``None``.
        enable_iip3 : bool, optional
            Whether to include IIP3 simulation and metrics. The default is
            ``True``.
        n_restricted : int, optional
            Number of restriction flags appended to the observation. The
            default is ``0``.
        s_param_netlist_path : str or os.PathLike or None, optional
            Explicit S-parameter netlist path. The default is ``None``.
        nf_netlist_path : str or os.PathLike or None, optional
            Explicit noise-figure netlist path. The default is ``None``.
        iip3_netlist_path : str or os.PathLike or None, optional
            Explicit IIP3 netlist path. The default is ``None``.
        dc_op_result_path : str or os.PathLike or None, optional
            Explicit DC operating-point result path. The default is ``None``.
        s_param_bandwidth_result_path : str or os.PathLike or None, optional
            Explicit S-parameter bandwidth result path. The default is
            ``None``.
        nf_result_path : str or os.PathLike or None, optional
            Explicit noise-figure result path. The default is ``None``.
        iip3_result_path : str or os.PathLike or None, optional
            Explicit IIP3 result path. The default is ``None``.
        workspace_root : str or os.PathLike or None, optional
            Root directory for simulator workspaces. If ``None``, a temporary
            workspace directory is created. The default is ``None``.
        worker_name : str or int, optional
            Worker label used in workspace naming. The default is
            ``"standalone"``.
        kernel_backend : str, optional
            ngspice execution backend identifier. The default is ``"batch"``.

        Returns
        -------
        None
            The constructor initializes normalized configuration, simulator
            service, internal dimensions, and Gym spaces in place.
        """

        super().__init__()

        self.circuit_type = circuit_type
        self.enable_iip3 = enable_iip3

        self.performance_names = ["S11", "S21", "S22", "NF", "PD", "IIP3"]
        self.active_performance_names = (
            self.performance_names
            if self.enable_iip3
            else self.performance_names[:-1]
        )

        self.bound_names = self._get_bound_names()
        self.fixed_value_names = ["v_dd", "r_b", "c_1", "l_m"]

        self.target_spec = self._normalize_named_array(
            target_spec,
            expected_names=self.active_performance_names,
            label="target_spec",
        )
        self.bound = self._normalize_bound_array(
            bound,
            expected_names=self.bound_names,
        )
        self.bound_decode_mode = self._normalize_bound_decode_mode(
            bound_decode_mode,
            expected_names=self.bound_names,
        )
        self.fixed_values = self._normalize_fixed_values(fixed_values)
        self.freq_range = self._normalize_freq_range(freq_range)

        self.max_steps = max_steps
        self.n_spec = len(self.target_spec)
        self.max_param = max_param

        self.project_name = str(circuit_type)
        self.experiment_run_id = create_run_id()
        self.workspace = None
        self.kernel_backend = str(kernel_backend)

        self.last_simulation_error = None
        self.last_simulation_profile = {}
        self.simulation_profiles = []
        self.last_reset_profile = {}
        self._last_valid_snapshot = None
        self.reset_fallback_after = 10
        self.profile_history_limit = 256

        default_workspace_root = (
            Path(workspace_root)
            if workspace_root is not None
            else (
                Path(tempfile.gettempdir())
                / "drl_spice_v4_ngspice"
                / self.circuit_type
                / self.experiment_run_id
            )
        )

        self.simulator = NgSpiceSimulationService(
            circuit_type=self.circuit_type,
            enable_iip3=self.enable_iip3,
            fixed_values=self.fixed_values,
            freq_range=self.freq_range,
            num_design_variables=len(self.bound_names) + len(self.fixed_value_names),
            workspace_root=default_workspace_root,
            worker_name=worker_name,
            kernel_backend=self.kernel_backend,
            project_name=self.project_name,
            run_id=self.experiment_run_id,
            profile_history_limit=self.profile_history_limit,
            s_param_netlist_path=s_param_netlist_path,
            nf_netlist_path=nf_netlist_path,
            iip3_netlist_path=iip3_netlist_path,
            dc_op_result_path=dc_op_result_path,
            s_param_bandwidth_result_path=s_param_bandwidth_result_path,
            nf_result_path=nf_result_path,
            iip3_result_path=iip3_result_path,
        )
        self._sync_simulator_state()

        if self.circuit_type == "CGCS":
            self.n_variables = 16
        elif self.circuit_type == "CS":
            self.n_variables = 9
        else:
            raise ValueError(
                f"Unsupported circuit_type: {self.circuit_type}. "
                f"Expected 'CS' or 'CGCS'."
            )

        self.N = self.n_variables
        self.M = len(self.target_spec)
        self.K = n_restricted
        self.n_fixed_values = len(self.fixed_values)
        self.num_design_variables = self.N + self.n_fixed_values

        self.observation_dim = self.M + self.N + self.K
        self.action_dim = self.N

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

    def _canonicalize_name(self, name: str) -> str:
        """Normalize a configuration key for name-insensitive matching.

        Parameters
        ----------
        name : str
            Configuration key or metric name to normalize.

        Returns
        -------
        str
            Canonicalized name used for robust key matching.
        """

        return canonicalize_name(name)

    def _get_bound_names(self) -> list[str]:
        """Return ordered design-variable names for the current circuit.

        Returns
        -------
        list[str]
            Ordered list of tunable parameter names for ``self.circuit_type``.
        """

        return get_bound_names(self.circuit_type)

    def _normalize_named_array(
        self,
        values,
        expected_names: list[str],
        label: str,
    ) -> np.ndarray:
        """Normalize a named or ordered metric vector.

        Parameters
        ----------
        values : sequence or mapping
            Input values to normalize. Mapping keys are matched against
            ``expected_names``; sequences are assumed to already follow the
            expected order.
        expected_names : list[str]
            Ordered names expected in the normalized output.
        label : str
            Human-readable label used in validation error messages.

        Returns
        -------
        numpy.ndarray
            Ordered one-dimensional array corresponding to ``expected_names``.
        """

        return normalize_named_array(
            values,
            expected_names=expected_names,
            label=label,
        )

    def _normalize_bound_array(self, bound, expected_names: list[str]) -> np.ndarray:
        """Normalize bound definitions into an ordered NumPy array.

        Parameters
        ----------
        bound : sequence or mapping
            Bound definitions for tunable circuit parameters.
        expected_names : list[str]
            Ordered parameter names expected in the normalized output.

        Returns
        -------
        numpy.ndarray
            Bound array with shape ``(len(expected_names), 2)``.
        """

        return normalize_bound_array(bound, expected_names=expected_names)

    def _normalize_bound_decode_mode(
        self,
        bound_decode_mode,
        expected_names: list[str],
    ) -> list[str]:
        """Normalize per-parameter decoding modes.

        Parameters
        ----------
        bound_decode_mode : mapping or sequence or None
            Decode-mode specification for each tunable parameter. Supported
            modes are defined by the LNA configuration utilities.
        expected_names : list[str]
            Ordered parameter names expected in the normalized output.

        Returns
        -------
        list[str]
            Ordered list of decode modes corresponding to ``expected_names``.
        """

        return normalize_bound_decode_mode(
            bound_decode_mode,
            expected_names=expected_names,
        )

    def _normalize_fixed_values(self, fixed_values) -> dict:
        """Normalize fixed simulator values into canonical key order.

        Parameters
        ----------
        fixed_values : mapping
            Mapping from fixed-value names to their numeric values.

        Returns
        -------
        dict[str, float]
            Normalized fixed values keyed by canonical fixed-value names.
        """

        return normalize_fixed_values(
            fixed_values,
            fixed_value_names=self.fixed_value_names,
        )

    def _normalize_freq_range(self, freq_range) -> tuple[float, float]:
        """Validate and normalize the simulation frequency range.

        Parameters
        ----------
        freq_range : sequence[float]
            Input frequency range in Hz.

        Returns
        -------
        tuple[float, float]
            Validated frequency range ``(f_start, f_stop)`` in Hz.
        """

        return normalize_freq_range(
            freq_range,
            min_frequency_hz=self.MIN_FREQUENCY_HZ,
            max_frequency_hz=self.MAX_FREQUENCY_HZ,
        )

    @torch.no_grad()
    def simulate(self, design_variables_config):
        """Run the simulator bridge for one candidate design.

        Parameters
        ----------
        design_variables_config : dict[str, float]
            Decoded simulator parameter mapping used to instantiate and
            evaluate a circuit design.

        Returns
        -------
        performances : numpy.ndarray
            Simulated performance vector ordered by active performance names.
        stability_factor : float
            Stability factor returned by the simulator bridge.
        """

        return bridge_simulate(self, design_variables_config)

    def step(self, *args, **kwargs):
        """Advance the environment by one step.

        Parameters
        ----------
        *args : tuple
            Subclass-specific positional arguments.
        **kwargs : dict[str, object]
            Subclass-specific keyword arguments.

        Returns
        -------
        object
            Subclass-defined Gym step result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class. Concrete subclasses must
            implement this method.
        """

        raise NotImplementedError()

    def reset(self, *args, **kwargs):
        """Reset the environment state.

        Parameters
        ----------
        *args : tuple
            Subclass-specific positional arguments.
        **kwargs : dict[str, object]
            Subclass-specific keyword arguments.

        Returns
        -------
        object
            Subclass-defined Gym reset result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class. Concrete subclasses must
            implement this method.
        """

        raise NotImplementedError()

    def render(self, mode="human", close=False):
        """Render the environment.

        Rendering is currently not implemented.

        Parameters
        ----------
        mode : str, optional
            Rendering mode requested by Gym. The default is ``"human"``.
        close : bool, optional
            Legacy Gym argument indicating whether rendering resources should
            be closed. The default is ``False``.

        Returns
        -------
        None
            This method currently performs no rendering.
        """

        pass

    def close(self):
        """Close simulator resources associated with the environment.

        Returns
        -------
        None
            Simulator resources are released in place.
        """

        self.simulator.close()

    def _reset_simulation_cache(self):
        """Clear the simulator-side memoized evaluation cache.

        Returns
        -------
        None
            Cached simulator evaluations associated with this environment are
            cleared.
        """

        bridge_reset_simulation_cache(self)

    def configure_workspace(
        self,
        *,
        project_name: str,
        run_id: str,
        run_root: Union[str, os.PathLike],
        worker_name: Union[str, int],
        scope: str = "train",
        clean: bool = False,
    ) -> None:
        """Bind the environment to a named simulator workspace.

        Parameters
        ----------
        project_name : str
            Project or experiment namespace used in workspace construction.
        run_id : str
            Unique run identifier.
        run_root : str or os.PathLike
            Root directory under which the simulator workspace is created.
        worker_name : str or int
            Worker identifier used to isolate simulator files.
        scope : str, optional
            Workspace scope, such as ``"train"`` or ``"eval"``. The default is
            ``"train"``.
        clean : bool, optional
            Whether to clean the workspace before use. The default is
            ``False``.

        Returns
        -------
        None
            Workspace configuration is applied in place.
        """

        bridge_configure_workspace(
            self,
            project_name=project_name,
            run_id=run_id,
            run_root=run_root,
            worker_name=worker_name,
            scope=scope,
            clean=clean,
        )

    def _make_simulation_cache_key(self, design_variables_config) -> tuple:
        """Build the simulator cache key for a decoded design.

        Parameters
        ----------
        design_variables_config : dict[str, float]
            Decoded simulator parameter mapping.

        Returns
        -------
        tuple
            Hashable cache key representing the candidate design and simulator
            context.
        """

        return bridge_make_simulation_cache_key(self, design_variables_config)

    def _get_cached_simulation(self, cache_key):
        """Return a cached simulator result if one exists.

        Parameters
        ----------
        cache_key : tuple
            Cache key produced by :meth:`_make_simulation_cache_key`.

        Returns
        -------
        object
            Cached simulation result if available. The exact return format is
            defined by the simulation bridge.
        """

        return bridge_get_cached_simulation(self, cache_key)

    def _store_cached_simulation(
        self,
        cache_key,
        performances,
        stability_factor,
    ) -> None:
        """Persist a simulator result in the environment cache.

        Parameters
        ----------
        cache_key : tuple
            Cache key associated with the evaluated design.
        performances : numpy.ndarray
            Simulated performance vector.
        stability_factor : float
            Stability factor returned by the simulator.

        Returns
        -------
        None
            The simulation result is stored in the cache.
        """

        bridge_store_cached_simulation(
            self,
            cache_key,
            performances,
            stability_factor,
        )

    def _sync_simulator_state(self):
        """Mirror simulator-side state onto environment attributes.

        Returns
        -------
        None
            Environment attributes are synchronized with the simulator bridge.
        """

        sync_simulator_state(self)

    def _restore_params(self, ps, target_bound, k=4):
        """Decode normalized parameters into physical design values.

        Parameters
        ----------
        ps : array-like or numpy.ndarray
            Normalized design parameters.
        target_bound : numpy.ndarray
            Physical parameter bounds used for decoding. Expected shape is
            ``(num_parameters, 2)``.
        k : float, optional
            Decoding sharpness or scaling parameter passed to the decoder. The
            default is ``4``.

        Returns
        -------
        numpy.ndarray
            Decoded physical design parameter values.
        """

        return restore_params(
            ps,
            target_bound,
            bound_decode_mode=self.bound_decode_mode,
            max_param=self.max_param,
            k=k,
        )

    def _make_design_variables_config(self, ps, target_bound):
        """Construct the simulator parameter mapping from normalized values.

        Parameters
        ----------
        ps : array-like or numpy.ndarray
            Normalized tunable design parameters.
        target_bound : numpy.ndarray
            Physical parameter bounds used to decode ``ps``.

        Returns
        -------
        dict[str, float]
            Simulator-ready design-variable mapping containing fixed values
            and decoded tunable parameters.
        """

        restored_params = self._restore_params(ps, target_bound)

        return make_design_variables_config(
            circuit_type=self.circuit_type,
            fixed_values=self.fixed_values,
            restored_params=restored_params,
        )