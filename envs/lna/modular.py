"""Concrete modular LNA environment implementation.

This module defines a concrete Gym-compatible LNA optimization environment
built on top of ``LNA_Environment_Base``. It combines simulator-backed state
updates, modular reward computation, reset handling, and termination logic.
"""

import numpy as np

from envs.lna.base import LNA_Environment_Base
from envs.lna.encoder import restore_last_valid_snapshot, update_state
from envs.lna.reset import reset_env
from envs.lna.reward import Reward
from envs.lna.termination import compute_step_flags


class LNAEnvBase(LNA_Environment_Base):
    """Concrete LNA optimization environment with modular reward and reset logic.

    This environment represents an LNA sizing task as a continuous-control
    optimization problem. Actions are interpreted as normalized parameter
    increments, candidate designs are evaluated through the simulator-backed
    state encoder, and rewards are computed through a configurable reward
    strategy.

    Parameters
    ----------
    circuit_type : str
        Supported circuit family identifier, such as ``"CS"`` or ``"CGCS"``.
    target_spec : sequence or mapping
        Target performance vector ordered by active performance names, or a
        mapping keyed by performance names.
    references : sequence or mapping
        Reference performance vector used to normalize simulated performance
        metrics.
    bound : sequence or mapping
        Per-parameter lower and upper bounds.
    fixed_values : mapping
        Fixed circuit values required by the simulator netlists.
    freq_range : sequence[float]
        Simulation frequency range in Hz, usually ``[f_start, f_stop]``.
    enable_iip3 : bool, optional
        Whether IIP3 simulation and reward terms are enabled. The default is
        ``True``.
    max_steps : int, optional
        Maximum number of environment steps before truncation. The default is
        ``20``.
    max_param : float, optional
        Upper bound of the normalized parameter space. The default is ``1.0``.
    bound_decode_mode : mapping or None, optional
        Per-parameter decoding mode, such as ``"log"`` or ``"lin"``. The
        default is ``None``.
    n_restricted : int, optional
        Number of invalidity flags appended to the observation. The default is
        ``2``.
    p : float, optional
        Exponent used by reward or violation shaping. The default is ``2.0``.
    beta : float, optional
        Reward-shaping or parameter-update coefficient. The default is
        ``0.2``.
    gamma : float, optional
        Discount factor or reward-shaping coefficient used by the environment.
        The default is ``0.99``.
    eta : float or None, optional
        Step size used to convert actions into normalized parameter updates.
        If ``None``, it is set to ``2 * max_param / max_steps``. The default is
        ``None``.
    reset_probability : float, optional
        Probability of forcing a random reset after an episode. The default is
        ``0.01``.
    allow_reset_fallback : bool, optional
        Whether reset may restore the last valid simulator state after
        repeated reset failures. The default is ``False``.
    reset_fallback_after : int, optional
        Number of failed random reset attempts before fallback is allowed. The
        default is ``10``.
    penal_viol : float, optional
        Reward penalty coefficient for constraint violation. The default is
        ``2.0``.
    penal_perf : float, optional
        Reward penalty or scaling coefficient for performance objective. The
        default is ``20.0``.
    lmda_viol : float, optional
        Weight for the violation-related reward term. The default is ``6.0``.
    lmda_perf : float, optional
        Weight for the performance-related reward term. The default is
        ``0.2``.
    lmda_var : float, optional
        Weight for the variance-related reward term. The default is ``0.1``.
    reward_strategy : object or None, optional
        Custom reward strategy object implementing the expected ``Reward``
        interface. The default is ``None``.
    reward_name : str or None, optional
        Name of a built-in reward strategy to instantiate. Currently supported
        values are ``"reward"`` and ``"default"``. The default is ``None``.
    s_param_netlist_path : str or os.PathLike or None, optional
        Explicit S-parameter netlist path. The default is ``None``.
    nf_netlist_path : str or os.PathLike or None, optional
        Explicit noise-figure netlist path. The default is ``None``.
    iip3_netlist_path : str or os.PathLike or None, optional
        Explicit IIP3 netlist path. The default is ``None``.
    dc_op_result_path : str or os.PathLike or None, optional
        Explicit DC operating-point result path. The default is ``None``.
    s_param_bandwidth_result_path : str or os.PathLike or None, optional
        Explicit S-parameter bandwidth result path. The default is ``None``.
    nf_result_path : str or os.PathLike or None, optional
        Explicit noise-figure result path. The default is ``None``.
    iip3_result_path : str or os.PathLike or None, optional
        Explicit IIP3 result path. The default is ``None``.
    workspace_root : str or os.PathLike or None, optional
        Root directory used by the ngspice workspace manager. The default is
        ``None``.
    worker_name : str or int, optional
        Worker label used in workspace naming. The default is ``"standalone"``.
    kernel_backend : str, optional
        ngspice execution backend identifier. The default is ``"batch"``.

    Attributes
    ----------
    references : numpy.ndarray
        Reference performance vector used for metric normalization.
    performances : numpy.ndarray
        Most recent simulated performance vector.
    parameters : numpy.ndarray
        Current normalized design-parameter vector.
    invalid_flag : numpy.ndarray
        Current invalidity-flag vector.
    action : numpy.ndarray
        Most recent action vector.
    state : numpy.ndarray
        Current encoded environment state.
    reward_strategy : object
        Reward strategy used to compute objective terms and scalar rewards.
    eta : float
        Step size used for normalized parameter updates.
    current_step : int
        Current episode step counter.
    is_non_convergent : bool
        Whether the latest simulation failed to converge.
    is_non_stable : bool
        Whether the latest design was marked unstable.
    is_invalid : bool
        Whether any invalidity flag is active.
    is_feasible : bool
        Whether the current design satisfies the feasibility criteria.
    """

    def __init__(
        self,
        circuit_type,
        target_spec,
        references,
        bound,
        fixed_values,
        freq_range,
        enable_iip3=True,
        max_steps=20,
        max_param=1.0,
        bound_decode_mode=None,
        n_restricted=2,
        p=2.0,
        beta=0.2,
        gamma=0.99,
        eta=None,
        reset_probability=0.01,
        allow_reset_fallback=False,
        reset_fallback_after=10,
        penal_viol=2.0,
        penal_perf=20.0,
        lmda_viol=6.0,
        lmda_perf=0.2,
        lmda_var=0.1,
        reward_strategy=None,
        reward_name=None,
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
        """Initialize the modular LNA optimization environment.

        Parameters
        ----------
        circuit_type : str
            Supported circuit family identifier.
        target_spec : sequence or mapping
            Target performance specification ordered by active performance
            names or keyed by performance names.
        references : sequence or mapping
            Reference baseline used to normalize performance metrics.
        bound : sequence or mapping
            Per-parameter lower and upper bounds.
        fixed_values : mapping
            Fixed circuit values required by the simulator.
        freq_range : sequence[float]
            Simulation frequency range in Hz.
        enable_iip3 : bool, optional
            Whether to include IIP3-related simulation and metrics. The
            default is ``True``.
        max_steps : int, optional
            Maximum number of environment steps before truncation. The default
            is ``20``.
        max_param : float, optional
            Upper bound of the normalized parameter space. The default is
            ``1.0``.
        bound_decode_mode : mapping or None, optional
            Per-parameter decode modes. The default is ``None``.
        n_restricted : int, optional
            Number of invalidity flags appended to the observation. The
            default is ``2``.
        p : float, optional
            Reward-shaping exponent. The default is ``2.0``.
        beta : float, optional
            Reward-shaping coefficient. The default is ``0.2``.
        gamma : float, optional
            Discount factor or reward-shaping coefficient. The default is
            ``0.99``.
        eta : float or None, optional
            Parameter-update step size. If ``None``, it is computed from
            ``max_param`` and ``max_steps``. The default is ``None``.
        reset_probability : float, optional
            Probability of forcing a random reset after an episode. The default
            is ``0.01``.
        allow_reset_fallback : bool, optional
            Whether reset fallback to the last valid snapshot is allowed. The
            default is ``False``.
        reset_fallback_after : int, optional
            Failed random reset threshold before fallback is allowed. The
            default is ``10``.
        penal_viol : float, optional
            Violation penalty coefficient. The default is ``2.0``.
        penal_perf : float, optional
            Performance penalty or scaling coefficient. The default is
            ``20.0``.
        lmda_viol : float, optional
            Violation reward weight. The default is ``6.0``.
        lmda_perf : float, optional
            Performance reward weight. The default is ``0.2``.
        lmda_var : float, optional
            Variance reward weight. The default is ``0.1``.
        reward_strategy : object or None, optional
            Custom reward strategy object. The default is ``None``.
        reward_name : str or None, optional
            Built-in reward strategy name. The default is ``None``.
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
            Root directory for simulator workspaces. The default is ``None``.
        worker_name : str or int, optional
            Worker label used in workspace naming. The default is
            ``"standalone"``.
        kernel_backend : str, optional
            ngspice execution backend identifier. The default is ``"batch"``.

        Returns
        -------
        None
            The constructor initializes the base environment, reward strategy,
            state vectors, parameters, and reward hyperparameters in place.
        """

        super().__init__(
            circuit_type=circuit_type,
            target_spec=target_spec,
            bound=bound,
            bound_decode_mode=bound_decode_mode,
            fixed_values=fixed_values,
            freq_range=freq_range,
            enable_iip3=enable_iip3,
            max_steps=max_steps,
            max_param=max_param,
            n_restricted=n_restricted,
            s_param_netlist_path=s_param_netlist_path,
            nf_netlist_path=nf_netlist_path,
            iip3_netlist_path=iip3_netlist_path,
            dc_op_result_path=dc_op_result_path,
            s_param_bandwidth_result_path=s_param_bandwidth_result_path,
            nf_result_path=nf_result_path,
            iip3_result_path=iip3_result_path,
            workspace_root=workspace_root,
            worker_name=worker_name,
            kernel_backend=kernel_backend,
        )

        self._initialization()

        self.references = self._normalize_named_array(
            references,
            expected_names=self.active_performance_names,
            label="references",
        )

        self.performances = np.zeros(self.M)
        self.parameters = np.zeros(self.N)
        self.invalid_flag = np.zeros(self.K)
        self.action = np.zeros(self.N)
        self.state = np.zeros(self.observation_dim, dtype=np.float32)

        if reward_strategy is not None and reward_name is not None:
            raise ValueError("Specify only one of reward_strategy or reward_name")

        if reward_strategy is None:
            reward_name_key = str(reward_name or "default").lower()

            if reward_name_key not in {"reward", "default"}:
                raise ValueError(f"Unsupported reward strategy: {reward_name}")

            reward_strategy = Reward()

        self.reward_strategy = reward_strategy

        self.p = p
        self.beta = beta
        self.gamma = gamma
        self.eta = 2.0 * self.max_param / self.max_steps if eta is None else eta

        self.reset_probability = float(reset_probability)
        self.allow_reset_fallback = bool(allow_reset_fallback)
        self.reset_fallback_after = int(reset_fallback_after)

        self.penal_viol = penal_viol
        self.penal_perf = penal_perf
        self.lmda_viol = lmda_viol
        self.lmda_perf = lmda_perf
        self.lmda_var = lmda_var

    def __str__(self):
        """Return the display name of the environment.

        Returns
        -------
        str
            Environment label, ``"LNA_Env_Base"``.
        """

        return "LNA_Env_Base"

    def _initialization(self, preserve_objectives=False):
        """Reset episode-local metrics and validity flags.

        Parameters
        ----------
        preserve_objectives : bool, optional
            Whether to preserve existing objective values ``fom``, ``perf``,
            and ``viol``. If ``False``, those values are reset to ``0.0``.
            The default is ``False``.

        Returns
        -------
        None
            Episode-local reward terms, step counters, validity flags, and
            simulation error state are reset in place.
        """

        prev_fom = getattr(self, "fom", 0.0)
        prev_perf = getattr(self, "perf", 0.0)
        prev_viol = getattr(self, "viol", 0.0)

        self.fom = prev_fom if preserve_objectives else 0.0
        self.perf = prev_perf if preserve_objectives else 0.0
        self.viol = prev_viol if preserve_objectives else 0.0

        self.var = 0.0
        self.reward = 0.0
        self.reward_viol = 0.0
        self.reward_perf = 0.0
        self.reward_var = 0.0
        self.pbrs_perf = 0.0
        self.pbrs_viol = 0.0

        self.current_step = 0

        self.is_non_convergent = False
        self.is_non_stable = False
        self.is_invalid = False
        self.is_feasible = False
        self.last_simulation_error = None

    def step(self, action):
        """Advance one design step in normalized parameter space.

        The action is interpreted as a normalized increment applied to the
        current design parameter vector. The updated design is clipped to the
        valid normalized parameter range before simulation and reward
        evaluation.

        Parameters
        ----------
        action : array-like
            Normalized action increment for each design variable. Expected
            shape is ``(self.N,)``. Values are typically in ``[-1, 1]``.

        Returns
        -------
        next_state : numpy.ndarray
            Encoded next observation with shape ``(self.observation_dim,)``.
        reward : float
            Scalar reward for the transition.
        terminated : bool
            Whether the episode has reached a terminal condition.
        truncated : bool
            Whether the episode was truncated, for example due to
            ``max_steps``.
        info : dict[str, object]
            Auxiliary diagnostic information for the transition.
        """

        self.current_step += 1

        x = np.clip(
            self.parameters + self.eta * action,
            0.0,
            self.max_param,
        )

        self.action = action

        o, next_state = update_state(self, x)
        reward = self._make_reward(o)
        terminated, truncated, info = compute_step_flags(self)

        return next_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment using the configured reset policy.

        Parameters
        ----------
        seed : int or None, optional
            Random seed passed by the Gym API. The default is ``None``.
        options : dict[str, object] or None, optional
            Optional reset configuration passed by the Gym API. The default is
            ``None``.

        Returns
        -------
        object
            Reset result returned by :func:`reset_env`. The exact structure is
            defined by the reset helper, typically an observation or an
            ``(observation, info)`` tuple depending on Gym compatibility.
        """

        return reset_env(self, seed=seed, options=options)

    def _update_state(self, x):
        """Update environment state from a candidate normalized design point.

        Parameters
        ----------
        x : array-like
            Candidate normalized design vector.

        Returns
        -------
        performances : numpy.ndarray
            Simulated performance vector.
        state : numpy.ndarray
            Updated encoded state vector.
        """

        return update_state(self, x)

    def _restore_last_valid_snapshot(self):
        """Restore the last simulator state that converged successfully.

        Returns
        -------
        performances : numpy.ndarray
            Restored performance vector.
        state : numpy.ndarray
            Restored encoded state vector.
        """

        return restore_last_valid_snapshot(self)

    def _get_reward_strategy(self):
        """Return the configured reward strategy instance.

        Returns
        -------
        object
            Reward strategy assigned to ``self.reward_strategy``. If the
            attribute is missing, a default :class:`Reward` instance is
            returned.
        """

        return getattr(self, "reward_strategy", Reward())

    def _make_f(self, performances):
        """Compute normalized performance progress values.

        Parameters
        ----------
        performances : array-like or numpy.ndarray
            Performance vector to normalize.

        Returns
        -------
        numpy.ndarray
            Normalized performance progress vector returned by the reward
            strategy.
        """

        return self._get_reward_strategy().make_f(self, performances)

    def _make_viol(self, performances=None, **kwargs):
        """Compute the feasibility-violation term.

        Parameters
        ----------
        performances : array-like or numpy.ndarray or None, optional
            Performance vector used for violation computation. If ``None``,
            the reward strategy may use environment state. The default is
            ``None``.
        **kwargs : dict[str, object]
            Additional keyword arguments forwarded to the reward strategy.

        Returns
        -------
        float
            Feasibility-violation value returned by the reward strategy.
        """

        return self._get_reward_strategy().make_viol(
            self,
            performances=performances,
            **kwargs,
        )

    def _make_perf(self, performances=None, eps=1e-9, **kwargs):
        """Compute the performance objective and figure of merit.

        Parameters
        ----------
        performances : array-like or numpy.ndarray or None, optional
            Performance vector used for performance computation. If ``None``,
            the reward strategy may use environment state. The default is
            ``None``.
        eps : float, optional
            Small positive constant used for numerical stability. The default
            is ``1e-9``.
        **kwargs : dict[str, object]
            Additional keyword arguments forwarded to the reward strategy.

        Returns
        -------
        perf : float
            Performance objective value.
        fom : float
            Figure-of-merit value. If the reward strategy returns a scalar
            instead of a tuple, the scalar is used for both ``perf`` and
            ``fom``.
        """

        perf = self._get_reward_strategy().make_perf(
            self,
            performances=performances,
            eps=eps,
            **kwargs,
        )

        if isinstance(perf, tuple):
            return perf

        return perf, perf

    def _make_var(self, performances=None, **kwargs):
        """Compute variance-based reward shaping metrics.

        Parameters
        ----------
        performances : array-like or numpy.ndarray or None, optional
            Performance vector used for variance computation. If ``None``,
            the reward strategy may use environment state. The default is
            ``None``.
        **kwargs : dict[str, object]
            Additional keyword arguments forwarded to the reward strategy.

        Returns
        -------
        float
            Variance-related shaping value returned by the reward strategy.
        """

        return self._get_reward_strategy().make_var(
            self,
            performances=performances,
            **kwargs,
        )

    def _make_reward(self, performances):
        """Compute the scalar reward for the current transition.

        This method ensures that reward-related attributes exist before
        delegating the actual reward computation to the configured reward
        strategy.

        Parameters
        ----------
        performances : array-like or numpy.ndarray
            Current performance vector used to compute the reward.

        Returns
        -------
        float
            Scalar reward value returned by the reward strategy.
        """

        if not hasattr(self, "perf"):
            self.perf = 0.0

        if not hasattr(self, "viol"):
            self.viol = 0.0

        if not hasattr(self, "var"):
            self.var = 0.0

        if not hasattr(self, "lmda_viol"):
            self.lmda_viol = 6.0

        if not hasattr(self, "lmda_perf"):
            self.lmda_perf = 0.2

        if not hasattr(self, "lmda_var"):
            self.lmda_var = 0.1

        return self._get_reward_strategy().make_reward(self, performances)


LNA_Modular_base = LNAEnvBase
"""type[LNAEnvBase]: Backward-compatible alias for the modular LNA environment."""