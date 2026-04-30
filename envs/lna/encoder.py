"""State encoding helpers for simulator-backed LNA environments.

This module provides helper functions for restoring valid simulator states
and updating environment observations after evaluating candidate LNA design
points. The encoded state combines normalized performance margins, normalized
design parameters, and optional invalidity flags.
"""

import logging

import numpy as np


logger = logging.getLogger(__name__)


def restore_last_valid_snapshot(env):
    """Restore the most recent converged simulator snapshot.

    This function restores performance values, design parameters, invalidity
    flags, encoded state, feasibility flags, and decoded simulator parameters
    from ``env._last_valid_snapshot``. It is intended to be used as a fallback
    when reset or simulation recovery requires a previously valid state.

    Parameters
    ----------
    env : LNA_Environment_Base
        Environment instance containing ``_last_valid_snapshot`` and mutable
        state attributes.

    Returns
    -------
    performances : numpy.ndarray
        Restored performance vector copied from the last valid snapshot.
    state : numpy.ndarray
        Restored encoded state vector copied from the last valid snapshot.

    Raises
    ------
    RuntimeError
        If no valid snapshot is available.
    """

    snapshot = env._last_valid_snapshot

    if snapshot is None:
        raise RuntimeError("No valid snapshot is available for reset fallback.")

    env.performances = np.array(snapshot["performances"], copy=True)
    env.parameters = np.array(snapshot["parameters"], copy=True)
    env.invalid_flag = np.array(snapshot["invalid_flag"], copy=True)
    env.state = np.array(snapshot["state"], copy=True)

    env.is_non_convergent = snapshot["is_non_convergent"]
    env.is_non_stable = snapshot["is_non_stable"]
    env.is_invalid = snapshot["is_invalid"]
    env.is_feasible = snapshot["is_feasible"]

    env.design_variables_config = dict(snapshot["design_variables_config"])
    env.last_simulation_error = None

    return np.array(env.performances, copy=True), np.array(env.state, copy=True)


def update_state(env, x):
    """Evaluate a candidate design point and update the environment state.

    This function decodes the normalized design vector, runs the simulator,
    updates performance values and invalidity flags, and constructs the next
    encoded state. If simulation fails, the previous performance and parameter
    values are preserved, and the invalidity flag is updated accordingly.

    The encoded state is formed by concatenating:

    - normalized performance vector,
    - normalized design parameter vector,
    - invalidity flag vector.

    Parameters
    ----------
    env : LNA_Environment_Base
        Environment instance to update. The environment is expected to provide
        target specifications, reference values, parameter bounds, simulator
        helpers, and mutable state attributes.
    x : array-like
        Candidate normalized design vector. Expected shape is
        ``(env.N,)`` or another shape compatible with the environment's
        parameter decoder.

    Returns
    -------
    performances : numpy.ndarray
        Simulated performance vector. If simulation fails, this is copied from
        the previous performance vector.
    state : numpy.ndarray
        Updated encoded state vector with dtype ``numpy.float32``.

    Raises
    ------
    ValueError
        If any element of ``env.target_spec`` is equal to the corresponding
        element of ``env.references``, because normalization would require
        division by zero.
    """

    attempted_x = np.array(x, copy=True)
    prev_performances = np.array(env.performances, copy=True)
    prev_parameters = np.array(env.parameters, copy=True)

    if np.any(env.target_spec == env.references):
        raise ValueError("target_spec and references must differ for every metric.")

    env.design_variables_config = env._make_design_variables_config(
        attempted_x,
        env.bound,
    )

    simulation_failed = False

    try:
        o, stability_factor = env.simulate(env.design_variables_config)
    except Exception as exc:
        simulation_failed = True
        env.last_simulation_error = str(exc)
        logger.warning("Simulation failed: %s", exc)
        o = np.array(prev_performances, copy=True)
        stability_factor = 0.0

    u = np.zeros(env.K)

    if env.K > 0 and simulation_failed:
        u[0] = 1

    if env.K > 1:
        u[1] = 1 if stability_factor < 1.0 else 0

    env.is_non_convergent = bool(env.K > 0 and u[0])
    env.is_non_stable = bool(env.K > 1 and u[1])
    env.is_invalid = bool(np.any(u == 1))
    env.invalid_flag = u

    if simulation_failed:
        env.performances = prev_performances
        env.parameters = prev_parameters

        f_prev = (
            (prev_performances - env.references)
            / (env.target_spec - env.references)
        )
        state = np.array(
            f_prev.tolist() + prev_parameters.tolist() + u.tolist(),
            dtype=np.float32,
        )
        env.state = state

        return o, state

    env.performances = o
    env.parameters = attempted_x

    f = (o - env.references) / (env.target_spec - env.references)
    state = np.array(
        f.tolist() + attempted_x.tolist() + u.tolist(),
        dtype=np.float32,
    )
    env.state = state

    if not env.is_non_convergent:
        env._last_valid_snapshot = {
            "performances": np.array(env.performances, copy=True),
            "parameters": np.array(env.parameters, copy=True),
            "invalid_flag": np.array(env.invalid_flag, copy=True),
            "state": np.array(env.state, copy=True),
            "is_non_convergent": env.is_non_convergent,
            "is_non_stable": env.is_non_stable,
            "is_invalid": env.is_invalid,
            "is_feasible": getattr(env, "is_feasible", False),
            "design_variables_config": dict(env.design_variables_config),
        }

    return o, state
