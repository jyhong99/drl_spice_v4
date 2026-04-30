"""Termination and truncation helpers for LNA environments.

This module provides helper functions for converting environment validity
state into Gymnasium-compatible termination and truncation flags.
"""


def compute_step_flags(env):
    """Derive Gymnasium termination flags from environment validity state.

    Parameters
    ----------
    env : LNAEnvBase
        Environment instance after a transition update. The environment is
        expected to expose validity flags, simulation-error state, current
        step count, and maximum episode length.

    Returns
    -------
    terminated : bool
        Whether the episode should terminate due to a terminal condition.
        In this implementation, non-convergent simulator evaluation triggers
        termination.
    truncated : bool
        Whether the episode should truncate due to an external limit. In this
        implementation, truncation occurs when ``current_step >= max_steps``
        and the episode has not already terminated.
    info : dict[str, object]
        Diagnostic dictionary containing feasibility flags, simulator error
        information, and termination or truncation reasons.
    """

    terminated = bool(env.is_non_convergent)
    truncated = bool((env.current_step >= env.max_steps) and not terminated)

    info = {
        "is_feasible": env.is_feasible,
        "is_non_convergent": env.is_non_convergent,
        "is_non_stable": env.is_non_stable,
        "is_invalid": env.is_invalid,
        "simulation_error": env.last_simulation_error,
        "termination_reason": "simulator_failure" if terminated else None,
        "truncation_reason": "time_limit" if truncated else None,
    }

    return terminated, truncated, info