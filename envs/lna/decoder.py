"""Decoding helpers that map normalized actions into circuit parameters.

This module provides helper functions for converting normalized design
parameters into physical circuit values and assembling simulator-ready
parameter dictionaries for supported LNA circuit topologies.
"""

import numpy as np


def _round_sig(x, k):
    """Round a scalar value to a given number of significant digits.

    Parameters
    ----------
    x : int, float, or numpy.ndarray
        Input scalar value to round.
    k : int
        Number of significant digits to preserve.

    Returns
    -------
    int, float, or numpy.ndarray
        Value rounded to ``k`` significant digits. If ``x`` is zero, ``0`` is
        returned directly.
    """

    x_arr = np.asarray(x)

    if x_arr == 0:
        return 0

    d = k - 1 - int(np.floor(np.log10(abs(x_arr))))
    return np.round(x_arr, d)


def restore_params(
    ps,
    target_bound,
    *,
    bound_decode_mode,
    max_param,
    k=4,
) -> np.ndarray:
    """Decode normalized parameters into physical circuit values.

    Each normalized parameter is decoded according to its corresponding
    decoding mode. Linear decoding maps the normalized value directly between
    the lower and upper physical bounds. Logarithmic decoding performs the
    interpolation in base-10 logarithmic space.

    Parameters
    ----------
    ps : sequence or numpy.ndarray
        Normalized parameter vector. Each value is expected to lie within the
        normalized design range, typically ``[0, max_param]``.
    target_bound : sequence or numpy.ndarray
        Per-parameter lower and upper physical bounds. Expected shape is
        ``(num_parameters, 2)``.
    bound_decode_mode : sequence[str]
        Per-parameter decoding mode. Each entry should be either ``"log"`` or
        ``"lin"``.
    max_param : float
        Upper bound of the normalized parameter space.
    k : int, optional
        Number of significant digits used for rounding decoded values. The
        default is ``4``.

    Returns
    -------
    numpy.ndarray
        Decoded physical parameter values with shape ``(num_parameters,)``.
    """

    restored = []

    for p, bound_values, decode_mode in zip(ps, target_bound, bound_decode_mode):
        p_min, p_max = bound_values
        p_arr = np.asarray(p)

        if decode_mode == "lin":
            raw_val = p_arr * (p_max - p_min) / max_param + p_min
        else:
            p_log = (
                p_arr * (np.log10(p_max) - np.log10(p_min)) / max_param
                + np.log10(p_min)
            )
            raw_val = 10 ** p_log

        restored.append(_round_sig(raw_val, k))

    return np.array(restored)


def make_design_variables_config(
    *,
    circuit_type: str,
    fixed_values: dict,
    restored_params,
) -> dict:
    """Build the simulator parameter mapping for a decoded design.

    This function combines fixed circuit values and decoded tunable
    parameters into the dictionary format expected by the simulator backend.

    Parameters
    ----------
    circuit_type : str
        Circuit family identifier. Supported values are ``"CGCS"`` and
        ``"CS"``.
    fixed_values : dict[str, float]
        Dictionary containing fixed simulator values. Required keys are
        ``"v_dd"``, ``"r_b"``, ``"c_1"``, and ``"l_m"``.
    restored_params : sequence or numpy.ndarray
        Decoded physical design parameter vector. For ``"CGCS"``, this must
        contain 16 values. For ``"CS"``, this must contain 9 values.

    Returns
    -------
    dict[str, float]
        Simulator-ready parameter dictionary containing both fixed and tunable
        circuit parameters.

    Raises
    ------
    ValueError
        If ``circuit_type`` is not supported.
    """

    if circuit_type == "CGCS":
        return {
            "v_dd": fixed_values["v_dd"],
            "r_b": fixed_values["r_b"],
            "c_1": fixed_values["c_1"],
            "l_m": fixed_values["l_m"],
            "v_b1": restored_params[0],
            "v_b2": restored_params[1],
            "v_b3": restored_params[2],
            "v_b4": restored_params[3],
            "r_d1": restored_params[4],
            "r_d4": restored_params[5],
            "r_s5": restored_params[6],
            "c_d1": restored_params[7],
            "c_d4": restored_params[8],
            "c_s3": restored_params[9],
            "c_s4": restored_params[10],
            "w_m1": restored_params[11],
            "w_m2": restored_params[12],
            "w_m3": restored_params[13],
            "w_m4": restored_params[14],
            "w_m5": restored_params[15],
        }

    if circuit_type == "CS":
        return {
            "v_dd": fixed_values["v_dd"],
            "r_b": fixed_values["r_b"],
            "c_1": fixed_values["c_1"],
            "l_m": fixed_values["l_m"],
            "v_b": restored_params[0],
            "r_d": restored_params[1],
            "l_d": restored_params[2],
            "l_g": restored_params[3],
            "l_s": restored_params[4],
            "c_d": restored_params[5],
            "c_ex": restored_params[6],
            "w_m1": restored_params[7],
            "w_m2": restored_params[8],
        }

    raise ValueError(f"Unsupported circuit_type: {circuit_type}")