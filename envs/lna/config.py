"""Configuration normalization helpers for LNA environments.

This module provides utility functions for validating and normalizing
configuration inputs used by low-noise-amplifier optimization environments.
It supports case-insensitive and separator-insensitive key matching,
circuit-specific design-variable ordering, metric-vector normalization,
parameter-bound normalization, decoding-mode normalization, fixed-value
normalization, and frequency-range validation.
"""

from collections.abc import Mapping

import numpy as np


def canonicalize_name(name: str) -> str:
    """Normalize a name for robust key matching.

    The normalization removes underscores, hyphens, and spaces, then converts
    the result to lowercase. This allows configuration keys such as
    ``"R_D"``, ``"r-d"``, and ``"r d"`` to be matched consistently.

    Parameters
    ----------
    name : str
        Input name or configuration key to normalize.

    Returns
    -------
    str
        Canonicalized lowercase string with common separators removed.
    """

    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


def get_bound_names(circuit_type: str) -> list[str]:
    """Return ordered design-variable names for a supported circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit family identifier. Supported values are ``"CGCS"`` and
        ``"CS"``.

    Returns
    -------
    list[str]
        Ordered list of tunable design-variable names used by the selected
        circuit type.

    Raises
    ------
    ValueError
        If ``circuit_type`` is not supported.
    """

    if circuit_type == "CGCS":
        return [
            "V_b1",
            "V_b2",
            "V_b3",
            "V_b4",
            "R_D1",
            "R_D4",
            "R_S5",
            "C_D1",
            "C_D4",
            "C_S3",
            "C_S4",
            "XM1",
            "XM2",
            "XM3",
            "XM4",
            "XM5",
        ]

    if circuit_type == "CS":
        return [
            "V_b",
            "R_D",
            "L_D",
            "L_G",
            "L_S",
            "C_D",
            "C_ex",
            "XM1",
            "XM2",
        ]

    raise ValueError(f"Unsupported circuit_type: {circuit_type}")


def normalize_named_array(
    values,
    *,
    expected_names: list[str],
    label: str,
) -> np.ndarray:
    """Normalize a named or ordered metric sequence into NumPy form.

    If ``values`` is a mapping, keys are canonicalized and reordered according
    to ``expected_names``. If ``values`` is a sequence, it is converted to a
    one-dimensional NumPy array and validated against the expected length.

    Parameters
    ----------
    values : Mapping or sequence
        Input values to normalize. Mapping inputs are matched by key.
        Sequence inputs are assumed to already follow ``expected_names``.
    expected_names : list[str]
        Ordered list of names expected in the normalized output.
    label : str
        Human-readable label used in validation error messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of normalized values ordered according to
        ``expected_names``.

    Raises
    ------
    ValueError
        If required mapping keys are missing, if a sequence is not
        one-dimensional, or if too few sequence values are provided.
    """

    if isinstance(values, Mapping):
        normalized = {canonicalize_name(k): v for k, v in values.items()}
        missing = [
            name
            for name in expected_names
            if canonicalize_name(name) not in normalized
        ]

        if missing:
            raise ValueError(f"Missing {label} keys: {missing}")

        ordered = [normalized[canonicalize_name(name)] for name in expected_names]
        return np.array(ordered, dtype=float)

    values_arr = np.array(values, dtype=float)

    if values_arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D sequence or mapping.")

    if len(values_arr) < len(expected_names):
        raise ValueError(f"{label} must have at least {len(expected_names)} values.")

    if len(values_arr) > len(expected_names):
        values_arr = values_arr[: len(expected_names)]

    return values_arr


def normalize_bound_array(bound, *, expected_names: list[str]) -> np.ndarray:
    """Normalize parameter-bound definitions into an ordered array.

    Mapping inputs are matched by canonicalized parameter names and reordered
    according to ``expected_names``. Sequence inputs must already have shape
    ``(N, 2)``, where ``N`` is the number of expected parameter names.

    Parameters
    ----------
    bound : Mapping or sequence
        Parameter-bound definitions. Mapping values should contain
        lower-upper pairs. Sequence inputs should have shape
        ``(len(expected_names), 2)``.
    expected_names : list[str]
        Ordered list of parameter names expected in the normalized output.

    Returns
    -------
    numpy.ndarray
        Two-dimensional bound array with shape ``(len(expected_names), 2)``.

    Raises
    ------
    ValueError
        If required mapping keys are missing, if the sequence is not
        two-dimensional with two columns, or if the number of bound entries
        does not match ``expected_names``.
    """

    if isinstance(bound, Mapping):
        normalized = {canonicalize_name(k): v for k, v in bound.items()}
        missing = [
            name
            for name in expected_names
            if canonicalize_name(name) not in normalized
        ]

        if missing:
            raise ValueError(f"Missing bound keys: {missing}")

        ordered = [normalized[canonicalize_name(name)] for name in expected_names]
        return np.array(ordered, dtype=float)

    bound_arr = np.array(bound, dtype=float)

    if bound_arr.ndim != 2 or bound_arr.shape[1] != 2:
        raise ValueError("bound must be a Nx2 sequence or mapping.")

    if bound_arr.shape[0] != len(expected_names):
        raise ValueError(f"bound must have {len(expected_names)} entries.")

    return bound_arr


def normalize_bound_decode_mode(
    bound_decode_mode,
    *,
    expected_names: list[str],
) -> list[str]:
    """Normalize per-parameter bound decoding modes.

    Each parameter can be decoded using either logarithmic or linear scaling.
    Missing parameter entries default to ``"log"``.

    Parameters
    ----------
    bound_decode_mode : Mapping or None
        Mapping from parameter names to decode modes. Supported values are
        ``"log"`` and ``"lin"``. If ``None``, all parameters use ``"log"``.
    expected_names : list[str]
        Ordered list of parameter names expected in the normalized output.

    Returns
    -------
    list[str]
        Ordered list of decode modes corresponding to ``expected_names``.

    Raises
    ------
    ValueError
        If ``bound_decode_mode`` is not a mapping when provided, or if an
        unsupported decode mode is specified.
    """

    default_modes = ["log"] * len(expected_names)

    if bound_decode_mode is None:
        return default_modes

    if not isinstance(bound_decode_mode, Mapping):
        raise ValueError("bound_decode_mode must be a mapping.")

    normalized = {
        canonicalize_name(k): str(v).strip().lower()
        for k, v in bound_decode_mode.items()
    }
    supported_modes = {"log", "lin"}
    modes = []

    for name in expected_names:
        mode = normalized.get(canonicalize_name(name), "log")

        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported decode mode for {name}: {mode}. "
                f"Supported modes: {sorted(supported_modes)}"
            )

        modes.append(mode)

    return modes


def normalize_fixed_values(fixed_values, *, fixed_value_names: list[str]) -> dict:
    """Normalize fixed simulator values into canonical key order.

    Parameters
    ----------
    fixed_values : Mapping
        Mapping from fixed-value names to numeric simulator values.
    fixed_value_names : list[str]
        Ordered list of required fixed-value names.

    Returns
    -------
    dict[str, object]
        Dictionary ordered by ``fixed_value_names``. Values are taken from the
        corresponding canonicalized input keys.

    Raises
    ------
    ValueError
        If ``fixed_values`` is not a mapping or if required keys are missing.
    """

    if not isinstance(fixed_values, Mapping):
        raise ValueError("fixed_values must be a mapping.")

    normalized = {canonicalize_name(k): v for k, v in fixed_values.items()}
    missing = [
        name
        for name in fixed_value_names
        if canonicalize_name(name) not in normalized
    ]

    if missing:
        raise ValueError(f"Missing fixed_values keys: {missing}")

    return {
        name: normalized[canonicalize_name(name)]
        for name in fixed_value_names
    }


def normalize_freq_range(
    freq_range,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> tuple[float, float]:
    """Validate and normalize the simulation frequency range.

    Parameters
    ----------
    freq_range : sequence[float]
        Two-element frequency range ``[f1, f2]`` in Hz.
    min_frequency_hz : float
        Minimum allowed frequency in Hz.
    max_frequency_hz : float
        Maximum allowed frequency in Hz.

    Returns
    -------
    tuple[float, float]
        Validated frequency range ``(f1, f2)`` in Hz.

    Raises
    ------
    ValueError
        If ``freq_range`` is missing, does not contain exactly two values,
        falls outside the allowed frequency range, or violates ``f1 <= f2``.
    """

    if freq_range is None:
        raise ValueError("freq_range must be provided.")

    if len(freq_range) != 2:
        raise ValueError("freq_range must contain exactly two values: [f1, f2].")

    f1 = float(freq_range[0])
    f2 = float(freq_range[1])

    if f1 < min_frequency_hz or f2 > max_frequency_hz:
        raise ValueError(
            f"freq_range must stay within "
            f"[{min_frequency_hz}, {max_frequency_hz}] Hz."
        )

    if f1 > f2:
        raise ValueError("freq_range must satisfy f1 <= f2.")

    return (f1, f2)