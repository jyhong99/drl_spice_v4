"""Static template metadata for supported LNA circuit families.

This module defines immutable template metadata used to parse annotated
ngspice netlists for supported LNA circuit topologies. Each template stores
the expected design-variable names, device names, and annotation markers used
by the circuit parser.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitTemplate:
    """Template metadata used to parse annotated ngspice netlists.

    Parameters
    ----------
    circuit_type : str
        Circuit family identifier associated with the template.
    designvar_names : tuple[str, ...]
        Ordered names of design variables expected in the annotated netlist.
    device_names : tuple[str, ...]
        Ordered names of devices expected in the annotated circuit section.
    designvar_annotation_start : str, optional
        Marker indicating the start of the design-variable annotation block.
        The default is ``"** design_variables_start"``.
    designvar_annotation_end : str, optional
        Marker indicating the end of the design-variable annotation block.
        The default is ``"** design_variables_end"``.
    device_annotation_start : str, optional
        Marker indicating the start of the device annotation block. The
        default is ``"**.subckt My_LNA"``.
    device_annotation_end : str, optional
        Marker indicating the end of the device annotation block. The default
        is ``"**** begin user architecture code"``.

    Attributes
    ----------
    circuit_type : str
        Circuit family identifier associated with the template.
    designvar_names : tuple[str, ...]
        Ordered design-variable names.
    device_names : tuple[str, ...]
        Ordered device names.
    designvar_annotation_start : str
        Start marker for the design-variable annotation block.
    designvar_annotation_end : str
        End marker for the design-variable annotation block.
    device_annotation_start : str
        Start marker for the device annotation block.
    device_annotation_end : str
        End marker for the device annotation block.
    """

    circuit_type: str
    designvar_names: tuple[str, ...]
    device_names: tuple[str, ...]
    designvar_annotation_start: str = "** design_variables_start"
    designvar_annotation_end: str = "** design_variables_end"
    device_annotation_start: str = "**.subckt My_LNA"
    device_annotation_end: str = "**** begin user architecture code"


_TEMPLATES = {
    "CGCS": CircuitTemplate(
        circuit_type="CGCS",
        designvar_names=(
            "v_dd",
            "r_b",
            "c_1",
            "l_m",
            "v_b1",
            "v_b2",
            "v_b3",
            "v_b4",
            "r_d1",
            "r_d4",
            "r_s5",
            "c_d1",
            "c_d4",
            "c_s3",
            "c_s4",
            "w_m1",
            "w_m2",
            "w_m3",
            "w_m4",
            "w_m5",
        ),
        device_names=(
            "V_DD",
            "R_b",
            "C_1",
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
        ),
    ),
    "CS": CircuitTemplate(
        circuit_type="CS",
        designvar_names=(
            "v_dd",
            "r_b",
            "c_1",
            "l_m",
            "v_b",
            "r_d",
            "l_d",
            "l_g",
            "l_s",
            "c_d",
            "c_g",
            "c_ex",
            "w_m1",
            "w_m2",
        ),
        device_names=(
            "V_DD",
            "R_b",
            "C_1",
            "V_b",
            "R_D",
            "L_D",
            "L_G",
            "L_S",
            "C_D",
            "C_G",
            "C_ex",
            "XM1",
            "XM2",
        ),
    ),
}


def get_circuit_template(circuit_type: str) -> CircuitTemplate:
    """Return template metadata for one supported circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit family identifier. Supported values are the keys returned by
        :func:`list_circuit_templates`.

    Returns
    -------
    CircuitTemplate
        Immutable template metadata for the requested circuit type.

    Raises
    ------
    ValueError
        If ``circuit_type`` is not registered in the template table.
    """

    try:
        return _TEMPLATES[str(circuit_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported circuit_type: {circuit_type}") from exc


def list_circuit_templates() -> tuple[str, ...]:
    """Return supported circuit template names.

    Returns
    -------
    tuple[str, ...]
        Tuple containing all registered circuit template identifiers.
    """

    return tuple(_TEMPLATES.keys())


__all__ = [
    "CircuitTemplate",
    "get_circuit_template",
    "list_circuit_templates",
]
"""list[str]: Public symbols exported by this module."""