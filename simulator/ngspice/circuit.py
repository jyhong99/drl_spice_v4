"""Circuit template metadata and netlist parsing helpers.

This module defines a circuit metadata wrapper that loads an annotated ngspice
netlist and extracts template-defined design-variable and device mappings.
"""

import re

from simulator.ngspice.templates import get_circuit_template


class Circuit(object):
    """Parsed circuit netlist plus template-derived metadata.

    This class loads a circuit netlist from disk, obtains the corresponding
    circuit template metadata, and parses annotated sections to recover
    design-variable default values and device-to-design-variable bindings.

    Parameters
    ----------
    netlist_path : str or pathlib.Path
        Path to the circuit netlist file.
    circuit_type : str
        Circuit template identifier used by
        :func:`simulator.ngspice.templates.get_circuit_template`.

    Attributes
    ----------
    netlist_path : str or pathlib.Path
        Path to the source netlist file.
    circuit_type : str
        Circuit template identifier.
    template : object
        Template metadata object for the selected circuit type.
    designvar_annotation_start : str
        Marker indicating the start of the design-variable annotation section.
    devignvar_annotation_end : str
        Marker indicating the end of the design-variable annotation section.
    device_annotation_start : str
        Marker indicating the start of the device annotation section.
    device_annotation_end : str
        Marker indicating the end of the device annotation section.
    designvar_names : list[str]
        Valid design-variable names declared by the template.
    device_names : list[str]
        Valid device names declared by the template.
    netlist : str
        Full netlist text loaded from disk.
    dsgnvar_to_val : dict[str, float]
        Mapping from design-variable names to parsed default values.
    dvc_to_dsgnvar : dict[str, str]
        Mapping from device identifiers to design-variable names or assignment
        expressions.
    dvc_to_val : dict[str, float or None]
        Mapping from device identifiers to resolved numeric values when
        available.
    """

    def __init__(self, netlist_path, circuit_type) -> None:
        """Initialize and parse a circuit netlist.

        Parameters
        ----------
        netlist_path : str or pathlib.Path
            Path to the circuit netlist file.
        circuit_type : str
            Circuit template identifier.

        Returns
        -------
        None
            Template metadata, netlist text, and parsed mappings are stored in
            the instance.
        """

        self.netlist_path = netlist_path
        self.circuit_type = str(circuit_type)
        self.template = get_circuit_template(self.circuit_type)

        self.designvar_annotation_start = self.template.designvar_annotation_start
        self.devignvar_annotation_end = self.template.designvar_annotation_end
        self.device_annotation_start = self.template.device_annotation_start
        self.device_annotation_end = self.template.device_annotation_end

        self.designvar_names = list(self.template.designvar_names)
        self.device_names = list(self.template.device_names)

        self.netlist = self._load_netlist()
        self.dsgnvar_to_val, self.dvc_to_dsgnvar, self.dvc_to_val = (
            self._update_circuit()
        )

    def _load_netlist(self) -> str:
        """Load the full netlist text from disk.

        Returns
        -------
        str
            Full text content of the netlist file.

        Raises
        ------
        Exception
            If the netlist file does not exist or cannot be read.
        """

        try:
            with open(self.netlist_path, "r") as f:
                netlist = f.read()
            return netlist
        except FileNotFoundError:
            raise Exception(f"Netlist file at {self.netlist_path} not found.")
        except IOError:
            raise Exception(f"Error reading the netlist file at {self.netlist_path}.")

    def _map_dsgnvar_to_val(self) -> dict:
        """Parse design-variable default values from annotated netlist text.

        The parser searches the design-variable annotation block and extracts
        assignments of the form ``name = value``.

        Returns
        -------
        dict[str, float]
            Mapping from design-variable names to numeric default values.

        Raises
        ------
        Exception
            If the design-variable annotation block cannot be found or parsed.
        """

        try:
            start_idx = self.netlist.index(
                self.designvar_annotation_start
            ) + len(self.designvar_annotation_start)
            end_idx = self.netlist.index(self.devignvar_annotation_end)
            raw_inform = self.netlist[start_idx:end_idx].strip().split("\n")

            dsgnvar_to_val = {}

            for raw_txt in raw_inform:
                match = re.search(r"(\w+)\s*=\s*([\d.]+)", raw_txt)

                if match:
                    var_name, var_value = match.groups()
                    dsgnvar_to_val[var_name] = float(var_value)

            return dsgnvar_to_val

        except ValueError:
            raise Exception("Error in parsing design variables from the netlist.")

    def _map_device_to_dsgnvar(self) -> dict:
        """Parse device-to-design-variable bindings from annotated netlist text.

        The parser searches the device annotation block and maps devices to
        either direct design-variable names or parameter assignment expressions
        found in the annotated device lines.

        Returns
        -------
        dict[str, str]
            Mapping from device identifiers to design-variable names or raw
            assignment expressions.

        Raises
        ------
        Exception
            If the device annotation block cannot be found or parsed.
        """

        try:
            start_idx = self.netlist.index(self.device_annotation_start) + len(
                self.device_annotation_start
            )
            end_idx = self.netlist.index(self.device_annotation_end)
            raw_inform = self.netlist[start_idx:end_idx].strip().split("\n")

            dvc_to_dsgnvar = {}

            for raw_txt in raw_inform:
                parts = raw_txt.strip().split()

                for dvc_name in self.device_names:
                    if dvc_name in parts:
                        name = dvc_name

                        for part in parts:
                            if "=" in part:
                                subfix, _ = part.split("=")
                                dvc_to_dsgnvar[f"{name}_{subfix}"] = part
                            elif part in self.designvar_names:
                                dvc_to_dsgnvar[name] = part

            return dvc_to_dsgnvar

        except ValueError:
            raise Exception("Error in parsing devices from the netlist.")

    def _update_circuit(self) -> tuple[dict, dict, dict]:
        """Refresh parsed design-variable and device-value mappings.

        Returns
        -------
        dsgnvar_to_val : dict[str, float]
            Mapping from design-variable names to numeric default values.
        dvc_to_dsgnvar : dict[str, str]
            Mapping from device identifiers to design-variable names or raw
            assignment expressions.
        dvc_to_val : dict[str, float or None]
            Mapping from device identifiers to resolved numeric values. Values
            are ``None`` when the corresponding device binding cannot be found
            directly in ``dsgnvar_to_val``.
        """

        dsgnvar_to_val = self._map_dsgnvar_to_val()
        dvc_to_dsgnvar = self._map_device_to_dsgnvar()

        dvc_to_val = {}

        for dvc, dsgvar in dvc_to_dsgnvar.items():
            dvc_to_val[dvc] = dsgnvar_to_val.get(dsgvar)

        return dsgnvar_to_val, dvc_to_dsgnvar, dvc_to_val