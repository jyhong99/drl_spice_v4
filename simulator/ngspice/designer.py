"""Circuit design-variable rewriting helpers for ngspice netlists.

This module defines a designer utility that updates annotated design-variable
sections in ngspice netlists and optionally rewrites simulator output paths.
"""

import os
import re
from typing import Dict, Optional

from simulator.ngspice.circuit import Circuit


class Designer(object):
    """Rewrite design variables and output paths in annotated circuit netlists.

    This class uses annotation markers from a source :class:`Circuit` object
    to locate the design-variable block in a netlist. It then rewrites that
    block using a provided design-variable configuration and saves the updated
    netlist to a target circuit path.

    Parameters
    ----------
    circuit : Circuit
        Parsed circuit object used as the template source. Its netlist and
        annotation markers define where the design-variable block begins and
        ends.
    num_design_variables : int
        Expected number of design variables in the target circuit.

    Attributes
    ----------
    circuit : Circuit
        Template circuit object.
    num_design_variables : int
        Expected number of design variables.
    design_variables_start_annot : str
        Marker indicating the start of the design-variable annotation block.
    design_variables_end_annot : str
        Marker indicating the end of the design-variable annotation block.
    start_idx : int
        Character index of the start annotation in the template netlist.
    end_idx : int
        Character index of the end annotation in the template netlist.
    _prefix : str
        Netlist text before the design-variable annotation block.
    _suffix : str
        Netlist text after the design-variable annotation block.
    """

    def __init__(self, circuit: Circuit, num_design_variables: int):
        """Initialize the designer from an annotated circuit template.

        Parameters
        ----------
        circuit : Circuit
            Parsed circuit object containing the source netlist and annotation
            metadata.
        num_design_variables : int
            Expected number of design variables.

        Returns
        -------
        None
            The designer stores annotation locations and reusable netlist
            prefix/suffix text in place.

        Raises
        ------
        ValueError
            If the design-variable annotation markers cannot be found in the
            circuit netlist.
        """

        self.circuit = circuit
        self.num_design_variables = num_design_variables
        self.design_variables_start_annot = self.circuit.designvar_annotation_start
        self.design_variables_end_annot = self.circuit.devignvar_annotation_end

        try:
            self.start_idx = self.circuit.netlist.index(
                self.design_variables_start_annot
            )
            self.end_idx = self.circuit.netlist.index(
                self.design_variables_end_annot
            )
        except ValueError:
            raise ValueError("Design variable annotations not found in the netlist.")

        self._prefix = self.circuit.netlist[: self.start_idx].rstrip("\n")
        suffix_start = self.end_idx + len(self.design_variables_end_annot)
        self._suffix = self.circuit.netlist[suffix_start:].lstrip("\n")

    def design_circuit(
        self,
        target_circuit: Circuit,
        design_variables_config: dict,
        output_path_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Rewrite the target circuit netlist with new design variables.

        This method validates the target circuit design-variable count, updates
        the annotated ``.param`` block, optionally rewrites output paths in
        ``write`` commands, and writes the resulting netlist to disk.

        Parameters
        ----------
        target_circuit : Circuit
            Circuit object whose ``netlist_path`` will be overwritten with the
            redesigned netlist.
        design_variables_config : dict[str, object]
            Mapping from design-variable names to values. Each item is written
            as a ``.param`` line.
        output_path_map : dict[str, str] or None, optional
            Mapping from original output filenames to rewritten output paths.
            If ``None`` or empty, output paths are left unchanged. The default
            is ``None``.

        Returns
        -------
        None
            The target circuit netlist file is overwritten in place.

        Raises
        ------
        ValueError
            If the target circuit does not contain the expected number of
            design variables.
        IOError
            If the rewritten netlist cannot be written to disk.
        """

        self._check_params(target_circuit)
        self._update_design_variables(
            circuit=target_circuit,
            design_variables_config=design_variables_config,
            output_path_map=output_path_map,
        )

    def _update_design_variables(
        self,
        circuit: Circuit,
        design_variables_config: dict,
        output_path_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Rewrite the design-variable block and save the netlist.

        Parameters
        ----------
        circuit : Circuit
            Target circuit whose netlist path should receive the rewritten
            text.
        design_variables_config : dict[str, object]
            Mapping from design-variable names to values written into the
            annotated design-variable section.
        output_path_map : dict[str, str] or None, optional
            Optional mapping used to rewrite ``write`` command output paths.
            The default is ``None``.

        Returns
        -------
        None
            The rewritten netlist is written to ``circuit.netlist_path``.

        Raises
        ------
        IOError
            If writing the updated netlist fails.
        """

        netlist_lines = [self._prefix, self.design_variables_start_annot]

        for dsgn_name, dsgn_value in design_variables_config.items():
            netlist_lines.append(f".param {dsgn_name.strip()} = {dsgn_value}")

        netlist_lines.append(self.design_variables_end_annot)

        netlist_txt = "\n".join(netlist_lines)

        if self._suffix:
            netlist_txt = f"{netlist_txt}\n{self._suffix}"

        netlist_txt = self._rewrite_output_paths(netlist_txt, output_path_map)

        save_path = circuit.netlist_path

        try:
            with open(save_path, "w") as f:
                f.write(netlist_txt)
        except IOError:
            raise IOError(f"Error writing to the netlist file at {save_path}.")

    def _rewrite_output_paths(
        self,
        netlist_txt: str,
        output_path_map: Optional[Dict[str, str]],
    ) -> str:
        """Rewrite ``write`` command output paths in a netlist.

        Each line beginning with ``write`` is inspected. If the basename of
        the current output path exists in ``output_path_map``, it is replaced
        with the mapped path. If no direct match is found, a normalized
        basename with a trailing numeric suffix removed is also checked.

        Parameters
        ----------
        netlist_txt : str
            Full netlist text to inspect and rewrite.
        output_path_map : dict[str, str] or None
            Mapping from output filename basenames to replacement paths.

        Returns
        -------
        str
            Netlist text with matching ``write`` output paths replaced.
        """

        if not output_path_map:
            return netlist_txt

        rewritten_lines = []

        for line in netlist_txt.splitlines():
            stripped = line.lstrip()

            if not stripped.lower().startswith("write "):
                rewritten_lines.append(line)
                continue

            parts = stripped.split(maxsplit=2)

            if len(parts) < 2:
                rewritten_lines.append(line)
                continue

            current_path = parts[1]
            current_name = os.path.basename(current_path)
            next_path = output_path_map.get(current_name)

            if next_path is None:
                normalized_name = re.sub(
                    r"_\d+(?=\.[^.]+$)",
                    "",
                    current_name,
                )
                next_path = output_path_map.get(normalized_name)

            if next_path is None:
                rewritten_lines.append(line)
                continue

            remainder = parts[2] if len(parts) > 2 else ""
            leading = line[: len(line) - len(stripped)]
            updated = f"{leading}write {next_path}"

            if remainder:
                updated += f" {remainder}"

            rewritten_lines.append(updated)

        rewritten = "\n".join(rewritten_lines)

        if netlist_txt.endswith("\n"):
            rewritten += "\n"

        return rewritten

    def _check_params(self, circuit: Circuit) -> None:
        """Validate the target circuit design-variable count.

        Parameters
        ----------
        circuit : Circuit
            Target circuit whose parsed design variables are checked.

        Returns
        -------
        None
            The method returns normally when the count matches.

        Raises
        ------
        ValueError
            If the number of parsed design variables does not match
            ``self.num_design_variables``.
        """

        if self.num_design_variables != len(circuit.dsgnvar_to_val):
            raise ValueError(
                "The number of design variables does not match the expected count."
            )