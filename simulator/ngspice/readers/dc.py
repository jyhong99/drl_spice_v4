"""Readers for DC, transient, and generic AC ngspice outputs.

This module defines concrete reader classes for parsing ngspice output files.
It includes readers for transient-analysis outputs, DC operating-point
outputs, and generic AC simulation outputs containing magnitude and phase
values.
"""

from io import StringIO
from pathlib import Path
from typing import Union

import pandas as pd

from simulator.ngspice.readers.base import BaseReader


class TransientReader(BaseReader):
    """Reader for transient-analysis tabular outputs.

    This reader parses ngspice transient-analysis output files into pandas
    dataframes. It expects an ngspice raw-text structure containing
    ``No. Variables``, ``No. Points``, ``Variables``, and ``Values`` sections.
    """

    def _read_impl(self, result_path: Union[str, Path]) -> pd.DataFrame:
        """Parse a transient-analysis output into a dataframe.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the transient-analysis output file.

        Returns
        -------
        pandas.DataFrame
            Parsed transient-analysis result. Columns correspond to variable
            names declared in the ngspice output header, and rows correspond to
            observation points.

        Raises
        ------
        ValueError
            If the file is empty, malformed, or the parsed number of variables
            or observations does not match the header metadata.
        IndexError
            If the expected header indices cannot be found.
        """

        try:
            raw = pd.read_csv(result_path, header=None)
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {result_path}")

        cond = (
            raw.squeeze().str.contains("No. Variables:")
            | raw.squeeze().str.contains("Variables")
            | raw.squeeze().str.contains("Values")
            | raw.squeeze().str.contains("No. Points")
        )
        information = raw[cond].squeeze()

        if information.empty:
            raise ValueError(
                f"Cannot parse the file content in {result_path}. "
                f"Please check the file format."
            )

        try:
            n_var, n_obs = (
                information.str.split(" ")
                .str[2]
                .iloc[0:2]
                .astype("int")
                .values
            )
            start_idx = information.index[2] + 1
            end_idx = information.index[3]
            val_name = list(
                raw[start_idx:end_idx].squeeze().str.split("\t").str[2].values
            )
        except IndexError:
            raise IndexError("Out of index")

        if n_var != len(val_name):
            raise ValueError(
                f"The number of variable names {len(val_name)} is not equal "
                f"to No.Variables {n_var} parsed from .cir file."
            )

        data = (
            raw.iloc[end_idx + 1 :, :]
            .reset_index(drop=True)
            .squeeze()
            .str.split("\t")
        )
        result = []

        for idx in range(0, len(data), n_var):
            obs = data[idx : idx + n_var].str[1].reset_index(drop=True)
            result.append(obs.values)

        result = pd.DataFrame(result, columns=val_name)

        if n_obs != len(result):
            raise ValueError(
                f"The number of observations {len(result)} is not equal to "
                f"No.Points {n_obs} parsed from .cir file"
            )

        return result


class DCReader(TransientReader):
    """Reader for DC operating-point outputs.

    This reader extends :class:`TransientReader` and additionally supports
    fast scalar metric extraction from DC operating-point output blocks.
    """

    def _read_metrics_impl(self, result_path: Union[str, Path]) -> dict:
        """Extract scalar DC operating-point metrics.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the DC operating-point output file.

        Returns
        -------
        dict[str, float]
            Mapping from parsed variable names to scalar DC values.

        Raises
        ------
        ValueError
            If no DC observations are found, if the data block is incomplete,
            or if a scalar value line cannot be parsed.
        """

        raw = self._read_text(result_path)
        n_var, n_obs, val_names, data_lines = self._parse_scalar_block(raw)

        if n_obs < 1:
            raise ValueError(f"No DC observations found in {result_path}")

        if len(data_lines) < n_var:
            raise ValueError(f"Incomplete DC data block in {result_path}")

        values = {}

        for idx, name in enumerate(val_names):
            parts = [part for part in data_lines[idx].split("\t") if part]

            if not parts:
                raise ValueError(f"Unexpected DC value line: {data_lines[idx]!r}")

            try:
                values[name] = float(parts[-1])
            except ValueError as exc:
                raise ValueError(
                    f"Unexpected DC value line: {data_lines[idx]!r}"
                ) from exc

        return values


class ACSimulationReader(BaseReader):
    """Reader for generic AC simulation outputs with magnitude and phase.

    This reader parses AC simulation outputs where each variable has magnitude
    and phase values. The returned dataframe contains magnitude columns and
    phase columns with the suffix ``"_phase"``.
    """

    def _read_impl(self, result_path: Union[str, Path]) -> pd.DataFrame:
        """Parse an AC simulation output into a dataframe.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the AC simulation output file.

        Returns
        -------
        pandas.DataFrame
            Parsed AC simulation result. Magnitude columns keep their original
            variable names, while phase columns are suffixed with
            ``"_phase"``. The ``frequency`` phase column is removed.

        Raises
        ------
        ValueError
            If the parsed number of variables or observations does not match
            the metadata declared in the output file.
        IndexError
            If expected header sections cannot be parsed.
        """

        raw = self._read_text(result_path)

        try:
            n_var_idx = raw.find("No. Variables")
            n_obs_idx = raw.find("No. Points")
            var_idx = raw.find("Variables", n_var_idx + len("No. Variables"))
            data_idx = raw.find("Values:")

            n_var = int(raw[n_var_idx:n_obs_idx].split(":")[1])
            n_obs = int(raw[n_obs_idx:var_idx].split(":")[1])
            variable_lines = [
                line.rstrip("\n")
                for line in raw[var_idx:data_idx].splitlines()[1:]
                if line.strip()
            ]
            val_name = [self._parse_variable_name(line) for line in variable_lines]
        except IndexError:
            raise IndexError("Out of index")

        if n_var != len(val_name):
            raise ValueError(
                f"The number of variable names {len(val_name)} is not equal "
                f"to No.Variables {n_var} parsed from .cir file."
            )

        data = pd.read_csv(StringIO(raw[data_idx:])).reset_index()
        data.columns = ["magnitude", "phase"]

        magnitude = (
            data["magnitude"]
            .str.split("\t")
            .str[1]
            .astype("float")
            .reset_index(drop=True)
        )
        phase = data["phase"].reset_index(drop=True)
        separated_data = [magnitude, phase]

        for idx, series in enumerate(separated_data):
            variable_data = {name: [] for name in val_name}

            for offset in range(0, len(magnitude), n_var):
                chunk = series[offset : offset + n_var]

                for name_idx, value in enumerate(chunk):
                    variable_data[val_name[name_idx]].append(value)

            separated_data[idx] = pd.DataFrame(variable_data)

        magnitude, phase = separated_data
        phase = phase.drop(columns=["frequency"])
        phase.columns = [name + "_phase" for name in list(phase.columns)]

        result = pd.concat([magnitude, phase], axis=1)

        if n_obs != len(result):
            raise ValueError(
                f"The number of observations {len(result)} is not equal to "
                f"No.Points {n_obs} parsed from .cir file"
            )

        return result


__all__ = ["TransientReader", "DCReader", "ACSimulationReader"]
"""list[str]: Public symbols exported by this module."""