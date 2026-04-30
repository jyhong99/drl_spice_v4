"""Reader for ngspice noise-analysis outputs.

This module defines a concrete reader for parsing ngspice noise-analysis
output files. It converts raw noise-analysis data into a dataframe containing
frequency and noise-figure columns, and supports fast extraction of peak
noise figure over a requested frequency range.
"""

from io import StringIO
from pathlib import Path
from typing import Union

import pandas as pd

from simulator.ngspice.readers.base import BaseReader


class NoiseReader(BaseReader):
    """Reader for noise-analysis outputs.

    This reader parses ngspice noise-analysis output files and returns
    frequency-dependent noise-figure data.
    """

    def _read_impl(self, result_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Parse a noise-analysis output into a dataframe.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the noise-analysis output file.
        **kwargs : dict[str, object]
            Additional keyword arguments accepted for interface compatibility.
            They are not used by this implementation.

        Returns
        -------
        pandas.DataFrame
            Parsed noise-analysis result with columns ``"frequency"`` and
            ``"NoiseFigure"``.

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

        data = pd.read_csv(StringIO(raw[data_idx:]))
        df = (
            data["Values:"]
            .str.split("\t")
            .str[1]
            .astype("float")
            .reset_index(drop=True)
        )

        cond_frequency = df.index % 2 == 0
        cond_noise_figure = df.index % 2 == 1

        result = pd.concat(
            (
                df[cond_frequency].reset_index(drop=True),
                df[cond_noise_figure].reset_index(drop=True),
            ),
            axis=1,
        )
        result.columns = ["frequency", "NoiseFigure"]

        if n_obs != len(result):
            raise ValueError(
                f"The number of observations {len(result)} is not equal to "
                f"No.Points {n_obs} parsed from .cir file"
            )

        return result

    def _read_metrics_impl(
        self,
        result_path: Union[str, Path],
        *,
        freq_range=None,
        **kwargs,
    ) -> dict:
        """Extract the peak noise figure over a frequency window.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the noise-analysis output file.
        freq_range : sequence[float] or None, optional
            Frequency range in Hz used to select samples. If ``None``, the
            default frequency range defined by :class:`BaseReader` is used.
            The default is ``None``.
        **kwargs : dict[str, object]
            Additional keyword arguments accepted for interface compatibility.
            They are not used by this implementation.

        Returns
        -------
        dict[str, float]
            Dictionary containing the maximum ``"NoiseFigure"`` value within
            the selected frequency window.
        """

        df = self._read_impl(result_path)
        window = self._select_frequency_window(
            df,
            freq_column="frequency",
            freq_range=freq_range,
        )

        return {"NoiseFigure": float(window["NoiseFigure"].max())}


__all__ = ["NoiseReader"]
"""list[str]: Public symbols exported by this module."""