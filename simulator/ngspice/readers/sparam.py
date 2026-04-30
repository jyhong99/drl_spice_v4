"""Readers for S-parameter and stability-analysis outputs.

This module defines concrete ngspice readers for S-parameter analysis outputs
and stability metrics derived from S-parameters. The readers parse raw ngspice
complex-valued output files and expose dataframe or scalar-metric interfaces.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from simulator.ngspice.readers.base import BaseReader


class SParamReader(BaseReader):
    """Reader for S-parameter analysis outputs.

    This reader parses complex-valued S-parameter output files and converts
    selected S-parameter magnitudes into dB.
    """

    def _load_sparam_dataframe(self, result_path: Union[str, Path]) -> pd.DataFrame:
        """Load an S-parameter output file into a dataframe.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the S-parameter analysis output file.

        Returns
        -------
        pandas.DataFrame
            Parsed S-parameter dataframe. The dataframe contains a
            ``"frequency"`` column and selected S-parameter magnitude columns
            in dB.

        Raises
        ------
        ValueError
            If the output data block contains fewer entries than expected from
            the declared number of variables and observation points.
        """

        raw = self._read_text(result_path)
        n_var, n_obs, val_name, lines = self._parse_complex_block(raw)

        expected_points = n_var * n_obs
        if len(lines) < expected_points:
            raise ValueError(f"Unexpected S-parameter data block in {result_path}")

        records = []

        for start in range(0, expected_points, n_var):
            block = lines[start : start + n_var]
            row = {"frequency": self._parse_complex_value(block[0]).real}

            for idx, name in enumerate(val_name[1:5], start=1):
                row[name] = 20 * np.log10(abs(self._parse_complex_value(block[idx])))

            records.append(row)

        return pd.DataFrame.from_records(records)

    def _read_impl(self, result_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Parse the full S-parameter dataframe.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the S-parameter analysis output file.
        **kwargs : dict[str, object]
            Additional keyword arguments accepted for interface compatibility.
            They are not used by this implementation.

        Returns
        -------
        pandas.DataFrame
            Parsed S-parameter dataframe.
        """

        return self._load_sparam_dataframe(result_path)

    def _read_metrics_impl(
        self,
        result_path: Union[str, Path],
        *,
        freq_range=None,
        **kwargs,
    ) -> dict:
        """Extract key scalar S-parameter metrics for a frequency window.

        The selected frequency window is reduced to worst-case scalar metrics:
        maximum input reflection, minimum forward gain, and maximum output
        reflection.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the S-parameter analysis output file.
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
            Dictionary containing scalar S-parameter metrics:

            ``"v(s_1_1)"``
                Maximum selected S11 magnitude in dB.
            ``"v(s_2_1)"``
                Minimum selected S21 magnitude in dB.
            ``"v(s_2_2)"``
                Maximum selected S22 magnitude in dB.
        """

        df = self._load_sparam_dataframe(result_path)
        window = self._select_frequency_window(
            df,
            freq_column="frequency",
            freq_range=freq_range,
        )

        return {
            "v(s_1_1)": float(window["v(s_1_1)"].max()),
            "v(s_2_1)": float(window["v(s_2_1)"].min()),
            "v(s_2_2)": float(window["v(s_2_2)"].max()),
        }


class StabilityReader(BaseReader):
    """Reader for stability-factor outputs derived from S-parameters.

    This reader parses complex S-parameter data and computes scalar stability
    metrics, including Rollett stability factor ``K``, ``mu``, ``mup``, the
    minimum of ``mu`` and ``mup``, and an estimated 3 dB bandwidth.
    """

    def _read_impl(self, result_path: Union[str, Path], **kwargs):
        """Parse and return stability metrics.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the S-parameter-derived stability output file.
        **kwargs : dict[str, object]
            Additional keyword arguments forwarded to
            :meth:`_read_metrics_impl`.

        Returns
        -------
        dict[str, float]
            Stability metric dictionary returned by
            :meth:`_read_metrics_impl`.
        """

        return self._read_metrics_impl(result_path, **kwargs)

    def _read_metrics_impl(self, result_path: Union[str, Path], **kwargs) -> dict:
        """Extract scalar stability metrics and 3 dB bandwidth.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the complex S-parameter output file.
        **kwargs : dict[str, object]
            Additional keyword arguments accepted for interface compatibility.
            They are not used by this implementation.

        Returns
        -------
        dict[str, float]
            Dictionary containing stability and bandwidth metrics.

            ``"K"``
                Minimum Rollett stability factor over the parsed frequency
                range.
            ``"mu"``
                Minimum input stability measure.
            ``"mup"``
                Minimum output stability measure.
            ``"mu_mup"``
                Minimum of ``mu`` and ``mup``.
            ``"bandwidth"``
                Estimated 3 dB bandwidth in Hz.

        Raises
        ------
        IndexError
            If expected header sections cannot be parsed.
        ValueError
            If the number of parsed variable names does not match the declared
            variable count.
        """

        raw = self._read_text(result_path)

        try:
            n_var_idx = raw.find("No. Variables")
            n_obs_idx = raw.find("No. Points")
            var_idx = raw.find("Variables", n_var_idx + len("No. Variables"))
            data_idx = raw.find("Values:\n")
            annot_len = len("Values:\n")

            n_var = int(raw[n_var_idx:n_obs_idx].split(":")[1])
            n_obs = int(raw[n_obs_idx:var_idx].split(":")[1])
            variable_lines = [
                line.rstrip("\n")
                for line in raw[var_idx:data_idx].splitlines()[1:]
                if line.strip()
            ]
            val_name = [self._parse_variable_name(line) for line in variable_lines]
        except IndexError:
            raise IndexError("Header parsing failed — malformed NGSPICE raw file.")

        if n_var != len(val_name):
            raise ValueError(
                f"The number of variable names {len(val_name)} != "
                f"No. Variables {n_var}"
            )

        lines = [
            line
            for line in raw[data_idx + annot_len :].splitlines()
            if line.strip() != ""
        ]

        freq, s11, s12, s21, s22 = [], [], [], [], []

        for idx in range(0, len(lines), n_var):
            freq.append(self._parse_complex_value(lines[idx]).real)
            s11.append(self._parse_complex_value(lines[idx + 1]))
            s12.append(self._parse_complex_value(lines[idx + 2]))
            s21.append(self._parse_complex_value(lines[idx + 3]))
            s22.append(self._parse_complex_value(lines[idx + 4]))

        freq, s11, s12, s21, s22 = map(
            np.array,
            (freq, s11, s12, s21, s22),
        )

        mask = freq >= 1e6
        freq, s11, s12, s21, s22 = [
            arr[mask] for arr in (freq, s11, s12, s21, s22)
        ]

        eps = 1e-12

        delta = s11 * s22 - s12 * s21

        num_k = 1 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(delta) ** 2
        den_k = 2 * np.abs(s12 * s21)
        k = num_k / np.maximum(den_k, eps)

        num_mu = 1 - np.abs(s11) ** 2
        den_mu = np.abs(s22 - delta * np.conj(s11)) + np.abs(s12 * s21)

        num_mup = 1 - np.abs(s22) ** 2
        den_mup = np.abs(s11 - delta * np.conj(s22)) + np.abs(s12 * s21)

        mu = num_mu / np.maximum(den_mu, eps)
        mup = num_mup / np.maximum(den_mup, eps)

        mag_s21_db = 20 * np.log10(np.abs(s21))
        peak_gain = np.max(mag_s21_db)
        threshold = peak_gain - 3.0
        above = mag_s21_db >= threshold
        indices = np.where(np.diff(above.astype(int)) != 0)[0]

        if len(indices) >= 2:
            f_low = freq[indices[0]]
            f_high = freq[indices[-1]]
            bandwidth = f_high - f_low
        else:
            bandwidth = 0.0

        return {
            "K": np.min(k),
            "mu": np.min(mu),
            "mup": np.min(mup),
            "mu_mup": min(np.min(mu), np.min(mup)),
            "bandwidth": bandwidth,
        }


__all__ = ["SParamReader", "StabilityReader"]
"""list[str]: Public symbols exported by this module."""