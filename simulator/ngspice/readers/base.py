"""Base parsing helpers for ngspice analysis outputs.

This module defines shared parsing utilities for ngspice analysis readers.
It provides freshness validation, raw-text reading, complex/scalar block
parsing, frequency-window selection, retry-based reading, and fast metric
extraction hooks for concrete reader subclasses.
"""

import re
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class StaleOutputError(RuntimeError):
    """Raised when a parsed output file is missing, empty, or stale.

    This exception is used when an ngspice output file exists but should not
    be trusted for the current simulation request, for example because it is
    empty, was generated before the current simulation started, or does not
    belong to the expected run identifier.
    """

    pass


class BaseReader:
    """Base class for ngspice output readers.

    This class provides common parsing and validation helpers used by concrete
    readers for S-parameter, noise-figure, IIP3, and other ngspice analysis
    outputs. Subclasses must implement :meth:`_read_impl` and may optionally
    implement :meth:`_read_metrics_impl`.

    Attributes
    ----------
    MIN_FREQUENCY_HZ : float
        Minimum supported frequency used when validating frequency ranges.
    MAX_FREQUENCY_HZ : float
        Maximum supported frequency used when validating frequency ranges.
    _GENERIC_VARIABLE_LABELS : set[str]
        Generic ngspice variable labels ignored when extracting semantic
        variable names from raw headers.
    """

    MIN_FREQUENCY_HZ = 1e6
    MAX_FREQUENCY_HZ = 8e9
    _GENERIC_VARIABLE_LABELS = {
        "voltage",
        "current",
        "frequency",
        "time",
        "notype",
    }

    def _parse_variable_name(self, line: str) -> str:
        """Extract a variable name from a raw ngspice header line.

        Parameters
        ----------
        line : str
            Raw variable-definition line from an ngspice output block.

        Returns
        -------
        str
            Parsed variable name. Preference is given to candidates containing
            parentheses, then to non-generic labels, and finally to the first
            available candidate.

        Raises
        ------
        ValueError
            If the line does not contain enough tab-separated fields to
            identify a variable name.
        """

        parts = [part.strip() for part in line.split("\t") if part.strip()]

        if len(parts) < 2:
            raise ValueError(f"Malformed variable line: {line!r}")

        candidates = parts[1:]

        for candidate in candidates:
            if "(" in candidate or ")" in candidate:
                return candidate

        for candidate in candidates:
            if candidate.lower() not in self._GENERIC_VARIABLE_LABELS:
                return candidate

        return candidates[0]

    def _assert_fresh_output(
        self,
        result_path: Union[str, Path],
        *,
        run_id: Optional[str] = None,
        started_at_ns: Optional[int] = None,
    ) -> None:
        """Validate that an output file exists, is non-empty, and is fresh.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the output file to validate.
        run_id : str or None, optional
            Expected run identifier. If provided, the output path must contain
            this run identifier. The default is ``None``.
        started_at_ns : int or None, optional
            Simulation start timestamp in nanoseconds. If provided, the output
            file modification time must be newer than or equal to this value.
            The default is ``None``.

        Returns
        -------
        None
            The method returns normally when the output file passes all
            freshness checks.

        Raises
        ------
        FileNotFoundError
            If ``result_path`` does not exist.
        StaleOutputError
            If the output file is empty, belongs to a different run identifier,
            or is older than ``started_at_ns``.
        """

        path = Path(result_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = path.stat()

        if stat.st_size <= 0:
            raise StaleOutputError(f"Output file is empty: {path}")

        if run_id is not None and run_id not in str(path):
            raise StaleOutputError(
                f"Output path does not contain run_id {run_id}: {path}"
            )

        if started_at_ns is not None and stat.st_mtime_ns < started_at_ns:
            raise StaleOutputError(f"Stale output detected for {path}")

    def _read_text(self, result_path: Union[str, Path]) -> str:
        """Read a text output file from disk.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the text output file.

        Returns
        -------
        str
            Full text contents of the file.

        Raises
        ------
        FileNotFoundError
            If ``result_path`` does not exist.
        """

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {result_path}")

    def _parse_complex_value(self, line: str) -> complex:
        """Parse one complex value from an ngspice raw-text line.

        The parser supports values represented as ``real,imag`` as well as
        whitespace- or tab-separated real and imaginary fields.

        Parameters
        ----------
        line : str
            Raw data line containing one complex value.

        Returns
        -------
        complex
            Parsed complex value.

        Raises
        ------
        ValueError
            If the line is empty or does not match a supported complex-value
            format.
        """

        stripped = line.strip()

        if not stripped:
            raise ValueError(f"Unexpected line format: {line!r}")

        fields = stripped.split()
        value_field = fields[-1]

        if "," in value_field:
            real_str, imag_str = value_field.split(",", 1)
            return complex(float(real_str), float(imag_str))

        parts = re.split(r"(?:\\t|[\t ,])+", stripped)

        if len(parts) >= 3:
            _, re_val, im_val = parts[:3]
            return complex(float(re_val), float(im_val))

        if len(parts) == 2:
            re_val, im_val = parts
            return complex(float(re_val), float(im_val))

        raise ValueError(f"Unexpected line format: {line!r}")

    def _parse_complex_block(self, raw: str):
        """Parse a complex-valued ngspice block header and payload lines.

        Parameters
        ----------
        raw : str
            Raw ngspice output text containing ``No. Variables``,
            ``No. Points``, ``Variables``, and ``Values`` sections.

        Returns
        -------
        n_var : int
            Number of variables declared in the block.
        n_obs : int
            Number of observation points declared in the block.
        val_name : list[str]
            Parsed variable names.
        lines : list[str]
            Non-empty data lines following the ``Values`` marker.

        Raises
        ------
        ValueError
            If the number of parsed variable names does not match the declared
            variable count.
        """

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

        if n_var != len(val_name):
            raise ValueError(
                f"The number of variable names {len(val_name)} is not equal "
                f"to No.Variables {n_var} parsed from .cir file."
            )

        lines = [
            line
            for line in raw[data_idx + annot_len :].splitlines()
            if line.strip()
        ]

        return n_var, n_obs, val_name, lines

    def _parse_scalar_block(self, raw: str):
        """Parse a scalar-valued ngspice block header and payload lines.

        Parameters
        ----------
        raw : str
            Raw ngspice output text containing ``No. Variables``,
            ``No. Points``, ``Variables:``, and ``Values:`` sections.

        Returns
        -------
        n_var : int
            Number of variables declared in the block.
        n_obs : int
            Number of observation points declared in the block.
        val_names : list[str]
            Parsed variable names.
        data_lines : list[str]
            Non-empty scalar data lines following the ``Values:`` marker.

        Raises
        ------
        ValueError
            If the scalar block is malformed or if the parsed variable count
            does not match the declared variable count.
        """

        lines = [line.rstrip("\n") for line in raw.splitlines()]
        n_var = None
        n_obs = None
        variables_idx = None
        values_idx = None

        for idx, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("No. Variables"):
                n_var = int(stripped.split(":")[1].strip())
            elif stripped.startswith("No. Points"):
                n_obs = int(stripped.split(":")[1].strip())
            elif stripped == "Variables:":
                variables_idx = idx
            elif stripped == "Values:":
                values_idx = idx

        if (
            n_var is None
            or n_obs is None
            or variables_idx is None
            or values_idx is None
        ):
            raise ValueError("Malformed NGSPICE scalar block.")

        var_lines = [
            line.rstrip("\n")
            for line in lines[variables_idx + 1 : values_idx]
            if line.strip()
        ]
        val_names = []

        for line in var_lines:
            val_names.append(self._parse_variable_name(line))

        if len(val_names) != n_var:
            raise ValueError(
                f"The number of variable names {len(val_names)} is not equal "
                f"to No.Variables {n_var} parsed from .cir file."
            )

        data_lines = [
            line.strip()
            for line in lines[values_idx + 1 :]
            if line.strip()
        ]

        return n_var, n_obs, val_names, data_lines

    def _normalize_freq_range(self, freq_range=None) -> tuple[float, float]:
        """Validate and normalize a frequency-range request.

        Parameters
        ----------
        freq_range : sequence[float] or None, optional
            Requested frequency range in Hz. If ``None``, the default target
            frequency ``(2.4e9, 2.4e9)`` is returned. The default is ``None``.

        Returns
        -------
        tuple[float, float]
            Normalized frequency range ``(f1, f2)`` in Hz.

        Raises
        ------
        ValueError
            If ``freq_range`` does not contain exactly two values, lies outside
            the allowed frequency range, or violates ``f1 <= f2``.
        """

        if freq_range is None:
            return (2.4e9, 2.4e9)

        if len(freq_range) != 2:
            raise ValueError("freq_range must contain exactly two values: [f1, f2].")

        f1 = float(freq_range[0])
        f2 = float(freq_range[1])

        if f1 < self.MIN_FREQUENCY_HZ or f2 > self.MAX_FREQUENCY_HZ:
            raise ValueError(
                f"freq_range must stay within "
                f"[{self.MIN_FREQUENCY_HZ}, {self.MAX_FREQUENCY_HZ}] Hz."
            )

        if f1 > f2:
            raise ValueError("freq_range must satisfy f1 <= f2.")

        return (f1, f2)

    def _select_frequency_window(
        self,
        df: pd.DataFrame,
        *,
        freq_column: str,
        freq_range=None,
    ) -> pd.DataFrame:
        """Select rows matching a target frequency or frequency window.

        If the requested frequency range is a single point, the nearest
        frequency sample is selected. Otherwise, all rows within the inclusive
        frequency window are returned.

        Parameters
        ----------
        df : pandas.DataFrame
            Parsed simulation result containing a frequency column.
        freq_column : str
            Name of the frequency column in ``df``.
        freq_range : sequence[float] or None, optional
            Requested frequency range in Hz. If ``None``, the default target
            frequency is used. The default is ``None``.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the selected frequency row or frequency
            window. The index is reset before returning.

        Raises
        ------
        ValueError
            If ``df`` is empty or if no simulation samples fall within the
            requested frequency range.
        """

        f1, f2 = self._normalize_freq_range(freq_range)

        if df.empty:
            raise ValueError("Simulation result is empty.")

        frequencies = df[freq_column].astype(float)

        if f1 == f2:
            nearest_idx = (frequencies - f1).abs().idxmin()
            return df.loc[[nearest_idx]].reset_index(drop=True)

        window = df[
            (frequencies >= f1)
            & (frequencies <= f2)
        ].reset_index(drop=True)

        if window.empty:
            raise ValueError(
                f"No simulation samples were found in the requested "
                f"freq_range [{f1}, {f2}] Hz."
            )

        return window

    def read(
        self,
        result_path: Union[str, Path],
        *,
        run_id: Optional[str] = None,
        started_at_ns: Optional[int] = None,
        max_retries: int = 1,
        **kwargs,
    ):
        """Read and parse a full analysis result with retry logic.

        This method validates the output file freshness and delegates parsing
        to :meth:`_read_impl`. Parsing is retried when transient parsing or
        freshness errors occur.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the ngspice output file.
        run_id : str or None, optional
            Expected run identifier. If provided, the result path must contain
            this identifier. The default is ``None``.
        started_at_ns : int or None, optional
            Simulation start timestamp in nanoseconds used for stale-output
            detection. The default is ``None``.
        max_retries : int, optional
            Maximum number of read attempts. The default is ``1``.
        **kwargs : dict[str, object]
            Additional keyword arguments forwarded to :meth:`_read_impl`.

        Returns
        -------
        object
            Parsed result returned by the concrete reader implementation.

        Raises
        ------
        RuntimeError
            If all read attempts fail.
        """

        for _ in range(max_retries):
            try:
                self._assert_fresh_output(
                    result_path,
                    run_id=run_id,
                    started_at_ns=started_at_ns,
                )
                return self._read_impl(result_path, **kwargs)
            except (
                ValueError,
                IndexError,
                pd.errors.EmptyDataError,
                FileNotFoundError,
                StaleOutputError,
            ) as exc:
                print(exc)
                time.sleep(0.05)

        raise RuntimeError(
            f"READER HAS FAILED TO READ THE RESULTS AFTER {max_retries} ATTEMPTS."
        )

    def read_metrics(
        self,
        result_path: Union[str, Path],
        *,
        run_id: Optional[str] = None,
        started_at_ns: Optional[int] = None,
        **kwargs,
    ):
        """Extract fast scalar metrics from an analysis result.

        This method validates output freshness and delegates metric extraction
        to :meth:`_read_metrics_impl`.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the ngspice output file.
        run_id : str or None, optional
            Expected run identifier. If provided, the result path must contain
            this identifier. The default is ``None``.
        started_at_ns : int or None, optional
            Simulation start timestamp in nanoseconds used for stale-output
            detection. The default is ``None``.
        **kwargs : dict[str, object]
            Additional keyword arguments forwarded to
            :meth:`_read_metrics_impl`.

        Returns
        -------
        object
            Fast metric payload returned by the concrete reader
            implementation.
        """

        self._assert_fresh_output(
            result_path,
            run_id=run_id,
            started_at_ns=started_at_ns,
        )
        return self._read_metrics_impl(result_path, **kwargs)

    def _read_impl(self, result_path: Union[str, Path], **kwargs):
        """Parse the full result payload.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the ngspice output file.
        **kwargs : dict[str, object]
            Reader-specific parsing options.

        Returns
        -------
        object
            Parsed reader-specific result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class. Concrete subclasses must
            implement this method.
        """

        raise NotImplementedError()

    def _read_metrics_impl(self, result_path: Union[str, Path], **kwargs):
        """Extract fast scalar metrics from a result payload.

        Parameters
        ----------
        result_path : str or pathlib.Path
            Path to the ngspice output file.
        **kwargs : dict[str, object]
            Reader-specific metric extraction options.

        Returns
        -------
        object
            Reader-specific metric payload.

        Raises
        ------
        ValueError
            Always raised by the base implementation. Concrete subclasses may
            override this method when fast metric extraction is supported.
        """

        raise ValueError(
            f"Fast metrics path is not supported for reader type: "
            f"{self.__class__.__name__}"
        )


__all__ = ["BaseReader", "StaleOutputError"]
"""list[str]: Public symbols exported by this module."""