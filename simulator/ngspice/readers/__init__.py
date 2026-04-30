"""Reader registry for ngspice analysis outputs.

This module exposes concrete ngspice output readers and provides a small
factory wrapper that dispatches parsing requests to the reader associated
with a requested analysis type.
"""

from simulator.ngspice.readers.base import BaseReader, StaleOutputError
from simulator.ngspice.readers.dc import ACSimulationReader, DCReader, TransientReader
from simulator.ngspice.readers.linearity import LinearityReader
from simulator.ngspice.readers.noise import NoiseReader
from simulator.ngspice.readers.sparam import SParamReader, StabilityReader


ANALYSIS_READERS = {
    "DC_Operating_Point": DCReader,
    "S-Parameter Analysis": SParamReader,
    "AC_simulation": ACSimulationReader,
    "Stability Factor": StabilityReader,
    "Transient": TransientReader,
    "Noise Analysis": NoiseReader,
    "Linearity": LinearityReader,
}
"""dict[str, type[BaseReader]]: Mapping from ngspice analysis names to readers."""


class Reader:
    """Factory wrapper that dispatches to an analysis-specific reader.

    Parameters
    ----------
    analysis_type : str
        Name of the ngspice analysis type to parse. The value must be one of
        the keys in ``ANALYSIS_READERS``.

    Attributes
    ----------
    type : str
        Selected analysis type.
    _reader : BaseReader
        Concrete reader instance associated with ``analysis_type``.

    Raises
    ------
    ValueError
        If ``analysis_type`` is not registered in ``ANALYSIS_READERS``.
    """

    def __init__(self, analysis_type: str):
        """Initialize an analysis-specific reader wrapper.

        Parameters
        ----------
        analysis_type : str
            Name of the analysis type to parse.

        Returns
        -------
        None
            The concrete reader instance is created in place.
        """

        if analysis_type not in ANALYSIS_READERS:
            raise ValueError(
                f"Analysis_type should be one of {list(ANALYSIS_READERS)}"
            )

        self.type = analysis_type
        self._reader = ANALYSIS_READERS[analysis_type]()

    def read(self, *args, **kwargs):
        """Delegate full-result parsing to the selected reader.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to the selected reader's
            :meth:`read` method.
        **kwargs : dict[str, object]
            Keyword arguments forwarded to the selected reader's
            :meth:`read` method.

        Returns
        -------
        object
            Parsed full-result payload returned by the concrete reader.
        """

        return self._reader.read(*args, **kwargs)

    def read_metrics(self, *args, **kwargs):
        """Delegate fast metric extraction to the selected reader.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to the selected reader's
            :meth:`read_metrics` method.
        **kwargs : dict[str, object]
            Keyword arguments forwarded to the selected reader's
            :meth:`read_metrics` method.

        Returns
        -------
        object
            Fast scalar metric payload returned by the concrete reader.
        """

        return self._reader.read_metrics(*args, **kwargs)


__all__ = [
    "StaleOutputError",
    "BaseReader",
    "TransientReader",
    "DCReader",
    "SParamReader",
    "StabilityReader",
    "NoiseReader",
    "ACSimulationReader",
    "LinearityReader",
    "Reader",
]
"""list[str]: Public symbols exported by this module."""