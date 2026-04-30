"""Abstract simulator backend interface.

This module defines the common simulator backend interface used by
simulator-backed environments. Concrete simulator services must implement
design evaluation, workspace configuration, and resource cleanup.
"""

from abc import ABC, abstractmethod


class SimulatorBackend(ABC):
    """Abstract interface for simulator backends used by environments.

    A simulator backend evaluates decoded design-variable configurations and
    returns simulator-specific result objects. Implementations may wrap SPICE,
    surrogate models, external simulators, or other evaluation engines.
    """

    @abstractmethod
    def evaluate(self, design_variables_config):
        """Evaluate one decoded design configuration.

        Parameters
        ----------
        design_variables_config : dict[str, object]
            Decoded simulator parameter mapping for one candidate design.

        Returns
        -------
        object
            Backend-specific simulation result. Concrete implementations
            should return a structured result object containing at least
            simulation status and evaluated performance values.

        Raises
        ------
        NotImplementedError
            Raised by abstract implementations.
        """

        raise NotImplementedError()

    @abstractmethod
    def configure_workspace(self, **kwargs):
        """Bind the backend to a concrete output workspace.

        Parameters
        ----------
        **kwargs : dict[str, object]
            Backend-specific workspace configuration arguments, such as
            project name, run identifier, root directory, worker name, scope,
            or cleanup options.

        Returns
        -------
        object
            Backend-specific return value. Implementations commonly return
            ``None`` after updating workspace state in place.

        Raises
        ------
        NotImplementedError
            Raised by abstract implementations.
        """

        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """Release backend resources.

        Returns
        -------
        object
            Backend-specific return value. Implementations commonly return
            ``None``.

        Raises
        ------
        NotImplementedError
            Raised by abstract implementations.
        """

        raise NotImplementedError()