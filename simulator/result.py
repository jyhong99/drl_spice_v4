"""Structured return type for simulator evaluations.

This module defines a lightweight dataclass used to represent the outcome of
one simulator evaluation. It stores simulation status, parsed performance
metrics, stability information, optional error messages, and arbitrary
metadata such as profiling details.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SimulationResult:
    """Outcome of a simulator evaluation.

    Parameters
    ----------
    status : str
        Simulation status string. The value ``"ok"`` indicates a successful
        evaluation.
    performances : numpy.ndarray or None, optional
        Parsed performance vector returned by the simulator. The default is
        ``None``.
    stability_factor : float or None, optional
        Parsed stability factor or stability-related scalar. The default is
        ``None``.
    error_message : str or None, optional
        Error message associated with a failed simulation. The default is
        ``None``.
    metadata : dict[str, Any], optional
        Additional simulator metadata, such as profiling results, cache state,
        or backend-specific diagnostics. The default is an empty dictionary.

    Attributes
    ----------
    status : str
        Simulation status string.
    performances : numpy.ndarray or None
        Parsed performance vector.
    stability_factor : float or None
        Parsed stability factor.
    error_message : str or None
        Optional failure message.
    metadata : dict[str, Any]
        Additional simulator metadata.
    """

    status: str
    performances: Optional[np.ndarray] = None
    stability_factor: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return whether the simulation completed successfully.

        Returns
        -------
        bool
            ``True`` if ``self.status`` is ``"ok"``; otherwise ``False``.
        """

        return self.status == "ok"