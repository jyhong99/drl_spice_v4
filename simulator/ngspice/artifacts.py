"""Output artifact management helpers for simulator runs.

This module defines a small utility class for creating unique output paths
for simulator-generated artifacts. It helps isolate outputs from different
simulation phases, processes, and run attempts.
"""

import os
import time
from pathlib import Path
from typing import Dict, Iterable


class OutputArtifactManager:
    """Manage unique output artifact paths under a run directory.

    This class creates per-phase output directories and generates unique
    filenames using a prefix, process ID, sequence number, and nanosecond
    timestamp.

    Parameters
    ----------
    run_root : str or pathlib.Path
        Root directory under which output artifacts are created.

    Attributes
    ----------
    run_root : pathlib.Path
        Root directory used to store generated output artifacts.
    _run_sequence : int
        Monotonically increasing sequence counter used to make run IDs unique.
    """

    def __init__(self, run_root):
        """Initialize the output artifact manager.

        Parameters
        ----------
        run_root : str or pathlib.Path
            Root directory under which output artifacts are created.

        Returns
        -------
        None
            The root directory is created if it does not already exist.
        """

        self._run_sequence = 0
        self.set_run_root(run_root)

    def set_run_root(self, run_root) -> None:
        """Set and create the root directory for output artifacts.

        Parameters
        ----------
        run_root : str or pathlib.Path
            Root directory under which generated output files are stored.

        Returns
        -------
        None
            ``self.run_root`` is updated and the directory is created in place.
        """

        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)

    def _next_run_id(self, prefix: str) -> str:
        """Generate a unique run identifier.

        Parameters
        ----------
        prefix : str
            Phase or artifact prefix included at the beginning of the run ID.

        Returns
        -------
        str
            Unique run identifier containing the prefix, process ID, sequence
            number, and current nanosecond timestamp.
        """

        self._run_sequence += 1
        return f"{prefix}_{os.getpid()}_{self._run_sequence}_{time.time_ns()}"

    def make_output_path_map(
        self,
        prefix: str,
        filenames: Iterable[str],
    ) -> Dict[str, object]:
        """Create unique output paths for a group of artifact filenames.

        A phase-specific directory named by ``prefix`` is created under
        ``self.run_root``. Each requested filename is prefixed with a unique
        run ID to avoid collisions across repeated simulator calls.

        Parameters
        ----------
        prefix : str
            Phase or artifact-group name used for the subdirectory and run ID.
        filenames : iterable[str]
            Iterable of base filenames for which unique output paths should be
            generated.

        Returns
        -------
        dict[str, object]
            Dictionary containing:

            ``"run_id"`` : str
                Unique run identifier shared by this output group.
            ``"paths"`` : dict[str, str]
                Mapping from original filename to string path.
            ``"actual_paths"`` : dict[str, pathlib.Path]
                Mapping from original filename to ``Path`` object.
        """

        run_id = self._next_run_id(prefix)
        phase_dir = self.run_root / prefix
        phase_dir.mkdir(parents=True, exist_ok=True)

        actual_paths = {
            filename: phase_dir / f"{run_id}_{filename}"
            for filename in filenames
        }

        return {
            "run_id": run_id,
            "paths": {
                name: str(path)
                for name, path in actual_paths.items()
            },
            "actual_paths": actual_paths,
        }

    def cleanup_outputs(self, output_paths) -> None:
        """Remove generated output files if they exist.

        Parameters
        ----------
        output_paths : iterable[str or pathlib.Path]
            Output file paths to remove.

        Returns
        -------
        None
            Existing files are deleted. Missing paths and directory paths are
            ignored.
        """

        for path in output_paths:
            try:
                Path(path).unlink()
            except (FileNotFoundError, IsADirectoryError):
                continue