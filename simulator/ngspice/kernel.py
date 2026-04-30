"""Process-level ngspice execution backends.

This module defines execution backends for launching ngspice simulations.
It provides a batch subprocess backend, a persistent-session backend with
batch fallback, and a factory helper for backend construction.
"""

import os
import shlex
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from simulator.ngspice.circuit import Circuit


@dataclass
class SpiceRunResult:
    """Structured result of one ngspice process invocation.

    Parameters
    ----------
    ok : bool
        Whether the ngspice invocation completed successfully.
    returncode : int or None
        Process return code. ``None`` is used when no return code is available,
        such as timeout or launch failure.
    timed_out : bool
        Whether the invocation timed out.
    command : list[str]
        Command used to launch or control ngspice.
    netlist_path : pathlib.Path
        Path to the circuit netlist executed by ngspice.
    run_id : str
        Unique identifier associated with this simulation run.
    started_at_ns : int
        Wall-clock start timestamp in nanoseconds.
    finished_at_ns : int
        Wall-clock finish timestamp in nanoseconds.
    stdout_tail : str
        Captured tail of standard output, if available.
    stderr_tail : str
        Captured tail of standard error, if available.
    output_paths : dict[str, pathlib.Path]
        Mapping from logical output names to generated output paths.
    backend : str, optional
        Backend name that produced the result. The default is ``"batch"``.
    error : str or None, optional
        Error message associated with a failed run. The default is ``None``.
    """

    ok: bool
    returncode: Optional[int]
    timed_out: bool
    command: List[str]
    netlist_path: Path
    run_id: str
    started_at_ns: int
    finished_at_ns: int
    stdout_tail: str
    stderr_tail: str
    output_paths: Dict[str, Path]
    backend: str = "batch"
    error: Optional[str] = None


class BaseSpiceBackend(ABC):
    """Abstract ngspice execution backend.

    Concrete backends must implement :meth:`run` and may override
    :meth:`close` if they hold persistent resources.

    Attributes
    ----------
    backend_name : str
        Human-readable backend identifier.
    """

    backend_name = "base"

    @abstractmethod
    def run(
        self,
        circuit: Circuit,
        *,
        run_id: str,
        output_paths: Dict[str, Union[str, Path]],
        timeout_s: float = 60.0,
    ) -> SpiceRunResult:
        """Run one circuit netlist and collect output metadata.

        Parameters
        ----------
        circuit : Circuit
            Circuit object whose netlist should be executed.
        run_id : str
            Unique identifier associated with the simulation run.
        output_paths : dict[str, str or pathlib.Path]
            Mapping from logical output names to expected output paths.
        timeout_s : float, optional
            Maximum allowed runtime in seconds. The default is ``60.0``.

        Returns
        -------
        SpiceRunResult
            Structured ngspice execution result.

        Raises
        ------
        NotImplementedError
            Raised by abstract or incomplete backend implementations.
        """

        raise NotImplementedError()

    def close(self) -> None:
        """Release backend resources.

        Returns
        -------
        None
            Base implementation has no persistent resources to release.
        """

        return None


class BatchSpiceBackend(BaseSpiceBackend):
    """Subprocess backend that launches one fresh ngspice process per run.

    This backend executes ``ngspice -b <netlist>`` for each simulation request
    and captures stderr for diagnostics.
    """

    backend_name = "batch"

    def _kill_process_group(self, proc: Optional[subprocess.Popen]) -> None:
        """Terminate a spawned process group.

        Parameters
        ----------
        proc : subprocess.Popen or None
            Process handle whose process group should be terminated.

        Returns
        -------
        None
            The process group is terminated when possible.
        """

        if proc is None:
            return

        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

    def _drain_stderr(self, proc: Optional[subprocess.Popen]) -> str:
        """Read remaining stderr output from a process.

        Parameters
        ----------
        proc : subprocess.Popen or None
            Process handle to inspect.

        Returns
        -------
        str
            Remaining stderr text if available; otherwise an empty string.
        """

        if proc is None or proc.stderr is None:
            return ""

        try:
            return proc.stderr.read() or ""
        except Exception:
            return ""

    def run(
        self,
        circuit: Circuit,
        *,
        run_id: str,
        output_paths: Dict[str, Union[str, Path]],
        timeout_s: float = 60.0,
    ) -> SpiceRunResult:
        """Run ngspice in batch mode for one netlist.

        Parameters
        ----------
        circuit : Circuit
            Circuit object whose netlist path is passed to ngspice.
        run_id : str
            Unique identifier associated with this simulation run.
        output_paths : dict[str, str or pathlib.Path]
            Mapping from logical output names to expected output paths.
        timeout_s : float, optional
            Maximum allowed runtime in seconds. The default is ``60.0``.

        Returns
        -------
        SpiceRunResult
            Structured execution result containing status, timestamps,
            command metadata, stderr tail, and output paths.
        """

        command = ["ngspice", "-b", str(circuit.netlist_path)]
        started_at_ns = time.time_ns()
        proc = None

        normalized_output_paths = {
            name: Path(path) for name, path in output_paths.items()
        }

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            _, stderr = proc.communicate(timeout=timeout_s)
            finished_at_ns = time.time_ns()
            ok = proc.returncode == 0

            return SpiceRunResult(
                ok=ok,
                returncode=proc.returncode,
                timed_out=False,
                command=command,
                netlist_path=Path(circuit.netlist_path),
                run_id=run_id,
                started_at_ns=started_at_ns,
                finished_at_ns=finished_at_ns,
                stdout_tail="",
                stderr_tail=(stderr or "")[-4000:],
                output_paths=normalized_output_paths,
                error=None if ok else f"ngspice exited with {proc.returncode}",
            )

        except subprocess.TimeoutExpired:
            self._kill_process_group(proc)
            finished_at_ns = time.time_ns()
            stderr_tail = self._drain_stderr(proc)

            return SpiceRunResult(
                ok=False,
                returncode=None,
                timed_out=True,
                command=command,
                netlist_path=Path(circuit.netlist_path),
                run_id=run_id,
                started_at_ns=started_at_ns,
                finished_at_ns=finished_at_ns,
                stdout_tail="",
                stderr_tail=stderr_tail[-4000:],
                output_paths=normalized_output_paths,
                error=f"timeout after {timeout_s} seconds",
            )

        except FileNotFoundError as exc:
            finished_at_ns = time.time_ns()

            return SpiceRunResult(
                ok=False,
                returncode=None,
                timed_out=False,
                command=command,
                netlist_path=Path(circuit.netlist_path),
                run_id=run_id,
                started_at_ns=started_at_ns,
                finished_at_ns=finished_at_ns,
                stdout_tail="",
                stderr_tail="",
                output_paths=normalized_output_paths,
                error=str(exc),
            )


class SessionPreparingSpiceBackend(BatchSpiceBackend):
    """Backend that prefers a persistent ngspice session with batch fallback.

    This backend attempts to keep a persistent ``ngspice -n -s`` process alive.
    If session startup or session execution fails, it disables session mode and
    falls back to the batch subprocess backend.

    Parameters
    ----------
    startup_timeout_s : float, optional
        Time in seconds allowed for the persistent session to remain alive
        after startup. The default is ``0.25``.
    output_stable_polls : int, optional
        Number of consecutive polling intervals for which output files must
        remain present and size-stable before a session run is considered
        complete. The default is ``2``.

    Attributes
    ----------
    startup_timeout_s : float
        Session startup observation timeout in seconds.
    output_stable_polls : int
        Required number of stable output-file polling rounds.
    _proc : subprocess.Popen or None
        Persistent ngspice process handle.
    _disabled_reason : str or None
        Reason session mode was disabled.
    _batch_fallback : BatchSpiceBackend
        Batch backend used when session mode is unavailable.
    """

    backend_name = "session"

    def __init__(
        self,
        *,
        startup_timeout_s: float = 0.25,
        output_stable_polls: int = 2,
    ) -> None:
        """Initialize the session-preparing backend.

        Parameters
        ----------
        startup_timeout_s : float, optional
            Session startup observation timeout in seconds. The default is
            ``0.25``.
        output_stable_polls : int, optional
            Required number of stable output-file polling rounds. Values below
            one are clipped to one. The default is ``2``.

        Returns
        -------
        None
            Session configuration and fallback backend are initialized in
            place.
        """

        self.startup_timeout_s = startup_timeout_s
        self.output_stable_polls = max(1, int(output_stable_polls))
        self._proc: Optional[subprocess.Popen] = None
        self._disabled_reason: Optional[str] = None
        self._batch_fallback = BatchSpiceBackend()

    def close(self) -> None:
        """Terminate any persistent ngspice session process.

        Returns
        -------
        None
            The persistent process is terminated when it exists.
        """

        if self._proc is not None:
            self._kill_process_group(self._proc)
            self._proc = None

    def _disable_session(self, reason: str) -> None:
        """Disable session mode after an unrecoverable session failure.

        Parameters
        ----------
        reason : str
            Human-readable reason recorded for disabling session mode.

        Returns
        -------
        None
            The disabled reason is stored and the persistent process is closed.
        """

        self._disabled_reason = reason
        self.close()

    def _session_available(self) -> bool:
        """Ensure that a persistent ngspice session is ready for use.

        Returns
        -------
        bool
            ``True`` if session mode is available and the persistent process is
            alive; otherwise ``False``.
        """

        if self._disabled_reason is not None:
            return False

        if self._proc is not None and self._proc.poll() is None:
            return True

        try:
            proc = subprocess.Popen(
                ["ngspice", "-n", "-s"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                bufsize=1,
            )
        except Exception as exc:
            self._disable_session(str(exc))
            return False

        deadline = time.monotonic() + self.startup_timeout_s

        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr_tail = self._drain_stderr(proc)[-4000:]
                self._disable_session(
                    stderr_tail or "ngspice session process exited during startup"
                )
                return False
            time.sleep(0.01)

        self._proc = proc
        return True

    def _session_result(
        self,
        *,
        ok: bool,
        returncode: Optional[int],
        timed_out: bool,
        command: List[str],
        circuit: Circuit,
        run_id: str,
        started_at_ns: int,
        finished_at_ns: int,
        output_paths: Dict[str, Path],
        error: Optional[str] = None,
        stderr_tail: str = "",
    ) -> SpiceRunResult:
        """Build a :class:`SpiceRunResult` for session-mode execution.

        Parameters
        ----------
        ok : bool
            Whether the session-mode run completed successfully.
        returncode : int or None
            Session process return code, if available.
        timed_out : bool
            Whether the session-mode run timed out.
        command : list[str]
            Logical command used for diagnostics.
        circuit : Circuit
            Circuit object whose netlist was executed.
        run_id : str
            Unique run identifier.
        started_at_ns : int
            Run start timestamp in nanoseconds.
        finished_at_ns : int
            Run finish timestamp in nanoseconds.
        output_paths : dict[str, pathlib.Path]
            Mapping from logical output names to output paths.
        error : str or None, optional
            Error message for failed runs. The default is ``None``.
        stderr_tail : str, optional
            Captured stderr tail. The default is ``""``.

        Returns
        -------
        SpiceRunResult
            Structured session-mode execution result.
        """

        return SpiceRunResult(
            ok=ok,
            returncode=returncode,
            timed_out=timed_out,
            command=command,
            netlist_path=Path(circuit.netlist_path),
            run_id=run_id,
            started_at_ns=started_at_ns,
            finished_at_ns=finished_at_ns,
            stdout_tail="",
            stderr_tail=stderr_tail,
            output_paths=output_paths,
            backend=self.backend_name,
            error=error,
        )

    def _run_via_session(
        self,
        circuit: Circuit,
        *,
        run_id: str,
        output_paths: Dict[str, Path],
        timeout_s: float,
    ) -> SpiceRunResult:
        """Execute a netlist through an already-running ngspice session.

        Parameters
        ----------
        circuit : Circuit
            Circuit object whose netlist should be sourced by the session.
        run_id : str
            Unique run identifier.
        output_paths : dict[str, pathlib.Path]
            Expected output paths used to detect run completion.
        timeout_s : float
            Maximum allowed runtime in seconds.

        Returns
        -------
        SpiceRunResult
            Session-mode execution result.
        """

        if self._proc is None or self._proc.stdin is None:
            finished_at_ns = time.time_ns()

            return self._session_result(
                ok=False,
                returncode=None,
                timed_out=False,
                command=["ngspice", "-n", "-s"],
                circuit=circuit,
                run_id=run_id,
                started_at_ns=finished_at_ns,
                finished_at_ns=finished_at_ns,
                output_paths=output_paths,
                error="session backend is not initialized",
            )

        started_at_ns = time.time_ns()
        quoted_netlist_path = shlex.quote(str(circuit.netlist_path))

        try:
            self._proc.stdin.write(f"source {quoted_netlist_path}\n")
            self._proc.stdin.flush()
        except Exception as exc:
            finished_at_ns = time.time_ns()
            self._disable_session(str(exc))

            return self._session_result(
                ok=False,
                returncode=None,
                timed_out=False,
                command=[
                    "ngspice",
                    "-n",
                    "-s",
                    f"source {quoted_netlist_path}",
                ],
                circuit=circuit,
                run_id=run_id,
                started_at_ns=started_at_ns,
                finished_at_ns=finished_at_ns,
                output_paths=output_paths,
                error=str(exc),
            )

        deadline = time.monotonic() + timeout_s
        stable_polls = 0
        last_snapshot = None

        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                finished_at_ns = time.time_ns()
                stderr_tail = self._drain_stderr(self._proc)[-4000:]
                self._disable_session(
                    stderr_tail or "ngspice session exited during run"
                )

                return self._session_result(
                    ok=False,
                    returncode=self._proc.returncode,
                    timed_out=False,
                    command=[
                        "ngspice",
                        "-n",
                        "-s",
                        f"source {quoted_netlist_path}",
                    ],
                    circuit=circuit,
                    run_id=run_id,
                    started_at_ns=started_at_ns,
                    finished_at_ns=finished_at_ns,
                    output_paths=output_paths,
                    error=stderr_tail or "ngspice session exited during run",
                    stderr_tail=stderr_tail,
                )

            outputs_ready = True
            snapshot = []

            for path in output_paths.values():
                if not path.exists():
                    outputs_ready = False
                    break

                stat = path.stat()

                if stat.st_size <= 0:
                    outputs_ready = False
                    break

                snapshot.append(
                    (
                        str(path),
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )

            if outputs_ready:
                if snapshot == last_snapshot:
                    stable_polls += 1
                else:
                    stable_polls = 1
                    last_snapshot = snapshot

                if stable_polls >= self.output_stable_polls:
                    finished_at_ns = time.time_ns()

                    return self._session_result(
                        ok=True,
                        returncode=0,
                        timed_out=False,
                        command=[
                            "ngspice",
                            "-n",
                            "-s",
                            f"source {quoted_netlist_path}",
                        ],
                        circuit=circuit,
                        run_id=run_id,
                        started_at_ns=started_at_ns,
                        finished_at_ns=finished_at_ns,
                        output_paths=output_paths,
                    )
            else:
                stable_polls = 0
                last_snapshot = None

            time.sleep(0.01)

        finished_at_ns = time.time_ns()
        self._disable_session(f"timeout after {timeout_s} seconds")

        return self._session_result(
            ok=False,
            returncode=None,
            timed_out=True,
            command=[
                "ngspice",
                "-n",
                "-s",
                f"source {quoted_netlist_path}",
            ],
            circuit=circuit,
            run_id=run_id,
            started_at_ns=started_at_ns,
            finished_at_ns=finished_at_ns,
            output_paths=output_paths,
            error=f"timeout after {timeout_s} seconds",
        )

    def run(
        self,
        circuit: Circuit,
        *,
        run_id: str,
        output_paths: Dict[str, Union[str, Path]],
        timeout_s: float = 60.0,
    ) -> SpiceRunResult:
        """Run a circuit through session mode or batch fallback.

        Parameters
        ----------
        circuit : Circuit
            Circuit object whose netlist should be executed.
        run_id : str
            Unique run identifier.
        output_paths : dict[str, str or pathlib.Path]
            Mapping from logical output names to expected output paths.
        timeout_s : float, optional
            Maximum allowed runtime in seconds. The default is ``60.0``.

        Returns
        -------
        SpiceRunResult
            Result from the session backend when successful, otherwise result
            from the batch fallback backend.
        """

        normalized_output_paths = {
            name: Path(path) for name, path in output_paths.items()
        }

        if self._session_available():
            result = self._run_via_session(
                circuit,
                run_id=run_id,
                output_paths=normalized_output_paths,
                timeout_s=timeout_s,
            )

            if result.ok:
                return result

        fallback_result = self._batch_fallback.run(
            circuit,
            run_id=run_id,
            output_paths=normalized_output_paths,
            timeout_s=timeout_s,
        )
        fallback_result.backend = f"{self.backend_name}-fallback"

        if self._disabled_reason and fallback_result.error is None:
            fallback_result.error = self._disabled_reason

        return fallback_result


def create_spice_backend(name: str = "batch") -> BaseSpiceBackend:
    """Create an ngspice backend from a backend identifier.

    Parameters
    ----------
    name : str, optional
        Backend identifier. Supported values are ``"batch"``,
        ``"subprocess"``, ``"session"``, and ``"persistent"``. The default is
        ``"batch"``.

    Returns
    -------
    BaseSpiceBackend
        Instantiated ngspice execution backend.

    Raises
    ------
    ValueError
        If ``name`` is not a supported backend identifier.
    """

    normalized = str(name).strip().lower()

    if normalized in {"batch", "subprocess"}:
        return BatchSpiceBackend()

    if normalized in {"session", "persistent"}:
        return SessionPreparingSpiceBackend()

    raise ValueError(f"Unsupported spice backend: {name}")


SpiceKernel = BatchSpiceBackend
"""type[BatchSpiceBackend]: Backward-compatible alias for the batch backend."""