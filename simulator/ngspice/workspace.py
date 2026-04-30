"""Workspace and artifact-path management for ngspice runs.

This module defines helpers for creating, preparing, cleaning, and describing
ngspice workspaces. A workspace contains copied circuit netlists, simulator
output files, scratch files, logs, and a manifest describing the resolved
file-system layout.
"""

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSTATION_ROOT = REPO_ROOT / "simulator" / "workstation"
SCHEMATIC_ROOT = REPO_ROOT / "simulator" / "ngspice" / "schematic"


@dataclass(frozen=True)
class NgSpiceWorkspace:
    """Resolved file-system layout for one ngspice worker workspace.

    Parameters
    ----------
    project_name : str
        Project or experiment namespace.
    run_id : str
        Unique experiment run identifier.
    run_root : pathlib.Path
        Root directory for the experiment run.
    worker_name : str
        Worker identifier associated with this workspace.
    scope : str
        Workspace scope label, such as ``"train"``, ``"eval"``, or ``"env"``.
    worker_root : pathlib.Path
        Root directory for this worker and scope.
    netlists_dir : pathlib.Path
        Directory containing copied and rewritten netlist files.
    outputs_dir : pathlib.Path
        Directory containing simulator output files.
    scratch_dir : pathlib.Path
        Directory for temporary simulation artifacts.
    logs_dir : pathlib.Path
        Directory for workspace logs and manifest files.
    s_param_netlist_path : pathlib.Path
        Path to the S-parameter simulation netlist.
    nf_netlist_path : pathlib.Path
        Path to the noise-figure simulation netlist.
    iip3_netlist_path : pathlib.Path or None
        Path to the IIP3 simulation netlist when enabled.
    dc_op_result_path : pathlib.Path
        Path to the DC operating-point result file.
    s_param_bandwidth_result_path : pathlib.Path
        Path to the S-parameter result file.
    nf_result_path : pathlib.Path
        Path to the noise-figure result file.
    iip3_result_path : pathlib.Path or None
        Path to the IIP3 result file when enabled.
    manifest_path : pathlib.Path
        Path to the generated workspace manifest JSON file.
    """

    project_name: str
    run_id: str
    run_root: Path
    worker_name: str
    scope: str
    worker_root: Path
    netlists_dir: Path
    outputs_dir: Path
    scratch_dir: Path
    logs_dir: Path
    s_param_netlist_path: Path
    nf_netlist_path: Path
    iip3_netlist_path: Optional[Path]
    dc_op_result_path: Path
    s_param_bandwidth_result_path: Path
    nf_result_path: Path
    iip3_result_path: Optional[Path]
    manifest_path: Path


def sanitize_path_component(value: str) -> str:
    """Sanitize an arbitrary label for safe path-component usage.

    Non-alphanumeric characters except ``.``, ``_``, and ``-`` are replaced
    with underscores. Leading and trailing dots or underscores are removed. If
    the resulting component is empty, ``"unnamed"`` is returned.

    Parameters
    ----------
    value : str
        Input label to sanitize.

    Returns
    -------
    str
        Sanitized path component safe for directory or file naming.
    """

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return sanitized or "unnamed"


def create_run_id() -> str:
    """Create a unique run identifier for simulator workspaces.

    Returns
    -------
    str
        Run identifier containing local time, process ID, and a short UUID
        suffix.
    """

    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def create_experiment_run_root(
    project_name: str,
    run_id: Optional[str] = None,
) -> Tuple[str, Path]:
    """Create the root directory for one experiment run.

    Parameters
    ----------
    project_name : str
        Project or experiment namespace used under the managed workstation
        root.
    run_id : str or None, optional
        Explicit run identifier. If ``None``, a new identifier is generated.
        The default is ``None``.

    Returns
    -------
    actual_run_id : str
        Run identifier used for the experiment.
    run_root : pathlib.Path
        Created experiment run-root directory.
    """

    actual_run_id = run_id or create_run_id()
    run_root = WORKSTATION_ROOT / sanitize_path_component(project_name) / actual_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return actual_run_id, run_root


def cleanup_experiment_run_root(run_root: Union[str, os.PathLike, None]) -> None:
    """Remove a run root when it lives under the managed workstation tree.

    This function refuses to delete paths outside ``WORKSTATION_ROOT`` to
    avoid accidental removal of arbitrary user directories.

    Parameters
    ----------
    run_root : str or os.PathLike or None
        Run-root directory to remove. If ``None``, no action is taken.

    Returns
    -------
    None
        The run-root directory is removed if it is inside the managed
        workstation tree.
    """

    if run_root is None:
        return

    run_root_path = Path(run_root)

    try:
        resolved_run_root = run_root_path.resolve(strict=False)
        resolved_workstation_root = WORKSTATION_ROOT.resolve(strict=False)
    except OSError:
        return

    try:
        resolved_run_root.relative_to(resolved_workstation_root)
    except ValueError:
        return

    shutil.rmtree(resolved_run_root, ignore_errors=True)


def get_template_dir(circuit_type: str) -> Path:
    """Return the schematic template directory for a circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit family identifier.

    Returns
    -------
    pathlib.Path
        Existing schematic template directory for ``circuit_type``.

    Raises
    ------
    FileNotFoundError
        If the template directory does not exist.
    """

    template_dir = SCHEMATIC_ROOT / sanitize_path_component(circuit_type)

    if not template_dir.exists():
        raise FileNotFoundError(f"Missing template directory: {template_dir}")

    return template_dir


def _copy_template(src: Path, dst: Path) -> None:
    """Copy one template file into a workspace location.

    Parameters
    ----------
    src : pathlib.Path
        Source template file path.
    dst : pathlib.Path
        Destination workspace file path.

    Returns
    -------
    None
        The source file is copied to ``dst`` and parent directories are
        created if needed.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _maybe_unlink(path: Optional[Path]) -> None:
    """Remove a file if it exists.

    Parameters
    ----------
    path : pathlib.Path or None
        File path to remove. If ``None``, no action is taken.

    Returns
    -------
    None
        The file is removed when it exists.
    """

    if path is None:
        return

    try:
        path.unlink()
    except FileNotFoundError:
        return


def prepare_worker_workspace(
    *,
    project_name: str,
    run_id: str,
    run_root: Path,
    circuit_type: str,
    worker_name: Union[str, int],
    scope: str = "train",
    enable_iip3: bool = True,
    clean: bool = False,
) -> NgSpiceWorkspace:
    """Create and populate a per-worker ngspice workspace.

    The function creates workspace directories, copies circuit template
    netlists, resolves output paths, optionally removes old outputs, and writes
    a JSON manifest describing the workspace.

    Parameters
    ----------
    project_name : str
        Project or experiment namespace.
    run_id : str
        Unique experiment run identifier.
    run_root : pathlib.Path
        Root directory under which the worker workspace is created.
    circuit_type : str
        Circuit family identifier used to select schematic templates.
    worker_name : str or int
        Worker identifier used to isolate workspace files.
    scope : str, optional
        Workspace scope label, such as ``"train"``, ``"eval"``, or ``"env"``.
        The default is ``"train"``.
    enable_iip3 : bool, optional
        Whether to copy and expose the IIP3/FFT netlist and output path. The
        default is ``True``.
    clean : bool, optional
        Whether to remove existing output and scratch artifacts after
        workspace creation. The default is ``False``.

    Returns
    -------
    NgSpiceWorkspace
        Dataclass containing the resolved workspace layout.

    Raises
    ------
    FileNotFoundError
        If the schematic template directory or required template files are
        missing.
    """

    worker_label = f"worker_{sanitize_path_component(worker_name)}"
    scope_label = sanitize_path_component(scope)

    worker_root = Path(run_root) / worker_label / scope_label
    netlists_dir = worker_root / "netlists"
    outputs_dir = worker_root / "outputs"
    scratch_dir = worker_root / "tmp"
    logs_dir = worker_root / "logs"

    for path in (netlists_dir, outputs_dir, scratch_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    template_dir = get_template_dir(circuit_type)
    template_files = {
        "s_param": template_dir / f"{circuit_type}_LNA_S_Param.spice",
        "noise": template_dir / f"{circuit_type}_LNA_NoiseFigure.spice",
        "iip3": template_dir / f"{circuit_type}_LNA_FFT.spice",
    }
    workspace_files = {
        "s_param": netlists_dir / f"{circuit_type}_LNA_S_Param.spice",
        "noise": netlists_dir / f"{circuit_type}_LNA_NoiseFigure.spice",
        "iip3": netlists_dir / f"{circuit_type}_LNA_FFT.spice",
    }

    for key in ("s_param", "noise"):
        _copy_template(template_files[key], workspace_files[key])

    if enable_iip3:
        _copy_template(template_files["iip3"], workspace_files["iip3"])
    else:
        _maybe_unlink(workspace_files["iip3"])

    workspace = NgSpiceWorkspace(
        project_name=project_name,
        run_id=run_id,
        run_root=Path(run_root),
        worker_name=str(worker_name),
        scope=scope_label,
        worker_root=worker_root,
        netlists_dir=netlists_dir,
        outputs_dir=outputs_dir,
        scratch_dir=scratch_dir,
        logs_dir=logs_dir,
        s_param_netlist_path=workspace_files["s_param"],
        nf_netlist_path=workspace_files["noise"],
        iip3_netlist_path=workspace_files["iip3"] if enable_iip3 else None,
        dc_op_result_path=outputs_dir / "DC_OP.csv",
        s_param_bandwidth_result_path=outputs_dir / "S_Param_Bandwidth.csv",
        nf_result_path=outputs_dir / "NoiseFigure.csv",
        iip3_result_path=outputs_dir / "FFT.csv" if enable_iip3 else None,
        manifest_path=logs_dir / "workspace_manifest.json",
    )

    if clean:
        cleanup_workspace_outputs(workspace)

    manifest = asdict(workspace)

    for key, value in list(manifest.items()):
        if isinstance(value, Path):
            manifest[key] = str(value)

    workspace.manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return workspace


def cleanup_workspace_outputs(workspace: NgSpiceWorkspace) -> None:
    """Delete generated output and scratch artifacts from a workspace.

    Parameters
    ----------
    workspace : NgSpiceWorkspace
        Workspace whose output and scratch directories should be cleaned.

    Returns
    -------
    None
        Files and directories under ``outputs_dir`` and ``scratch_dir`` are
        removed.
    """

    for path in workspace.outputs_dir.glob("*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()

    for path in workspace.scratch_dir.glob("*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()