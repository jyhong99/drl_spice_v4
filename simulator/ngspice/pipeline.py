"""Simulation pipeline helpers for ngspice-backed LNA evaluation.

This module defines the end-to-end simulation pipeline used to evaluate one
decoded LNA design. The pipeline creates unique output paths, executes the
required ngspice phases, parses scalar metrics, assembles the performance
vector, and returns profiling metadata.
"""

import time
from pathlib import Path


def run_simulation_pipeline(
    *,
    design_variables_config,
    artifacts,
    executor,
    readers,
    circuits,
    output_paths,
    fixed_values,
    freq_range,
    enable_iip3,
    kernel_backend,
    simulation_cache_stats,
):
    """Run the full simulator pipeline for one decoded design.

    The pipeline always executes S-parameter and noise-analysis phases. If
    ``enable_iip3`` is ``True``, it additionally executes the FFT/IIP3 phase.
    Parsed metrics are combined into the performance vector expected by the
    LNA environment.

    Parameters
    ----------
    design_variables_config : dict[str, float]
        Decoded simulator parameter mapping for the candidate circuit design.
    artifacts : OutputArtifactManager
        Artifact manager used to create unique output paths for simulator
        result files.
    executor : PhaseExecutor
        Phase executor used to rewrite netlists and run ngspice phases.
    readers : dict[str, object]
        Mapping from reader names to reader instances. Expected keys are
        ``"dc"``, ``"s_param"``, ``"stability"``, ``"nf"``, and optionally
        ``"iip3"`` when ``enable_iip3`` is ``True``.
    circuits : dict[str, Circuit]
        Mapping from phase names to circuit objects. Expected keys are
        ``"s_param"``, ``"nf"``, and optionally ``"iip3"`` when
        ``enable_iip3`` is ``True``.
    output_paths : dict[str, str or pathlib.Path]
        Mapping from logical output names to base output paths. Expected keys
        are ``"dc_op"``, ``"s_param_bandwidth"``, ``"nf"``, and optionally
        ``"iip3"`` when ``enable_iip3`` is ``True``.
    fixed_values : dict[str, float]
        Fixed circuit values used to compute derived metrics such as power
        dissipation.
    freq_range : sequence[float]
        Frequency range in Hz used when extracting S-parameter and noise
        metrics.
    enable_iip3 : bool
        Whether to run and parse the IIP3 simulation phase.
    kernel_backend : str
        Configured simulator backend name used as a fallback profile label.
    simulation_cache_stats : dict[str, object]
        Current simulation-cache statistics copied into the returned profile.

    Returns
    -------
    dict[str, object]
        Simulation result dictionary containing:

        ``"performances"`` : list[float]
            Performance vector ordered as ``S11``, ``S21``, ``S22``, ``NF``,
            ``PD``, and optionally ``IIP3``.
        ``"stability_factor"`` : float
            Parsed minimum Rollett stability factor ``K``.
        ``"profile"`` : dict[str, object]
            Timing and execution metadata for the full pipeline.
        ``"cleanup_paths"`` : list[pathlib.Path]
            Generated output files that may be removed after parsing.
    """

    sim_t0 = time.perf_counter_ns()
    phase_profiles = []
    cleanup_paths = []

    s_outputs = artifacts.make_output_path_map(
        prefix="sparam",
        filenames=[
            Path(output_paths["dc_op"]).name,
            Path(output_paths["s_param_bandwidth"]).name,
        ],
    )
    cleanup_paths.extend(s_outputs["actual_paths"].values())

    s_result, s_profile = executor.execute(
        circuits["s_param"],
        design_variables_config,
        phase_name="sparam",
        run_id=s_outputs["run_id"],
        output_path_map=s_outputs["paths"],
    )
    phase_profiles.append(s_profile)

    nf_outputs = artifacts.make_output_path_map(
        prefix="noise",
        filenames=[Path(output_paths["nf"]).name],
    )
    cleanup_paths.extend(nf_outputs["actual_paths"].values())

    nf_run_result, nf_profile = executor.execute(
        circuits["nf"],
        design_variables_config,
        phase_name="noise",
        run_id=nf_outputs["run_id"],
        output_path_map=nf_outputs["paths"],
    )
    phase_profiles.append(nf_profile)

    iip3_outputs = None
    iip3_run_result = None

    if enable_iip3:
        iip3_outputs = artifacts.make_output_path_map(
            prefix="fft",
            filenames=[Path(output_paths["iip3"]).name],
        )
        cleanup_paths.extend(iip3_outputs["actual_paths"].values())

        iip3_run_result, iip3_profile = executor.execute(
            circuits["iip3"],
            design_variables_config,
            phase_name="iip3",
            run_id=iip3_outputs["run_id"],
            output_path_map=iip3_outputs["paths"],
        )
        phase_profiles.append(iip3_profile)

    parse_t0 = time.perf_counter_ns()

    dc_result = readers["dc"].read_metrics(
        s_outputs["actual_paths"][Path(output_paths["dc_op"]).name],
        run_id=s_result.run_id,
        started_at_ns=s_result.started_at_ns,
    )

    s_param_result = readers["s_param"].read_metrics(
        s_outputs["actual_paths"][Path(output_paths["s_param_bandwidth"]).name],
        run_id=s_result.run_id,
        started_at_ns=s_result.started_at_ns,
        freq_range=freq_range,
    )

    stability_result = readers["stability"].read_metrics(
        s_outputs["actual_paths"][Path(output_paths["s_param_bandwidth"]).name],
        run_id=s_result.run_id,
        started_at_ns=s_result.started_at_ns,
    )

    nf_result = readers["nf"].read_metrics(
        nf_outputs["actual_paths"][Path(output_paths["nf"]).name],
        run_id=nf_run_result.run_id,
        started_at_ns=nf_run_result.started_at_ns,
        freq_range=freq_range,
    )

    stability_factor = float(stability_result["K"])

    performances = [
        float(s_param_result["v(s_1_1)"]),
        float(s_param_result["v(s_2_1)"]),
        float(s_param_result["v(s_2_2)"]),
        float(nf_result["NoiseFigure"]),
        abs(fixed_values["v_dd"] * float(dc_result["i(v_dd)"]) * 1e3),
    ]

    if enable_iip3:
        iip3_result = readers["iip3"].read_metrics(
            iip3_outputs["actual_paths"][Path(output_paths["iip3"]).name],
            run_id=iip3_run_result.run_id,
            started_at_ns=iip3_run_result.started_at_ns,
        )
        performances.append(float(iip3_result["IIP3_dBm"]))

    parse_ms = (time.perf_counter_ns() - parse_t0) / 1e6
    total_ms = (time.perf_counter_ns() - sim_t0) / 1e6

    profile = {
        "total_ms": total_ms,
        "parse_ms": parse_ms,
        "phases": phase_profiles,
        "enable_iip3": enable_iip3,
        "kernel_backend": (
            phase_profiles[0]["backend"]
            if phase_profiles
            else kernel_backend
        ),
        "cache_hit": False,
        "cache_stats": dict(simulation_cache_stats),
    }

    return {
        "performances": performances,
        "stability_factor": stability_factor,
        "profile": profile,
        "cleanup_paths": cleanup_paths,
    }