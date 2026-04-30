"""Structured log normalization, partitioning, and persistence helpers.

This module provides utilities for collecting, normalizing, partitioning, and
saving structured reinforcement-learning experiment logs. It converts raw
training, epoch-level, and observation-level entries into schema-tagged records
and writes both raw and partitioned artifacts to disk.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from loggers.schema import (
    EVAL_EVENT,
    LEARNER_EVENT,
    OBS_OTHER_EVENT,
    RESET_EVENT,
    ROLLOUT_CHUNK_EVENT,
    SCHEMA_VERSION,
    SCHEDULER_EVENT,
    SIMULATION_PROFILE_EVENT,
    normalize_epoch_entry,
    normalize_obs_entry,
    normalize_train_entry,
)


NORMALIZERS = {
    "train": normalize_train_entry,
    "epoch": normalize_epoch_entry,
    "obs": normalize_obs_entry,
}


class EventLog(list):
    """List-like container that normalizes entries on insertion.

    This class behaves like a Python list but applies a stream-specific
    normalizer whenever entries are appended through :meth:`append` or
    :meth:`extend`.

    Parameters
    ----------
    normalizer_key : {"train", "epoch", "obs"}
        Normalizer namespace used to select the schema-normalization function.
    entries : sequence[dict] or None, optional
        Initial entries used to populate the log container without
        re-normalizing. This is mainly used during pickle restoration. The
        default is ``None``.

    Attributes
    ----------
    _normalizer_key : str
        Key used to identify the selected normalizer.
    _normalizer : callable
        Normalization function applied to newly inserted entries.
    """

    def __init__(self, normalizer_key, entries=None):
        """Initialize an event log container.

        Parameters
        ----------
        normalizer_key : {"train", "epoch", "obs"}
            Key used to select the entry normalizer.
        entries : sequence[dict] or None, optional
            Initial already-normalized entries. The default is ``None``.

        Returns
        -------
        None
            The list container and normalizer metadata are initialized in
            place.
        """

        super().__init__()
        self._normalizer_key = normalizer_key
        self._normalizer = NORMALIZERS[normalizer_key]

        if entries:
            super().extend(entries)

    def append(self, entry):
        """Normalize and append one log entry.

        Parameters
        ----------
        entry : dict
            Raw log entry to normalize and append.

        Returns
        -------
        None
            The normalized entry is appended in place.
        """

        super().append(self._normalizer(entry))

    def extend(self, entries):
        """Normalize and append multiple log entries.

        Parameters
        ----------
        entries : iterable[dict]
            Raw log entries to normalize and append.

        Returns
        -------
        None
            All normalized entries are appended in order.
        """

        for entry in entries:
            self.append(entry)

    def __reduce__(self):
        """Return pickle reconstruction instructions.

        Returns
        -------
        tuple
            Pickle-compatible reconstruction tuple containing the restore
            callable and its arguments.
        """

        return (self.__class__._restore, (self._normalizer_key, list(self)))

    @classmethod
    def _restore(cls, normalizer_key, entries):
        """Rebuild an :class:`EventLog` during pickle restoration.

        Parameters
        ----------
        normalizer_key : {"train", "epoch", "obs"}
            Normalizer key restored from the serialized object.
        entries : sequence[dict]
            Already-normalized entries restored from the serialized object.

        Returns
        -------
        EventLog
            Reconstructed event log container.
        """

        return cls(normalizer_key, entries=entries)


def json_safe(value):
    """Convert nested objects into JSON-safe representations.

    This helper recursively converts NumPy scalars, NumPy arrays, PyTorch
    tensors, dictionaries, lists, and tuples into values that can be serialized
    by ``json.dumps``.

    Parameters
    ----------
    value : Any
        Arbitrary Python object or nested container.

    Returns
    -------
    Any
        JSON-serializable representation of ``value``.
    """

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return str(value)


def flatten_mapping(mapping, prefix=""):
    """Flatten a nested mapping using underscore-delimited keys.

    Nested dictionaries are recursively flattened. NumPy arrays are converted
    to lists, and NumPy scalar values are converted to Python scalars.

    Parameters
    ----------
    mapping : dict
        Nested mapping to flatten.
    prefix : str, optional
        Prefix prepended to emitted keys. The default is ``""``.

    Returns
    -------
    dict[str, object]
        Flattened mapping with underscore-delimited keys.
    """

    flattened = {}

    for key, value in mapping.items():
        key_name = f"{prefix}{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, prefix=f"{key_name}_"))
        elif isinstance(value, np.ndarray):
            flattened[key_name] = value.tolist()
        elif isinstance(value, np.generic):
            flattened[key_name] = value.item()
        else:
            flattened[key_name] = value

    return flattened


def partition_epoch_logs(epoch_logger):
    """Split epoch logs into evaluation, learner, scheduler, and other streams.

    Parameters
    ----------
    epoch_logger : sequence[dict]
        Normalized epoch-level log entries.

    Returns
    -------
    dict[str, list[dict]]
        Partitioned epoch records with keys ``"eval"``, ``"learner"``,
        ``"scheduler"``, and ``"other"``.
    """

    partitions = {
        "eval": [],
        "learner": [],
        "scheduler": [],
        "other": [],
    }

    for entry in epoch_logger:
        event_type = entry.get("_event_type")

        if event_type == EVAL_EVENT or (
            "max_ep_ret" in entry and "mean_ep_ret" in entry
        ):
            partitions["eval"].append(json_safe(dict(entry)))

        elif event_type == LEARNER_EVENT or "result" in entry:
            learner_entry = dict(entry)
            result_data = learner_entry.pop("result")

            if isinstance(result_data, dict):
                learner_entry.update(flatten_mapping(json_safe(result_data)))
            else:
                learner_entry["result"] = json_safe(result_data)

            partitions["learner"].append(json_safe(learner_entry))

        elif event_type == SCHEDULER_EVENT or (
            "learner_updates_total" in entry and "chunk_env_steps" in entry
        ):
            partitions["scheduler"].append(json_safe(dict(entry)))

        else:
            partitions["other"].append(json_safe(dict(entry)))

    return partitions


def partition_obs_logs(obs_logger):
    """Split observation logs into rollout, simulation, reset, and other streams.

    Rollout chunk entries are expanded into per-step records when they contain
    a ``"results"`` list. Nested ``info``, ``simulation_profile``, and
    ``reset_info`` dictionaries are flattened into dedicated structured
    records when available.

    Parameters
    ----------
    obs_logger : sequence[dict]
        Normalized observation-level log entries.

    Returns
    -------
    dict[str, list[dict]]
        Partitioned observation records with keys ``"rollout_chunks"``,
        ``"rollout_steps"``, ``"simulation_profiles"``, ``"reset_events"``,
        and ``"other"``.
    """

    partitions = {
        "rollout_chunks": [],
        "rollout_steps": [],
        "simulation_profiles": [],
        "reset_events": [],
        "other": [],
    }

    for entry in obs_logger:
        event_type = entry.get("_event_type")

        if event_type == ROLLOUT_CHUNK_EVENT or "results" in entry:
            chunk_entry = {k: v for k, v in entry.items() if k != "results"}
            partitions["rollout_chunks"].append(json_safe(chunk_entry))

            for result in entry.get("results", []):
                result_entry = json_safe(dict(result))
                combined = json_safe(
                    {
                        "chunk_time_steps": entry.get("time_steps"),
                        "runner_name": entry.get("runner_name"),
                        "runner_policy_version": entry.get("policy_version"),
                        "runner_env_steps_total": entry.get("env_steps_total"),
                        "runner_learner_updates_total": entry.get(
                            "learner_updates_total"
                        ),
                        "runner_effective_utd": entry.get("effective_utd"),
                        **result_entry,
                    }
                )

                info = result_entry.get("info")
                if isinstance(info, dict):
                    combined.update(flatten_mapping(info, prefix="info_"))

                partitions["rollout_steps"].append(combined)

                simulation_profile = result_entry.get("simulation_profile")
                if isinstance(simulation_profile, dict):
                    sim_record = {
                        "chunk_time_steps": entry.get("time_steps"),
                        "runner_name": entry.get("runner_name"),
                        "agent_timesteps": result_entry.get("agent_timesteps"),
                    }
                    sim_record.update(
                        flatten_mapping(
                            simulation_profile,
                            prefix="simulation_",
                        )
                    )
                    partitions["simulation_profiles"].append(json_safe(sim_record))

                reset_info = result_entry.get("reset_info")
                if isinstance(reset_info, dict) and reset_info:
                    reset_record = {
                        "chunk_time_steps": entry.get("time_steps"),
                        "runner_name": entry.get("runner_name"),
                        "agent_timesteps": result_entry.get("agent_timesteps"),
                    }
                    reset_record.update(
                        flatten_mapping(reset_info, prefix="reset_")
                    )
                    partitions["reset_events"].append(json_safe(reset_record))

            continue

        if "results" not in entry:
            step_entry = json_safe(dict(entry))
            info = step_entry.get("info")

            if isinstance(info, dict):
                step_entry.update(flatten_mapping(info, prefix="info_"))

            if event_type == RESET_EVENT:
                partitions["other"].append(step_entry)
            elif event_type == SIMULATION_PROFILE_EVENT:
                partitions["other"].append(step_entry)
            elif event_type == OBS_OTHER_EVENT:
                partitions["other"].append(step_entry)
            else:
                partitions["rollout_steps"].append(step_entry)

            simulation_profile = step_entry.get("simulation_profile")
            if isinstance(simulation_profile, dict):
                sim_record = {"time_steps": step_entry.get("time_steps")}
                sim_record.update(
                    flatten_mapping(simulation_profile, prefix="simulation_")
                )
                partitions["simulation_profiles"].append(json_safe(sim_record))

            reset_info = step_entry.get("reset_info")
            if isinstance(reset_info, dict) and reset_info:
                reset_record = {"time_steps": step_entry.get("time_steps")}
                reset_record.update(flatten_mapping(reset_info, prefix="reset_"))
                partitions["reset_events"].append(json_safe(reset_record))

            continue

        partitions["other"].append(json_safe(dict(entry)))

    return partitions


def save_partitioned_logs(project_name, train_logger, epoch_logger, obs_logger):
    """Persist raw and partitioned structured logs under ``./log/<project>``.

    This function writes raw pickle logs, optional raw observation CSV logs,
    structured metadata, JSONL partitions, and CSV partitions for non-empty
    structured record groups.

    Parameters
    ----------
    project_name : str
        Experiment project name. Files are saved under ``./log/{project_name}``.
    train_logger : sequence[dict]
        Training metadata log entries.
    epoch_logger : sequence[dict]
        Epoch-level metrics and events.
    obs_logger : sequence[dict]
        Observation-level or rollout-level log entries.

    Returns
    -------
    None
        Log artifacts are written to disk.
    """

    save_path = f"./log/{project_name}"
    os.makedirs(save_path, exist_ok=True)

    structured_path = os.path.join(save_path, "structured")
    os.makedirs(structured_path, exist_ok=True)

    with open(os.path.join(save_path, f"{project_name}_train_logs.pkl"), "wb") as f:
        pickle.dump(
            {
                "epoch_logger": epoch_logger,
                "train_logger": train_logger,
            },
            f,
        )

    if obs_logger:
        pd.DataFrame([json_safe(entry) for entry in obs_logger]).to_csv(
            os.path.join(save_path, f"{project_name}_obs_logs.csv"),
            index=False,
        )

    with open(
        os.path.join(structured_path, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "train_logs": json_safe(train_logger),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    epoch_partitions = partition_epoch_logs(epoch_logger)
    obs_partitions = partition_obs_logs(obs_logger)

    partition_map = {
        "eval_metrics": epoch_partitions["eval"],
        "learner_metrics": epoch_partitions["learner"],
        "scheduler_metrics": epoch_partitions["scheduler"],
        "epoch_other": epoch_partitions["other"],
        "rollout_chunks": obs_partitions["rollout_chunks"],
        "rollout_steps": obs_partitions["rollout_steps"],
        "simulation_profiles": obs_partitions["simulation_profiles"],
        "reset_events": obs_partitions["reset_events"],
        "obs_other": obs_partitions["other"],
    }

    for name, records in partition_map.items():
        json_path = os.path.join(structured_path, f"{name}.jsonl")

        with open(json_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(json_safe(record), ensure_ascii=False))
                f.write("\n")

        if records:
            pd.DataFrame(records).to_csv(
                os.path.join(structured_path, f"{name}.csv"),
                index=False,
            )


class TrainingLogger:
    """Container for train, epoch, and observation log streams.

    This class manages three schema-normalized log streams and provides helper
    methods for appending entries and saving all streams as structured
    artifacts.

    Parameters
    ----------
    project_name : str
        Experiment project name used when persisting logs.

    Attributes
    ----------
    project_name : str
        Experiment project name.
    train_logs : EventLog
        Training metadata log stream.
    epoch_logs : EventLog
        Epoch-level log stream.
    obs_logs : EventLog
        Observation-level log stream.
    """

    def __init__(self, project_name):
        """Initialize a training logger.

        Parameters
        ----------
        project_name : str
            Experiment project name used when saving log artifacts.

        Returns
        -------
        None
            Log streams are initialized in place.
        """

        self.project_name = project_name
        self.train_logs = EventLog("train")
        self.epoch_logs = EventLog("epoch")
        self.obs_logs = EventLog("obs")

    def log_train(self, entry):
        """Append one training-metadata entry.

        Parameters
        ----------
        entry : dict
            Training metadata entry to normalize and append.

        Returns
        -------
        None
            The normalized entry is appended to ``self.train_logs``.
        """

        self.train_logs.append(entry)

    def log_epoch(self, entry):
        """Append one epoch-level entry.

        Parameters
        ----------
        entry : dict
            Epoch-level entry to normalize and append.

        Returns
        -------
        None
            The normalized entry is appended to ``self.epoch_logs``.
        """

        self.epoch_logs.append(entry)

    def log_obs(self, entry):
        """Append one observation-level entry.

        Parameters
        ----------
        entry : dict
            Observation-level entry to normalize and append.

        Returns
        -------
        None
            The normalized entry is appended to ``self.obs_logs``.
        """

        self.obs_logs.append(entry)

    def get_logs(self):
        """Return the three managed log streams.

        Returns
        -------
        train_logs : EventLog
            Training metadata log stream.
        epoch_logs : EventLog
            Epoch-level log stream.
        obs_logs : EventLog
            Observation-level log stream.
        """

        return self.train_logs, self.epoch_logs, self.obs_logs

    def save_all(self):
        """Persist all managed log streams using structured partitions.

        Returns
        -------
        None
            Raw and partitioned log artifacts are written under
            ``./log/{self.project_name}``.
        """

        save_partitioned_logs(
            self.project_name,
            self.train_logs,
            self.epoch_logs,
            self.obs_logs,
        )