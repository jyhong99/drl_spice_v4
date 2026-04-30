"""Schema tagging and event classification helpers for structured logs.

This module defines schema constants and helper functions used to normalize
structured log entries. Each normalized entry is tagged with a schema version,
stream name, and event type so that downstream readers, plotters, and
analysis tools can route records consistently.
"""

from typing import Dict


SCHEMA_VERSION = 1

TRAIN_STREAM = "train"
EPOCH_STREAM = "epoch"
OBS_STREAM = "obs"

EVAL_EVENT = "eval"
LEARNER_EVENT = "learner"
SCHEDULER_EVENT = "scheduler"
EPOCH_OTHER_EVENT = "epoch_other"

ROLLOUT_CHUNK_EVENT = "rollout_chunk"
ROLLOUT_STEP_EVENT = "rollout_step"
SIMULATION_PROFILE_EVENT = "simulation_profile"
RESET_EVENT = "reset_event"
OBS_OTHER_EVENT = "obs_other"


def _with_schema_fields(entry: Dict, *, stream: str, event_type: str) -> Dict:
    """Attach schema metadata to a log entry.

    Parameters
    ----------
    entry : dict
        Original log entry to normalize. The input dictionary is copied before
        schema fields are added.
    stream : str
        Stream label assigned to the log entry, such as ``"train"``,
        ``"epoch"``, or ``"obs"``.
    event_type : str
        Event type assigned to the log entry.

    Returns
    -------
    dict
        New dictionary containing the original entry fields plus schema
        metadata fields. Existing schema fields are preserved because
        ``setdefault`` is used.
    """

    normalized = dict(entry)
    normalized.setdefault("_schema_version", SCHEMA_VERSION)
    normalized.setdefault("_stream", stream)
    normalized.setdefault("_event_type", event_type)
    return normalized


def classify_epoch_event(entry: Dict) -> str:
    """Classify an epoch-level log entry.

    The classification is based on the presence of known fields produced by
    evaluation, learner, and scheduler logging components.

    Parameters
    ----------
    entry : dict
        Epoch-level log entry to classify.

    Returns
    -------
    str
        Event type string. Returns one of ``EVAL_EVENT``, ``LEARNER_EVENT``,
        ``SCHEDULER_EVENT``, or ``EPOCH_OTHER_EVENT``.
    """

    if "max_ep_ret" in entry and "mean_ep_ret" in entry:
        return EVAL_EVENT

    if "result" in entry:
        return LEARNER_EVENT

    if "learner_updates_total" in entry and "chunk_env_steps" in entry:
        return SCHEDULER_EVENT

    return EPOCH_OTHER_EVENT


def classify_obs_event(entry: Dict) -> str:
    """Classify an observation-level log entry.

    The classification is based on the presence of fields associated with
    rollout chunks and individual rollout steps.

    Parameters
    ----------
    entry : dict
        Observation-level log entry to classify.

    Returns
    -------
    str
        Event type string. Returns one of ``ROLLOUT_CHUNK_EVENT``,
        ``ROLLOUT_STEP_EVENT``, or ``OBS_OTHER_EVENT``.
    """

    if "results" in entry:
        return ROLLOUT_CHUNK_EVENT

    if "time_steps" in entry:
        return ROLLOUT_STEP_EVENT

    return OBS_OTHER_EVENT


def normalize_train_entry(entry: Dict) -> Dict:
    """Normalize a training-metadata log entry.

    Parameters
    ----------
    entry : dict
        Training-level metadata entry to normalize.

    Returns
    -------
    dict
        Normalized entry tagged with train-stream schema fields.
    """

    return _with_schema_fields(
        entry,
        stream=TRAIN_STREAM,
        event_type="train_metadata",
    )


def normalize_epoch_entry(entry: Dict) -> Dict:
    """Normalize an epoch-level log entry.

    Parameters
    ----------
    entry : dict
        Epoch-level entry to normalize.

    Returns
    -------
    dict
        Normalized entry tagged with epoch-stream schema fields and an event
        type inferred by :func:`classify_epoch_event`.
    """

    return _with_schema_fields(
        entry,
        stream=EPOCH_STREAM,
        event_type=classify_epoch_event(entry),
    )


def normalize_obs_entry(entry: Dict) -> Dict:
    """Normalize an observation-level log entry.

    Parameters
    ----------
    entry : dict
        Observation-level entry to normalize.

    Returns
    -------
    dict
        Normalized entry tagged with observation-stream schema fields and an
        event type inferred by :func:`classify_obs_event`.
    """

    return _with_schema_fields(
        entry,
        stream=OBS_STREAM,
        event_type=classify_obs_event(entry),
    )