"""Structured logging, reading, and plotting helpers.

This package-level module re-exports the main utilities used for structured
experiment logging. It provides a single import location for schema metadata,
log writing, log partitioning, structured log reading, and plot generation.

Examples
--------
Import logging utilities directly from the package:

>>> from loggers import TrainingLogger, plot_saved_training_logs
>>> from loggers import load_structured_csv
"""

from loggers.plotter import plot_saved_training_logs
from loggers.reader import load_structured_csv, load_structured_jsonl
from loggers.schema import SCHEMA_VERSION
from loggers.writer import (
    TrainingLogger,
    partition_epoch_logs,
    partition_obs_logs,
    save_partitioned_logs,
)


__all__ = [
    "SCHEMA_VERSION",
    "TrainingLogger",
    "partition_epoch_logs",
    "partition_obs_logs",
    "save_partitioned_logs",
    "load_structured_csv",
    "load_structured_jsonl",
    "plot_saved_training_logs",
]
"""list[str]: Public symbols exported when using ``from loggers import *``.

The exported names are:

- ``SCHEMA_VERSION``: Structured log schema version.
- ``TrainingLogger``: Container for train, epoch, and observation log streams.
- ``partition_epoch_logs``: Helper for splitting epoch logs into structured
  event groups.
- ``partition_obs_logs``: Helper for splitting observation logs into
  structured event groups.
- ``save_partitioned_logs``: Helper for saving raw and partitioned logs.
- ``load_structured_csv``: Helper for reading structured CSV artifacts.
- ``load_structured_jsonl``: Helper for reading structured JSONL artifacts.
- ``plot_saved_training_logs``: Helper for generating plots from structured
  logs.
"""