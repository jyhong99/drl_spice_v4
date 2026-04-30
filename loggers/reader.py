"""Helpers for reading structured log artifacts back into memory.

This module provides lightweight readers for structured CSV and JSONL log
files produced during training. Missing or empty log files are handled safely
by returning empty containers.
"""

import json
from pathlib import Path

import pandas as pd


def load_structured_csv(project_name: str, dataset_name: str) -> pd.DataFrame:
    """Load one structured CSV dataset for a project.

    Parameters
    ----------
    project_name : str
        Experiment project name. The function looks for logs under
        ``./log/{project_name}/structured``.
    dataset_name : str
        Dataset stem inside the ``structured`` directory, without the
        ``.csv`` suffix.

    Returns
    -------
    pandas.DataFrame
        Loaded CSV dataset. If the file does not exist or is empty, an empty
        ``pandas.DataFrame`` is returned.
    """

    path = Path(f"./log/{project_name}/structured/{dataset_name}.csv")

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    return pd.read_csv(path)


def load_structured_jsonl(project_name: str, dataset_name: str):
    """Load one structured JSONL dataset for a project.

    Parameters
    ----------
    project_name : str
        Experiment project name. The function looks for logs under
        ``./log/{project_name}/structured``.
    dataset_name : str
        Dataset stem inside the ``structured`` directory, without the
        ``.jsonl`` suffix.

    Returns
    -------
    list[dict]
        List of parsed JSON objects. If the file does not exist or is empty,
        an empty list is returned.
    """

    path = Path(f"./log/{project_name}/structured/{dataset_name}.jsonl")

    if not path.exists() or path.stat().st_size == 0:
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]