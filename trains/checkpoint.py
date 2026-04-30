"""Checkpoint path, loading, and saving helpers for training runs.

This module provides small utility functions for constructing checkpoint
paths and persisting or restoring learner and replay-buffer state during
training.
"""

import os

import ray


def get_checkpoint_paths(project_name):
    """Return standard checkpoint paths for a project.

    Parameters
    ----------
    project_name : str
        Experiment project name. Checkpoints are stored under
        ``./log/{project_name}``.

    Returns
    -------
    dict[str, str]
        Dictionary containing checkpoint paths.

        ``"root"`` : str
            Root checkpoint directory.
        ``"model"`` : str
            Path to the learner model checkpoint file.
        ``"buffer"`` : str
            Path to the replay-buffer checkpoint file.
    """

    path = f"./log/{project_name}"
    os.makedirs(path, exist_ok=True)

    return {
        "root": path,
        "model": os.path.join(path, "model.pth"),
        "buffer": os.path.join(path, "buffer.pkl"),
    }


def load_learner_checkpoint(learner, load_path):
    """Load a learner checkpoint from a run directory.

    Parameters
    ----------
    learner : object
        Learner or agent object exposing a ``load`` method.
    load_path : str or os.PathLike or None
        Directory containing ``model.pth``. If falsy, loading is skipped.

    Returns
    -------
    str or None
        Path to the loaded model checkpoint if loading was attempted;
        otherwise ``None``.
    """

    if not load_path:
        return None

    model_path = os.path.join(load_path, "model.pth")
    learner.load(model_path)

    return model_path


def load_buffer_checkpoint(buffer_actor, load_path):
    """Load a replay-buffer checkpoint into a Ray buffer actor.

    Parameters
    ----------
    buffer_actor : ray.actor.ActorHandle
        Remote buffer actor exposing a ``load.remote`` method.
    load_path : str or os.PathLike or None
        Directory containing ``buffer.pkl``. If falsy or if the file does not
        exist, loading is skipped.

    Returns
    -------
    str or None
        Path to the loaded buffer checkpoint if available; otherwise ``None``.
    """

    if not load_path:
        return None

    buffer_path = os.path.join(load_path, "buffer.pkl")

    if not os.path.exists(buffer_path):
        return None

    ray.get(buffer_actor.load.remote(buffer_path))

    return buffer_path


def save_learner_checkpoint(learner, model_path):
    """Save a learner checkpoint.

    Parameters
    ----------
    learner : object
        Learner or agent object exposing a ``save`` method.
    model_path : str or os.PathLike
        Destination path for the model checkpoint.

    Returns
    -------
    None
        The learner checkpoint is written to ``model_path``.
    """

    learner.save(model_path)


def save_buffer_checkpoint(buffer_actor, buffer_path, wait=False):
    """Save a replay-buffer checkpoint through a Ray buffer actor.

    Parameters
    ----------
    buffer_actor : ray.actor.ActorHandle
        Remote buffer actor exposing a ``save.remote`` method.
    buffer_path : str or os.PathLike
        Destination path for the buffer checkpoint.
    wait : bool, optional
        Whether to block until the remote save operation completes. If
        ``False``, the Ray object reference is returned immediately. The
        default is ``False``.

    Returns
    -------
    ray.ObjectRef
        Ray object reference for the remote save operation.
    """

    ref = buffer_actor.save.remote(buffer_path)

    if wait:
        ray.get(ref)

    return ref