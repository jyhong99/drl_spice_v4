"""Shared storage primitives and mixins for replay buffers.

This module defines common storage utilities used by local and Ray-remote
replay buffers. It provides shape helpers, usage-statistics tracking,
pickle-based persistence, and a base NumPy-backed transition buffer.
"""

import pickle
import random

import numpy as np


def combined_shape(length, shape=None):
    """Build a tuple containing a leading dimension and optional trailing shape.

    Parameters
    ----------
    length : int
        Leading dimension of the resulting shape. This is typically the buffer
        capacity or batch size.
    shape : int or tuple[int, ...] or None, optional
        Trailing element shape. If ``None``, the resulting shape is
        ``(length,)``. If an integer is provided, the resulting shape is
        ``(length, shape)``. If a tuple is provided, the resulting shape is
        ``(length, *shape)``. The default is ``None``.

    Returns
    -------
    tuple[int, ...]
        Combined shape tuple.
    """

    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class BufferStatsMixin:
    """Mixin that tracks basic buffer usage counters.

    This mixin provides counters for store operations, sample operations,
    total inserted transitions, and overwrite events. Classes using this
    mixin must define ``buffer_size`` and a ``size`` interface, either as a
    property or as a method.
    """

    def _init_stats(self):
        """Initialize buffer usage counters.

        Returns
        -------
        None
            Store, sample, transition, and overwrite counters are initialized
            in place.
        """

        self.total_store_calls = 0
        self.total_sample_calls = 0
        self.total_stored_transitions = 0
        self.total_overwrite_count = 0

    def _record_store(self):
        """Record one store operation.

        Returns
        -------
        None
            Store-related counters are incremented in place.
        """

        self.total_store_calls += 1
        self.total_stored_transitions += 1

    def _record_sample(self):
        """Record one sample operation.

        Returns
        -------
        None
            Sample-call counter is incremented in place.
        """

        self.total_sample_calls += 1

    def stats(self):
        """Return current buffer usage statistics.

        Returns
        -------
        dict[str, int or float]
            Dictionary containing the current valid buffer size, fill ratio,
            number of store calls, number of sample calls, total number of
            inserted transitions, and overwrite count.
        """

        size = (
            self.size
            if isinstance(getattr(type(self), "size", None), property)
            else self.size()
        )

        return {
            "size": int(size),
            "fill_ratio": float(size / max(self.buffer_size, 1)),
            "store_calls": int(self.total_store_calls),
            "sample_calls": int(self.total_sample_calls),
            "stored_transitions": int(self.total_stored_transitions),
            "overwrite_count": int(self.total_overwrite_count),
        }


class PicklePersistenceMixin:
    """Mixin that provides pickle-based save and load helpers.

    This mixin serializes an object's internal ``__dict__`` state using
    Python pickle. It is intended for lightweight checkpointing of replay
    buffers and related storage objects.
    """

    def _serialize_payload(self):
        """Build a serialization payload for the current object state.

        Returns
        -------
        dict[str, object]
            Dictionary containing the class name and a shallow copy of the
            object's internal attribute dictionary.
        """

        return {
            "class_name": type(self).__name__,
            "state": dict(self.__dict__),
        }

    def save(self, save_path):
        """Persist object state to disk.

        Parameters
        ----------
        save_path : str or path-like
            Destination file path where the pickle payload will be written.

        Returns
        -------
        None
            The current object state is written to ``save_path``.
        """

        with open(save_path, "wb") as f:
            pickle.dump(self._serialize_payload(), f)

    def load(self, load_path):
        """Restore object state from disk.

        Parameters
        ----------
        load_path : str or path-like
            Source file path containing a serialized pickle payload.

        Returns
        -------
        None
            Object attributes are restored in place.
        """

        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "state" in payload:
            self.__dict__.update(payload["state"])
            return

        self.__dict__.update(payload.__dict__)


class BaseBuffer(BufferStatsMixin, PicklePersistenceMixin):
    """Base in-process buffer implementation backed by NumPy arrays.

    This class stores transitions in fixed-size NumPy arrays. It supports
    circular insertion, transition overwrite counting, batch-size metadata,
    action reshaping, and pickle-based persistence. Concrete subclasses must
    implement the :meth:`sample` method.

    Parameters
    ----------
    state_dim : int or tuple[int, ...]
        Dimension or shape of stored observations.
    action_dim : int or tuple[int, ...]
        Dimension or shape of stored actions.
    buffer_size : int
        Maximum number of transitions stored in the buffer.
    batch_size : int or None
        Number of transitions returned by :meth:`sample`. On-policy rollout
        buffers may use ``None`` because they return the entire rollout.
    device : torch.device or str
        Target device used by concrete subclasses when converting sampled
        arrays into tensors.
    action_storage_shape : int or tuple[int, ...] or None, optional
        Explicit internal storage shape for actions. If ``None``, the action
        storage shape is inferred from ``action_dim``. The default is
        ``None``.

    Attributes
    ----------
    state_dim : int or tuple[int, ...]
        Stored observation dimension or shape.
    action_dim : int or tuple[int, ...]
        Action dimension or descriptor.
    action_storage_shape : int or tuple[int, ...]
        Internal action-storage shape.
    device : torch.device or str
        Device used for tensor conversion in concrete subclasses.
    batch_size : int or None
        Number of samples returned per batch.
    buffer_size : int
        Maximum buffer capacity.
    states : numpy.ndarray
        Array storing current observations.
    actions : numpy.ndarray
        Array storing actions.
    rewards : numpy.ndarray
        Array storing rewards.
    next_states : numpy.ndarray
        Array storing next observations.
    dones : numpy.ndarray
        Array storing terminal flags.
    truncateds : numpy.ndarray
        Array storing truncation flags.
    ptr : int
        Current circular insertion pointer.
    batch_ptr : int
        Number of valid transitions available for sampling, capped at
        ``buffer_size``.
    is_full : bool
        Whether the buffer has reached full capacity.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size,
        batch_size,
        device,
        action_storage_shape=None,
    ):
        """Initialize the base transition buffer.

        Parameters
        ----------
        state_dim : int or tuple[int, ...]
            Dimension or shape of stored observations.
        action_dim : int or tuple[int, ...]
            Dimension or shape of stored actions.
        buffer_size : int
            Maximum number of transitions stored in the buffer.
        batch_size : int or None
            Number of transitions returned per sampled batch.
        device : torch.device or str
            Target device used by concrete subclasses for tensor conversion.
        action_storage_shape : int or tuple[int, ...] or None, optional
            Explicit internal storage shape for actions. The default is
            ``None``.

        Returns
        -------
        None
            The constructor initializes metadata, statistics, and storage
            arrays in place.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_storage_shape = (
            action_dim if action_storage_shape is None else action_storage_shape
        )
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self._init_stats()
        self.reset()

    def store(self, state, action, reward, next_state, terminated, truncated=False):
        """Store a single transition.

        Parameters
        ----------
        state : array-like
            Observation before applying the action.
        action : array-like or int or float
            Action taken in the environment.
        reward : float
            Scalar reward observed after the transition.
        next_state : array-like
            Observation reached after applying the action.
        terminated : bool or float
            Terminal flag. ``True`` or ``1.0`` indicates that the episode ended
            due to a terminal state.
        truncated : bool or float, optional
            Truncation flag. ``True`` or ``1.0`` indicates that the episode
            ended due to an external limit such as a time limit. The default is
            ``False``.

        Returns
        -------
        None
            The transition is written into the current buffer slot in place.
        """

        if self.is_full:
            self.total_overwrite_count += 1

        self.states[self.ptr] = state
        self.actions[self.ptr] = self._format_action(action)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = terminated
        self.truncateds[self.ptr] = truncated

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.batch_ptr = min(self.batch_ptr + 1, self.buffer_size)

        if self.batch_ptr == self.buffer_size:
            self.is_full = True

        self._record_store()

    def sample(self, *args, **kwargs):
        """Sample one batch of transitions.

        Parameters
        ----------
        *args : tuple
            Subclass-specific positional arguments.
        **kwargs : dict[str, object]
            Subclass-specific keyword arguments.

        Returns
        -------
        object
            Subclass-defined sample payload.

        Raises
        ------
        NotImplementedError
            Always raised by the base class. Concrete subclasses must
            implement this method.
        """

        raise NotImplementedError()

    def store_many(self, transitions):
        """Store multiple transitions in insertion order.

        Parameters
        ----------
        transitions : iterable[tuple]
            Iterable of transition tuples. Each tuple is passed directly to
            :meth:`store`, so it should match the argument order accepted by
            ``store``.

        Returns
        -------
        None
            All transitions are stored in the order provided.
        """

        for transition in transitions:
            self.store(*transition)

    def reset(self):
        """Clear all stored transitions and reinitialize storage arrays.

        Returns
        -------
        None
            Storage arrays, insertion pointer, fullness flag, and valid-size
            counter are reset in place.
        """

        self.states = np.zeros(
            combined_shape(self.buffer_size, self.state_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            combined_shape(self.buffer_size, self.action_storage_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros(
            combined_shape(self.buffer_size, self.state_dim),
            dtype=np.float32,
        )
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.truncateds = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.ptr, self.is_full = 0, False
        self.batch_ptr = 0

    def _format_action(self, action):
        """Normalize an action into the configured storage shape.

        Parameters
        ----------
        action : array-like or int or float
            Raw action value to store.

        Returns
        -------
        numpy.ndarray
            Action array reshaped to ``self.action_storage_shape``.
        """

        action_array = np.asarray(action, dtype=np.float32)

        if np.isscalar(self.action_storage_shape):
            return action_array.reshape(int(self.action_storage_shape))

        return action_array.reshape(tuple(self.action_storage_shape))

    @property
    def size(self):
        """Return the number of valid transitions currently stored.

        Returns
        -------
        int
            Number of valid transitions in the buffer.
        """

        return self.buffer_size if self.is_full else self.ptr


class SharedBufferBase(BufferStatsMixin, PicklePersistenceMixin):
    """Base mixin for Ray-remote buffers with reproducible RNG state.

    This class provides random-number-generator initialization and loading
    behavior shared by Ray-remote buffer implementations.
    """

    def _set_rng(self, seed):
        """Initialize NumPy and Python random-number generators.

        Parameters
        ----------
        seed : int or None
            Seed used to initialize both NumPy and Python random-number
            generators. If ``None``, non-deterministic entropy is used.

        Returns
        -------
        None
            ``np_rng`` and ``py_rng`` are initialized in place.
        """

        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def load(self, load_path):
        """Restore remote buffer state and ensure RNG availability.

        Parameters
        ----------
        load_path : str or path-like
            Source file path containing a serialized buffer state.

        Returns
        -------
        None
            Buffer attributes are restored in place. If RNG attributes are
            missing from the loaded state, they are recreated from ``seed``.
        """

        super().load(load_path)

        if not hasattr(self, "np_rng"):
            self._set_rng(getattr(self, "seed", None))