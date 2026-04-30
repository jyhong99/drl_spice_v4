"""Ray-remote buffer implementations used in distributed training.

This module defines Ray actor-based rollout and replay buffers for
distributed reinforcement-learning workflows. The shared buffers store
transitions as NumPy arrays and can be accessed remotely by rollout workers,
trainers, or runtime managers.
"""

import numpy as np
import ray

from buffers.storage import SharedBufferBase, combined_shape


@ray.remote
class SharedRolloutBuffer(SharedBufferBase):
    """Ray-remote rollout buffer for on-policy distributed collection.

    This buffer stores transitions collected by distributed workers and
    returns the full collected rollout as NumPy arrays. After sampling, the
    buffer is reset so that new on-policy trajectories can be collected.

    Parameters
    ----------
    state_dim : int or tuple[int, ...]
        Dimension or shape of stored observations.
    action_dim : int or tuple[int, ...]
        Dimension or shape of stored actions.
    buffer_size : int
        Maximum number of transitions stored in the rollout buffer.
    device : str or torch.device
        Device argument kept for interface compatibility. This shared buffer
        stores and returns NumPy arrays rather than device tensors.
    seed : int or None, optional
        Random seed used by the shared buffer. The default is ``None``.
    action_storage_shape : int or tuple[int, ...] or None, optional
        Explicit internal storage shape for actions. If ``None``, the action
        storage shape is inferred from ``action_dim``. The default is
        ``None``.

    Attributes
    ----------
    state_dim : int or tuple[int, ...]
        Stored observation dimension or shape.
    action_dim : int or tuple[int, ...]
        Stored action dimension or shape.
    action_storage_shape : int or tuple[int, ...]
        Internal action-storage shape.
    buffer_size : int
        Maximum rollout-buffer capacity.
    seed : int or None
        Random seed used by the buffer.
    states : numpy.ndarray
        Array used to store observations.
    actions : numpy.ndarray
        Array used to store actions.
    rewards : numpy.ndarray
        Array used to store rewards.
    next_states : numpy.ndarray
        Array used to store next observations.
    dones : numpy.ndarray
        Array used to store terminal flags.
    truncateds : numpy.ndarray
        Array used to store truncation flags.
    ptr : int
        Current insertion pointer.
    is_full : bool
        Whether the buffer has wrapped around at least once.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size,
        device,
        seed=None,
        action_storage_shape=None,
    ):
        """Initialize the shared rollout buffer.

        Parameters
        ----------
        state_dim : int or tuple[int, ...]
            Dimension or shape of stored observations.
        action_dim : int or tuple[int, ...]
            Dimension or shape of stored actions.
        buffer_size : int
            Maximum number of transitions stored in the rollout buffer.
        device : str or torch.device
            Device argument kept for interface compatibility.
        seed : int or None, optional
            Random seed used by the buffer. The default is ``None``.
        action_storage_shape : int or tuple[int, ...] or None, optional
            Explicit internal storage shape for actions. The default is
            ``None``.

        Returns
        -------
        None
            The constructor initializes metadata, random state, usage
            statistics, and storage arrays in place.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_storage_shape = (
            action_dim if action_storage_shape is None else action_storage_shape
        )
        self.buffer_size = buffer_size
        self.seed = seed

        self._init_stats()
        self._set_rng(seed)
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

        self.ptr += 1

        if self.ptr == self.buffer_size:
            self.ptr = 0
            self.is_full = True

        self._record_store()

    def sample(self):
        """Return the full rollout as NumPy arrays and clear the buffer.

        The sampled arrays are copied before returning so that later buffer
        writes do not mutate the returned rollout batch.

        Returns
        -------
        states : numpy.ndarray
            Array of stored states with shape determined by
            ``combined_shape(size, state_dim)``.
        actions : numpy.ndarray
            Array of stored actions with shape determined by
            ``combined_shape(size, action_storage_shape)``.
        rewards : numpy.ndarray
            Array of rewards with shape ``(size, 1)``.
        next_states : numpy.ndarray
            Array of next states with shape determined by
            ``combined_shape(size, state_dim)``.
        dones : numpy.ndarray
            Array of terminal flags with shape ``(size, 1)``.
        truncateds : numpy.ndarray
            Array of truncation flags with shape ``(size, 1)``.
        """

        self._record_sample()

        size = self.size()

        states = self.states[:size].copy()
        actions = self.actions[:size].copy()
        rewards = self.rewards[:size].copy()
        next_states = self.next_states[:size].copy()
        dones = self.dones[:size].copy()
        truncateds = self.truncateds[:size].copy()

        self.reset()

        return states, actions, rewards, next_states, dones, truncateds

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
            Storage arrays, insertion pointer, and fullness flag are reset in
            place.
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

    def size(self):
        """Return the number of valid transitions currently stored.

        Returns
        -------
        int
            Number of valid transitions in the buffer.
        """

        return self.buffer_size if self.is_full else self.ptr

    def load(self, load_path):
        """Restore buffer state from disk and ensure RNG availability.

        Parameters
        ----------
        load_path : str or pathlib.Path
            File path of the serialized buffer state.

        Returns
        -------
        None
            Buffer state is restored in place. If the RNG is missing from the
            loaded object, it is recreated using the stored seed.
        """

        super().load(load_path)

        if not hasattr(self, "np_rng"):
            self._set_rng(getattr(self, "seed", None))

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


@ray.remote
class SharedReplayBuffer(SharedBufferBase):
    """Ray-remote uniform replay buffer for off-policy distributed training.

    This buffer stores transitions in a Ray actor and returns uniformly
    sampled mini-batches as NumPy arrays. It is intended for distributed
    off-policy algorithms such as DDPG, TD3, and SAC.

    Parameters
    ----------
    state_dim : int or tuple[int, ...]
        Dimension or shape of stored observations.
    action_dim : int or tuple[int, ...]
        Dimension or shape of stored actions.
    buffer_size : int
        Maximum number of transitions stored in the replay buffer.
    batch_size : int
        Number of transitions sampled per mini-batch.
    device : str or torch.device
        Device argument kept for interface compatibility. This shared buffer
        stores and returns NumPy arrays rather than device tensors.
    seed : int or None, optional
        Random seed used by the shared buffer. The default is ``None``.
    action_storage_shape : int or tuple[int, ...] or None, optional
        Explicit internal storage shape for actions. If ``None``, the action
        storage shape is inferred from ``action_dim``. The default is
        ``None``.

    Attributes
    ----------
    state_dim : int or tuple[int, ...]
        Stored observation dimension or shape.
    action_dim : int or tuple[int, ...]
        Stored action dimension or shape.
    action_storage_shape : int or tuple[int, ...]
        Internal action-storage shape.
    batch_size : int
        Number of transitions sampled per mini-batch.
    buffer_size : int
        Maximum replay-buffer capacity.
    seed : int or None
        Random seed used by the buffer.
    states : numpy.ndarray
        Array used to store observations.
    actions : numpy.ndarray
        Array used to store actions.
    rewards : numpy.ndarray
        Array used to store rewards.
    next_states : numpy.ndarray
        Array used to store next observations.
    dones : numpy.ndarray
        Array used to store terminal flags.
    truncateds : numpy.ndarray
        Array used to store truncation flags.
    ptr : int
        Current insertion pointer.
    batch_ptr : int
        Number of valid transitions available for sampling, capped at
        ``buffer_size``.
    is_full : bool
        Whether the replay buffer has reached full capacity.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size,
        batch_size,
        device,
        seed=None,
        action_storage_shape=None,
    ):
        """Initialize the shared replay buffer.

        Parameters
        ----------
        state_dim : int or tuple[int, ...]
            Dimension or shape of stored observations.
        action_dim : int or tuple[int, ...]
            Dimension or shape of stored actions.
        buffer_size : int
            Maximum number of transitions stored in the replay buffer.
        batch_size : int
            Number of transitions sampled per mini-batch.
        device : str or torch.device
            Device argument kept for interface compatibility.
        seed : int or None, optional
            Random seed used by the buffer. The default is ``None``.
        action_storage_shape : int or tuple[int, ...] or None, optional
            Explicit internal storage shape for actions. The default is
            ``None``.

        Returns
        -------
        None
            The constructor initializes metadata, random state, usage
            statistics, and storage arrays in place.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_storage_shape = (
            action_dim if action_storage_shape is None else action_storage_shape
        )
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed

        self._init_stats()
        self._set_rng(seed)
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

    def sample(self):
        """Sample a uniformly random mini-batch as NumPy arrays.

        The sampled arrays are copied before returning so that later buffer
        writes do not mutate the returned mini-batch.

        Returns
        -------
        states : numpy.ndarray
            Array of sampled states with shape determined by
            ``combined_shape(batch_size, state_dim)``.
        actions : numpy.ndarray
            Array of sampled actions with shape determined by
            ``combined_shape(batch_size, action_storage_shape)``.
        rewards : numpy.ndarray
            Array of sampled rewards with shape ``(batch_size, 1)``.
        next_states : numpy.ndarray
            Array of sampled next states with shape determined by
            ``combined_shape(batch_size, state_dim)``.
        dones : numpy.ndarray
            Array of sampled terminal flags with shape ``(batch_size, 1)``.
        truncateds : numpy.ndarray
            Array of sampled truncation flags with shape ``(batch_size, 1)``.
        """

        self._record_sample()

        idxs = self.np_rng.integers(0, self.batch_ptr, size=self.batch_size)

        states = self.states[idxs].copy()
        actions = self.actions[idxs].copy()
        rewards = self.rewards[idxs].copy()
        next_states = self.next_states[idxs].copy()
        dones = self.dones[idxs].copy()
        truncateds = self.truncateds[idxs].copy()

        return states, actions, rewards, next_states, dones, truncateds

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

    def size(self):
        """Return the number of valid transitions currently stored.

        Returns
        -------
        int
            Number of valid transitions in the buffer.
        """

        return self.buffer_size if self.is_full else self.ptr

    def load(self, load_path):
        """Restore buffer state from disk and ensure RNG availability.

        Parameters
        ----------
        load_path : str or pathlib.Path
            File path of the serialized buffer state.

        Returns
        -------
        None
            Buffer state is restored in place. If the RNG is missing from the
            loaded object, it is recreated using the stored seed.
        """

        super().load(load_path)

        if not hasattr(self, "np_rng"):
            self._set_rng(getattr(self, "seed", None))

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