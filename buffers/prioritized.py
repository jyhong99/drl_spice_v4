"""Prioritized replay-buffer implementations."""

import random

import numpy as np
import ray
import torch

from buffers.segment_tree import MinSegmentTree, SumSegmentTree
from buffers.storage import BaseBuffer, SharedBufferBase


class PrioritizedReplayBuffer(BaseBuffer):
    """In-process prioritized replay buffer.

    This buffer samples transitions according to their priorities rather than
    uniformly. It is intended for local off-policy learning with prioritized
    experience replay.

    Parameters
    ----------
    state_dim : int or tuple[int, ...]
        Dimension or shape of stored observations.
    action_dim : int or tuple[int, ...]
        Dimension or shape of stored actions.
    buffer_size : int
        Maximum number of transitions stored in the buffer.
    batch_size : int
        Number of transitions sampled per mini-batch.
    device : torch.device or str
        Device on which sampled tensors are allocated.
    alpha : float, optional
        Priority exponent controlling how strongly sampling depends on
        priorities. A value of ``0`` corresponds to uniform sampling. The
        default is ``0.6``.
    action_storage_shape : int or tuple[int, ...] or None, optional
        Explicit internal storage shape for actions. If ``None``, the base
        buffer determines the storage shape from ``action_dim``. The default
        is ``None``.

    Attributes
    ----------
    max_prio : float
        Maximum priority observed so far. New transitions are inserted with
        this priority when no explicit priority is available.
    tree_ptr : int
        Current segment-tree insertion pointer.
    alpha : float
        Priority exponent used when storing priorities.
    sum_tree : SumSegmentTree
        Segment tree storing priority sums for proportional sampling.
    min_tree : MinSegmentTree
        Segment tree storing minimum priorities for importance-weight
        normalization.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size,
        batch_size,
        device,
        alpha=0.6,
        action_storage_shape=None,
    ):
        """Initialize the prioritized replay buffer.

        Parameters
        ----------
        state_dim : int or tuple[int, ...]
            Dimension or shape of stored observations.
        action_dim : int or tuple[int, ...]
            Dimension or shape of stored actions.
        buffer_size : int
            Maximum number of transitions stored in the buffer.
        batch_size : int
            Number of transitions sampled per mini-batch.
        device : torch.device or str
            Device on which sampled tensors are allocated.
        alpha : float, optional
            Priority exponent controlling sampling skew. The default is
            ``0.6``.
        action_storage_shape : int or tuple[int, ...] or None, optional
            Explicit internal storage shape for actions. The default is
            ``None``.

        Returns
        -------
        None
            The constructor initializes base storage, segment trees, and
            priority metadata in place.
        """

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            action_storage_shape=action_storage_shape,
        )

        self.max_prio, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, state, action, reward, next_state, terminated, truncated=False):
        """Store a transition and initialize its priority.

        The transition is first stored through the base buffer. Its priority
        is then initialized to the current maximum priority so that newly
        inserted transitions are likely to be sampled at least once.

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
            The transition and its initial priority are stored in place.
        """

        super().store(state, action, reward, next_state, terminated, truncated)

        self.sum_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample(self, beta):
        """Sample a prioritized mini-batch.

        This method samples indices proportionally to stored priorities,
        converts the corresponding transitions to tensors, and computes
        normalized importance-sampling weights.

        Parameters
        ----------
        beta : float
            Importance-sampling correction exponent. Larger values apply
            stronger correction for the bias introduced by prioritized
            sampling.

        Returns
        -------
        states : torch.Tensor
            Tensor of sampled states with shape ``(batch_size, state_dim)``.
        actions : torch.Tensor
            Tensor of sampled actions with shape determined by the action
            storage configuration.
        rewards : torch.Tensor
            Tensor of sampled rewards with shape ``(batch_size, 1)``.
        next_states : torch.Tensor
            Tensor of sampled next states with shape
            ``(batch_size, state_dim)``.
        dones : torch.Tensor
            Tensor of sampled terminal flags with shape ``(batch_size, 1)``.
        truncateds : torch.Tensor
            Tensor of sampled truncation flags with shape ``(batch_size, 1)``.
        weights : torch.Tensor
            Importance-sampling weights with shape ``(batch_size, 1)``.
        idxs : list[int]
            Indices sampled from the replay buffer. These indices are used
            later to update priorities.
        """

        idxs = self._sample_proportional()

        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        truncateds = torch.FloatTensor(self.truncateds[idxs]).to(self.device)
        weights = torch.FloatTensor(
            [self._calculate_weight(i, beta) for i in idxs]
        ).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones, truncateds, weights, idxs

    def update_priorities(self, idxs, prios):
        """Update priorities for sampled transitions.

        Parameters
        ----------
        idxs : iterable[int]
            Buffer indices whose priorities should be updated.
        prios : iterable[float]
            New positive priority values corresponding to ``idxs``.

        Returns
        -------
        None
            The sum tree, min tree, and maximum priority are updated in place.
        """

        for idx, prio in zip(idxs, prios):
            self.sum_tree[idx] = prio ** self.alpha
            self.min_tree[idx] = prio ** self.alpha
            self.max_prio = max(self.max_prio, prio)

    def _sample_proportional(self):
        """Draw indices proportionally to stored priorities.

        The total priority mass is divided into ``batch_size`` equal segments.
        One random value is sampled from each segment, and the corresponding
        index is retrieved from the sum segment tree.

        Returns
        -------
        list[int]
            List of sampled replay-buffer indices with length
            ``self.batch_size``.
        """

        idxs = []
        p_total = self.sum_tree.sum(0, self.batch_ptr - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idxs.append(idx)

        return idxs

    def _calculate_weight(self, idx, beta):
        """Compute the importance-sampling weight for one index.

        Parameters
        ----------
        idx : int
            Replay-buffer index whose importance-sampling weight should be
            computed.
        beta : float
            Importance-sampling correction exponent.

        Returns
        -------
        float
            Normalized importance-sampling weight for the given index.
        """

        total = self.sum_tree.sum(0, self.batch_ptr - 1)
        p_min = self.min_tree.min(0, self.batch_ptr - 1) / total
        max_weight = (p_min * self.batch_ptr) ** (-beta)
        p_sample = self.sum_tree[idx] / total
        weight = (p_sample * self.batch_ptr) ** (-beta)
        weight = weight / max_weight
        return weight


@ray.remote
class SharedPrioritizedReplayBuffer(SharedBufferBase):
    """Ray-remote prioritized replay buffer for distributed training.

    This class stores transitions in a Ray actor so that multiple workers can
    share a prioritized replay buffer during distributed off-policy training.

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
    device : torch.device or str
        Device argument kept for interface compatibility. Shared buffers
        return NumPy arrays rather than device tensors.
    alpha : float
        Priority exponent controlling how strongly sampling depends on
        priorities.
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
    max_prio : float
        Maximum priority observed so far.
    tree_ptr : int
        Current segment-tree insertion pointer.
    alpha : float
        Priority exponent used when storing priorities.
    sum_tree : SumSegmentTree
        Segment tree storing priority sums for proportional sampling.
    min_tree : MinSegmentTree
        Segment tree storing minimum priorities for importance-weight
        normalization.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size,
        batch_size,
        device,
        alpha,
        seed=None,
        action_storage_shape=None,
    ):
        """Initialize the shared prioritized replay buffer.

        Parameters
        ----------
        state_dim : int or tuple[int, ...]
            Dimension or shape of stored observations.
        action_dim : int or tuple[int, ...]
            Dimension or shape of stored actions.
        buffer_size : int
            Maximum number of transitions stored in the buffer.
        batch_size : int
            Number of transitions sampled per mini-batch.
        device : torch.device or str
            Device argument kept for interface compatibility.
        alpha : float
            Priority exponent controlling sampling skew.
        seed : int or None, optional
            Random seed used for reproducible sampling. The default is
            ``None``.
        action_storage_shape : int or tuple[int, ...] or None, optional
            Explicit internal storage shape for actions. The default is
            ``None``.

        Returns
        -------
        None
            The constructor initializes storage arrays, segment trees, random
            state, and usage statistics in place.
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

        self.max_prio, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        self.reset()

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated=False,
        priority=None,
    ):
        """Store one transition and its priority.

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
        priority : float or None, optional
            Initial transition priority. If ``None``, the current maximum
            priority is used. The default is ``None``.

        Returns
        -------
        None
            The transition and priority are stored in place.
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

        if priority is None:
            priority = self.max_prio

        self.sum_tree[self.tree_ptr] = priority ** self.alpha
        self.min_tree[self.tree_ptr] = priority ** self.alpha
        self.max_prio = max(self.max_prio, priority)
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample(self, beta):
        """Sample a prioritized mini-batch as NumPy arrays.

        Parameters
        ----------
        beta : float
            Importance-sampling correction exponent.

        Returns
        -------
        states : numpy.ndarray
            Array of sampled states with shape ``(batch_size, state_dim)``.
        actions : numpy.ndarray
            Array of sampled actions with shape determined by the action
            storage configuration.
        rewards : numpy.ndarray
            Array of sampled rewards with shape ``(batch_size, 1)``.
        next_states : numpy.ndarray
            Array of sampled next states with shape
            ``(batch_size, state_dim)``.
        dones : numpy.ndarray
            Array of sampled terminal flags with shape ``(batch_size, 1)``.
        truncateds : numpy.ndarray
            Array of sampled truncation flags with shape ``(batch_size, 1)``.
        weights : numpy.ndarray
            Importance-sampling weights with shape ``(batch_size, 1)``.
        idxs : numpy.ndarray
            Sampled replay-buffer indices with shape ``(batch_size,)``.
        """

        self._record_sample()

        idxs = self._sample_proportional()

        states = self.states[idxs].copy()
        actions = self.actions[idxs].copy()
        rewards = self.rewards[idxs].copy()
        next_states = self.next_states[idxs].copy()
        dones = self.dones[idxs].copy()
        truncateds = self.truncateds[idxs].copy()
        weights = np.asarray(
            [self._calculate_weight(i, beta) for i in idxs],
            dtype=np.float32,
        ).reshape(-1, 1)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            truncateds,
            weights,
            np.asarray(idxs, dtype=np.int64),
        )

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

        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(
            (self.buffer_size, self.action_storage_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros(
            (self.buffer_size, self.state_dim),
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

    def update_priorities(self, idxs, prios):
        """Update priorities for sampled transitions.

        Parameters
        ----------
        idxs : iterable[int]
            Buffer indices whose priorities should be updated.
        prios : iterable[float]
            New positive priority values corresponding to ``idxs``.

        Returns
        -------
        None
            The sum tree, min tree, and maximum priority are updated in place.
        """

        for idx, prio in zip(idxs, prios):
            self.sum_tree[idx] = prio ** self.alpha
            self.min_tree[idx] = prio ** self.alpha
            self.max_prio = max(self.max_prio, prio)

    def size(self):
        """Return the number of valid transitions currently stored.

        Returns
        -------
        int
            Number of valid transitions in the buffer.
        """

        return self.buffer_size if self.is_full else self.ptr

    def stats(self):
        """Return current usage statistics.

        Returns
        -------
        dict[str, object]
            Buffer statistics returned by the base class, augmented with
            ``"max_priority"``.
        """

        stats = super().stats()
        stats["max_priority"] = float(self.max_prio)
        return stats

    def _sample_proportional(self):
        """Draw indices proportionally to stored priorities.

        Returns
        -------
        list[int]
            List of sampled replay-buffer indices with length
            ``self.batch_size``.
        """

        idxs = []
        p_total = self.sum_tree.sum(0, self.batch_ptr - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = self.np_rng.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idxs.append(idx)

        return idxs

    def _calculate_weight(self, idx, beta):
        """Compute the importance-sampling weight for one index.

        Parameters
        ----------
        idx : int
            Replay-buffer index whose importance-sampling weight should be
            computed.
        beta : float
            Importance-sampling correction exponent.

        Returns
        -------
        float
            Normalized importance-sampling weight for the given index.
        """

        total = self.sum_tree.sum(0, self.batch_ptr - 1)
        p_min = self.min_tree.min(0, self.batch_ptr - 1) / total
        max_weight = (p_min * self.batch_ptr) ** (-beta)
        p_sample = self.sum_tree[idx] / total
        weight = (p_sample * self.batch_ptr) ** (-beta)
        weight = weight / max_weight
        return weight