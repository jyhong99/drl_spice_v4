"""Local in-process replay-buffer implementations.

This module defines local replay-buffer classes used by the training stack.
The buffers store transitions in memory and return PyTorch tensors for
agent updates.

Two buffer types are provided:

- ``RolloutBuffer`` for on-policy algorithms.
- ``ReplayBuffer`` for off-policy algorithms.
"""

import torch
import numpy as np

from buffers.storage import BaseBuffer


class RolloutBuffer(BaseBuffer):
    """On-policy rollout buffer that returns the full collected trajectory.

    This buffer is intended for on-policy algorithms such as PPO. It stores
    transitions collected under the current policy and returns the entire
    rollout when sampled. After sampling, the buffer is reset.

    Parameters
    ----------
    state_dim : int
        Dimension of the state or observation vector.
    action_dim : int or numpy.ndarray
        Dimension of the action space. For continuous actions, this is usually
        an integer. For multi-discrete actions, this may be an array-like
        action-dimension descriptor.
    buffer_size : int
        Maximum number of transitions stored before sampling.
    device : str or torch.device
        Device on which sampled tensors are allocated.
    action_storage_shape : int or tuple[int, ...] or None, optional
        Shape used to store actions internally. If ``None``, the base buffer
        determines the storage shape from ``action_dim``. The default is
        ``None``.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size,
        device,
        action_storage_shape=None,
    ):
        """Initialize the rollout buffer.

        Parameters
        ----------
        state_dim : int
            Dimension of the state or observation vector.
        action_dim : int or numpy.ndarray
            Dimension or descriptor of the action space.
        buffer_size : int
            Maximum rollout length stored in the buffer.
        device : str or torch.device
            Device on which sampled tensors are created.
        action_storage_shape : int or tuple[int, ...] or None, optional
            Internal action-storage shape. The default is ``None``.

        Returns
        -------
        None
            The constructor initializes the underlying buffer storage in place.
        """

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=None,
            device=device,
            action_storage_shape=action_storage_shape,
        )

    def sample(self):
        """Return the full rollout as tensors and clear the buffer.

        This method converts all currently stored transitions into
        ``torch.FloatTensor`` objects on ``self.device``. After the rollout is
        sampled, the buffer is reset so that new on-policy data can be
        collected.

        Returns
        -------
        states : torch.Tensor
            Tensor of stored states with shape ``(size, state_dim)``.
        actions : torch.Tensor
            Tensor of stored actions with shape determined by the buffer's
            action-storage configuration.
        rewards : torch.Tensor
            Tensor of rewards with shape ``(size, 1)``.
        next_states : torch.Tensor
            Tensor of next states with shape ``(size, state_dim)``.
        dones : torch.Tensor
            Tensor of terminal flags with shape ``(size, 1)``.
        truncateds : torch.Tensor
            Tensor of truncation flags with shape ``(size, 1)``.
        """

        self._record_sample()
        size = self.size

        states = torch.FloatTensor(self.states[:size]).to(self.device)
        actions = torch.FloatTensor(self.actions[:size]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[:size]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[:size]).to(self.device)
        dones = torch.FloatTensor(self.dones[:size]).to(self.device)
        truncateds = torch.FloatTensor(self.truncateds[:size]).to(self.device)

        self.reset()

        return states, actions, rewards, next_states, dones, truncateds


class ReplayBuffer(BaseBuffer):
    """Uniform replay buffer for off-policy learning.

    This buffer is intended for off-policy algorithms such as DDPG, TD3, and
    SAC. It stores transitions in memory and returns uniformly sampled
    mini-batches for optimization.

    Parameters
    ----------
    state_dim : int
        Dimension of the state or observation vector.
    action_dim : int or numpy.ndarray
        Dimension of the action space. For continuous actions, this is usually
        an integer. For multi-discrete actions, this may be an array-like
        action-dimension descriptor.
    buffer_size : int
        Maximum number of transitions stored in the replay buffer.
    batch_size : int
        Number of transitions sampled per mini-batch.
    device : str or torch.device
        Device on which sampled tensors are allocated.
    action_storage_shape : int or tuple[int, ...] or None, optional
        Shape used to store actions internally. If ``None``, the base buffer
        determines the storage shape from ``action_dim``. The default is
        ``None``.
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
        """Initialize the replay buffer.

        Parameters
        ----------
        state_dim : int
            Dimension of the state or observation vector.
        action_dim : int or numpy.ndarray
            Dimension or descriptor of the action space.
        buffer_size : int
            Maximum number of transitions stored in the buffer.
        batch_size : int
            Number of transitions sampled per mini-batch.
        device : str or torch.device
            Device on which sampled tensors are created.
        action_storage_shape : int or tuple[int, ...] or None, optional
            Internal action-storage shape. The default is ``None``.

        Returns
        -------
        None
            The constructor initializes the underlying buffer storage in place.
        """

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            action_storage_shape=action_storage_shape,
        )

    def sample(self):
        """Sample a uniformly random mini-batch as tensors.

        This method randomly selects ``self.batch_size`` transition indices
        from the currently available replay storage and converts the selected
        transitions into ``torch.FloatTensor`` objects on ``self.device``.

        Returns
        -------
        states : torch.Tensor
            Tensor of sampled states with shape ``(batch_size, state_dim)``.
        actions : torch.Tensor
            Tensor of sampled actions with shape determined by the buffer's
            action-storage configuration.
        rewards : torch.Tensor
            Tensor of sampled rewards with shape ``(batch_size, 1)``.
        next_states : torch.Tensor
            Tensor of sampled next states with shape
            ``(batch_size, state_dim)``.
        dones : torch.Tensor
            Tensor of sampled terminal flags with shape ``(batch_size, 1)``.
        truncateds : torch.Tensor
            Tensor of sampled truncation flags with shape ``(batch_size, 1)``.
        """

        self._record_sample()

        idxs = np.random.randint(0, self.batch_ptr, size=self.batch_size)

        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        truncateds = torch.FloatTensor(self.truncateds[idxs]).to(self.device)

        return states, actions, rewards, next_states, dones, truncateds