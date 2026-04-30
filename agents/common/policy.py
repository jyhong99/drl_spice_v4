"""Shared policy abstractions and serialization helpers for agents.

This module provides common base classes and utility functions shared by
on-policy and off-policy reinforcement-learning algorithms.

The utilities mainly support checkpoint serialization, CPU-safe deep copies
for distributed workers, rollout/replay runtime integration, policy-state
synchronization, and common optimization diagnostics.
"""

import gym
import torch
import numpy as np
from copy import deepcopy
from trains.runtime import (
    create_on_policy_runtime,
    create_off_policy_runtime,
)


def to_cpu_serializable(value):
    """Recursively convert tensors in a nested object to CPU tensors.

    This function creates a serialization-safe copy of a possibly nested
    object. Any ``torch.Tensor`` found inside dictionaries, lists, or tuples
    is detached from the computation graph, moved to CPU, and cloned. Other
    non-container Python objects are deep-copied.

    This is useful when saving checkpoints or sending model states to remote
    workers that should not depend on GPU memory.

    Parameters
    ----------
    value : Any
        Arbitrary Python object to convert. Supported nested containers
        include ``dict``, ``list``, and ``tuple``. Tensor values inside these
        containers are converted to detached CPU clones.

    Returns
    -------
    Any
        CPU-safe serialized copy of ``value``. Tensor values are returned as
        detached CPU clones, containers are recursively copied, and other
        objects are deep-copied.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_serializable(item) for item in value)
    return deepcopy(value)


def make_cpu_safe(value, memo=None):
    """Normalize an object graph so it can be deserialized on CPU workers.

    This function recursively walks through an object graph and moves tensors,
    devices, and PyTorch modules to CPU. Unlike :func:`to_cpu_serializable`,
    this function may modify mutable containers and object attributes in
    place.

    A memoization dictionary is used to preserve shared references and avoid
    infinite recursion when the object graph contains cycles.

    Parameters
    ----------
    value : Any
        Object graph to normalize. This may include tensors, modules, devices,
        dictionaries, lists, tuples, or user-defined objects with a
        ``__dict__`` attribute.
    memo : dict[int, Any] or None, optional
        Mapping from object identity ``id(value)`` to the already-converted
        object. This is used internally to preserve shared references and
        prevent repeated conversion of cyclic object graphs. The default is
        ``None``.

    Returns
    -------
    Any
        CPU-normalized object graph. Tensors are detached, cloned, and moved
        to CPU. Non-CPU ``torch.device`` objects are converted to
        ``torch.device("cpu")``. PyTorch modules are moved to CPU in place.

    Notes
    -----
    This function may mutate mutable objects in place. In this module, it is
    called after ``deepcopy`` in ``make_remote_copy`` so that the original
    agent instance is not modified.
    """

    if memo is None:
        memo = {}

    immutable_types = (str, bytes, int, float, bool, type(None))
    if isinstance(value, immutable_types):
        return value

    object_id = id(value)
    if object_id in memo:
        return memo[object_id]

    if isinstance(value, torch.Tensor):
        converted = value.detach().cpu().clone()
        converted.requires_grad_(value.requires_grad)
        memo[object_id] = converted
        return converted

    if isinstance(value, torch.device):
        converted = torch.device("cpu") if value.type != "cpu" else value
        memo[object_id] = converted
        return converted

    if isinstance(value, dict):
        memo[object_id] = value
        for key, item in list(value.items()):
            value[key] = make_cpu_safe(item, memo)
        return value

    if isinstance(value, list):
        memo[object_id] = value
        for index, item in enumerate(value):
            value[index] = make_cpu_safe(item, memo)
        return value

    if isinstance(value, tuple):
        converted = tuple(make_cpu_safe(item, memo) for item in value)
        memo[object_id] = converted
        return converted

    if isinstance(value, torch.nn.Module):
        memo[object_id] = value
        value.to(torch.device("cpu"))

    if hasattr(value, "__dict__"):
        memo[object_id] = value
        for attr_name, attr_value in vars(value).items():
            converted = make_cpu_safe(attr_value, memo)
            try:
                setattr(value, attr_name, converted)
            except (AttributeError, TypeError):
                # Some deep-copied objects in the environment graph are frozen
                # dataclasses. Fall back to object.__setattr__ so Ray worker
                # copies can still be CPU-normalized without mutating the
                # original instance.
                object.__setattr__(value, attr_name, converted)
        return value

    memo[object_id] = value
    return value


class OnPolicyAlgorithm(object):
    """Base class for on-policy reinforcement-learning algorithms.

    This class stores common metadata, hyperparameters, runtime objects, and
    utility methods shared by on-policy agents such as PPO or A2C-style
    methods. It infers observation and action dimensions from the provided
    Gym environment and delegates rollout collection and training execution
    to the runtime layer.

    Parameters
    ----------
    env : gym.Env
        Environment used for training and for inferring observation and
        action-space dimensions.
    actor_size : tuple[int, ...]
        Hidden-layer sizes of the actor network.
    critic_size : tuple[int, ...]
        Hidden-layer sizes of the critic or value network.
    actor_activation : type[torch.nn.Module] or callable
        Activation function used in the actor network.
    critic_activation : type[torch.nn.Module] or callable
        Activation function used in the critic network.
    buffer_size : int
        Maximum number of transitions stored in the rollout buffer.
    update_after : int
        Number of collected transitions required before a policy update is
        allowed.
    actor_lr : float
        Learning rate for the actor optimizer.
    critic_lr : float
        Learning rate for the critic optimizer.
    gamma : float
        Discount factor for future rewards.
    lmda : float
        Generalized advantage estimation coefficient.
    vf_coef : float
        Weight applied to the value-function loss term.
    ent_coef : float
        Weight applied to the entropy regularization term.
    adv_norm : bool
        Whether computed advantages should be normalized before optimization.

    Attributes
    ----------
    env : gym.Env
        Training environment.
    state_dim : int
        Dimension of the observation vector.
    action_dim : int or numpy.ndarray
        Action-space dimension. For ``Box`` spaces this is the number of
        continuous action dimensions. For ``Discrete`` spaces this is the
        number of discrete actions. For ``MultiDiscrete`` spaces this is the
        ``nvec`` array.
    action_storage_shape : int
        Number of values stored per action in the rollout buffer.
    action_type : {"continuous", "discrete", "multidiscrete"}
        Type of action space inferred from the environment.
    device : torch.device
        Device used for model computation.
    timesteps : int
        Number of environment steps collected so far.
    epsilon : float
        Small numerical constant used to avoid division by zero.
    last_action_info : dict[str, object]
        Dictionary storing diagnostic information about the most recent
        action selection.
    actor_size : tuple[int, ...]
        Hidden-layer sizes of the actor network.
    critic_size : tuple[int, ...]
        Hidden-layer sizes of the critic network.
    buffer_size : int
        Rollout-buffer capacity.
    update_after : int
        Minimum number of transitions required before optimization.
    actor_lr : float
        Actor optimizer learning rate.
    critic_lr : float
        Critic optimizer learning rate.
    gamma : float
        Discount factor.
    lmda : float
        Generalized advantage estimation coefficient.
    vf_coef : float
        Value-function loss coefficient.
    ent_coef : float
        Entropy regularization coefficient.
    adv_norm : bool
        Advantage normalization flag.
    """

    def __init__(
        self,
        env,
        actor_size,
        critic_size,
        actor_activation,
        critic_activation,
        buffer_size,
        update_after,
        actor_lr,
        critic_lr,
        gamma,
        lmda,
        vf_coef,
        ent_coef,
        adv_norm,
    ):
        """Initialize common on-policy agent state.

        Parameters
        ----------
        env : gym.Env
            Environment instance used to infer observation and action shapes.
        actor_size : tuple[int, ...]
            Hidden-layer widths for the actor network.
        critic_size : tuple[int, ...]
            Hidden-layer widths for the critic network.
        actor_activation : type[torch.nn.Module] or callable
            Activation applied inside actor hidden layers.
        critic_activation : type[torch.nn.Module] or callable
            Activation applied inside critic hidden layers.
        buffer_size : int
            Rollout-buffer capacity in transitions.
        update_after : int
            Minimum number of collected transitions before optimization.
        actor_lr : float
            Learning rate for the actor optimizer.
        critic_lr : float
            Learning rate for the critic optimizer.
        gamma : float
            Discount factor used in return computation.
        lmda : float
            Lambda parameter used by generalized advantage estimation.
        vf_coef : float
            Weight applied to the critic loss term.
        ent_coef : float
            Weight applied to the entropy regularization term.
        adv_norm : bool
            Whether to normalize computed advantages before updates.

        Returns
        -------
        None
            This constructor populates shared agent metadata and runtime
            state.
        """

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
            self.action_storage_shape = self.action_dim
            self.action_type = "continuous"
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.action_storage_shape = 1
            self.action_type = "discrete"
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.action_type = "multidiscrete"
            self.action_dim = self.env.action_space.nvec
            self.action_storage_shape = len(self.action_dim)
        else:
            raise TypeError(
                f"Unsupported action space type: "
                f"{type(self.env.action_space).__name__}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = 0
        self.epsilon = 1e-8
        self.last_action_info = {}

        self.actor_size = actor_size
        self.critic_size = critic_size
        self.actor_activation = actor_activation
        self.critic_activation = critic_activation
        self.buffer_size = buffer_size
        self.update_after = update_after

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lmda = lmda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.adv_norm = adv_norm

        self._runtime = create_on_policy_runtime(
            self,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_storage_shape=self.action_storage_shape,
            buffer_size=self.buffer_size,
            device=self.device,
        )

    @property
    def buffer(self):
        """Return the rollout buffer managed by the runtime.

        Returns
        -------
        object
            Runtime-managed rollout buffer instance.
        """

        return self._runtime.buffer

    @property
    def trainer(self):
        """Return the active trainer instance.

        Returns
        -------
        object or None
            Trainer attached to the runtime. Returns ``None`` if training has
            not been launched or if the runtime has not attached a trainer.
        """

        return self._runtime.trainer

    @torch.no_grad()
    def act(self, *args, **kwargs):
        """Produce an action using the current policy.

        This is an abstract method and must be implemented by each concrete
        on-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs, usually including the
            current state or observation.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs used during action selection.

        Returns
        -------
        object
            Action or action-related payload defined by the subclass.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()

    def GAE(self, values, next_values, rewards, dones):
        """Compute generalized advantage estimates and return targets.

        This method computes temporal-difference residuals and applies the
        generalized advantage estimation recursion. The returned targets are
        computed as

        ``returns = advantages + values``.

        If ``self.adv_norm`` is ``True``, the advantages are normalized using
        their batch mean and standard deviation.

        Parameters
        ----------
        values : torch.Tensor
            Value estimates for the current states. Expected shape is
            ``(T, 1)`` or ``(T, batch_size, 1)``, depending on how rollouts are
            stored.
        next_values : torch.Tensor
            Value estimates for the next states. Must have the same shape as
            ``values``.
        rewards : torch.Tensor
            Reward tensor for each transition. Must be broadcast-compatible
            with ``values``.
        dones : torch.Tensor
            Terminal indicator tensor. A value of ``1.0`` indicates that the
            transition ended an episode, and ``0.0`` indicates a non-terminal
            transition.

        Returns
        -------
        rets : torch.Tensor
            Return targets used for value-function learning. Has the same
            shape as ``values``.
        advs : torch.Tensor
            Generalized advantage estimates. Has the same shape as
            ``rewards`` and may be normalized depending on ``self.adv_norm``.
        """

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        advs = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[-1])

        for i in reversed(range(len(rewards))):
            gae = delta[i] + (1.0 - dones[i]) * self.gamma * self.lmda * gae
            advs[i] = gae

        rets = advs + values

        if self.adv_norm:
            advs = (advs - advs.mean()) / (advs.std() + self.epsilon)

        return rets, advs

    @staticmethod
    def _grad_norm(parameters):
        """Compute the global L2 norm of parameter gradients.

        Parameters
        ----------
        parameters : iterable[torch.nn.Parameter]
            Iterable of parameters whose gradients should be aggregated.
            Parameters with ``grad is None`` are ignored.

        Returns
        -------
        float
            Euclidean norm of all available gradients. Returns ``0.0`` if no
            parameter has a gradient.
        """

        total_norm_sq = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            param_norm = param.grad.detach().data.norm(2)
            total_norm_sq += float(param_norm.item() ** 2)
        return float(total_norm_sq ** 0.5)

    @staticmethod
    def _tensor_mean(tensor):
        """Return the scalar mean of a tensor as a Python float.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor whose elements will be averaged. The tensor is detached
            before conversion.

        Returns
        -------
        float
            Mean of ``tensor`` converted to a Python scalar.
        """

        return float(tensor.detach().mean().item())

    def learn(self, *args, **kwargs):
        """Run one policy update step.

        This is an abstract method and must be implemented by each concrete
        on-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs for learning.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs for learning.

        Returns
        -------
        object
            Update statistics or another subclass-defined result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()

    def get_policy_state(self):
        """Collect a CPU-safe policy state for synchronization.

        This method collects available actor and critic state dictionaries
        from the agent. Tensor values are detached, moved to CPU, and cloned
        so that the returned state can be serialized or transferred to remote
        workers.

        Returns
        -------
        dict[str, object]
            CPU-safe state dictionary. Possible keys are
            ``"actor_state_dict"`` and ``"critic_state_dict"``, depending on
            which modules are present in the concrete algorithm.
        """

        state = {}
        if hasattr(self, "actor"):
            state["actor_state_dict"] = to_cpu_serializable(self.actor.state_dict())
        if hasattr(self, "critic"):
            state["critic_state_dict"] = to_cpu_serializable(self.critic.state_dict())
        return state

    def set_policy_state(self, state):
        """Load synchronized policy parameters into local modules.

        Parameters
        ----------
        state : dict[str, object] or None
            Serialized policy state produced by :meth:`get_policy_state`. If
            ``None`` is provided, the method returns without changing the
            current agent.

        Returns
        -------
        None
            Available actor and critic module parameters are updated in place.
        """

        if state is None:
            return

        if "actor_state_dict" in state and hasattr(self, "actor"):
            self.actor.load_state_dict(state["actor_state_dict"])

        if "critic_state_dict" in state and hasattr(self, "critic"):
            self.critic.load_state_dict(state["critic_state_dict"])

    def train(self, project_name, **config):
        """Launch training through the runtime layer.

        Parameters
        ----------
        project_name : str
            Logging or experiment namespace passed to the runtime.
        **config : dict[str, object]
            Runtime-specific training configuration. Accepted keys depend on
            the implementation of the runtime launcher.

        Returns
        -------
        object
            Result returned by the runtime training launcher.
        """

        self.policy_type = "on_policy"
        return self._runtime.launch_training(project_name, **config)

    def step(self, state, action, reward, next_state, terminated, truncated=False):
        """Store one transition and trigger learning when ready.

        The transition is delegated to the runtime object. Depending on the
        runtime implementation, this method may only store the transition or
        may also trigger a learning step once enough data has been collected.

        Parameters
        ----------
        state : array-like or torch.Tensor
            Observation before applying the action.
        action : array-like or torch.Tensor or int
            Action sampled from the policy.
        reward : float
            Scalar reward observed after the transition.
        next_state : array-like or torch.Tensor
            Observation reached after applying the action.
        terminated : bool
            Environment termination flag. ``True`` means the episode ended
            because a terminal state was reached.
        truncated : bool, optional
            Environment truncation flag. ``True`` means the episode ended due
            to an external limit such as a time limit. The default is
            ``False``.

        Returns
        -------
        object
            Runtime-defined result, typically learning metrics or ``None``.
        """

        return self._runtime.step(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )

    def make_remote_copy(self):
        """Create a CPU-safe copy suitable for Ray worker processes.

        The current agent is first deep-copied. The copied object is then
        normalized so that all tensors, devices, and modules are moved to CPU.
        This prevents remote workers from depending on GPU-specific object
        state during deserialization.

        Returns
        -------
        OnPolicyAlgorithm
            Deep-copied agent instance with tensors and modules moved to CPU.
        """

        remote_copy = deepcopy(self)
        return make_cpu_safe(remote_copy)

    def save(self, *args, **kwargs):
        """Persist model or algorithm state.

        This is an abstract method and must be implemented by each concrete
        on-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs for persistence.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs for persistence.

        Returns
        -------
        object
            Subclass-defined persistence result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()

    def load(self, *args, **kwargs):
        """Restore model or algorithm state.

        This is an abstract method and must be implemented by each concrete
        on-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs for loading.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs for loading.

        Returns
        -------
        object
            Subclass-defined loading result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()


class OffPolicyAlgorithm(object):
    """Base class for off-policy reinforcement-learning algorithms.

    This class stores common metadata, hyperparameters, replay-buffer
    configuration, policy synchronization utilities, and target-network
    update logic shared by off-policy agents such as DDPG, TD3, SAC, and
    DQN-style methods.

    Parameters
    ----------
    env : gym.Env
        Environment used for training and for inferring observation and
        action-space dimensions.
    actor_size : tuple[int, ...]
        Hidden-layer sizes of the actor network.
    critic_size : tuple[int, ...]
        Hidden-layer sizes of the critic or Q-network.
    actor_activation : type[torch.nn.Module] or callable
        Activation function used in the actor network.
    critic_activation : type[torch.nn.Module] or callable
        Activation function used in the critic network.
    buffer_size : int
        Maximum number of transitions stored in the replay buffer.
    batch_size : int
        Number of transitions sampled from the replay buffer per update.
    update_after : int
        Minimum number of stored transitions required before learning starts.
    actor_lr : float
        Learning rate for the actor optimizer.
    critic_lr : float
        Learning rate for the critic optimizer.
    gamma : float
        Discount factor used in Bellman target computation.
    tau : float
        Polyak averaging coefficient used for target-network soft updates.
    prioritized_mode : bool
        Whether prioritized experience replay is enabled.
    prio_alpha : float
        Priority exponent controlling how strongly sampling depends on
        transition priorities.
    prio_beta : float
        Initial importance-sampling correction exponent.
    prio_eps : float
        Small positive constant added to priorities for numerical stability.

    Attributes
    ----------
    env : gym.Env
        Training environment.
    state_dim : int
        Dimension of the observation vector.
    action_dim : int or numpy.ndarray
        Action-space dimension or multi-discrete action vector.
    action_storage_shape : int
        Number of values stored per action in the replay buffer.
    action_type : {"continuous", "discrete", "multidiscrete"}
        Type of action space inferred from the environment.
    device : torch.device
        Device used for model computation.
    timesteps : int
        Number of environment steps collected so far.
    epsilon : float
        Small numerical constant used to avoid division by zero.
    last_action_info : dict[str, object]
        Dictionary storing diagnostic information about the most recent
        action selection.
    actor_size : tuple[int, ...]
        Hidden-layer sizes of the actor network.
    critic_size : tuple[int, ...]
        Hidden-layer sizes of the critic network.
    buffer_size : int
        Replay-buffer capacity.
    batch_size : int
        Minibatch size used for optimization.
    update_after : int
        Warm-up threshold before learning begins.
    gamma : float
        Discount factor.
    tau : float
        Target-network soft-update coefficient.
    prioritized_mode : bool
        Whether prioritized replay is enabled.
    prio_alpha : float
        Priority exponent.
    prio_beta : float
        Current importance-sampling correction exponent.
    prio_beta_start : float
        Initial importance-sampling correction exponent.
    prio_eps : float
        Small numerical constant added to priorities.
    """

    def __init__(
        self,
        env,
        actor_size,
        critic_size,
        actor_activation,
        critic_activation,
        buffer_size,
        batch_size,
        update_after,
        actor_lr,
        critic_lr,
        gamma,
        tau,
        prioritized_mode,
        prio_alpha,
        prio_beta,
        prio_eps,
    ):
        """Initialize common off-policy agent state.

        Parameters
        ----------
        env : gym.Env
            Environment instance used to infer observation and action shapes.
        actor_size : tuple[int, ...]
            Hidden-layer widths for the actor network.
        critic_size : tuple[int, ...]
            Hidden-layer widths for the critic network.
        actor_activation : type[torch.nn.Module] or callable
            Activation applied inside actor hidden layers.
        critic_activation : type[torch.nn.Module] or callable
            Activation applied inside critic hidden layers.
        buffer_size : int
            Replay-buffer capacity in transitions.
        batch_size : int
            Number of transitions sampled per optimization step.
        update_after : int
            Warm-up threshold before learning is allowed.
        actor_lr : float
            Learning rate for the actor optimizer.
        critic_lr : float
            Learning rate for the critic optimizer.
        gamma : float
            Discount factor used in Bellman targets.
        tau : float
            Interpolation factor for target-network updates.
        prioritized_mode : bool
            Whether prioritized replay is enabled.
        prio_alpha : float
            Exponent controlling replay prioritization strength.
        prio_beta : float
            Initial exponent for importance-sampling correction.
        prio_eps : float
            Small constant added to priorities for numerical stability.

        Returns
        -------
        None
            This constructor populates shared agent metadata and runtime
            state.
        """

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
            self.action_storage_shape = self.action_dim
            self.action_type = "continuous"
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.action_storage_shape = 1
            self.action_type = "discrete"
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.action_type = "multidiscrete"
            self.action_dim = self.env.action_space.nvec
            self.action_storage_shape = len(self.action_dim)
        else:
            raise TypeError(
                f"Unsupported action space type: "
                f"{type(self.env.action_space).__name__}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = 0
        self.epsilon = 1e-8
        self.last_action_info = {}

        self.actor_size = actor_size
        self.critic_size = critic_size
        self.actor_activation = actor_activation
        self.critic_activation = critic_activation

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_after = update_after

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau

        self.prioritized_mode = prioritized_mode
        self.prio_alpha = prio_alpha
        self.prio_beta = prio_beta
        self.prio_beta_start = prio_beta
        self.prio_eps = prio_eps

        self._runtime = create_off_policy_runtime(
            self,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_storage_shape=self.action_storage_shape,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            prioritized_mode=self.prioritized_mode,
            prio_alpha=self.prio_alpha,
        )

    @property
    def buffer(self):
        """Return the replay buffer managed by the runtime.

        Returns
        -------
        object
            Runtime-managed replay buffer instance.
        """

        return self._runtime.buffer

    @property
    def trainer(self):
        """Return the active trainer instance.

        Returns
        -------
        object or None
            Trainer attached to the runtime. Returns ``None`` if training has
            not been launched or if the runtime has not attached a trainer.
        """

        return self._runtime.trainer

    @torch.no_grad()
    def act(self, *args, **kwargs):
        """Produce an action using the current policy.

        This is an abstract method and must be implemented by each concrete
        off-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs, usually including the
            current state or observation.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs used during action selection.

        Returns
        -------
        object
            Action or action-related payload defined by the subclass.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()

    def random_action(self):
        """Sample a random action for replay-buffer warm-up.

        For continuous action spaces, this method samples uniformly from
        ``[-1, 1]`` for each action dimension. For discrete and multi-discrete
        action spaces, it delegates sampling to ``env.action_space.sample()``.

        The method also updates ``self.last_action_info`` with diagnostic
        information indicating that a random warm-up action was used.

        Returns
        -------
        numpy.ndarray or int
            Random action compatible with the environment action space. For
            continuous actions, the returned value has shape
            ``(action_dim,)``. For discrete actions, the returned value is an
            integer action index. For multi-discrete actions, the returned
            value follows the environment action-space format.
        """

        if self.action_type == "continuous":
            action = np.random.uniform(-1.0, 1.0, self.action_dim)
        else:
            action = self.env.action_space.sample()

        action_array = np.asarray(action, dtype=np.float32)
        self.last_action_info = {
            "used_random_action": True,
            "action_norm": float(np.linalg.norm(action_array)),
            "buffer_warmup": True,
        }
        return action

    @staticmethod
    def _grad_norm(parameters):
        """Compute the global L2 norm of parameter gradients.

        Parameters
        ----------
        parameters : iterable[torch.nn.Parameter]
            Iterable of parameters whose gradients should be aggregated.
            Parameters with ``grad is None`` are ignored.

        Returns
        -------
        float
            Euclidean norm of all available gradients. Returns ``0.0`` if no
            parameter has a gradient.
        """

        total_norm_sq = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            param_norm = param.grad.detach().data.norm(2)
            total_norm_sq += float(param_norm.item() ** 2)
        return float(total_norm_sq ** 0.5)

    @staticmethod
    def _tensor_mean(tensor):
        """Return the scalar mean of a tensor as a Python float.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor whose elements will be averaged. The tensor is detached
            before conversion.

        Returns
        -------
        float
            Mean of ``tensor`` converted to a Python scalar.
        """

        return float(tensor.detach().mean().item())

    def learn(self, *args, **kwargs):
        """Run one optimization step.

        This is an abstract method and must be implemented by each concrete
        off-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs for learning.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs for learning.

        Returns
        -------
        object
            Update statistics or another subclass-defined result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()

    def train(self, project_name, **config):
        """Launch training through the runtime layer.

        Parameters
        ----------
        project_name : str
            Logging or experiment namespace passed to the runtime.
        **config : dict[str, object]
            Runtime-specific training configuration. Accepted keys depend on
            the implementation of the runtime launcher.

        Returns
        -------
        object
            Result returned by the runtime training launcher.
        """

        self.policy_type = "off_policy"
        return self._runtime.launch_training(project_name, **config)

    def step(self, state, action, reward, next_state, terminated, truncated=False):
        """Store one transition and trigger learning when ready.

        The transition is delegated to the runtime object. Depending on the
        runtime implementation, this method may only store the transition or
        may also trigger a learning step once enough data has been collected.

        Parameters
        ----------
        state : array-like or torch.Tensor
            Observation before applying the action.
        action : array-like or torch.Tensor or int
            Action sampled from the policy, random warm-up policy, or
            exploration process.
        reward : float
            Scalar reward observed after the transition.
        next_state : array-like or torch.Tensor
            Observation reached after applying the action.
        terminated : bool
            Environment termination flag. ``True`` means the episode ended
            because a terminal state was reached.
        truncated : bool, optional
            Environment truncation flag. ``True`` means the episode ended due
            to an external limit such as a time limit. The default is
            ``False``.

        Returns
        -------
        object
            Runtime-defined result, typically learning metrics or ``None``.
        """

        return self._runtime.step(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )

    def soft_update(self, main_model, target_model):
        """Soft-update target-network parameters.

        This method performs Polyak averaging from ``main_model`` to
        ``target_model`` using the coefficient ``self.tau``:

        ``target = tau * main + (1 - tau) * target``

        Parameters
        ----------
        main_model : torch.nn.Module
            Source network whose parameters provide the new information.
        target_model : torch.nn.Module
            Target network updated toward ``main_model``.

        Returns
        -------
        None
            Parameters of ``target_model`` are updated in place.
        """

        for target_param, main_param in zip(
            target_model.parameters(),
            main_model.parameters(),
        ):
            target_param.data.copy_(
                self.tau * main_param.data
                + (1.0 - self.tau) * target_param.data
            )

    def get_policy_state(self):
        """Collect a CPU-safe policy state for synchronization.

        This method collects available model state dictionaries from the
        agent, including actor, critic, target actor, target critic, and
        adaptive entropy-temperature variables when present. Tensor values
        are converted to CPU-safe serialized copies.

        Returns
        -------
        dict[str, object]
            Dictionary containing available policy components. Possible keys
            include ``"actor_state_dict"``, ``"critic_state_dict"``,
            ``"target_actor_state_dict"``, ``"target_critic_state_dict"``,
            ``"log_alpha"``, and ``"alpha"``.
        """

        state = {}

        if hasattr(self, "actor"):
            state["actor_state_dict"] = to_cpu_serializable(self.actor.state_dict())

        if hasattr(self, "critic"):
            state["critic_state_dict"] = to_cpu_serializable(self.critic.state_dict())

        if hasattr(self, "target_actor"):
            state["target_actor_state_dict"] = to_cpu_serializable(
                self.target_actor.state_dict()
            )

        if hasattr(self, "target_critic"):
            state["target_critic_state_dict"] = to_cpu_serializable(
                self.target_critic.state_dict()
            )

        if hasattr(self, "adaptive_alpha_mode") and getattr(
            self,
            "adaptive_alpha_mode",
            False,
        ):
            if hasattr(self, "log_alpha"):
                state["log_alpha"] = to_cpu_serializable(self.log_alpha)

            if hasattr(self, "alpha"):
                state["alpha"] = self.alpha

        return state

    def set_policy_state(self, state):
        """Load synchronized policy state into local modules.

        Parameters
        ----------
        state : dict[str, object] or None
            Policy-state dictionary produced by :meth:`get_policy_state`.
            If ``None`` is given, the method returns without modifying the
            agent.

        Returns
        -------
        None
            Available actor, critic, target-network, and entropy-temperature
            states are loaded in place.
        """

        if state is None:
            return

        if "actor_state_dict" in state and hasattr(self, "actor"):
            self.actor.load_state_dict(state["actor_state_dict"])

        if "critic_state_dict" in state and hasattr(self, "critic"):
            self.critic.load_state_dict(state["critic_state_dict"])

        if "target_actor_state_dict" in state and hasattr(self, "target_actor"):
            self.target_actor.load_state_dict(state["target_actor_state_dict"])

        if "target_critic_state_dict" in state and hasattr(self, "target_critic"):
            self.target_critic.load_state_dict(state["target_critic_state_dict"])

        if "log_alpha" in state and hasattr(self, "log_alpha"):
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))

        if "alpha" in state and hasattr(self, "alpha"):
            self.alpha = state["alpha"]

    def save(self, *args, **kwargs):
        """Persist model or algorithm state.

        This is an abstract method and must be implemented by each concrete
        off-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs for persistence.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs for persistence.

        Returns
        -------
        object
            Subclass-defined persistence result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()

    def load(self, *args, **kwargs):
        """Restore model or algorithm state.

        This is an abstract method and must be implemented by each concrete
        off-policy algorithm.

        Parameters
        ----------
        *args : tuple
            Algorithm-specific positional inputs for loading.
        **kwargs : dict[str, object]
            Algorithm-specific keyword inputs for loading.

        Returns
        -------
        object
            Subclass-defined loading result.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.
        """

        raise NotImplementedError()

    def make_remote_copy(self):
        """Create a CPU-safe copy suitable for Ray worker processes.

        The current agent is first deep-copied. The copied object is then
        normalized so that all tensors, devices, and modules are moved to CPU.
        This prevents remote workers from depending on GPU-specific object
        state during deserialization.

        Returns
        -------
        OffPolicyAlgorithm
            Deep-copied agent instance with tensors and modules moved to CPU.
        """

        remote_copy = deepcopy(self)
        return make_cpu_safe(remote_copy)