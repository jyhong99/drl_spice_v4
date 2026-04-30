"""Deep Deterministic Policy Gradient agent."""

import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam

from agents.common.policy import OffPolicyAlgorithm, to_cpu_serializable
from agents.common.network import MLPDeterministicPolicy, MLPQFunction
from agents.common.noise import GaussianNoise, OrnsteinUhlenbeckNoise


class DDPG(OffPolicyAlgorithm):
    """Deep Deterministic Policy Gradient agent for continuous control.

    This class implements the DDPG algorithm using a deterministic actor,
    a Q-function critic, target networks, and exploration noise.

    Parameters
    ----------
    env : gym.Env
        Environment instance used to infer observation and action dimensions.
        The action space is expected to be continuous.
    **config : dict[str, object]
        Configuration dictionary containing network sizes, activation
        functions, optimizer settings, replay-buffer options, target-network
        update settings, prioritized replay settings, and exploration-noise
        settings.

    Attributes
    ----------
    actor : MLPDeterministicPolicy
        Deterministic actor network used to generate continuous actions.
    target_actor : MLPDeterministicPolicy
        Target actor network used for critic target computation.
    critic : MLPQFunction
        Q-function network that estimates state-action values.
    target_critic : MLPQFunction
        Target critic network used for stable Bellman target computation.
    actor_optim : torch.optim.Adam
        Optimizer for the actor network.
    critic_optim : torch.optim.Adam
        Optimizer for the critic network.
    noise : GaussianNoise or OrnsteinUhlenbeckNoise
        Exploration-noise process used during training.
    max_grad_norm : float or None
        Maximum gradient norm for gradient clipping. If ``None``, gradient
        clipping is disabled.
    action_noise_std : float
        Standard deviation of the exploration noise.
    noise_type : str
        Type of exploration noise. Supported values are ``"normal"`` and
        ``"ou"``.
    config : dict[str, object]
        Original configuration dictionary passed to the constructor.
    """

    def __init__(self, env, **config):
        """Initialize the DDPG agent.

        Parameters
        ----------
        env : gym.Env
            Environment instance used for training and dimension inference.
        **config : dict[str, object]
            Configuration options for the agent.

            Supported keys include:

            ``actor_size`` : tuple[int, ...], optional
                Hidden-layer sizes of the actor network. The default is
                ``(256, 256)``.
            ``critic_size`` : tuple[int, ...], optional
                Hidden-layer sizes of the critic network. The default is
                ``(256, 256)``.
            ``actor_activation`` : callable, optional
                Activation function used by the actor network. The default is
                ``torch.relu``.
            ``critic_activation`` : callable, optional
                Activation function used by the critic network. The default is
                ``torch.relu``.
            ``buffer_size`` : int, optional
                Replay-buffer capacity. The default is ``int(1e6)``.
            ``batch_size`` : int, optional
                Batch size used for optimization. The default is ``256``.
            ``update_after`` : int or float, optional
                Number of transitions required before learning starts. The
                default is ``1e3``.
            ``actor_lr`` : float, optional
                Learning rate for the actor optimizer. The default is
                ``3e-4``.
            ``critic_lr`` : float, optional
                Learning rate for the critic optimizer. The default is
                ``3e-4``.
            ``gamma`` : float, optional
                Discount factor. The default is ``0.99``.
            ``tau`` : float, optional
                Soft-update coefficient for target networks. The default is
                ``0.005``.
            ``prioritized_mode`` : bool, optional
                Whether prioritized replay is enabled. The default is
                ``False``.
            ``prio_alpha`` : float, optional
                Priority exponent for prioritized replay. The default is
                ``0.6``.
            ``prio_beta`` : float, optional
                Importance-sampling correction exponent. The default is
                ``0.4``.
            ``prio_eps`` : float, optional
                Small constant added to priorities. The default is ``1e-6``.
            ``max_grad_norm`` : float or None, optional
                Maximum gradient norm for clipping. The default is ``None``.
            ``action_noise_std`` : float, optional
                Standard deviation of exploration noise. The default is
                ``0.1``.
            ``noise_type`` : {"normal", "ou"}, optional
                Exploration-noise type. ``"normal"`` uses Gaussian noise and
                ``"ou"`` uses Ornstein-Uhlenbeck noise. The default is
                ``"normal"``.

        Returns
        -------
        None
            The constructor initializes networks, optimizers, target networks,
            and exploration noise in place.
        """

        super().__init__(
            env=env,
            actor_size=config.get("actor_size", (256, 256)),
            critic_size=config.get("critic_size", (256, 256)),
            actor_activation=config.get("actor_activation", torch.relu),
            critic_activation=config.get("critic_activation", torch.relu),
            buffer_size=config.get("buffer_size", int(1e6)),
            batch_size=config.get("batch_size", 256),
            update_after=config.get("update_after", 1e3),
            actor_lr=config.get("actor_lr", 3e-4),
            critic_lr=config.get("critic_lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            tau=config.get("tau", 0.005),
            prioritized_mode=config.get("prioritized_mode", False),
            prio_alpha=config.get("prio_alpha", 0.6),
            prio_beta=config.get("prio_beta", 0.4),
            prio_eps=config.get("prio_eps", 1e-6),
        )

        self.max_grad_norm = config.get("max_grad_norm", None)
        self.action_noise_std = config.get("action_noise_std", 0.1)
        self.noise_type = config.get("noise_type", "normal")
        self.config = config

        self.actor = MLPDeterministicPolicy(
            self.state_dim,
            self.action_dim,
            self.actor_size,
            self.actor_activation,
        ).to(self.device)
        self.target_actor = deepcopy(self.actor)

        self.critic = MLPQFunction(
            self.state_dim,
            self.action_dim,
            self.critic_size,
            self.critic_activation,
        ).to(self.device)

        self.target_critic = deepcopy(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        if self.noise_type == "normal":
            self.noise = GaussianNoise(
                self.action_dim,
                sigma=self.action_noise_std,
                device=self.device,
            )
        elif self.noise_type == "ou":
            self.noise = OrnsteinUhlenbeckNoise(
                self.action_dim,
                sigma=self.action_noise_std,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unsupported noise_type: {self.noise_type}. "
                f"Expected 'normal' or 'ou'."
            )

    def __str__(self):
        """Return the display name of the algorithm.

        Returns
        -------
        str
            Algorithm name, ``"DDPG"``.
        """

        return "DDPG"

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
        """Select an action using the current policy.

        During early replay-buffer warm-up, a random action is returned when
        training is enabled. After warm-up, the actor network generates a
        deterministic action. Exploration noise is added during training.

        Parameters
        ----------
        state : array-like or torch.Tensor
            Current environment observation. Expected shape is ``(state_dim,)``
            for a single observation or ``(batch_size, state_dim)`` for batched
            observations.
        training : bool, optional
            Whether the agent is acting in training mode. If ``True``,
            replay-buffer warm-up and exploration noise are enabled. If
            ``False``, the deterministic actor output is used directly. The
            default is ``True``.
        global_buffer_size : int or None, optional
            Optional global replay-buffer size used for distributed training.
            If provided, this value is used instead of ``self.buffer.size`` for
            the warm-up check. The default is ``None``.

        Returns
        -------
        numpy.ndarray
            Selected action clipped to the range ``[-1, 1]`` and moved to CPU.
            For a single state, the shape is typically ``(action_dim,)``.
        """

        if global_buffer_size is None:
            if (self.buffer.size < self.update_after) and training:
                return self.random_action()
        else:
            if (global_buffer_size < self.update_after) and training:
                return self.random_action()

        self.actor.train(training)

        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)

        noise_norm = 0.0
        if training:
            noise = self.noise.sample()
            noise_norm = float(noise.norm().item())
            action += noise

        self.last_action_info = {
            "used_random_action": False,
            "buffer_warmup": False,
            "training_action": bool(training),
            "noise_type": self.noise_type,
            "noise_norm": noise_norm,
            "action_norm": float(action.norm().item()),
        }

        return torch.clamp(action, -1.0, 1.0).cpu().numpy()

    def learn(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        truncateds=None,
        weights=None,
        global_timesteps=None,
    ):
        """Perform one DDPG optimization step.

        This method updates the critic, updates the actor, applies optional
        gradient clipping, performs target-network soft updates, and returns
        training diagnostics.

        Parameters
        ----------
        states : torch.Tensor
            Batch of current states with shape ``(batch_size, state_dim)``.
        actions : torch.Tensor
            Batch of actions with shape ``(batch_size, action_dim)``.
        rewards : torch.Tensor
            Batch of rewards with shape ``(batch_size, 1)`` or a
            broadcast-compatible shape.
        next_states : torch.Tensor
            Batch of next states with shape ``(batch_size, state_dim)``.
        dones : torch.Tensor
            Batch of terminal flags with shape ``(batch_size, 1)`` or a
            broadcast-compatible shape. A value of ``1.0`` indicates terminal
            transition, and ``0.0`` indicates non-terminal transition.
        truncateds : torch.Tensor or None, optional
            Batch of truncation flags. This argument is accepted for runtime
            interface compatibility but is not used in this method. The
            default is ``None``.
        weights : torch.Tensor or None, optional
            Importance-sampling weights used for prioritized replay. Required
            when ``self.prioritized_mode`` is ``True``. The default is
            ``None``.
        global_timesteps : int or None, optional
            Optional global timestep value used to synchronize diagnostics in
            distributed training. The default is ``None``.

        Returns
        -------
        dict[str, object]
            Dictionary containing update diagnostics.

            ``"agent_timesteps"`` : int
                Current agent timestep counter.
            ``"actor_loss"`` : float
                Actor loss value.
            ``"critic_loss"`` : float
                Critic loss value.
            ``"td_error"`` : torch.Tensor or None
                Absolute TD error when prioritized replay is enabled;
                otherwise ``None``.
            ``"actor_updated"`` : bool
                Whether the actor was updated.
            ``"actor_grad_norm"`` : float
                Global L2 norm of actor gradients before clipping.
            ``"critic_grad_norm"`` : float
                Global L2 norm of critic gradients before clipping.
            ``"mean_q"`` : float
                Mean current Q-value estimate.
            ``"mean_target_q"`` : float
                Mean target Q-value.
        """

        self.actor.train()
        self.critic.train()

        if global_timesteps is not None:
            self.timesteps = global_timesteps

        with torch.no_grad():
            next_q_values = self.target_critic(
                next_states,
                self.target_actor(next_states),
            )
            target_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values

        td_error = None
        q_values = self.critic(states, actions)

        if self.prioritized_mode:
            td_error = (target_q_values - q_values).abs()
            critic_loss = (weights * td_error ** 2).mean()
        else:
            critic_loss = F.mse_loss(q_values, target_q_values)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._grad_norm(self.critic.parameters())

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
            )

        self.critic_optim.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._grad_norm(self.actor.parameters())

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm,
            )

        self.actor_optim.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return {
            "agent_timesteps": self.timesteps,
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": td_error,
            "actor_updated": True,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "mean_q": self._tensor_mean(q_values),
            "mean_target_q": self._tensor_mean(target_q_values),
        }

    @torch.no_grad()
    def calculate_td_error(self, state, action, reward, next_state, terminated):
        """Compute the absolute TD error for prioritized replay.

        Parameters
        ----------
        state : torch.Tensor
            Current state tensor with shape ``(batch_size, state_dim)`` or
            ``(state_dim,)``.
        action : torch.Tensor
            Action tensor with shape ``(batch_size, action_dim)`` or
            ``(action_dim,)``.
        reward : torch.Tensor
            Reward tensor with shape ``(batch_size, 1)`` or a
            broadcast-compatible shape.
        next_state : torch.Tensor
            Next-state tensor with shape ``(batch_size, state_dim)`` or
            ``(state_dim,)``.
        terminated : torch.Tensor
            Terminal indicator tensor. A value of ``1.0`` indicates terminal
            transition, and ``0.0`` indicates non-terminal transition.

        Returns
        -------
        torch.Tensor
            Absolute TD-error tensor with the same batch dimension as the
            input transition batch.
        """

        next_q_value = self.target_critic(
            next_state,
            self.target_actor(next_state),
        )
        target_q_value = reward + (1.0 - terminated) * self.gamma * next_q_value
        q_value = self.critic(state, action)
        td_error = (target_q_value - q_value).abs()
        return td_error

    def save(self, save_path):
        """Save model and optimizer states to disk.

        Parameters
        ----------
        save_path : str or pathlib.Path
            File path where the checkpoint will be saved.

        Returns
        -------
        None
            The checkpoint is written to ``save_path``.
        """

        torch.save(
            {
                "actor_state_dict": to_cpu_serializable(self.actor.state_dict()),
                "critic_state_dict": to_cpu_serializable(self.critic.state_dict()),
                "target_actor_state_dict": to_cpu_serializable(
                    self.target_actor.state_dict()
                ),
                "target_critic_state_dict": to_cpu_serializable(
                    self.target_critic.state_dict()
                ),
                "actor_optim_state_dict": to_cpu_serializable(
                    self.actor_optim.state_dict()
                ),
                "critic_optim_state_dict": to_cpu_serializable(
                    self.critic_optim.state_dict()
                ),
            },
            save_path,
        )

    def load(self, load_path):
        """Load model and optimizer states from disk.

        Parameters
        ----------
        load_path : str or pathlib.Path
            File path of the checkpoint to load.

        Returns
        -------
        None
            Actor, critic, target-network, and optimizer states are restored
            in place.
        """

        checkpoint = torch.load(
            load_path,
            map_location=self.device,
            weights_only=True,
        )

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])