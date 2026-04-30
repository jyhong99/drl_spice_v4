"""Twin Delayed DDPG agent."""

import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam

from agents.common.policy import OffPolicyAlgorithm, to_cpu_serializable
from agents.common.network import MLPDeterministicPolicy, MLPDoubleQFunction
from agents.common.noise import GaussianNoise, OrnsteinUhlenbeckNoise


class TD3(OffPolicyAlgorithm):
    """Twin Delayed Deep Deterministic Policy Gradient agent.

    This class implements TD3 for continuous-control tasks using a
    deterministic actor, twin Q-function critics, target policy smoothing,
    delayed actor updates, target networks, and exploration noise.

    Parameters
    ----------
    env : gym.Env
        Environment instance used to infer observation and action dimensions.
        The action space is expected to be continuous.
    **config : dict[str, object]
        Configuration dictionary containing network sizes, activation
        functions, optimizer settings, replay-buffer options, target-network
        update settings, target smoothing settings, prioritized replay
        settings, and exploration-noise settings.

    Attributes
    ----------
    actor : MLPDeterministicPolicy
        Deterministic actor network used to generate continuous actions.
    critic : MLPDoubleQFunction
        Twin Q-function critic network used to estimate state-action values.
    target_actor : MLPDeterministicPolicy
        Target actor network used for critic target computation.
    target_critic : MLPDoubleQFunction
        Target twin Q-function critic network used for stable target
        computation.
    actor_optim : torch.optim.Adam
        Optimizer for the actor network.
    critic_optim : torch.optim.Adam
        Optimizer for the critic network.
    noise : GaussianNoise or OrnsteinUhlenbeckNoise
        Exploration-noise process used during training.
    update_freq : int
        Frequency of delayed actor and target-network updates.
    max_grad_norm : float or None
        Maximum gradient norm for gradient clipping. If ``None``, gradient
        clipping is disabled.
    action_noise_std : float
        Standard deviation of exploration noise added to actor actions.
    target_noise_std : float
        Standard deviation of smoothing noise added to target actions.
    noise_clip : float
        Absolute clipping bound for target smoothing noise.
    noise_type : str
        Type of exploration noise. Supported values are ``"normal"`` and
        ``"ou"``.
    config : dict[str, object]
        Original configuration dictionary passed to the constructor.
    """

    def __init__(self, env, **config):
        """Initialize the TD3 agent.

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
                Hidden-layer sizes of the twin critic networks. The default is
                ``(256, 256)``.
            ``actor_activation`` : callable, optional
                Activation function used by the actor network. The default is
                ``torch.relu``.
            ``critic_activation`` : callable, optional
                Activation function used by the critic networks. The default is
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
            ``update_freq`` : int, optional
                Frequency of delayed actor and target-network updates. The
                default is ``2``.
            ``max_grad_norm`` : float or None, optional
                Maximum gradient norm for clipping. The default is ``None``.
            ``action_noise_std`` : float, optional
                Standard deviation of exploration noise. The default is
                ``0.1``.
            ``target_noise_std`` : float, optional
                Standard deviation of target policy smoothing noise. The
                default is ``0.2``.
            ``noise_clip`` : float, optional
                Absolute clipping bound for target policy smoothing noise.
                The default is ``0.5``.
            ``noise_type`` : {"normal", "ou"}, optional
                Exploration-noise type. ``"normal"`` uses Gaussian noise and
                ``"ou"`` uses Ornstein-Uhlenbeck noise. The default is
                ``"normal"``.

        Returns
        -------
        None
            The constructor initializes networks, target networks, optimizers,
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

        self.update_freq = config.get("update_freq", 2)
        self.max_grad_norm = config.get("max_grad_norm", None)
        self.action_noise_std = config.get("action_noise_std", 0.1)
        self.target_noise_std = config.get("target_noise_std", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.noise_type = config.get("noise_type", "normal")
        self.config = config

        self.actor = MLPDeterministicPolicy(
            self.state_dim,
            self.action_dim,
            self.actor_size,
            self.actor_activation,
        ).to(self.device)

        self.critic = MLPDoubleQFunction(
            self.state_dim,
            self.action_dim,
            self.critic_size,
            self.critic_activation,
        ).to(self.device)

        self.target_actor = deepcopy(self.actor)
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
            Algorithm name, ``"TD3"``.
        """

        return "TD3"

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
        """Select an action using the current actor policy.

        During early replay-buffer warm-up, this method returns a random
        action when training is enabled. After warm-up, the actor network
        generates a deterministic action. Exploration noise is added during
        training, and the final action is clipped to ``[-1, 1]``.

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
        """Perform one TD3 optimization step.

        This method updates the twin critics at every call. The actor and
        target networks are updated only when ``self.timesteps`` is divisible
        by ``self.update_freq``. Optional prioritized replay weights and
        gradient clipping are supported.

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
            Dictionary containing TD3 update diagnostics.

            ``"agent_timesteps"`` : int
                Current agent timestep counter.
            ``"actor_loss"`` : float
                Actor loss value. This is ``0.0`` when the actor is not
                updated at the current step.
            ``"critic_loss"`` : float
                Twin-critic loss value.
            ``"td_error"`` : torch.Tensor or None
                TD-error tensor used for prioritized replay, or ``None`` when
                prioritized replay is disabled.
            ``"actor_updated"`` : bool
                Whether the actor and target networks were updated at the
                current step.
            ``"actor_grad_norm"`` : float
                Global L2 norm of actor gradients before clipping.
            ``"critic_grad_norm"`` : float
                Global L2 norm of critic gradients before clipping.
            ``"mean_q1"`` : float
                Mean value of the first Q-function estimate.
            ``"mean_q2"`` : float
                Mean value of the second Q-function estimate.
            ``"mean_target_q"`` : float
                Mean target Q-value.
        """

        self.actor.train()
        self.critic.train()

        if global_timesteps is not None:
            self.timesteps = global_timesteps

        with torch.no_grad():
            target_noises = torch.clamp(
                self.target_noise_std * torch.randn_like(actions),
                -self.noise_clip,
                self.noise_clip,
            )
            target_next_actions = torch.clamp(
                self.target_actor(next_states) + target_noises,
                -1.0,
                1.0,
            )
            next_target_q1_values, next_target_q2_values = self.target_critic(
                next_states,
                target_next_actions,
            )
            next_target_q_values = torch.min(
                next_target_q1_values,
                next_target_q2_values,
            )
            target_q_values = (
                rewards + (1.0 - dones) * self.gamma * next_target_q_values
            )

        td_error = None
        q1_values, q2_values = self.critic(states, actions)

        if self.prioritized_mode:
            td_error1 = target_q_values - q1_values
            td_error2 = target_q_values - q2_values
            td_error = 0.5 * (td_error1.abs() + td_error2.abs())
            critic_loss = (weights * td_error1 ** 2).mean() + (
                weights * td_error2 ** 2
            ).mean()
        else:
            critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(
                q2_values,
                target_q_values,
            )

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._grad_norm(self.critic.parameters())

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
            )

        self.critic_optim.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        actor_grad_norm = 0.0
        actor_updated = False

        if (self.timesteps % self.update_freq) == 0:
            actor_updated = True
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

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
            "actor_updated": actor_updated,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "mean_q1": self._tensor_mean(q1_values),
            "mean_q2": self._tensor_mean(q2_values),
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
            Mean absolute TD-error tensor computed from both Q-functions.
        """

        target_noise = torch.clamp(
            self.target_noise_std * torch.randn_like(action),
            -self.noise_clip,
            self.noise_clip,
        )
        target_next_action = torch.clamp(
            self.target_actor(next_state) + target_noise,
            -1.0,
            1.0,
        )
        next_target_q1_value, next_target_q2_value = self.target_critic(
            next_state,
            target_next_action,
        )
        next_target_q_value = torch.min(next_target_q1_value, next_target_q2_value)
        target_q_value = (
            reward + (1.0 - terminated) * self.gamma * next_target_q_value
        )

        q1_value, q2_value = self.critic(state, action)
        td_error1 = target_q_value - q1_value
        td_error2 = target_q_value - q2_value
        td_error = 0.5 * (td_error1.abs() + td_error2.abs())
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