"""Soft Actor-Critic agent."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from agents.common.network import MLPGaussianPolicy, MLPDoubleQFunction, Normal
from agents.common.policy import OffPolicyAlgorithm, to_cpu_serializable


class SAC(OffPolicyAlgorithm):
    """Soft Actor-Critic agent for continuous-control tasks.

    This class implements SAC using a stochastic Gaussian actor, a double
    Q-function critic, a target critic, entropy regularization, optional
    adaptive entropy-temperature tuning, and optional prioritized replay.

    Parameters
    ----------
    env : gym.Env
        Environment instance used to infer observation and action dimensions.
        The action space is expected to be continuous.
    **config : dict[str, object]
        Configuration dictionary containing network sizes, optimizer settings,
        replay-buffer options, target-network update settings, entropy
        settings, and prioritized replay settings.

    Attributes
    ----------
    actor : MLPGaussianPolicy
        Gaussian policy network used to sample continuous actions.
    critic : MLPDoubleQFunction
        Double Q-function network used to estimate state-action values.
    target_critic : MLPDoubleQFunction
        Target double Q-function network used for stable target computation.
    actor_optim : torch.optim.Adam
        Optimizer for the actor network.
    critic_optim : torch.optim.Adam
        Optimizer for the critic network.
    alpha_optim : torch.optim.Adam
        Optimizer for the entropy-temperature parameter. This attribute is
        only created when ``adaptive_alpha_mode`` is enabled.
    update_freq : int
        Frequency of actor, entropy-temperature, and target-critic updates.
    max_grad_norm : float or None
        Maximum gradient norm for gradient clipping. If ``None``, gradient
        clipping is disabled.
    alpha : float
        Entropy-temperature coefficient.
    adaptive_alpha_mode : bool
        Whether the entropy-temperature coefficient is learned automatically.
    ent_lr : float
        Learning rate for the entropy-temperature optimizer.
    target_entropy : float
        Target policy entropy used for adaptive entropy-temperature tuning.
        This attribute is only created when ``adaptive_alpha_mode`` is enabled.
    log_alpha : torch.Tensor
        Learnable log entropy-temperature parameter. This attribute is only
        created when ``adaptive_alpha_mode`` is enabled.
    config : dict[str, object]
        Original configuration dictionary passed to the constructor.
    """

    def __init__(self, env, **config):
        """Initialize the SAC agent.

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
                Soft-update coefficient for the target critic. The default is
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
                Frequency of actor, entropy-temperature, and target-critic
                updates. The default is ``1``.
            ``max_grad_norm`` : float or None, optional
                Maximum gradient norm for clipping. The default is ``None``.
            ``alpha`` : float, optional
                Initial or fixed entropy-temperature coefficient. The default
                is ``0.2``.
            ``adaptive_alpha_mode`` : bool, optional
                Whether to learn ``alpha`` automatically. The default is
                ``True``.
            ``ent_lr`` : float, optional
                Learning rate for the entropy-temperature optimizer. The
                default is ``3e-4``.

        Returns
        -------
        None
            The constructor initializes networks, optimizers, target networks,
            and entropy-temperature parameters in place.
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

        self.update_freq = config.get("update_freq", 1)
        self.max_grad_norm = config.get("max_grad_norm", None)
        self.alpha = config.get("alpha", 0.2)
        self.adaptive_alpha_mode = config.get("adaptive_alpha_mode", True)
        self.ent_lr = config.get("ent_lr", 3e-4)
        self.config = config

        self.actor = MLPGaussianPolicy(
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

        if self.adaptive_alpha_mode:
            self.target_entropy = -float(self.action_dim)

        self._initialize_sac_modules()

    @staticmethod
    def _all_finite(*tensors):
        """Check whether all provided tensors contain finite values.

        Parameters
        ----------
        *tensors : torch.Tensor
            One or more tensors to inspect.

        Returns
        -------
        bool
            ``True`` if every tensor contains only finite values; otherwise
            ``False``.
        """

        return all(torch.isfinite(tensor).all() for tensor in tensors)

    def _should_use_random_action(self, *, training, global_buffer_size=None):
        """Decide whether random warm-up actions should be used.

        Parameters
        ----------
        training : bool
            Whether the agent is currently acting in training mode.
        global_buffer_size : int or None, optional
            Optional global replay-buffer size used in distributed training.
            If provided, this value is used instead of ``self.buffer.size``.
            The default is ``None``.

        Returns
        -------
        bool
            ``True`` if training is enabled and the replay buffer has not yet
            reached ``self.update_after`` transitions; otherwise ``False``.
        """

        if not training:
            return False
        if global_buffer_size is None:
            return self.buffer.size < self.update_after
        return global_buffer_size < self.update_after

    def _initialize_sac_modules(self):
        """Build the target critic and optimizers.

        This helper initializes the target critic as a deep copy of the critic,
        creates actor and critic Adam optimizers, and initializes adaptive
        entropy-temperature parameters if enabled.

        Returns
        -------
        None
            Target critic, optimizers, and optional entropy-temperature
            parameters are created in place.
        """

        self.target_critic = deepcopy(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        if self.adaptive_alpha_mode:
            self._initialize_alpha_parameters()

    def _initialize_alpha_parameters(self):
        """Initialize learnable entropy-temperature parameters.

        This method creates ``self.log_alpha`` as a learnable scalar tensor and
        an Adam optimizer for entropy-temperature tuning.

        Returns
        -------
        None
            ``log_alpha`` and ``alpha_optim`` are created in place.
        """

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.ent_lr)

    def _save_sac_checkpoint(self, save_path):
        """Save all SAC model and optimizer states to disk.

        Parameters
        ----------
        save_path : str or pathlib.Path
            File path where the SAC checkpoint will be saved.

        Returns
        -------
        None
            The checkpoint is written to ``save_path``.
        """

        checkpoint = {
            "actor_state_dict": to_cpu_serializable(self.actor.state_dict()),
            "critic_state_dict": to_cpu_serializable(self.critic.state_dict()),
            "target_critic_state_dict": to_cpu_serializable(
                self.target_critic.state_dict()
            ),
            "actor_optim_state_dict": to_cpu_serializable(
                self.actor_optim.state_dict()
            ),
            "critic_optim_state_dict": to_cpu_serializable(
                self.critic_optim.state_dict()
            ),
        }

        if self.adaptive_alpha_mode:
            checkpoint["log_alpha"] = to_cpu_serializable(self.log_alpha)
            checkpoint["alpha"] = self.alpha
            checkpoint["alpha_optim_state_dict"] = to_cpu_serializable(
                self.alpha_optim.state_dict()
            )

        torch.save(checkpoint, save_path)

    def _load_sac_checkpoint(self, load_path):
        """Load all SAC model and optimizer states from disk.

        Parameters
        ----------
        load_path : str or pathlib.Path
            File path of the SAC checkpoint to load.

        Returns
        -------
        None
            Actor, critic, target critic, optimizer, and optional
            entropy-temperature states are restored in place.
        """

        checkpoint = torch.load(
            load_path,
            map_location=self.device,
            weights_only=True,
        )

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])

        if self.adaptive_alpha_mode and "alpha_optim_state_dict" in checkpoint:
            if "log_alpha" in checkpoint:
                self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))

            if "alpha" in checkpoint:
                self.alpha = checkpoint["alpha"]
            else:
                self.alpha = self.log_alpha.exp().item()

            self.alpha_optim.load_state_dict(checkpoint["alpha_optim_state_dict"])

    def _log_alpha_value(self):
        """Return the entropy-temperature value in log space.

        Returns
        -------
        float
            Current log entropy-temperature value. If adaptive
            entropy-temperature tuning is enabled, this is ``log_alpha``.
            Otherwise, this is ``log(self.alpha)``.
        """

        if self.adaptive_alpha_mode:
            return float(self.log_alpha.item())
        return float(np.log(self.alpha))

    def __str__(self):
        """Return the display name of the algorithm.

        Returns
        -------
        str
            Algorithm name, ``"SAC_Continuous"``.
        """

        return "SAC_Continuous"

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
        """Sample or predict an action from the SAC policy.

        During replay-buffer warm-up, this method returns random actions when
        training is enabled. After warm-up, actions are sampled from the
        Gaussian policy during training and set to the policy mean during
        evaluation. The selected action is squashed by ``tanh`` before being
        returned.

        Parameters
        ----------
        state : array-like or torch.Tensor
            Current environment observation. Expected shape is ``(state_dim,)``
            for a single observation or ``(batch_size, state_dim)`` for batched
            observations.
        training : bool, optional
            Whether the agent is acting in training mode. If ``True``, random
            warm-up actions and stochastic policy sampling are enabled. If
            ``False``, the policy mean is used. The default is ``True``.
        global_buffer_size : int or None, optional
            Optional global replay-buffer size used in distributed training.
            If provided, this value is used instead of ``self.buffer.size`` for
            warm-up action selection. The default is ``None``.

        Returns
        -------
        numpy.ndarray
            Tanh-squashed action tensor converted to a CPU NumPy array. For a
            single state, the shape is typically ``(action_dim,)``.
        """

        if self._should_use_random_action(
            training=training,
            global_buffer_size=global_buffer_size,
        ):
            return self.random_action()

        self.actor.train(training)

        state = torch.FloatTensor(state).to(self.device)
        mu, std = self.actor(state)
        action = torch.normal(mu, std) if training else mu
        squashed = torch.tanh(action)

        self.last_action_info = {
            "used_random_action": False,
            "buffer_warmup": False,
            "training_action": bool(training),
            "policy_mode": "gaussian",
            "action_norm": float(squashed.norm().item()),
            "policy_mean_norm": float(mu.norm().item()),
            "policy_std_mean": float(std.mean().item()),
        }

        return squashed.cpu().numpy()

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
        """Perform one SAC optimization step.

        This method updates the double Q-function critic at every call. The
        actor, adaptive entropy-temperature parameter, and target critic are
        updated according to ``self.update_freq``. It returns scalar training
        diagnostics and optional TD errors for prioritized replay.

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
            Dictionary containing SAC update diagnostics.

            ``"agent_timesteps"`` : int
                Current agent timestep counter.
            ``"actor_loss"`` : float
                Actor loss value. This is ``0.0`` when the actor is not
                updated at the current step.
            ``"critic_loss"`` : float
                Critic loss value.
            ``"alpha_loss"`` : float
                Entropy-temperature loss value. This is ``0.0`` when adaptive
                entropy tuning is disabled or not updated at the current step.
            ``"entropy"`` : float
                Mean entropy of the current Gaussian policy.
            ``"alpha"`` : float
                Current entropy-temperature coefficient.
            ``"log_alpha"`` : float
                Current entropy-temperature value in log space.
            ``"td_error"`` : torch.Tensor or None
                TD-error tensor used for prioritized replay, or ``None`` when
                prioritized replay is disabled.
            ``"actor_updated"`` : bool
                Whether the actor was updated at the current step.
            ``"actor_grad_norm"`` : float
                Global L2 norm of actor gradients before clipping.
            ``"critic_grad_norm"`` : float
                Global L2 norm of critic gradients before clipping.
            ``"alpha_grad_norm"`` : float
                Gradient norm of the entropy-temperature parameter.
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
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1_values, next_q2_values = self.target_critic(
                next_states,
                next_actions,
            )
            next_q_values = torch.min(next_q1_values, next_q2_values)
            target_q_values = rewards + (1.0 - dones) * self.gamma * (
                next_q_values - self.alpha * next_log_probs
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
        alpha_loss = torch.tensor(0.0, device=self.device)
        actor_grad_norm = 0.0
        alpha_grad_norm = 0.0
        actor_updated = False

        if self.timesteps % self.update_freq == 0:
            actor_updated = True

            sample_actions, log_probs = self.actor.sample(states)
            q1_values, q2_values = self.critic(states, sample_actions)
            q_val = torch.min(q1_values, q2_values)

            if self.prioritized_mode:
                actor_loss = -(weights * (q_val - self.alpha * log_probs)).mean()
            else:
                actor_loss = -(q_val - self.alpha * log_probs).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            actor_grad_norm = self._grad_norm(self.actor.parameters())

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )

            self.actor_optim.step()

            if self.adaptive_alpha_mode:
                alpha_loss = -(
                    self.log_alpha * (log_probs + self.target_entropy).detach()
                ).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_grad_norm = self._grad_norm([self.log_alpha])
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp().item()

            self.soft_update(self.critic, self.target_critic)

        entropy = torch.tensor(0.0, device=self.device)
        if self._all_finite(states):
            mu, std = self.actor(states)
            if self._all_finite(mu, std):
                entropy = Normal(mu, std).entropy().mean()

        return {
            "agent_timesteps": self.timesteps,
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": entropy.item(),
            "alpha": self.alpha,
            "log_alpha": self._log_alpha_value(),
            "td_error": td_error,
            "actor_updated": actor_updated,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "alpha_grad_norm": alpha_grad_norm,
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

        next_action, next_log_prob = self.actor.sample(next_state)
        next_q1_value, next_q2_value = self.target_critic(next_state, next_action)
        next_q_value = torch.min(next_q1_value, next_q2_value)
        target_q_value = reward + (1.0 - terminated) * self.gamma * (
            next_q_value - self.alpha * next_log_prob
        )

        q1_value, q2_value = self.critic(state, action)
        td_error1 = target_q_value - q1_value
        td_error2 = target_q_value - q2_value
        td_error = 0.5 * (td_error1.abs() + td_error2.abs())
        return td_error

    def save(self, save_path):
        """Save all SAC states to disk.

        Parameters
        ----------
        save_path : str or pathlib.Path
            File path where the checkpoint will be saved.

        Returns
        -------
        None
            The checkpoint is written to ``save_path``.
        """

        self._save_sac_checkpoint(save_path)

    def load(self, load_path):
        """Load all SAC states from disk.

        Parameters
        ----------
        load_path : str or pathlib.Path
            File path of the checkpoint to load.

        Returns
        -------
        None
            Actor, critic, target critic, optimizer, and optional
            entropy-temperature states are restored in place.
        """

        self._load_sac_checkpoint(load_path)