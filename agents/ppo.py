"""Proximal Policy Optimization agent."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from agents.common.network import MLPGaussianPolicy, MLPVFunction
from agents.common.policy import OnPolicyAlgorithm, to_cpu_serializable


class PPO(OnPolicyAlgorithm):
    """Proximal Policy Optimization agent for continuous action spaces.

    This class implements PPO using a Gaussian actor policy, a value-function
    critic, generalized advantage estimation, clipped policy updates, optional
    value-function clipping, entropy regularization, and optional KL-based
    early stopping.

    Parameters
    ----------
    env : gym.Env
        Environment instance used to infer observation and action dimensions.
        The action space is expected to be continuous.
    **config : dict[str, object]
        Configuration dictionary containing network sizes, activation
        functions, rollout-buffer settings, optimizer settings, advantage
        estimation options, and PPO-specific optimization parameters.

    Attributes
    ----------
    actor : MLPGaussianPolicy
        Gaussian policy network used to sample continuous actions.
    critic : MLPVFunction
        Value-function network used to estimate state values.
    actor_optim : torch.optim.Adam
        Optimizer for the actor network.
    critic_optim : torch.optim.Adam
        Optimizer for the critic network.
    train_iters : int
        Number of optimization epochs performed over each rollout batch.
    batch_size : int
        Minibatch size used during PPO optimization.
    clip_range : float
        Clipping range for the PPO policy-ratio objective.
    clip_range_vf : float or None
        Optional clipping range for value-function updates. If ``None``,
        value-function clipping is disabled.
    target_kl : float or None
        Optional KL-divergence threshold for early stopping. If ``None``,
        KL-based early stopping is disabled.
    max_grad_norm : float or None
        Maximum gradient norm for gradient clipping. If ``None``, gradient
        clipping is disabled.
    config : dict[str, object]
        Original configuration dictionary passed to the constructor.
    """

    def __init__(self, env, **config):
        """Initialize the PPO agent.

        Parameters
        ----------
        env : gym.Env
            Environment instance used for training and dimension inference.
        **config : dict[str, object]
            Configuration options for the agent.

            Supported keys include:

            ``actor_size`` : tuple[int, ...], optional
                Hidden-layer sizes of the actor network. The default is
                ``(128, 128)``.
            ``critic_size`` : tuple[int, ...], optional
                Hidden-layer sizes of the critic network. The default is
                ``(128, 128)``.
            ``actor_activation`` : callable, optional
                Activation function used by the actor network. The default is
                ``torch.tanh``.
            ``critic_activation`` : callable, optional
                Activation function used by the critic network. The default is
                ``torch.tanh``.
            ``buffer_size`` : int, optional
                Rollout-buffer capacity. The default is ``2048``.
            ``update_after`` : int, optional
                Number of collected transitions required before learning
                starts. The default is ``2048``.
            ``actor_lr`` : float, optional
                Learning rate for the actor optimizer. The default is
                ``3e-4``.
            ``critic_lr`` : float, optional
                Learning rate for the critic optimizer. The default is
                ``3e-4``.
            ``gamma`` : float, optional
                Discount factor. The default is ``0.99``.
            ``lmda`` : float, optional
                Generalized advantage estimation coefficient. The default is
                ``0.95``.
            ``vf_coef`` : float, optional
                Coefficient applied to the value-function loss. The default is
                ``0.5``.
            ``ent_coef`` : float, optional
                Coefficient applied to the policy entropy bonus. The default
                is ``0.01``.
            ``adv_norm`` : bool, optional
                Whether to normalize advantages before optimization. The
                default is ``True``.
            ``train_iters`` : int, optional
                Number of training epochs over the rollout batch. The default
                is ``10``.
            ``batch_size`` : int, optional
                Minibatch size used during optimization. The default is
                ``256``.
            ``clip_range`` : float, optional
                PPO policy-ratio clipping range. The default is ``0.2``.
            ``clip_range_vf`` : float or None, optional
                Value-function clipping range. If ``None``, value clipping is
                disabled. The default is ``None``.
            ``target_kl`` : float or None, optional
                KL threshold for early stopping. If ``None``, KL-based early
                stopping is disabled. The default is ``None``.
            ``max_grad_norm`` : float or None, optional
                Maximum gradient norm for clipping. The default is ``0.5``.

        Returns
        -------
        None
            The constructor initializes shared on-policy state, actor network,
            critic network, and optimizers in place.
        """

        super().__init__(
            env=env,
            actor_size=config.get("actor_size", (128, 128)),
            critic_size=config.get("critic_size", (128, 128)),
            actor_activation=config.get("actor_activation", torch.tanh),
            critic_activation=config.get("critic_activation", torch.tanh),
            buffer_size=config.get("buffer_size", 2048),
            update_after=config.get("update_after", 2048),
            actor_lr=config.get("actor_lr", 3e-4),
            critic_lr=config.get("critic_lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            lmda=config.get("lmda", 0.95),
            vf_coef=config.get("vf_coef", 0.5),
            ent_coef=config.get("ent_coef", 0.01),
            adv_norm=config.get("adv_norm", True),
        )

        self.train_iters = config.get("train_iters", 10)
        self.batch_size = config.get("batch_size", 256)
        self.clip_range = config.get("clip_range", 0.2)
        self.clip_range_vf = config.get("clip_range_vf", None)
        self.target_kl = config.get("target_kl", None)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.config = config

        self.actor = MLPGaussianPolicy(
            self.state_dim,
            self.action_dim,
            self.actor_size,
            self.actor_activation,
        ).to(self.device)

        self._initialize_ppo_modules()

    def _initialize_ppo_modules(self):
        """Build the critic network and optimizers.

        This helper is called after the actor is constructed. It initializes
        the value-function critic and creates separate Adam optimizers for the
        actor and critic networks.

        Returns
        -------
        None
            The critic and optimizers are created and assigned in place.
        """

        self.critic = MLPVFunction(
            self.state_dim,
            self.critic_size,
            self.critic_activation,
        ).to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

    def learn(self, states, actions, rewards, next_states, dones, truncateds=None):
        """Perform one PPO training phase over a rollout batch.

        This method computes value estimates, return targets, advantages, and
        old action log-probabilities, then performs multiple minibatch
        optimization epochs using PPO's clipped objective. It returns scalar
        diagnostics summarizing the update.

        Parameters
        ----------
        states : torch.Tensor
            Batch of current states with shape ``(batch_size, state_dim)``.
        actions : torch.Tensor
            Batch of sampled actions with shape ``(batch_size, action_dim)``.
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
            interface compatibility but is not used directly in this method.
            The default is ``None``.

        Returns
        -------
        dict[str, object]
            Dictionary containing PPO update diagnostics.

            ``"agent_timesteps"`` : int
                Current agent timestep counter.
            ``"actor_loss"`` : float
                Mean actor loss across processed minibatches.
            ``"critic_loss"`` : float
                Mean critic loss across processed minibatches.
            ``"total_loss"`` : float
                Mean combined optimization loss.
            ``"entropy"`` : float
                Mean policy entropy.
            ``"clip_frac"`` : float
                Mean fraction of samples whose probability ratio exceeded the
                clipping range.
            ``"approx_kl"`` : float
                Mean approximate KL divergence between old and new policies.
            ``"actor_grad_norm"`` : float
                Global L2 norm of actor gradients before clipping.
            ``"critic_grad_norm"`` : float
                Global L2 norm of critic gradients before clipping.
            ``"mean_value"`` : float
                Mean value estimate from the last processed minibatch.
            ``"mean_return_target"`` : float
                Mean return target over the rollout batch.
            ``"mean_advantage"`` : float
                Mean advantage over the rollout batch.
            ``"early_stop"`` : bool
                Whether KL-based early stopping was triggered.
            ``"completed_train_iters"`` : int
                Number of completed full training epochs.
            ``"completed_minibatches"`` : int
                Number of processed minibatches.
        """

        self.actor.train()
        self.critic.train()

        with torch.no_grad():
            values, next_values = self.critic(states), self.critic(next_states)
            rets, advs = self.GAE(values, next_values, rewards, dones)
            log_prob_olds = self.actor.log_prob(states, actions)

        continue_training = True
        actor_losses, critic_losses, entropies = [], [], []
        losses, clip_fracs, approx_kls = [], [], []
        completed_train_iters = 0
        completed_minibatches = 0
        early_stop = False
        actor_grad_norm = 0.0
        critic_grad_norm = 0.0

        dataset = TensorDataset(states, actions, values, rets, advs, log_prob_olds)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(self.train_iters):
            for batch in dataloader:
                (
                    batch_states,
                    batch_actions,
                    batch_values,
                    batch_rets,
                    batch_advs,
                    batch_log_prob_olds,
                ) = batch

                log_probs = self.actor.log_prob(batch_states, batch_actions)
                ratios = (log_probs - batch_log_prob_olds).exp()

                surr1 = batch_advs * ratios
                surr2 = batch_advs * torch.clamp(
                    ratios,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_losses.append(actor_loss.item())

                clip_frac = torch.mean(
                    (torch.abs(ratios - 1.0) > self.clip_range).float()
                )
                clip_fracs.append(clip_frac.item())

                values = self.critic(batch_states)

                if self.clip_range_vf is not None:
                    clipped_values = batch_values + torch.clamp(
                        values - batch_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    critic_loss1 = F.mse_loss(values, batch_rets)
                    critic_loss2 = F.mse_loss(clipped_values, batch_rets)
                    critic_loss = torch.max(critic_loss1, critic_loss2)
                else:
                    critic_loss = F.mse_loss(values, batch_rets)

                critic_losses.append(critic_loss.item())

                entropy = self.actor.entropy(batch_states)
                entropies.append(entropy.item())

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                losses.append(loss.item())

                with torch.no_grad():
                    log_ratios = log_probs - batch_log_prob_olds
                    approx_kl = torch.mean(
                        (torch.exp(log_ratios) - 1.0) - log_ratios
                    ).cpu().numpy()
                    approx_kls.append(approx_kl)

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                loss.backward()

                actor_grad_norm = self._grad_norm(self.actor.parameters())
                critic_grad_norm = self._grad_norm(self.critic.parameters())

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        self.max_grad_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.max_grad_norm,
                    )

                self.actor_optim.step()
                self.critic_optim.step()

                completed_minibatches += 1

            if not continue_training:
                early_stop = True
                break

            completed_train_iters += 1

        if continue_training and completed_minibatches > 0:
            completed_train_iters = self.train_iters

        return {
            "agent_timesteps": self.timesteps,
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "total_loss": np.mean(losses),
            "entropy": np.mean(entropies),
            "clip_frac": np.mean(clip_fracs),
            "approx_kl": np.mean(approx_kls),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "mean_value": self._tensor_mean(values),
            "mean_return_target": self._tensor_mean(rets),
            "mean_advantage": self._tensor_mean(advs),
            "early_stop": early_stop,
            "completed_train_iters": completed_train_iters,
            "completed_minibatches": completed_minibatches,
        }

    def save(self, save_path):
        """Save actor, critic, and optimizer states to disk.

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
        """Load actor, critic, and optimizer states from disk.

        Parameters
        ----------
        load_path : str or pathlib.Path
            File path of the checkpoint to load.

        Returns
        -------
        None
            Actor, critic, actor optimizer, and critic optimizer states are
            restored in place.
        """

        checkpoint = torch.load(
            load_path,
            map_location=self.device,
            weights_only=True,
        )

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])

    def __str__(self):
        """Return the display name of the algorithm.

        Returns
        -------
        str
            Algorithm name, ``"PPO_Continuous"``.
        """

        return "PPO_Continuous"

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
        """Sample or predict an action from the Gaussian policy.

        During training, an action is sampled from the Gaussian policy. During
        evaluation, the policy mean is used directly. The selected action is
        squashed by ``tanh`` before being returned.

        Parameters
        ----------
        state : array-like or torch.Tensor
            Current environment observation. Expected shape is ``(state_dim,)``
            for a single observation or ``(batch_size, state_dim)`` for batched
            observations.
        training : bool, optional
            Whether to sample stochastically from the policy. If ``True``, an
            action is sampled from the Gaussian distribution. If ``False``,
            the policy mean is used. The default is ``True``.
        global_buffer_size : int or None, optional
            Optional global buffer size for interface compatibility with
            distributed runtime code. This argument is not used by this method.
            The default is ``None``.

        Returns
        -------
        numpy.ndarray
            Tanh-squashed action tensor converted to a CPU NumPy array. For a
            single state, the shape is typically ``(action_dim,)``.
        """

        self.actor.train(training)

        state = torch.FloatTensor(state).to(self.device)
        mu, std = self.actor(state)
        action = torch.normal(mu, std) if training else mu
        squashed = torch.tanh(action)

        self.last_action_info = {
            "used_random_action": False,
            "training_action": bool(training),
            "policy_mode": "gaussian",
            "action_norm": float(squashed.norm().item()),
            "policy_mean_norm": float(mu.norm().item()),
            "policy_std_mean": float(std.mean().item()),
        }

        return squashed.cpu().numpy()