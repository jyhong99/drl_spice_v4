"""Neural-network building blocks used by agent implementations.

This module defines reusable PyTorch neural-network components for
reinforcement-learning agents, including deterministic policies,
categorical policies, Gaussian policies, value functions, Q-functions,
double Q-functions, and dueling Q-networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

MAX_LOG_STD = 2
MIN_LOG_STD = -20


class TanhBijector(object):
    """Bijective tanh transform helper for squashed Gaussian policies.

    This class provides the forward tanh transformation, its numerically
    stable inverse, and the log-probability correction term required when
    applying a tanh squashing function to Gaussian policy samples.

    Parameters
    ----------
    epsilon : float, optional
        Small positive constant added to the Jacobian correction term to
        prevent numerical instability from taking the logarithm of zero.
        The default is 1e-7.
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Apply the forward tanh transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor in the unconstrained pre-tanh space. This is
            typically a Gaussian policy sample with shape
            ``(..., action_dim)``.

        Returns
        -------
        torch.Tensor
            Tanh-squashed tensor with the same shape as ``x``. Each value
            is bounded in the range ``[-1, 1]``.
        """

        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """Compute the inverse hyperbolic tangent.

        This method maps values from the bounded tanh output space back
        to the unconstrained pre-tanh space.

        Parameters
        ----------
        x : torch.Tensor
            Tensor whose values should lie in the open interval ``(-1, 1)``.
            The shape is typically ``(..., action_dim)``.

        Returns
        -------
        torch.Tensor
            Inverse hyperbolic tangent of ``x`` with the same shape as the
            input tensor.
        """

        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """Map a squashed action back to the pre-tanh space.

        The input is clamped slightly away from ``-1`` and ``1`` to avoid
        numerical overflow in the inverse hyperbolic tangent.

        Parameters
        ----------
        y : torch.Tensor
            Tanh-squashed tensor, usually an action tensor with values in
            ``[-1, 1]`` and shape ``(..., action_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor in the unconstrained pre-tanh space with the same shape
            as ``y``.
        """

        eps = torch.finfo(y.dtype).eps
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the tanh Jacobian correction for log-probabilities.

        When a Gaussian sample is transformed by tanh, the probability
        density must be adjusted by the log absolute determinant of the
        Jacobian of the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Pre-tanh tensor, typically a Gaussian policy sample before
            squashing. The shape is usually ``(..., action_dim)``.

        Returns
        -------
        torch.Tensor
            Element-wise log-Jacobian correction term with the same shape
            as ``x``.
        """

        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)


def weights_init_(module: nn.Module) -> None:
    """Initialize linear-layer weights and biases.

    Xavier uniform initialization is applied to the weight matrix of each
    ``nn.Linear`` layer, and the corresponding bias vector is initialized
    to zero.

    Parameters
    ----------
    module : nn.Module
        PyTorch module to initialize. Initialization is only applied when
        ``module`` is an instance of ``nn.Linear``.

    Returns
    -------
    None
        This function modifies the given module in place and does not
        return a value.
    """

    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        torch.nn.init.constant_(module.bias, 0.0)


class MLPBase(nn.Module):
    """Base multilayer perceptron encoder.

    This class implements a shared feedforward feature extractor used by
    policy networks, value networks, and Q-networks.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_sizes : tuple[int, ...]
        Sizes of hidden layers. The first element is used for the input
        layer output size, and each subsequent element defines an
        additional hidden layer.
    activation : type[nn.Module] or callable
        Activation function constructor or callable applied after each
        linear layer. Examples include ``nn.ReLU`` and ``nn.Tanh``.
    """

    def __init__(self, input_dim, hidden_sizes, activation):
        super(MLPBase, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.activation = activation

    def mlp(self, state: torch.Tensor) -> torch.Tensor:
        """Encode input features through the MLP hidden stack.

        Parameters
        ----------
        state : torch.Tensor
            Input feature tensor. For RL models, this is usually a state
            or observation tensor with shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded feature tensor with shape
            ``(batch_size, hidden_sizes[-1])``.
        """

        x = self.activation(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return x


class MLPDeterministicPolicy(MLPBase):
    """Deterministic actor network for continuous action spaces.

    The network maps states directly to continuous actions and applies
    tanh squashing so that each action component lies in ``[-1, 1]``.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Dimension of the continuous action vector.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    """

    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super().__init__(state_dim, hidden_sizes, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute a deterministic continuous action.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Tanh-squashed action tensor with shape
            ``(batch_size, action_dim)``. Each action component lies in
            ``[-1, 1]``.
        """

        x = self.mlp(state)
        return torch.tanh(self.output_layer(x))


class MLPCategoricalPolicy(MLPBase):
    """Categorical policy network for discrete action spaces.

    The network outputs a probability distribution over discrete actions.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Number of discrete actions.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    epsilon : float, optional
        Minimum probability value used to avoid exact zero probabilities.
        The default is 1e-6.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        epsilon=1e-6,
    ):
        super(MLPCategoricalPolicy, self).__init__(state_dim, hidden_sizes, activation)
        self.epsilon = epsilon
        self.output_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Probability tensor over discrete actions with shape
            ``(batch_size, action_dim)``. Values are clamped to be at
            least ``epsilon``.
        """

        x = self.mlp(state)
        probs = F.softmax(self.output_layer(x), dim=-1)
        return probs.clamp_min(self.epsilon)

    def dist(self, state: torch.Tensor) -> Categorical:
        """Construct a categorical action distribution.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.distributions.Categorical
            Categorical distribution parameterized by the action
            probabilities produced by this policy.
        """

        return Categorical(self.forward(state))

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of discrete actions.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.
        action : torch.Tensor
            Discrete action tensor. Expected shape is ``(batch_size,)`` or
            ``(batch_size, 1)``. Values are converted to integer action
            indices.

        Returns
        -------
        torch.Tensor
            Log-probability tensor with shape ``(batch_size, 1)``.
        """

        dist = self.dist(state)
        if action.ndim > 1:
            action = action.squeeze(-1)
        return dist.log_prob(action.long()).unsqueeze(-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the mean entropy of the categorical policy.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the mean entropy over the batch.
        """

        dist = self.dist(state)
        return dist.entropy().mean()

    def sample(self, state: torch.Tensor) -> torch.Tensor:
        """Return normalized action probabilities.

        This method does not directly sample an action. Instead, it returns
        normalized probabilities so that the caller can perform sampling or
        action selection externally.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Normalized action-probability tensor with shape
            ``(batch_size, action_dim)``.
        """

        probs = self.forward(state)
        return probs / probs.sum(dim=-1, keepdim=True)


class MLPMultiCategoricalPolicy(MLPBase):
    """Factorized categorical policy for multi-discrete action spaces.

    The network has one categorical output head per action factor. This is
    useful when the action is represented as a vector of independent
    discrete choices.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dims : Sequence[int]
        Number of discrete choices for each action factor.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    epsilon : float, optional
        Minimum probability value used to avoid exact zero probabilities.
        The default is 1e-6.
    """

    def __init__(
        self,
        state_dim,
        action_dims,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        epsilon=1e-6,
    ):
        super(MLPMultiCategoricalPolicy, self).__init__(
            state_dim, hidden_sizes, activation
        )
        self.epsilon = epsilon
        self.output_layers = nn.ModuleList(
            [nn.Linear(hidden_sizes[-1], action_dim) for action_dim in action_dims]
        )
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Return action probabilities for each discrete action factor.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        list[torch.Tensor]
            List of probability tensors. The ``i``-th tensor has shape
            ``(batch_size, action_dims[i])`` and represents the categorical
            distribution over the ``i``-th action factor.
        """

        x = self.mlp(state)
        probs = [
            F.softmax(layer(x), dim=-1).clamp_min(self.epsilon)
            for layer in self.output_layers
        ]
        return [prob / prob.sum(dim=-1, keepdim=True) for prob in probs]

    def dist(self, state: torch.Tensor) -> list[Categorical]:
        """Construct categorical distributions for all action factors.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        list[torch.distributions.Categorical]
            List of categorical distributions, one for each discrete action
            factor.
        """

        probs = self.forward(state)
        return [Categorical(prob) for prob in probs]

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute joint log-probabilities for multi-discrete actions.

        The joint log-probability is computed as the sum of the
        log-probabilities of all factorized categorical distributions.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.
        action : torch.Tensor
            Multi-discrete action tensor. Expected shape is
            ``(batch_size, num_action_factors)``. If the shape is
            ``(batch_size,)``, it is expanded to ``(batch_size, 1)``.

        Returns
        -------
        torch.Tensor
            Joint log-probability tensor with shape ``(batch_size, 1)``.
        """

        dists = self.dist(state)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        log_probs = torch.stack(
            [dist.log_prob(a.long()) for dist, a in zip(dists, action.T)],
            dim=1,
        )
        return log_probs.sum(dim=1, keepdim=True)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the mean entropy across action factors and batch samples.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the mean entropy.
        """

        dists = self.dist(state)
        entropies = torch.stack([dist.entropy() for dist in dists], dim=1)
        return entropies.mean()

    def sample(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Return normalized action probabilities for all action factors.

        This method does not directly sample actions. Instead, it returns
        the probability tensors used for external sampling or action
        selection.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        list[torch.Tensor]
            List of normalized probability tensors, one for each action
            factor.
        """

        return self.forward(state)


class MLPGaussianPolicy(MLPBase):
    """Gaussian policy network with tanh-squashed continuous actions.

    The network predicts the mean and standard deviation of a Gaussian
    action distribution. Samples from this Gaussian are transformed by tanh
    to produce bounded continuous actions.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Dimension of the continuous action vector.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    epsilon : float, optional
        Small positive constant used for numerical stability in the
        standard deviation and tanh correction terms. The default is 1e-6.
    mu_scale : float, optional
        Scaling factor used to bound the predicted mean before sampling.
        The default is 5.0.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        epsilon=1e-6,
        mu_scale=5.0,
    ):
        super(MLPGaussianPolicy, self).__init__(state_dim, hidden_sizes, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.bijector = TanhBijector(epsilon)
        self.epsilon = epsilon
        self.mu_scale = mu_scale
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian mean and standard deviation.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        mu : torch.Tensor
            Mean tensor of the Gaussian action distribution with shape
            ``(batch_size, action_dim)``.
        std : torch.Tensor
            Standard-deviation tensor of the Gaussian action distribution
            with shape ``(batch_size, action_dim)``.
        """

        x = self.mlp(state)
        mu = self.mu_scale * torch.tanh(self.mu_layer(x) / self.mu_scale)
        log_std = torch.clamp(self.log_std_layer(x), MIN_LOG_STD, MAX_LOG_STD)
        std = F.softplus(log_std) + self.epsilon
        return mu, std

    def dist(self, state: torch.Tensor) -> Normal:
        """Construct the Gaussian action distribution.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.distributions.Normal
            Independent element-wise Gaussian distribution parameterized by
            the predicted mean and standard deviation.
        """

        mu, std = self.forward(state)
        return Normal(mu, std)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute corrected log-probabilities of tanh-squashed actions.

        The input action is first mapped back to the pre-tanh space. The
        Gaussian log-probability is then corrected using the tanh Jacobian
        term.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.
        action : torch.Tensor
            Tanh-squashed action tensor with shape
            ``(batch_size, action_dim)``. Values are expected to lie in
            ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Corrected log-probability tensor with shape
            ``(batch_size, 1)``.
        """

        dist = self.dist(state)
        x = self.bijector.inverse(action)
        log_prob = dist.log_prob(x)
        log_prob -= self.bijector.log_prob_correction(x)
        return log_prob.sum(dim=-1, keepdims=True)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the mean entropy of the Gaussian policy.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the mean entropy over all action
            dimensions and batch samples.
        """

        dist = self.dist(state)
        return dist.entropy().mean()

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tanh-squashed actions using reparameterization.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        action : torch.Tensor
            Tanh-squashed action tensor with shape
            ``(batch_size, action_dim)``.
        log_prob : torch.Tensor
            Corrected log-probability tensor of the sampled action with
            shape ``(batch_size, 1)``.
        """

        dist = self.dist(state)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample)
        log_prob -= self.bijector.log_prob_correction(sample)
        return self.bijector.forward(sample), log_prob.sum(-1, keepdim=True)


class MLPVFunction(MLPBase):
    """State-value function approximator.

    This network predicts a scalar value ``V(s)`` for each input state.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    """

    def __init__(self, state_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPVFunction, self).__init__(state_dim, hidden_sizes, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict scalar state values.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted state-value tensor with shape ``(batch_size, 1)``.
        """

        return self.output_layer(self.mlp(state))


class MLPQFunction(MLPBase):
    """State-action value function approximator.

    This network predicts a scalar Q-value ``Q(s, a)`` for each
    state-action pair.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Dimension of the action vector.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    """

    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPQFunction, self).__init__(state_dim + action_dim, hidden_sizes, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict scalar Q-values for state-action pairs.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.
        action : torch.Tensor
            Action tensor with shape ``(batch_size, action_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted Q-value tensor with shape ``(batch_size, 1)``.
        """

        x = torch.cat([state, action], 1)
        return self.output_layer(self.mlp(x))


class MLPDoubleQFunction(nn.Module):
    """Pair of independent state-action Q-functions.

    This module is commonly used in double-critic algorithms such as TD3
    and SAC to reduce overestimation bias.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Dimension of the action vector.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes used by each Q-function. The default is
        ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    """

    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPDoubleQFunction, self).__init__()
        self.q1 = MLPQFunction(state_dim, action_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(state_dim, action_dim, hidden_sizes, activation)
        self.apply(weights_init_)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independent Q-value estimates.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.
        action : torch.Tensor
            Action tensor with shape ``(batch_size, action_dim)``.

        Returns
        -------
        q1_value : torch.Tensor
            First Q-value estimate with shape ``(batch_size, 1)``.
        q2_value : torch.Tensor
            Second Q-value estimate with shape ``(batch_size, 1)``.
        """

        return self.q1(state, action), self.q2(state, action)


class MLPQNetwork(MLPBase):
    """Discrete-action Q-network with optional dueling architecture.

    The network predicts one Q-value for each discrete action. When
    ``dueling_mode`` is enabled, it separately estimates a scalar state
    value and per-action advantages.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Number of discrete actions.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    dueling_mode : bool, optional
        If ``True``, use dueling value and advantage heads. If ``False``,
        use a single output head. The default is ``False``.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        dueling_mode=False,
    ):
        super(MLPQNetwork, self).__init__(state_dim, hidden_sizes, activation)
        self.dueling_mode = dueling_mode

        if dueling_mode:
            self.value_layer = nn.Linear(hidden_sizes[-1], 1)
            self.advantage_layer = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.output_layer = nn.Linear(hidden_sizes[-1], action_dim)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict per-action Q-values.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Q-value tensor with shape ``(batch_size, action_dim)``.
        """

        x = self.mlp(state)

        if self.dueling_mode:
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)
            output = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            output = self.output_layer(x)

        return output


class MLPDoubleQNetwork(nn.Module):
    """Pair of discrete-action Q-networks.

    This module returns two independent Q-value vectors for each state.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Number of discrete actions.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes used by each Q-network. The default is
        ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    dueling_mode : bool, optional
        If ``True``, each Q-network uses a dueling architecture. The
        default is ``False``.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        dueling_mode=False,
    ):
        super(MLPDoubleQNetwork, self).__init__()
        self.q1 = MLPQNetwork(state_dim, action_dim, hidden_sizes, activation, dueling_mode)
        self.q2 = MLPQNetwork(state_dim, action_dim, hidden_sizes, activation, dueling_mode)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independent per-action Q-value tensors.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        q1_values : torch.Tensor
            First per-action Q-value tensor with shape
            ``(batch_size, action_dim)``.
        q2_values : torch.Tensor
            Second per-action Q-value tensor with shape
            ``(batch_size, action_dim)``.
        """

        return self.q1(state), self.q2(state)


class MLPMultiQNetwork(MLPBase):
    """Q-network for multi-discrete action spaces.

    The network predicts Q-values for multiple discrete action factors.
    The output tensors for all action factors are concatenated along the
    last dimension. When ``dueling_mode`` is enabled, a shared state-value
    head and factor-specific advantage heads are used.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dims : Sequence[int]
        Number of discrete choices for each action factor.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the MLP encoder. The default is ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    dueling_mode : bool, optional
        If ``True``, use a dueling architecture with a shared value head
        and factor-specific advantage heads. The default is ``False``.
    """

    def __init__(
        self,
        state_dim,
        action_dims,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        dueling_mode=False,
    ):
        super(MLPMultiQNetwork, self).__init__(state_dim, hidden_sizes, activation)
        self.dueling_mode = dueling_mode

        if dueling_mode:
            self.value_layer = nn.Linear(hidden_sizes[-1], 1)
            self.advantage_layers = nn.ModuleList(
                [nn.Linear(hidden_sizes[-1], dim) for dim in action_dims]
            )
        else:
            self.output_layers = nn.ModuleList(
                [nn.Linear(hidden_sizes[-1], dim) for dim in action_dims]
            )

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict concatenated Q-values for multi-discrete actions.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        torch.Tensor
            Concatenated Q-value tensor with shape
            ``(batch_size, sum(action_dims))``.
        """

        x = self.mlp(state)

        if self.dueling_mode:
            value = self.value_layer(x)
            advantages = [adv_layer(x) for adv_layer in self.advantage_layers]
            advantage = torch.cat(advantages, dim=-1)
            output = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            outputs = [layer(x) for layer in self.output_layers]
            output = torch.cat(outputs, dim=-1)

        return output


class MLPMultiDoubleQNetwork(nn.Module):
    """Pair of multi-discrete Q-networks.

    This module returns two independent concatenated Q-value tensors for
    multi-discrete action spaces.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dims : Sequence[int]
        Number of discrete choices for each action factor.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes used by each Q-network. The default is
        ``(64, 64)``.
    activation : type[nn.Module] or callable, optional
        Activation function used after each hidden linear layer. The
        default is ``nn.ReLU``.
    dueling_mode : bool, optional
        If ``True``, each Q-network uses a dueling architecture. The
        default is ``False``.
    """

    def __init__(
        self,
        state_dim,
        action_dims,
        hidden_sizes=(64, 64),
        activation=nn.ReLU,
        dueling_mode=False,
    ):
        super(MLPMultiDoubleQNetwork, self).__init__()
        self.q1 = MLPMultiQNetwork(
            state_dim, action_dims, hidden_sizes, activation, dueling_mode
        )
        self.q2 = MLPMultiQNetwork(
            state_dim, action_dims, hidden_sizes, activation, dueling_mode
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independent multi-discrete Q-value tensors.

        Parameters
        ----------
        state : torch.Tensor
            State tensor with shape ``(batch_size, state_dim)``.

        Returns
        -------
        q1_values : torch.Tensor
            First concatenated Q-value tensor with shape
            ``(batch_size, sum(action_dims))``.
        q2_values : torch.Tensor
            Second concatenated Q-value tensor with shape
            ``(batch_size, sum(action_dims))``.
        """

        return self.q1(state), self.q2(state)
