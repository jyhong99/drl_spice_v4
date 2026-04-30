"""Action-noise processes used by continuous-control agents.

This module defines stochastic exploration-noise processes commonly used
in continuous-control reinforcement-learning algorithms such as DDPG and
TD3. It includes independent Gaussian noise and temporally correlated
Ornstein-Uhlenbeck noise.
"""

import torch
import numpy as np


class GaussianNoise(object):
    """Gaussian action-noise generator.

    This class generates independent Gaussian noise samples. It is mainly
    used for exploration in continuous-action reinforcement-learning agents.

    Each call to :meth:`sample` returns a tensor sampled from

    ``N(mu, sigma^2)``

    with the configured output shape.

    Parameters
    ----------
    size : int or tuple[int, ...]
        Shape of the sampled noise tensor. If an integer is provided, the
        output shape becomes ``(size,)``. If a tuple is provided, it should
        directly represent the desired tensor shape.
    mu : float, optional
        Mean of the Gaussian distribution. The default is 0.0.
    sigma : float, optional
        Standard deviation of the Gaussian distribution. The default is 0.2.
    device : str or torch.device, optional
        Device on which the sampled noise tensor is allocated. Examples are
        ``"cpu"``, ``"cuda"``, or ``torch.device("cuda")``. The default is
        ``"cpu"``.

    Attributes
    ----------
    size : tuple[int, ...]
        Shape of the generated noise tensor.
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.
    device : str or torch.device
        Device used for tensor allocation.
    """

    def __init__(self, size, mu=0.0, sigma=0.2, device='cpu'):
        """Initialize the Gaussian noise generator.

        Parameters
        ----------
        size : int or tuple[int, ...]
            Shape of each generated noise sample. If ``size`` is an integer,
            it is converted to ``(size,)``. If ``size`` is already a tuple,
            it is used as the sample shape.
        mu : float, optional
            Mean of the Gaussian distribution. The default is 0.0.
        sigma : float, optional
            Standard deviation of the Gaussian distribution. Larger values
            produce stronger exploration noise. The default is 0.2.
        device : str or torch.device, optional
            Device on which generated noise tensors are allocated. The
            default is ``"cpu"``.

        Returns
        -------
        None
            This constructor initializes the noise generator in place.
        """

        self.size = (size,) if isinstance(size, int) else tuple(size)
        self.mu = mu
        self.sigma = sigma
        self.device = device

    def sample(self):
        """Sample an independent Gaussian noise tensor.

        Returns
        -------
        torch.Tensor
            Noise tensor sampled from a Gaussian distribution with mean
            ``mu`` and standard deviation ``sigma``. The returned tensor has
            shape ``self.size`` and is allocated on ``self.device``.
        """

        noise = torch.normal(
            mean=self.mu,
            std=self.sigma,
            size=self.size,
            device=self.device,
        )
        return noise


class OrnsteinUhlenbeckNoise(object):
    """Ornstein-Uhlenbeck noise process.

    This class implements the Ornstein-Uhlenbeck process, which generates
    temporally correlated noise. It is often used in deterministic
    continuous-control algorithms, especially DDPG, to encourage smoother
    exploration trajectories.

    The process is updated using an Euler-Maruyama discretization:

    ``dx = theta * (mu - x) * dt + sigma * sqrt(dt) * N(0, 1)``

    where ``x`` is the current noise state.

    Parameters
    ----------
    size : int or tuple[int, ...]
        Shape of the internal OU process state. If an integer is provided,
        the state shape becomes ``(size,)``. If a tuple is provided, it is
        used directly as the state shape.
    mu : float, optional
        Long-run mean of the OU process. The process tends to drift back
        toward this value over time. The default is 0.0.
    theta : float, optional
        Mean-reversion coefficient. Larger values make the process return
        to ``mu`` more quickly. The default is 0.15.
    sigma : float, optional
        Diffusion coefficient controlling the stochastic noise amplitude.
        Larger values produce more random fluctuations. The default is 0.2.
    dt : float, optional
        Discrete time step used for process updates. The default is 1e-2.
    device : str or torch.device, optional
        Device on which the internal state and sampled tensors are stored.
        The default is ``"cpu"``.

    Attributes
    ----------
    size : tuple[int, ...]
        Shape of the OU process state.
    mu : torch.Tensor
        Tensor containing the long-run mean of the process. Its shape is
        ``self.size``.
    theta : float
        Mean-reversion strength.
    sigma : float
        Diffusion coefficient.
    dt : float
        Discrete time step.
    device : str or torch.device
        Device used for tensor allocation.
    noise : torch.Tensor
        Current internal OU process state.
    """

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, device='cpu'):
        """Initialize the Ornstein-Uhlenbeck noise process.

        Parameters
        ----------
        size : int or tuple[int, ...]
            Shape of the internal noise state. If ``size`` is an integer,
            it is converted to ``(size,)``. If ``size`` is a tuple, it is
            used directly.
        mu : float, optional
            Long-run mean of the OU process. The internal state is reset to
            this value when :meth:`reset` is called. The default is 0.0.
        theta : float, optional
            Mean-reversion coefficient. Higher values increase the speed at
            which the process returns toward ``mu``. The default is 0.15.
        sigma : float, optional
            Diffusion coefficient controlling the scale of the random
            perturbation at each step. The default is 0.2.
        dt : float, optional
            Time-step size used in the discrete OU update equation. The
            default is 1e-2.
        device : str or torch.device, optional
            Device on which the OU state and random samples are allocated.
            The default is ``"cpu"``.

        Returns
        -------
        None
            This constructor initializes the process parameters and resets
            the internal noise state.
        """

        self.size = (size,) if isinstance(size, int) else tuple(size)
        self.mu = torch.full(self.size, mu, device=device)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.device = device
        self.reset()

    def reset(self):
        """Reset the internal OU process state.

        The internal state is reset to the long-run mean ``mu``. This is
        usually called at the beginning of each episode so that exploration
        noise starts from a stable initial value.

        Returns
        -------
        None
            The internal noise state ``self.noise`` is overwritten in place.
        """

        self.noise = torch.clone(self.mu)

    def sample(self):
        """Advance the OU process by one step and return the new noise state.

        The update follows the discrete Ornstein-Uhlenbeck equation:

        ``x_{t+1} = x_t + theta * (mu - x_t) * dt
        + sigma * sqrt(dt) * N(0, 1)``

        Returns
        -------
        torch.Tensor
            Updated OU noise tensor with shape ``self.size``. The returned
            tensor is also stored internally as ``self.noise``.
        """

        x = self.noise
        dx = (
            self.theta * (self.mu - x) * self.dt
            + self.sigma
            * torch.sqrt(torch.tensor(self.dt, device=self.device))
            * torch.randn_like(x)
        )
        self.noise = x + dx
        return self.noise